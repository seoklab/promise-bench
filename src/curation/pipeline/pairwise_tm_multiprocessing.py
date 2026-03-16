import itertools
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import click
import joblib
import numpy as np
from joblib import Parallel, delayed
from nuri.tools import tm
from nuri.tools.chimera import match_maker
from tqdm import tqdm

from ..utils._config import pipeline_cfg as C
from ..utils._data_root import DataRootCommand
from ..utils.typedefs import TMScoreResult


def _score(
    query_data,
    templ_data,
):
    qn, qc, query_map, query_len = _unpack_entry(query_data)
    tn, tc, target_map, target_len = _unpack_entry(templ_data)

    lshort, llong = sorted((query_len, target_len))
    if lshort == 0 or (llong / lshort) > 1.5:
        return 0.0, float("inf")

    lmin = min(len(query_map), len(target_map))

    common = sorted(set(query_map.keys()) & set(target_map.keys()))
    if len(common) < lmin * 0.8:
        return 0.0, float("inf")

    qi = [query_map[i] for i in common]
    ti = [target_map[i] for i in common]
    qc_common = qc[qi]
    tc_common = tc[ti]

    try:
        mm_result = match_maker(qc_common, tc_common)
    except RuntimeError:
        print(f"MATCH_MAKER_FAILED\t{qn}\t{tn}", flush=True)
        return 0.0, float("inf")
    alignment = [(i, i) for i in mm_result.selected]

    # Transform query coordinates and calculate full RMSD
    qc_homogeneous = np.hstack([qc_common, np.ones((len(qc_common), 1))])
    qc_transformed = (mm_result.transform @ qc_homogeneous.T).T[:, :3]
    full_rmsd = np.sqrt(np.mean(np.sum((qc_transformed - tc_common) ** 2, axis=1)))

    _, tms = tm.tm_score(qc_common, tc_common, alignment, l_norm=lmin)
    return tms, full_rmsd


def _unpack_entry(data):
    if len(data) == 4:
        chain, coords, align_map, seq_len = data
        return chain, coords, align_map, int(seq_len)

    chain, coords, align_map = data
    return chain, coords, align_map, len(align_map)


def _coord_len(data) -> int:
    return len(data[1])


def process_group(group: Path, scores: Path, tmpdir: Path):
    cluster = np.load(group, allow_pickle=True)["coords"]
    cluster = cluster[[_coord_len(entry) >= 100 for entry in cluster]]
    if len(cluster) < 2:
        return None

    if len(cluster) >= 100:
        joblib.dump(cluster, tmpdir / f"{group.stem}.npy")
        return group.stem

    pairs = [
        _score(cluster[i], cluster[j])
        for i, j in itertools.combinations(range(len(cluster)), 2)
    ]
    tm_scores = np.array([p[0] for p in pairs], dtype=np.float32)
    rmsds = np.array([p[1] for p in pairs], dtype=np.float32)

    result = TMScoreResult(
        chains=np.array([c for c, *_ in cluster], dtype=np.str_),
        tm_scores=tm_scores,
        rmsds=rmsds,
    )

    scores.parent.mkdir(exist_ok=True)
    np.savez_compressed(scores, **asdict(result))

    return None


def _score_inplace(
    pool, tm_out: np.memmap, rmsd_out: np.memmap, i: int, j: int, k: int
):
    tms, rmsd = _score(pool[i], pool[j])
    tm_out[k] = tms
    rmsd_out[k] = rmsd


def run_tmscore_parallel(
    center: str,
    scores: Path,
    tmpdir: Path,
    nproc: int,
):
    cluster = joblib.load(tmpdir / f"{center}.npy", mmap_mode="r")

    n_pairs = len(cluster) * (len(cluster) - 1) // 2

    pscores = np.memmap(
        tmpdir / "pscores.npy",
        dtype=np.float32,
        mode="w+",
        shape=n_pairs,
    )
    prmsds = np.memmap(
        tmpdir / "prmsds.npy",
        dtype=np.float32,
        mode="w+",
        shape=n_pairs,
    )

    Parallel(n_jobs=nproc, batch_size=n_pairs // nproc, backend="multiprocessing")(
        delayed(_score_inplace)(cluster, pscores, prmsds, i, j, k)
        for k, (i, j) in enumerate(itertools.combinations(range(len(cluster)), 2))
    )

    result = TMScoreResult(
        chains=np.array([c for c, *_ in cluster], dtype=np.str_),
        tm_scores=pscores,
        rmsds=prmsds,
    )

    scores.parent.mkdir(exist_ok=True)
    np.savez_compressed(scores, **asdict(result))


@click.command(cls=DataRootCommand)
@click.option("--nproc", "-n", type=int, default=8)
@click.option(
    "--scores",
    type=click.Path(file_okay=False, path_type=Path),
    default=C.dir("scores"),
    show_default=True,
)
@click.argument(
    "coords",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=str(C.dir("coords")),
)
def main(nproc: int, scores: Path, coords: Path):
    for key in [
        "NUMEXPR_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ]:
        os.environ.pop(key, None)

    scores.mkdir(exist_ok=True, parents=True)

    groups = list(coords.rglob("*.npz"))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        large_groups = Parallel(n_jobs=nproc, backend="multiprocessing")(
            delayed(process_group)(
                group,
                scores / group.stem[1:3] / f"{group.stem}.npz",
                tmpdir,
            )
            for group in tqdm(groups)
        )
        large_groups = list(filter(None, large_groups))

        for center in tqdm(large_groups):
            run_tmscore_parallel(
                center,
                scores / center[1:3] / f"{center}.npz",
                tmpdir,
                nproc,
            )


if __name__ == "__main__":
    main()

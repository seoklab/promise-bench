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
from scipy.spatial import distance as D
from tqdm import tqdm

from ..utils.typedefs import TMScoreResult


def _tmscore(
    query_data: tuple[str, np.ndarray, dict[int, int]],
    templ_data: tuple[str, np.ndarray, dict[int, int]],
):
    _, qc, query_map = query_data
    _, tc, target_map = templ_data

    lmin = min(len(query_map), len(target_map))

    common = sorted(set(query_map.keys()) & set(target_map.keys()))
    alignment = [(query_map[i], target_map[i]) for i in common]
    if len(alignment) < lmin * 0.8:
        return 0

    _, tm_score = tm.tm_score(qc, tc, alignment, l_norm=lmin)
    return tm_score


def process_group(group: Path, scores: Path, tmpdir: Path):
    cluster = np.load(group, allow_pickle=True)["coords"]
    cluster = cluster[[len(crd) >= 100 for _, crd, _ in cluster]]
    if len(cluster) < 2:
        return None

    if len(cluster) >= 100:
        joblib.dump(cluster, tmpdir / f"{group.stem}.npy")
        return group.stem

    result = TMScoreResult(
        chains=np.array([c for c, *_ in cluster], dtype=np.str_),
        tm_scores=D.pdist(cluster, _tmscore),
    )

    scores.parent.mkdir(exist_ok=True)
    np.savez_compressed(scores, **asdict(result))

    return None


def _tmscore_inplace(pool, output: np.memmap, i: int, j: int, k: int):
    output[k] = _tmscore(pool[i], pool[j])


def run_tmscore_parallel(
    center: str,
    scores: Path,
    tmpdir: Path,
    nproc: int,
):
    cluster = joblib.load(tmpdir / f"{center}.npy", mmap_mode="r")

    pscores = np.memmap(
        tmpdir / "pscores.npy",
        dtype=np.float32,
        mode="w+",
        shape=len(cluster) * (len(cluster) - 1) // 2,
    )

    Parallel(n_jobs=nproc, batch_size=len(pscores) // nproc)(
        delayed(_tmscore_inplace)(cluster, pscores, i, j, k)
        for k, (i, j) in enumerate(itertools.combinations(range(len(cluster)), 2))
    )

    result = TMScoreResult(
        chains=np.array([c for c, *_ in cluster], dtype=np.str_),
        tm_scores=pscores,
    )

    scores.parent.mkdir(exist_ok=True)
    np.savez_compressed(scores, **asdict(result))


@click.command()
@click.option("--nproc", "-n", type=int, default=8)
@click.option(
    "--scores",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/scores"),
    show_default=True,
)
@click.argument(
    "coords",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default="data/coords",
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

        large_groups = Parallel(n_jobs=nproc)(
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

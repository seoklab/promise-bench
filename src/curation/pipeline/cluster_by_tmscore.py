from pathlib import Path

import click
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import distance as D
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from utils._config import pipeline_cfg as C
from utils._data_root import DataRootCommand
from ..utils.typedefs import TMScoreResult


def cluster_by_tmscore(
    tmscore_result: Path,
    clusters: Path,
    cutoff: float = 0.8,
):
    data = TMScoreResult(**np.load(tmscore_result))
    scoremap = D.squareform(data.tm_scores)
    rmsdmap = D.squareform(data.rmsds)

    cluster = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.2,
        metric="precomputed",
        linkage="complete",
    ).fit(1 - scoremap)

    df = pd.DataFrame.from_dict(
        {"chain": data.chains, "label": cluster.labels_}
    ).sort_values("label")

    clusters.parent.mkdir(exist_ok=True)
    df.to_csv(clusters, index=False)

    # Only consider pairs from different clusters
    diff_cluster = cluster.labels_[:, None] != cluster.labels_[None, :]
    upper = np.triu(np.ones_like(scoremap, dtype=bool), k=1)
    base_mask = diff_cluster & upper

    # 1st pass: TM < 0.8
    mask1 = base_mask & (scoremap < cutoff) & (scoremap > 0)

    # 2nd pass: 0.8 < TM < 0.9 and RMSD > 3.0
    mask2 = base_mask & (scoremap > cutoff) & (scoremap < 0.90) & (rmsdmap > 3.0)

    x1, y1 = np.nonzero(mask1)
    x2, y2 = np.nonzero(mask2)

    df_main = None
    df_high = None

    if len(x1) > 0:
        # Ensure cluster-x < cluster-y
        swap = cluster.labels_[x1] > cluster.labels_[y1]
        x1[swap], y1[swap] = y1[swap], x1[swap]

        df_main = pd.DataFrame.from_dict(
            {
                "center": tmscore_result.stem,
                "cluster-x": cluster.labels_[x1],
                "cluster-y": cluster.labels_[y1],
                "chain-x": data.chains[x1],
                "chain-y": data.chains[y1],
                "TMscore": scoremap[x1, y1],
                "RMSD": rmsdmap[x1, y1],
            }
        ).sort_values(["cluster-x", "cluster-y", "TMscore"])

    if len(x2) > 0:
        swap = cluster.labels_[x2] > cluster.labels_[y2]
        x2[swap], y2[swap] = y2[swap], x2[swap]

        df_high = pd.DataFrame.from_dict(
            {
                "center": tmscore_result.stem,
                "cluster-x": cluster.labels_[x2],
                "cluster-y": cluster.labels_[y2],
                "chain-x": data.chains[x2],
                "chain-y": data.chains[y2],
                "TMscore": scoremap[x2, y2],
                "RMSD": rmsdmap[x2, y2],
            }
        ).sort_values(["cluster-x", "cluster-y", "TMscore"])

    return df_main, df_high


@click.command(cls=DataRootCommand)
@click.option("--nproc", "-n", type=int, default=8)
@click.option(
    "--clusters",
    type=click.Path(file_okay=False, path_type=Path),
    default=C.dir("clusters"),
    show_default=True,
)
@click.option(
    "--filtered",
    type=click.Path(dir_okay=False, path_type=Path),
    default=C.file("filtered_pairs"),
    show_default=True,
)
@click.option(
    "--high-tm",
    type=click.Path(dir_okay=False, path_type=Path),
    default=C.file("high_tm"),
    show_default=True,
)
@click.option(
    "--scores",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=C.dir("scores"),
    show_default=True,
)
def main(nproc: int, clusters: Path, filtered: Path, high_tm: Path, scores: Path):
    clusters.mkdir(exist_ok=True, parents=True)

    results = Parallel(n_jobs=nproc)(
        delayed(cluster_by_tmscore)(
            tm_result,
            clusters / tm_result.stem[1:3] / f"{tm_result.stem}.csv",
        )
        for tm_result in tqdm(list(scores.rglob("*.npz")))
    )

    dfs_main = [r[0] for r in results if r[0] is not None]
    dfs_high = [r[1] for r in results if r[1] is not None]

    filtered.parent.mkdir(exist_ok=True, parents=True)
    if dfs_main:
        df = pd.concat(dfs_main).reset_index(drop=True).sort_values("center")
        df.to_csv(filtered, index=False)

    if dfs_high:
        df_high = pd.concat(dfs_high).reset_index(drop=True).sort_values("center")
        df_high.to_csv(high_tm, index=False)


if __name__ == "__main__":
    main()

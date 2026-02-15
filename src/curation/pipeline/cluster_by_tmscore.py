from pathlib import Path

import click
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import distance as D
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

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
        distance_threshold=0.1,
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

    # 2nd pass: 0.8 <= TM < 0.9 and RMSD > 3.0
    mask2 = base_mask & (scoremap >= cutoff) & (scoremap < 0.9) & (rmsdmap > 3.0)

    mask = mask1 | mask2
    x, y = np.nonzero(mask)
    if len(x) == 0:
        return None

    # Ensure cluster-x < cluster-y
    swap = cluster.labels_[x] > cluster.labels_[y]
    x[swap], y[swap] = y[swap], x[swap]

    df = pd.DataFrame.from_dict(
        {
            "center": tmscore_result.stem,
            "cluster-x": cluster.labels_[x],
            "cluster-y": cluster.labels_[y],
            "chain-x": data.chains[x],
            "chain-y": data.chains[y],
            "TMscore": scoremap[x, y],
            "RMSD": rmsdmap[x, y],
        }
    ).sort_values(["cluster-x", "cluster-y", "TMscore"])
    return df


@click.command()
@click.option("--nproc", "-n", type=int, default=8)
@click.option(
    "--clusters",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/clusters"),
    show_default=True,
)
@click.option(
    "--filtered",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/filtered-pairs.csv"),
    show_default=True,
)
@click.option(
    "--scores",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/scores"),
    show_default=True,
)
def main(nproc: int, clusters: Path, filtered: Path, scores: Path):
    clusters.mkdir(exist_ok=True, parents=True)

    dfs = Parallel(n_jobs=nproc)(
        delayed(cluster_by_tmscore)(
            tm_result,
            clusters / tm_result.stem[1:3] / f"{tm_result.stem}.csv",
        )
        for tm_result in tqdm(list(scores.rglob("*.npz")))
    )

    filtered.parent.mkdir(exist_ok=True, parents=True)
    df = (
        pd.concat(df for df in dfs if df is not None)
        .reset_index(drop=True)
        .sort_values("center")
    )
    df.to_csv(filtered, index=False)


if __name__ == "__main__":
    main()

from pathlib import Path

import click
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import distance as D
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from .typedefs import TMScoreResult


def cluster_by_tmscore(
    tmscore_result: Path,
    clusters: Path,
    cutoff: float = 0.8,
):
    data = TMScoreResult(**np.load(tmscore_result))
    scoremap = D.squareform(data.tm_scores)

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

    pairs = np.stack(np.nonzero((scoremap < cutoff) & (scoremap > 0)))
    if len(pairs) == 0:
        return None

    x, y = pairs[:, cluster.labels_[pairs[0]] < cluster.labels_[pairs[1]]]
    df = pd.DataFrame.from_dict(
        {
            "center": tmscore_result.stem,
            "cluster-x": cluster.labels_[x],
            "cluster-y": cluster.labels_[y],
            "chain-x": data.chains[x],
            "chain-y": data.chains[y],
            "TMscore": scoremap[x, y],
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

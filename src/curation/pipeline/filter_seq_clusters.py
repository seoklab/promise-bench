from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import click
import pandas as pd

from ._data_root import DataRootCommand


def _cluster_stem(cluster_csv: str) -> str:
    """'A2/8A27_1' -> '8A27_1'"""
    return cluster_csv.split("/")[-1] if "/" in cluster_csv else cluster_csv


def _cluster_stats(df: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    """Return {cluster_stem: (n_conf_labels, n_entries)} from a pair CSV."""
    stats: Dict[str, Tuple[int, int]] = {}
    for stem, g in df.groupby(df["cluster_csv"].apply(_cluster_stem)):
        # Unique conf labels across both sides
        confs: set = set()
        entries: set = set()
        for side in ("a", "b"):
            confs.update(g[f"{side}_conf_label"].dropna().unique())
            entries.update(
                g[[f"{side}_pdb", f"{side}_assembly_id", f"{side}_chain"]].apply(
                    lambda r: f"{r.iloc[0]}_{r.iloc[1]}_{r.iloc[2]}", axis=1
                )
            )
        stats[stem] = (len(confs), len(entries))
    return stats


def _run_mmseqs(
    fasta_path: Path,
    work_dir: Path,
    min_seq_id: float,
    mmseqs: str,
) -> Dict[str, str]:
    """Run MMseqs2 easy-cluster and return {member_stem: rep_stem}.

    Every cluster member maps to the cluster's representative sequence name.
    """
    out_prefix = work_dir / "mmseqs_out"
    tmp_dir = work_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    cmd = [
        mmseqs,
        "easy-cluster",
        str(fasta_path),
        str(out_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        "0.8",
        "--cov-mode",
        "0",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Parse _cluster.tsv: rep\tmember
    tsv = Path(f"{out_prefix}_cluster.tsv")
    member_to_rep: Dict[str, str] = {}
    with tsv.open() as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                member_to_rep[member] = rep
    return member_to_rep


def _select_best(
    mmseqs_groups: Dict[str, List[str]],
    stats: Dict[str, Tuple[int, int]],
) -> Set[str]:
    """Given mmseqs cluster groups, pick one cluster per group.

    Selection: max conf_labels, then max entries.
    Returns set of cluster stems to keep.
    """
    keep: Set[str] = set()
    for members in mmseqs_groups.values():
        best = max(
            members,
            key=lambda s: stats.get(s, (0, 0)),
        )
        keep.add(best)
    return keep


# ---------------------------------------------------------------------------
# Process one category
# ---------------------------------------------------------------------------
def process_category(
    name: str,
    csv_path: Path,
    rep_seqs: Dict[str, dict],
    work_dir: Path,
    min_seq_id: float,
    mmseqs: str,
) -> Tuple[pd.DataFrame, int, int]:
    """Filter a single category CSV. Returns (filtered_df, n_before, n_clusters_removed)."""

    df = pd.read_csv(csv_path)
    if df.empty or "cluster_csv" not in df.columns:
        return df, 0, 0

    n_before = len(df)

    # Cluster stats for this category
    stats = _cluster_stats(df)
    cluster_stems = set(stats.keys())

    # Build FASTA with representative sequences
    fasta_path = work_dir / f"{name}.fasta"
    n_written = 0
    with fasta_path.open("w") as fh:
        for stem in sorted(cluster_stems):
            if stem not in rep_seqs:
                continue
            seq = rep_seqs[stem]["seq"]
            if not seq:
                continue
            fh.write(f">{stem}\n{seq}\n")
            n_written += 1

    if n_written < 2:
        # Nothing to cluster
        return df, n_before, 0

    # Run MMseqs2
    cat_work = work_dir / name
    cat_work.mkdir(exist_ok=True)
    member_to_rep = _run_mmseqs(fasta_path, cat_work, min_seq_id, mmseqs)

    # Group by mmseqs representative
    groups: Dict[str, List[str]] = defaultdict(list)
    for member, rep in member_to_rep.items():
        groups[rep].append(member)

    # Select best per group
    keep = _select_best(groups, stats)

    removed_stems = cluster_stems - keep
    n_removed = len(removed_stems)

    if not removed_stems:
        return df, n_before, 0

    # Filter pairs
    df_out = df[df["cluster_csv"].apply(lambda c: _cluster_stem(c) in keep)]
    return df_out, n_before, n_removed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command(
    cls=DataRootCommand, context_settings={"help_option_names": ["-h", "--help"]}
)
@click.option(
    "--dataset-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/combinations"),
    show_default=True,
    help="Directory with pair CSVs.",
)
@click.option(
    "--rep-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/representative_sequences.json"),
    show_default=True,
    help="Representative sequences JSON from select_representative.py.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("data/dataset-pipeline"),
    show_default=True,
    help="Output directory for the final filtered dataset.",
)
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=Path("data/seqcluster_work"),
    show_default=True,
    help="Working directory for FASTA / MMseqs2 files.",
)
@click.option(
    "--min-seq-id",
    type=float,
    default=0.4,
    show_default=True,
    help="MMseqs2 minimum sequence identity threshold.",
)
@click.option(
    "--mmseqs",
    type=str,
    default="mmseqs",
    show_default=True,
    help="Path to mmseqs binary.",
)
def main(
    dataset_dir: Path,
    rep_json: Path,
    out_dir: Path,
    work_dir: Path,
    min_seq_id: float,
    mmseqs: str,
):
    """Filter redundant clusters per category via MMseqs2 sequence identity."""

    # Load representative sequences
    with rep_json.open() as fh:
        rep_seqs: Dict[str, dict] = json.load(fh)
    click.echo(f"Loaded {len(rep_seqs)} representative sequences")

    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = ["ligand-induced", "protein-induced", "intrinsic"]

    for cat in categories:
        csv_path = dataset_dir / f"{cat}.csv"
        if not csv_path.exists():
            click.echo(f"  [{cat}] CSV not found, skipping")
            continue

        df_out, n_before, n_removed = process_category(
            cat,
            csv_path,
            rep_seqs,
            work_dir,
            min_seq_id,
            mmseqs,
        )
        df_out.to_csv(out_dir / f"{cat}.csv", index=False)

        n_clusters_before = (
            pd.read_csv(csv_path)["cluster_csv"].apply(_cluster_stem).nunique()
        )
        n_clusters_after = n_clusters_before - n_removed

        click.echo(
            f"  [{cat}] clusters: {n_clusters_before} -> {n_clusters_after} "
            f"(removed {n_removed})  |  "
            f"pairs: {n_before} -> {len(df_out)} (dropped {n_before - len(df_out)})"
        )

    click.echo("Done.")


if __name__ == "__main__":
    main()

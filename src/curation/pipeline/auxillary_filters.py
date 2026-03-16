"""Post-pipeline auxiliary filters for dataset CSVs.

Each public ``filter_*`` function takes a :class:`~pandas.DataFrame`,
applies one specific filter, and returns the filtered frame together
with a short human-readable summary string.

Running as a CLI (via ``click``) applies **all** registered filters to
the dataset-pipeline directory and overwrites the files in-place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click
import pandas as pd

from ..utils._data_root import DataRootCommand


# ---------------------------------------------------------------------------
# Individual filter functions
# ---------------------------------------------------------------------------


def filter_unique_conf_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Keep only unique ``(a_conf_label, b_conf_label)`` per ``cluster_csv``.

    When multiple rows in the same cluster share identical conformational
    label pairs, only the first occurrence is retained.

    :param df: DataFrame with at least ``cluster_csv``, ``a_conf_label``,
        and ``b_conf_label`` columns.
    :return: Filtered DataFrame and a one-line summary.
    """
    before = len(df)
    out = df.drop_duplicates(
        subset=["cluster_csv", "a_conf_label", "b_conf_label"],
        keep="first",
    )
    after = len(out)
    summary = (
        f"filter_unique_conf_labels: {before} -> {after} "
        f"(dropped {before - after})"
    )
    return out, summary


def filter_no_contact_ligands(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Keep only rows where **both** sides have no contact ligands.

    Rows where either ``a_contact_ligands`` or ``b_contact_ligands`` is
    non-empty are dropped.  This ensures the pair's conformational change
    is purely protein-induced (not ligand-mediated).

    :param df: DataFrame with ``a_contact_ligands`` and
        ``b_contact_ligands`` columns.
    :return: Filtered DataFrame and a one-line summary.
    """
    before = len(df)
    mask = df["a_contact_ligands"].isna() & df["b_contact_ligands"].isna()
    out = df[mask].reset_index(drop=True)
    after = len(out)
    summary = (
        f"filter_no_contact_ligands: {before} -> {after} "
        f"(dropped {before - after})"
    )
    return out, summary


# ---------------------------------------------------------------------------
# Ligand-similarity filter
# ---------------------------------------------------------------------------


def _build_ligand_to_cluster(
    lig_clusters: Dict[str, Dict[str, list]],
) -> Dict[str, str]:
    """Build a reverse map: ligand_name -> cluster_id.

    Expected JSON structure::

        {
            "simple": {
                "0": ["PAR", "GLC-GLC"],
                "1": ["CH1", "U3H"],
                ...
            }
        }

    Only the ``"simple"`` key is used.  The returned dict maps every
    individual ligand name to ``"simple:<cluster_id>"``.
    """
    lig2cluster: Dict[str, str] = {}
    clusters = lig_clusters.get("simple", {})
    for cid, members in clusters.items():
        label = f"simple:{cid}"
        for lig in members:
            lig2cluster[lig.strip()] = label
    return lig2cluster


def _contact_ligands_signature(
    raw: str,
    lig2cluster: Dict[str, str],
) -> str:
    """Map a semicolon-separated ``b_contact_ligands`` value to a canonical
    signature based on ligand clusters.

    Each individual ligand is replaced by its cluster id (or kept as-is if
    unmapped).  Duplicates within the same cluster are collapsed, then
    sorted to form a deterministic string key.

    Examples
    --------
    >>> lig2c = {"ATP": "simple:0", "ADP": "simple:0", "MG": "simple:5"}
    >>> _contact_ligands_signature("ATP;ADP;MG", lig2c)
    'simple:0|simple:0|simple:5'
    >>> _contact_ligands_signature("ATP", lig2c)
    'simple:0'
    """
    if pd.isna(raw) or not str(raw).strip():
        return ""
    parts = [p.strip() for p in str(raw).split(";") if p.strip()]
    mapped = sorted(lig2cluster.get(p, p) for p in parts)
    return "|".join(mapped)


def filter_excluded_ligands(
    df: pd.DataFrame,
    excluded: set[str],
    columns: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Strip excluded ligands from contact-ligand columns, then drop rows
    that no longer qualify as ligand-induced.

    Instead of dropping every row that mentions an excluded ligand, only
    the individual excluded CCD codes are removed from the semicolon-
    separated ligand lists.  A row is dropped only when **both**
    ``a_contact_ligands`` and ``b_contact_ligands`` become empty after
    stripping, meaning it no longer meets the ligand-induced criterion.

    :param df: DataFrame.
    :param excluded: Set of ligand names to exclude.
    :param columns: Columns to clean (default: both ``a_contact_ligands``
        and ``b_contact_ligands`` if present).
    :return: Filtered DataFrame and a one-line summary.
    """
    if columns is None:
        columns = [
            c for c in ("a_contact_ligands", "b_contact_ligands")
            if c in df.columns
        ]

    before = len(df)
    df = df.copy()
    n_stripped = 0

    def _strip(raw: object) -> object:
        nonlocal n_stripped
        if pd.isna(raw) or not str(raw).strip():
            return raw
        parts = [p.strip() for p in str(raw).split(";") if p.strip()]
        remaining = [p for p in parts if p not in excluded]
        n_stripped += len(parts) - len(remaining)
        if not remaining:
            return pd.NA
        return ";".join(remaining)

    for col in columns:
        df[col] = df[col].map(_strip)

    # Drop rows that no longer qualify as ligand-induced:
    # at least one side must still have contact ligands.
    a_empty = (
        df["a_contact_ligands"].isna()
        if "a_contact_ligands" in df.columns
        else True
    )
    b_empty = (
        df["b_contact_ligands"].isna()
        if "b_contact_ligands" in df.columns
        else True
    )
    mask = ~(a_empty & b_empty)
    out = df[mask].reset_index(drop=True)
    after = len(out)
    summary = (
        f"filter_excluded_ligands: {before} -> {after} "
        f"(dropped {before - after} rows, "
        f"stripped {n_stripped} ligand occurrences, "
        f"{len(excluded)} ligands in exclusion list)"
    )
    return out, summary


def filter_ligand_similarity(
    df: pd.DataFrame,
    lig2cluster: Dict[str, str],
) -> Tuple[pd.DataFrame, str]:
    """Deduplicate rows by ligand-cluster signature per structural cluster.

    Within each ``cluster_csv`` group, rows whose ``b_contact_ligands``
    map to the same ligand-cluster signature are considered redundant;
    only the first occurrence (highest TM-score if pre-sorted) is kept.

    :param df: DataFrame with ``cluster_csv`` and ``b_contact_ligands``.
    :param lig2cluster: Ligand-name -> cluster-id mapping (from
        :func:`_build_ligand_to_cluster`).
    :return: Filtered DataFrame and a one-line summary.
    """
    before = len(df)
    df = df.copy()
    df["_lig_sig"] = df["b_contact_ligands"].map(
        lambda x: _contact_ligands_signature(x, lig2cluster)
    )
    out = df.drop_duplicates(
        subset=["cluster_csv", "_lig_sig"],
        keep="first",
    ).drop(columns=["_lig_sig"])
    after = len(out)
    n_unmapped = sum(
        1
        for v in df["b_contact_ligands"].dropna()
        for p in str(v).split(";")
        if p.strip() and p.strip() not in lig2cluster
    )
    summary = (
        f"filter_ligand_similarity: {before} -> {after} "
        f"(dropped {before - after}, "
        f"{len(lig2cluster)} ligands mapped, "
        f"{n_unmapped} individual occurrences unmapped)"
    )
    return out, summary


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Static filters: (csv_stem, [filter_func, ...])
# These take only (df) -> (df, str).
FILTERS: list[Tuple[str, list]] = [
    ("intrinsic", [filter_unique_conf_labels]),
    ("protein-induced", [filter_no_contact_ligands]),
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_exclusion_list(path: Path) -> set[str]:
    """Load a newline/whitespace-separated ligand exclusion list.

    Lines starting with ``#`` are treated as comments and ignored.
    """
    excluded: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for tok in line.split():
                if tok:
                    excluded.add(tok)
    return excluded


def apply_all_filters(
    dataset_dir: Path,
    ligand_clusters_path: Optional[Path] = None,
    exclude_ligands_path: Optional[Path] = None,
) -> list[str]:
    """Apply every registered filter and overwrite CSVs in *dataset_dir*.

    :param dataset_dir: Path to ``dataset-pipeline/`` directory.
    :param ligand_clusters_path: Optional path to a JSON file containing
        ligand-similarity clusters.  If provided, the ligand-similarity
        filter is applied to ``ligand-induced.csv``.
    :param exclude_ligands_path: Optional path to a text file listing
        ligand names to exclude.  Rows in ``ligand-induced.csv`` whose
        ``b_contact_ligands`` contain any of these are dropped.
    :return: List of summary strings (one per applied filter).
    """
    summaries: list[str] = []

    # --- static filters ---
    for stem, funcs in FILTERS:
        csv_path = dataset_dir / f"{stem}.csv"
        if not csv_path.exists():
            summaries.append(f"{stem}.csv not found -- skipped")
            continue
        df = pd.read_csv(csv_path)
        for fn in funcs:
            df, msg = fn(df)
            summaries.append(msg)
        df.to_csv(csv_path, index=False)
        summaries.append(f"  -> wrote {csv_path}")

    # --- ligand-induced filters (parameterised) ---
    lig_csv = dataset_dir / "ligand-induced.csv"
    if lig_csv.exists():
        df_lig = pd.read_csv(lig_csv)
        lig_modified = False

        # exclusion list
        if exclude_ligands_path is not None:
            excluded = _load_exclusion_list(exclude_ligands_path)
            summaries.append(
                f"Loaded {len(excluded)} excluded ligands "
                f"from {exclude_ligands_path}"
            )
            df_lig, msg = filter_excluded_ligands(df_lig, excluded)
            summaries.append(msg)
            lig_modified = True

        # ligand-similarity clustering
        if ligand_clusters_path is not None:
            with open(ligand_clusters_path) as f:
                lig_clusters_json: Dict[str, Any] = json.load(f)
            lig2cluster = _build_ligand_to_cluster(lig_clusters_json)
            summaries.append(
                f"Loaded {len(lig2cluster)} ligand->cluster mappings "
                f"from {ligand_clusters_path}"
            )
            df_lig, msg = filter_ligand_similarity(df_lig, lig2cluster)
            summaries.append(msg)
            lig_modified = True

        if lig_modified:
            df_lig.to_csv(lig_csv, index=False)
            summaries.append(f"  -> wrote {lig_csv}")
        else:
            summaries.append(
                "ligand-induced.csv: no --ligand-clusters or "
                "--exclude-ligands provided, skipped"
            )

    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(cls=DataRootCommand)
@click.option(
    "--dataset-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default="data-95/dataset-pipeline",
    show_default=True,
    help="Path to dataset-pipeline/ directory.",
)
@click.option(
    "--ligand-clusters",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="/home.galaxy4/seeun/works/projects/casp-pipe/data/cluster-lig_ccd/cutoff_0.4.json",
    show_default=True,
    help=(
        "JSON file with ligand-similarity clusters. "
        'Expected format: {"namespace": {"cluster_id": ["LIG1", "LIG2"], ...}}. '
        "If provided, ligand-induced.csv is deduplicated by cluster signature."
    ),
)
@click.option(
    "--exclude-ligands",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="/home/bonjae02/projects/promise-bench/data-90/excluded_ccd_codes.txt",
    show_default=True,
    help=(
        "Text file listing ligand names to exclude (one per line). "
        "Rows in ligand-induced.csv containing any of these are dropped."
    ),
)
def auxillary_filters(
    dataset_dir: Path,
    ligand_clusters: Optional[Path],
    exclude_ligands: Optional[Path],
) -> None:
    """Apply auxiliary post-pipeline filters to dataset CSVs."""
    click.echo(f"Filtering CSVs in {dataset_dir}")
    for msg in apply_all_filters(
        dataset_dir,
        ligand_clusters_path=ligand_clusters,
        exclude_ligands_path=exclude_ligands,
    ):
        click.echo(f"  {msg}")
    click.echo("Done.")


if __name__ == "__main__":
    auxillary_filters()

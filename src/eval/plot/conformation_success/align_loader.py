"""Load ``align_part*.json`` from promise-bench's alignment stage.

Provides utilities for:
- streaming-loading all ``align_part*.json`` under a directory,
- grouping rows by ``(cluster_id, prediction_method, mobile_cif)``
  (i.e. per prediction sample),
- collapsing each sample's per-reference TM values into a per-conformation
  view (max over multiple references in the same conf, see ``_common``).

Schema of one row (relevant fields)
-----------------------------------
- ``pair_type``: "intrinsic" | "ligand-induced" | "protein-induced"
- ``cluster_id``: e.g. ``"4KBF_1"``
- ``prediction_method``: dashed label, e.g. ``"boltz-1"`` (use
  ``normalize_method`` -> ``"boltz1"``)
- ``mobile_cif``: prediction CIF path; uniquely identifies one
  (seed, sample) prediction
- ``mobile_entity``: prediction yaml tag, e.g. ``"4kbg_2_B1_m"``
- ``model_entity``: reference yaml tag the prediction was aligned to
- ``reference_conformation``: ``"conf_0"`` / ``"conf_1"`` / ...
- ``ok``: alignment success flag
- ``tm_score_ca``: CA TM-score
- ``valid_pair``: ``[ref1_tag, ref2_tag]`` (set-level metadata)

CLI (single-cluster inspection)
-------------------------------

::

    PYTHONPATH=src uv run --project . \\
      python -m eval.plot.conformation_success.align_loader \\
        --align-dir data_eval/output/align/job_batches_v2/align_results \\
        --cluster 4KBF_1
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from ._common import normalize_method, normalize_set_type

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Row = Dict[str, Any]
SampleKey = Tuple[str, str, str]  # (cluster_id, method [dash-less], mobile_cif)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def iter_align_files(align_dir: Path) -> Iterator[Path]:
    """Yield ``align_part*.json`` files under ``align_dir`` (sorted)."""
    yield from sorted(align_dir.glob("align_part*.json"))


def load_align_rows(
    align_dir: Path,
    *,
    set_filter: Optional[Iterable[str]] = None,
    cluster_filter: Optional[Iterable[str]] = None,
    method_filter: Optional[Iterable[str]] = None,
    ok_only: bool = True,
) -> List[Row]:
    """Load and concatenate rows from every ``align_part*.json`` under ``align_dir``.

    All string-set filters are matched after normalisation (``intrinsic``,
    dash-less method keys). ``ok_only`` drops rows with ``ok=False``
    (alignment failures).
    """
    set_set: Optional[set] = set(set_filter) if set_filter is not None else None
    cluster_set: Optional[set] = (
        set(cluster_filter) if cluster_filter is not None else None
    )
    method_set: Optional[set] = (
        set(normalize_method(m) for m in method_filter)
        if method_filter is not None
        else None
    )

    rows: List[Row] = []
    for fp in iter_align_files(align_dir):
        try:
            data = json.load(open(fp))
        except Exception as e:
            print(f"  [WARN] failed to load {fp}: {e}")
            continue
        if not isinstance(data, list):
            continue
        for r in data:
            if ok_only and not r.get("ok"):
                continue
            pt = normalize_set_type(r.get("pair_type"))
            if set_set is not None and pt not in set_set:
                continue
            if cluster_set is not None and r.get("cluster_id") not in cluster_set:
                continue
            m_norm = normalize_method(r.get("prediction_method"))
            if method_set is not None and m_norm not in method_set:
                continue
            # Cache normalised fields back onto the row (cheap & convenient
            # for downstream consumers); leave the original strings intact
            # under their existing keys.
            r["_set_type"] = pt
            r["_method"] = m_norm
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def sample_key(row: Row) -> SampleKey:
    """``(cluster_id, dash-less method, mobile_cif)`` -- identifies one prediction sample."""
    return (
        row["cluster_id"],
        row.get("_method") or normalize_method(row.get("prediction_method")),
        row.get("mobile_cif", ""),
    )


def group_by_sample(rows: Iterable[Row]) -> Dict[SampleKey, List[Row]]:
    """Group rows by ``(cluster_id, method, mobile_cif)``."""
    out: Dict[SampleKey, List[Row]] = defaultdict(list)
    for r in rows:
        if not r.get("mobile_cif"):
            continue
        out[sample_key(r)].append(r)
    return out


def cluster_expected_confs(
    rows: Iterable[Row],
) -> Dict[Tuple[str, str], set]:
    """Per ``(cluster_id, method)``, set of reference_conformation labels seen.

    Used as the ground-truth "what we want to cover" for cluster-level
    success.
    """
    out: Dict[Tuple[str, str], set] = defaultdict(set)
    for r in rows:
        conf = r.get("reference_conformation")
        if not conf:
            continue
        m = r.get("_method") or normalize_method(r.get("prediction_method"))
        out[(r["cluster_id"], m)].add(conf)
    return out


def conf_to_tags(rows: Iterable[Row]) -> Dict[str, set]:
    """Build ``{conf_label: {reference_tag, ...}}`` from a row collection.

    A "reference_tag" is the ``mobile_entity`` field, which (per the
    promise-bench align row convention) names the experimental reference
    structure used as the alignment target. Useful when a single conf
    label is shared across multiple reference structures.
    """
    out: Dict[str, set] = defaultdict(set)
    for r in rows:
        c = r.get("reference_conformation")
        tag = r.get("mobile_entity")
        if c and tag:
            out[c].add(tag)
    return out


# ---------------------------------------------------------------------------
# Per-sample TM by conformation
# ---------------------------------------------------------------------------


def tm_by_conf(sample_rows: Iterable[Row]) -> Dict[str, List[float]]:
    """``{conf_label: [tm, ...]}`` for one sample (multiple refs per conf possible)."""
    out: Dict[str, List[float]] = defaultdict(list)
    for r in sample_rows:
        conf = r.get("reference_conformation")
        tm = r.get("tm_score_ca")
        if conf is None or tm is None:
            continue
        out[conf].append(float(tm))
    return out


def best_tm_by_conf(sample_rows: Iterable[Row]) -> Dict[str, float]:
    """``{conf_label: max TM}`` for one sample.

    "Best" only matters when a conf has more than one reference; for the
    common 1-ref-per-conf case this is just the single TM value.
    """
    return {c: max(tms) for c, tms in tm_by_conf(sample_rows).items() if tms}


# ---------------------------------------------------------------------------
# CLI -- single-cluster inspection
# ---------------------------------------------------------------------------


def _format_per_sample(
    sample_rows: List[Row], expected_confs: Optional[set]
) -> str:
    lines = []
    btm = best_tm_by_conf(sample_rows)
    missing = sorted(expected_confs - set(btm.keys())) if expected_confs else []
    for conf in sorted(btm):
        lines.append(f"    {conf}: best_tm = {btm[conf]:.4f}")
    for conf in missing:
        lines.append(f"    {conf}: MISSING (no alignment row)")
    return "\n".join(lines)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect alignment rows for a single cluster: per-sample best "
            "TM per conformation."
        )
    )
    parser.add_argument(
        "--align-dir",
        type=str,
        default=None,
        help="Dir of align_part*.json (default: eval.dirs.align_results)",
    )
    parser.add_argument(
        "--cluster", "-c", type=str, required=True, help="Cluster id, e.g. 4KBF_1"
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default=None,
        help="Filter by method (af3 / boltz1 / boltz2 / chai / bioemu)",
    )
    parser.add_argument(
        "--set",
        type=str,
        default="intrinsic",
        help="pair_type filter (default: intrinsic). Use '' to keep all sets.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Limit number of samples printed (default 10).",
    )
    args = parser.parse_args()

    # Lazy import so non-CLI consumers don't need the full config.
    from utils._config import eval_cfg as E

    align_dir = E.align_results_dir(args.align_dir)
    set_filter = [args.set] if args.set else None
    method_filter = [args.method] if args.method else None

    rows = load_align_rows(
        align_dir,
        set_filter=set_filter,
        cluster_filter=[args.cluster],
        method_filter=method_filter,
    )
    print(f"loaded {len(rows)} rows from {align_dir}")
    if not rows:
        return

    grouped = group_by_sample(rows)
    expected = cluster_expected_confs(rows)
    print(f"samples in cluster {args.cluster}: {len(grouped)}")
    print(f"expected confs per method: {dict(expected)}")
    print()

    shown = 0
    for key, srows in sorted(grouped.items()):
        cluster, method, mobile_cif = key
        exp = expected.get((cluster, method), set())
        sample_name = Path(mobile_cif).name
        print(f"--- {method}  sample={sample_name}")
        print(_format_per_sample(srows, exp))
        shown += 1
        if shown >= args.max_samples:
            print(f"... (truncated at {args.max_samples} samples)")
            break


if __name__ == "__main__":
    _cli()

"""Compute pair-level conformation-success results.

Port of ``dynamic_set/final/step8a_compute_pair_results.py`` to promise-bench.
Replaces the legacy on-disk ``rmsd_tmscore_real_final.json`` IO with
in-memory consumption of ``align_part*.json`` (loaded via ``align_loader``)
and uses promise-bench's ``valid_pairs.json`` schema (list-of-lists).

Success criteria (TM_THRESHOLD = 0.8)
-------------------------------------

**Intrinsic (apo-monomers)** -- cluster-level success
    Each sample (prediction CIF) is aligned to every cluster reference.
    Per sample, ``best_tm[conf]`` is the max TM over references in ``conf``.
    Sample *covers* ``conf_X`` iff
        best_tm[conf_X] >= 0.8  AND  best_tm[conf_X] > best_tm[conf_Y]
    for every other conf_Y. Cluster *succeeds* iff every expected conf
    label is covered by at least one sample. (One sample need not cover
    all confs.)

**Induced (ligand-induced / protein-induced)** -- pair-level flags
    Each pair has an apo-prediction set and a holo-prediction set.

    For an apo prediction's sample (relaxed apo rule):
        ``target_tm = max TM over EVERY apo reference in the cluster``
        ``other_tm  = TM(designated holo ref)``
    i.e. the apo prediction is considered successful as long as it matches
    *any* apo conformation in the same cluster, not only the pair's
    designated apo. The cluster-wide lookup is built from all rows with
    ``method_type='apo' AND reference_state='apo'`` for that (cluster,
    method), so it automatically picks up cross-pair / intrinsic
    alignments without re-aligning.

    For a holo prediction's sample (unchanged):
        ``target_tm = TM(designated holo ref)``
        ``other_tm  = TM(designated apo ref)``
    The holo target must exactly match the pair's holo (no relaxation).

    Sample succeeds iff ``target_tm >= 0.8`` AND ``target_tm > other_tm``.
    ``apo_success`` = any apo sample succeeded; ``holo_success`` = any holo
    sample succeeded; ``overall_success`` = both.

Output
------
``<output_dir>/pair_level_results.json`` (canonical), plus optional CSVs:
``sample_details_intrinsic.csv``, ``sample_details_induced.csv``.

CLI
---
::

    PYTHONPATH=src uv run --project . \\
      python -m eval.plot.conformation_success.pair_results \\
        --output-dir data_eval/plots/conformation_success
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ._common import (
    METHODS,
    SET_TYPES,
    TM_THRESHOLD,
    normalize_method,
    normalize_set_type,
)
from .align_loader import (
    Row,
    SampleKey,
    best_tm_by_conf,
    conf_to_tags,
    group_by_sample,
    load_align_rows,
    tm_by_conf,
)
from .confidence_loader import attach_confidence_to_results


# ---------------------------------------------------------------------------
# Intrinsic
# ---------------------------------------------------------------------------


def _check_sample_conf_success(
    sample_btm: Dict[str, float],
    target_conf: str,
    expected_confs: Iterable[str],
    threshold: float = TM_THRESHOLD,
) -> bool:
    """Apply the per-sample conf-cover rule to one sample's best-tm-by-conf dict.

    A sample *covers* ``target_conf`` iff
        ``best_tm(target_conf) >= threshold`` AND
        ``best_tm(target_conf) > best_tm(other_conf)`` for every other conf
    in ``expected_confs``. Missing conf TM is treated as 0.0.
    """
    target_tm = sample_btm.get(target_conf, 0.0)
    if target_tm < threshold:
        return False
    for other in expected_confs:
        if other == target_conf:
            continue
        if sample_btm.get(other, 0.0) >= target_tm:
            return False
    return True


def analyze_intrinsic_cluster_method(
    rows: List[Row], cluster_id: str, method: str
) -> Dict[str, Any]:
    """Compute intrinsic cluster-level success for one (cluster, method).

    ``rows`` must already be pre-filtered to that cluster+method (no extra
    filtering done here).
    """
    expected_confs: Set[str] = set()
    c2t = conf_to_tags(rows)
    for c in c2t:
        expected_confs.add(c)

    samples_by_key = group_by_sample(rows)
    samples_out: List[Dict[str, Any]] = []
    covered: Set[str] = set()

    for key, srows in samples_by_key.items():
        _, _, mobile_cif = key
        btm = best_tm_by_conf(srows)
        per_conf = tm_by_conf(srows)
        conf_successes: Dict[str, Dict[str, Any]] = {}
        for conf in expected_confs:
            ok = _check_sample_conf_success(btm, conf, expected_confs)
            conf_successes[conf] = {
                "success": ok,
                "best_tm": btm.get(conf, 0.0),
                "tm_values": per_conf.get(conf, []),
            }
            if ok:
                covered.add(conf)
        samples_out.append(
            {
                "mobile_cif": mobile_cif,
                "sample_name": Path(mobile_cif).name,
                "model_entity": srows[0].get("model_entity"),
                "best_tm_by_conf": btm,
                "conf_successes": conf_successes,
            }
        )

    return {
        "method": method,
        "cluster_id": cluster_id,
        "success": bool(expected_confs) and covered == expected_confs,
        "expected_confs": sorted(expected_confs),
        "covered_confs": sorted(covered),
        "missing_confs": sorted(expected_confs - covered),
        "conf_to_tags": {k: sorted(v) for k, v in c2t.items()},
        "n_samples": len(samples_out),
        "samples": samples_out,
    }


# ---------------------------------------------------------------------------
# Induced (ligand-induced / protein-induced)
# ---------------------------------------------------------------------------


def _split_pair_tags(valid_pair: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (apo_tag, holo_tag) from a ``[apo_tag_m, holo_tag_x]`` pair."""
    apo_tag = holo_tag = None
    for t in valid_pair:
        if t.endswith("_m"):
            apo_tag = t
        elif t.endswith("_x"):
            holo_tag = t
    return apo_tag, holo_tag


def _collect_entity_samples(
    rows: List[Row],
    want: str,
    *,
    cluster_apo_tm_lookup: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Group ``rows`` (one entity side) by sample and compute target/other stats.

    ``want`` is ``"apo"`` or ``"holo"`` and decides which ``reference_state``
    counts as the target. ``rows`` should already be filtered to
    ``method_type == want`` for one (cluster, method, pair).

    When ``want == "apo"`` and ``cluster_apo_tm_lookup`` is provided, the apo
    target_tm for each sample is upgraded to ``max TM over EVERY apo reference
    in the cluster`` (relaxed apo rule). The lookup maps ``mobile_cif ->
    best_tm_against_any_cluster_apo_ref``. For samples missing from the lookup
    the per-pair apo TM is used as a safe fallback.
    """
    by_sample: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_sample[r.get("mobile_cif", "")].append(r)

    samples_out: List[Dict[str, Any]] = []
    stats = {
        "success_exists": False,
        "collapse_exists": False,
        "max_target_tm": 0.0,
        "max_other_tm": 0.0,
        "max_margin_other_minus_target": 0.0,
        "max_target_sample": None,
        "max_other_sample": None,
    }

    for mobile_cif, srows in by_sample.items():
        apo_tm = 0.0
        holo_tm = 0.0
        for r in srows:
            tm = r.get("tm_score_ca")
            if tm is None:
                continue
            rs = r.get("reference_state")
            if rs == "apo":
                apo_tm = max(apo_tm, float(tm))
            elif rs == "holo":
                holo_tm = max(holo_tm, float(tm))

        apo_tm_pair = apo_tm
        if want == "apo" and cluster_apo_tm_lookup is not None:
            apo_tm = max(apo_tm, cluster_apo_tm_lookup.get(mobile_cif, 0.0))

        if want == "apo":
            target_tm, other_tm = apo_tm, holo_tm
        else:
            target_tm, other_tm = holo_tm, apo_tm

        if target_tm > stats["max_target_tm"]:
            stats["max_target_tm"] = target_tm
            stats["max_target_sample"] = mobile_cif
        if other_tm > stats["max_other_tm"]:
            stats["max_other_tm"] = other_tm
            stats["max_other_sample"] = mobile_cif
        margin = other_tm - target_tm
        if margin > stats["max_margin_other_minus_target"]:
            stats["max_margin_other_minus_target"] = margin

        sample_success = target_tm >= TM_THRESHOLD and target_tm > other_tm
        sample_collapse = other_tm >= TM_THRESHOLD and other_tm > target_tm
        if sample_success:
            stats["success_exists"] = True
        if sample_collapse:
            stats["collapse_exists"] = True

        samples_out.append(
            {
                "mobile_cif": mobile_cif,
                "sample_name": Path(mobile_cif).name,
                "apo_tm": apo_tm,
                "apo_tm_pair": apo_tm_pair,
                "holo_tm": holo_tm,
                "success": sample_success,
                "collapse": sample_collapse,
            }
        )

    return samples_out, stats


def analyze_induced_pair(
    rows: List[Row],
    cluster_id: str,
    method: str,
    pair_type: str,
    valid_pair: List[str],
    *,
    cluster_apo_tm_lookup: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute induced-pair success for one (cluster, method, valid_pair).

    ``cluster_apo_tm_lookup`` is the relaxed-apo lookup for this (cluster,
    method): ``mobile_cif -> max TM over every cluster apo reference``.
    Passed through to ``_collect_entity_samples`` for the apo side only.
    """
    apo_tag, holo_tag = _split_pair_tags(valid_pair)
    pair_set = set(valid_pair)
    apo_rows = [
        r
        for r in rows
        if set(r.get("valid_pair") or []) == pair_set and r.get("method_type") == "apo"
    ]
    holo_rows = [
        r
        for r in rows
        if set(r.get("valid_pair") or []) == pair_set and r.get("method_type") == "holo"
    ]

    apo_samples, apo_stats = _collect_entity_samples(
        apo_rows, want="apo", cluster_apo_tm_lookup=cluster_apo_tm_lookup
    )
    holo_samples, holo_stats = _collect_entity_samples(holo_rows, want="holo")

    return {
        "cluster_id": cluster_id,
        "method": method,
        "pair_type": pair_type,
        "valid_pair": valid_pair,
        "apo_tag": apo_tag,
        "holo_tag": holo_tag,
        "apo_success": apo_stats["success_exists"],
        "holo_success": holo_stats["success_exists"],
        "overall_success": apo_stats["success_exists"] and holo_stats["success_exists"],
        "apo_collapse": apo_stats["collapse_exists"],
        "holo_collapse": holo_stats["collapse_exists"],
        "apo_max_apo_tm": apo_stats["max_target_tm"],
        "apo_max_holo_tm": apo_stats["max_other_tm"],
        "apo_max_margin_holo_minus_apo": apo_stats["max_margin_other_minus_target"],
        "holo_max_holo_tm": holo_stats["max_target_tm"],
        "holo_max_apo_tm": holo_stats["max_other_tm"],
        "holo_max_margin_apo_minus_holo": holo_stats["max_margin_other_minus_target"],
        "apo_max_apo_tm_sample": apo_stats["max_target_sample"],
        "apo_max_holo_tm_sample": apo_stats["max_other_sample"],
        "holo_max_holo_tm_sample": holo_stats["max_target_sample"],
        "holo_max_apo_tm_sample": holo_stats["max_other_sample"],
        "n_apo_samples": len(apo_samples),
        "n_holo_samples": len(holo_samples),
        "apo_samples": apo_samples,
        "holo_samples": holo_samples,
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def _rows_index(rows: List[Row]) -> Dict[Tuple[str, str, str], List[Row]]:
    """Index rows by ``(set_type, cluster_id, dash-less method)``."""
    idx: Dict[Tuple[str, str, str], List[Row]] = defaultdict(list)
    for r in rows:
        idx[(r["_set_type"], r["cluster_id"], r["_method"])].append(r)
    return idx


def _build_cluster_apo_tm_lookup(rows_cm: Iterable[Row]) -> Dict[str, float]:
    """Per (cluster, method): ``mobile_cif -> max TM against any cluster apo ref``.

    Scans every row (intrinsic + induced, all pair_types) and keeps only those
    with ``method_type='apo'`` and ``reference_state='apo'``, which by
    construction align a prediction CIF to *some* cluster apo reference. Taking
    the max over all such rows for the same ``mobile_cif`` yields the relaxed
    apo target TM for that sample.
    """
    out: Dict[str, float] = {}
    for r in rows_cm:
        if r.get("method_type") != "apo":
            continue
        if r.get("reference_state") != "apo":
            continue
        tm = r.get("tm_score_ca")
        cif = r.get("mobile_cif")
        if tm is None or not cif:
            continue
        v = float(tm)
        if v > out.get(cif, 0.0):
            out[cif] = v
    return out


def _index_rows_by_cluster_method(
    rows: List[Row],
) -> Dict[Tuple[str, str], List[Row]]:
    """Index rows by ``(cluster_id, dash-less method)`` (across pair_types)."""
    idx: Dict[Tuple[str, str], List[Row]] = defaultdict(list)
    for r in rows:
        idx[(r["cluster_id"], r["_method"])].append(r)
    return idx


def compute_pair_results(
    align_dir: Path,
    valid_pairs_path: Path,
    *,
    cluster_filter: Optional[Iterable[str]] = None,
    method_filter: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Run the full analysis and return the nested results dict.

    ``align_dir`` -- directory containing ``align_part*.json`` files.
    ``valid_pairs_path`` -- promise-bench ``valid_pairs.json``.
    ``cluster_filter`` / ``method_filter`` -- optional subset (for testing).
    """
    methods = (
        [normalize_method(m) for m in method_filter] if method_filter else list(METHODS)
    )

    print(f"[1/3] Loading align rows from {align_dir}", flush=True)
    rows = load_align_rows(
        align_dir,
        cluster_filter=list(cluster_filter) if cluster_filter else None,
        method_filter=methods,
    )
    print(f"      -> {len(rows)} rows after filtering", flush=True)
    idx = _rows_index(rows)
    idx_cm = _index_rows_by_cluster_method(rows)

    print(f"[2/3] Loading valid_pairs from {valid_pairs_path}", flush=True)
    with open(valid_pairs_path) as fh:
        valid_pairs_raw = json.load(fh)
    valid_pairs: Dict[str, Dict[str, List[List[str]]]] = {}
    for raw_set, cmap in valid_pairs_raw.items():
        valid_pairs[normalize_set_type(raw_set)] = cmap

    all_results: Dict[str, Any] = {
        st: {} for st in SET_TYPES
    }
    all_results["metadata"] = {
        "tm_threshold": TM_THRESHOLD,
        "methods": methods,
        "set_types": list(SET_TYPES),
        "align_dir": str(align_dir),
        "valid_pairs": str(valid_pairs_path),
    }

    print("[3/3] Processing clusters", flush=True)

    # ------ intrinsic ------
    intrinsic_clusters = sorted(valid_pairs.get("intrinsic", {}).keys())
    if cluster_filter:
        cf = set(cluster_filter)
        intrinsic_clusters = [c for c in intrinsic_clusters if c in cf]
    for ci, cluster_id in enumerate(intrinsic_clusters, 1):
        all_results["intrinsic"].setdefault(cluster_id, {})
        for method in methods:
            cluster_rows = idx.get(("intrinsic", cluster_id, method), [])
            if not cluster_rows:
                all_results["intrinsic"][cluster_id][method] = {
                    "method": method,
                    "cluster_id": cluster_id,
                    "success": False,
                    "expected_confs": [],
                    "covered_confs": [],
                    "missing_confs": [],
                    "conf_to_tags": {},
                    "n_samples": 0,
                    "samples": [],
                    "no_data": True,
                }
                continue
            all_results["intrinsic"][cluster_id][method] = (
                analyze_intrinsic_cluster_method(cluster_rows, cluster_id, method)
            )
        if ci % 50 == 0 or ci == len(intrinsic_clusters):
            print(
                f"      intrinsic: {ci}/{len(intrinsic_clusters)} clusters",
                flush=True,
            )

    # ------ induced ------
    for set_type in ("ligand-induced", "protein-induced"):
        clusters = sorted(valid_pairs.get(set_type, {}).keys())
        if cluster_filter:
            cf = set(cluster_filter)
            clusters = [c for c in clusters if c in cf]
        for ci, cluster_id in enumerate(clusters, 1):
            all_results[set_type].setdefault(cluster_id, {})
            pairs = valid_pairs[set_type][cluster_id]
            for method in methods:
                if method == "bioemu":
                    # bioemu has no per-sample induced predictions
                    continue
                rows_cm = idx.get((set_type, cluster_id, method), [])
                apo_tm_lookup = _build_cluster_apo_tm_lookup(
                    idx_cm.get((cluster_id, method), [])
                )
                pair_results: List[Dict[str, Any]] = []
                for vp in pairs:
                    pair_results.append(
                        analyze_induced_pair(
                            rows_cm,
                            cluster_id,
                            method,
                            set_type,
                            list(vp),
                            cluster_apo_tm_lookup=apo_tm_lookup,
                        )
                    )
                all_results[set_type][cluster_id][method] = pair_results
            if ci % 100 == 0 or ci == len(clusters):
                print(
                    f"      {set_type}: {ci}/{len(clusters)} clusters",
                    flush=True,
                )

    return all_results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _summary_lines(results: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    methods = results["metadata"]["methods"]

    lines.append("INTRINSIC (cluster-level success):")
    total = len(results.get("intrinsic", {}))
    for m in methods:
        ok = sum(
            1
            for c in results["intrinsic"]
            if results["intrinsic"][c].get(m, {}).get("success")
        )
        rate = ok / total if total else 0.0
        lines.append(f"  {m:<8} {ok}/{total} = {rate:.1%}")

    for st in ("ligand-induced", "protein-induced"):
        lines.append("")
        lines.append(f"{st.upper()} (pair-level success):")
        for m in methods:
            if m == "bioemu":
                continue
            total = 0
            ok = 0
            for c in results.get(st, {}):
                pairs = results[st][c].get(m, [])
                total += len(pairs)
                ok += sum(1 for p in pairs if p.get("overall_success"))
            rate = ok / total if total else 0.0
            lines.append(f"  {m:<8} {ok}/{total} = {rate:.1%}")
    return lines


def _write_sample_csvs(results: Dict[str, Any], output_dir: Path) -> None:
    """Write per-sample CSVs (intrinsic + induced) using pure csv module."""
    import csv

    intr_rows = []
    for cluster_id, mmap in results.get("intrinsic", {}).items():
        for method, res in mmap.items():
            if res.get("no_data"):
                continue
            for s in res.get("samples", []):
                for conf, info in s.get("conf_successes", {}).items():
                    intr_rows.append(
                        {
                            "cluster_id": cluster_id,
                            "method": method,
                            "sample_name": s["sample_name"],
                            "model_entity": s.get("model_entity"),
                            "conf": conf,
                            "best_tm": info["best_tm"],
                            "conf_success": info["success"],
                        }
                    )
    # Attach per-sample confidence to the flat CSV rows (if available)
    cs_map: Dict[Tuple[str, str], Optional[float]] = {}
    for cluster_id, mmap in results.get("intrinsic", {}).items():
        for method, res in mmap.items():
            for s in res.get("samples", []) or []:
                cs_map[(s["sample_name"], method)] = s.get("confidence_score")
    for r in intr_rows:
        r["confidence_score"] = cs_map.get((r["sample_name"], r["method"]))

    if intr_rows:
        fp = output_dir / "sample_details_intrinsic.csv"
        with open(fp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(intr_rows[0].keys()))
            w.writeheader()
            w.writerows(intr_rows)
        print(f"  Saved: {fp} ({len(intr_rows)} rows)")

    ind_rows = []
    for st in ("ligand-induced", "protein-induced"):
        for cluster_id, mmap in results.get(st, {}).items():
            for method, pairs in mmap.items():
                for p in pairs:
                    for side, samples in (
                        ("apo", p.get("apo_samples", [])),
                        ("holo", p.get("holo_samples", [])),
                    ):
                        for s in samples:
                            ind_rows.append(
                                {
                                    "cluster_id": cluster_id,
                                    "method": method,
                                    "set_type": st,
                                    "entity": side,
                                    "apo_tag": p.get("apo_tag"),
                                    "holo_tag": p.get("holo_tag"),
                                    "sample_name": s["sample_name"],
                                    "apo_tm": s.get("apo_tm"),
                                    "holo_tm": s.get("holo_tm"),
                                    "success": s.get("success"),
                                    "collapse": s.get("collapse"),
                                    "confidence_score": s.get("confidence_score"),
                                }
                            )
    if ind_rows:
        fp = output_dir / "sample_details_induced.csv"
        with open(fp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ind_rows[0].keys()))
            w.writeheader()
            w.writerows(ind_rows)
        print(f"  Saved: {fp} ({len(ind_rows)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Compute pair-level conformation-success results."
    )
    parser.add_argument(
        "--align-dir",
        type=str,
        default=None,
        help="Dir of align_part*.json (default: eval.dirs.align_results).",
    )
    parser.add_argument(
        "--valid-pairs",
        type=str,
        default=None,
        help="Path to valid_pairs.json (default: eval.files.valid_pairs).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dir (default: eval.dirs.conformation_success).",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default=None,
        help="Comma-separated cluster ids to limit processing (testing).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Comma-separated methods (testing).",
    )
    parser.add_argument(
        "--no-csv", action="store_true", help="Skip CSV output."
    )
    parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="Skip per-sample confidence_score lookup (faster).",
    )
    args = parser.parse_args()

    from utils._config import eval_cfg as E

    align_dir = E.align_results_dir(args.align_dir)
    valid_pairs_path = E.distogram_valid_pairs_path(args.valid_pairs)
    output_dir = E.conformation_success_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONFORMATION-SUCCESS  pair_results")
    print("=" * 80)
    print(f"align_dir   = {align_dir}")
    print(f"valid_pairs = {valid_pairs_path}")
    print(f"output_dir  = {output_dir}")

    cluster_filter = (
        [c.strip() for c in args.cluster.split(",") if c.strip()]
        if args.cluster
        else None
    )
    method_filter = (
        [m.strip() for m in args.method.split(",") if m.strip()]
        if args.method
        else None
    )

    results = compute_pair_results(
        align_dir,
        valid_pairs_path,
        cluster_filter=cluster_filter,
        method_filter=method_filter,
    )

    if not args.no_confidence:
        print("[4/3] Attaching per-sample confidence_score")
        attach_confidence_to_results(results)

    json_path = output_dir / "pair_level_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=list)
    print(f"\nSaved: {json_path}")

    if not args.no_csv:
        _write_sample_csvs(results, output_dir)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for line in _summary_lines(results):
        print(line)


if __name__ == "__main__":
    _cli()

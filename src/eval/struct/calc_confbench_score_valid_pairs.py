#!/usr/bin/env python3
"""
ConfBench score calculation for valid pairs (structure-based; RMSD).

This script matches the JSON schema produced by
``foundation_model/dynamic_set/final/step4.calc_confbench_score_valid_pairs.py``,
but reads prediction↔reference RMSDs from ProMiSE-bench alignment-batch outputs.

Inputs
------
- ``valid_pairs.json``: expected to be keyed by:
  ``intrinsic``, ``ligand-induced``, ``protein-induced``.
- Alignment results: JSON emitted by ``eval.align.struct_align_batch`` (typically
  via ``eval.align.split_alignment_jobs``) containing per-task ``rmsd_ca``.
- reference↔reference metrics: JSON emitted by ``eval.struct.calc_reference_structural_metrics``
  under ``<ref_metrics_dir>/aligned_references/<cluster_id>/**/*_metrics.json``.

Output
------
- ``confbench_scores_valid_pairs.json`` compatible with ``eval/merge_all.py``.
- ``confbench_summary_valid_pairs.csv`` and ``confbench_validation_report.json``.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils._config import eval_cfg as E


MODELS = ["af3", "boltz-1", "boltz-2", "chai-1", "bioemu"]


def parse_tag_to_base(tag: str) -> str:
    """Remove suffix (_m or _x) from tag."""
    if tag.endswith("_m") or tag.endswith("_x"):
        return tag[:-2]
    return tag


def parse_ref_key_to_tag(ref_key: str) -> str:
    """Convert ref key like 'asm_1cke_1_A1' to tag like '1cke_1_A1'."""
    if ref_key.startswith("asm_"):
        return ref_key[4:]
    return ref_key


def calculate_confbench_score(
    rmsd_pred_ref1: float, rmsd_pred_ref2: float, rmsd_ref1_ref2: float
) -> float:
    numerator = rmsd_pred_ref1 - rmsd_pred_ref2
    denominator = sqrt(
        0.5 * (rmsd_pred_ref1**2 + rmsd_pred_ref2**2 + rmsd_ref1_ref2**2)
    )
    if denominator == 0:
        return 0.0
    return numerator / denominator


def parse_modeled_key(modeled_key: str, mobile_cif: Optional[str] = None) -> Tuple[int, int]:
    """
    Parse seed and model number from modeled_key or mobile_cif path.
    Mirrors the original script behavior.
    """
    seed_num = -1
    model_num = -1

    seed_match = re.search(r"seed_(\d+)", modeled_key)
    if seed_match:
        seed_num = int(seed_match.group(1))

    model_match = re.search(r"model_(\d+)_aligned", modeled_key)
    if model_match:
        model_num = int(model_match.group(1))
    else:
        model_match = re.search(r"_model_(\d+)", modeled_key)
        if model_match:
            model_num = int(model_match.group(1))

    if mobile_cif and (seed_num == -1 or model_num == -1):
        if seed_num == -1:
            seed_match_cif = re.search(r"/seed_(\d+)/", mobile_cif)
            if seed_match_cif:
                seed_num = int(seed_match_cif.group(1))
        if model_num == -1:
            model_match_cif = re.search(r"_model_(\d+)\.(cif|pdb)$", mobile_cif)
            if model_match_cif:
                model_num = int(model_match_cif.group(1))
            else:
                sample_match = re.search(r"sample-(\d+)", mobile_cif)
                if sample_match:
                    model_num = int(sample_match.group(1))
    return seed_num, model_num


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# In-memory caches: align rows (1.4GB-ish) and ref-metrics files are otherwise
# scanned once per (pair × model × group), turning the script into a multi-hour
# disk-bound job. We load each artefact exactly once, index it, and let
# subsequent lookups hit the index.
# ---------------------------------------------------------------------------


def _normalize_align_sources(
    align_results: Any,
) -> Tuple[str, ...]:
    """Coerce ``--align-results`` value(s) into an immutable, hashable tuple.

    Accepts a single :class:`Path`/str (legacy) or a list/tuple of them
    (multi-source mode used to merge separately-run alignment batches such as
    ``job_batches/align_results`` + ``job_batches_boltz2/align_results``).
    """
    if align_results is None:
        return ()
    if isinstance(align_results, (str, Path)):
        return (str(align_results),)
    return tuple(str(p) for p in align_results)


@functools.lru_cache(maxsize=8)
def _load_all_align_rows_cached(
    align_results_strs: Tuple[str, ...],
) -> Tuple[Dict[str, Any], ...]:
    """Load every row from one or more align_results sources ONCE.

    Each source may be a directory containing ``align_part*.json`` files or a
    single JSON file. Results from all sources are concatenated. Returns an
    immutable tuple so :class:`functools.lru_cache` can hold the result
    safely for repeated calls within the same process.
    """
    rows: List[Dict[str, Any]] = []
    for s in align_results_strs:
        align_results = Path(s)
        if align_results.is_file():
            try:
                rows.extend(_load_json(align_results))
            except Exception as e:
                print(f"  Warning: failed to load {align_results}: {e}")
        else:
            files = sorted(align_results.glob("align_part*.json"))
            if not files:
                print(f"  Warning: no align_part*.json under {align_results}")
            for p in files:
                try:
                    rows.extend(_load_json(p))
                except Exception as e:
                    print(f"  Warning: failed to load {p}: {e}")
    return tuple(rows)


@functools.lru_cache(maxsize=8)
def _align_index_cached(
    align_results_strs: Tuple[str, ...],
) -> Dict[Tuple[str, str, str], Tuple[Dict[str, Any], ...]]:
    """Group rows by ``(prediction_method, pair_type, cluster_id)``.

    Only rows with ``ok == True`` are kept, so the per-call filter loop in
    :func:`_load_prediction_rmsds_from_align_results` can scan a small bucket
    instead of the full 800k-row list.
    """
    rows = _load_all_align_rows_cached(align_results_strs)
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("ok"):
            continue
        key = (
            str(row.get("prediction_method", "")),
            str(row.get("pair_type", "")),
            str(row.get("cluster_id", "")),
        )
        buckets[key].append(row)
    return {k: tuple(v) for k, v in buckets.items()}


def _iter_align_rows(align_results: Any) -> Iterable[Dict[str, Any]]:
    """Backward-compatible iterator (no longer used on the hot path)."""
    yield from _load_all_align_rows_cached(_normalize_align_sources(align_results))


def _infer_model_entity_from_row(row: Dict[str, Any]) -> Optional[str]:
    """
    ProMiSE-bench does not emit model_entity in alignment results; recover it from output_cif:
      <...>/<model>/<set>/<cluster>/<model_entity>/<file>.cif
    """
    outp = row.get("output_cif")
    if not outp:
        return None
    p = Path(str(outp))
    return p.parent.name if p.parent.name else None


def _load_prediction_rmsds_from_align_results(
    align_results: Any,
    model: str,
    set_name: str,
    cluster_id: str,
    model_entity: Optional[str],
    target_valid_pair: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """
    Return a list of dicts matching the original script's expectations:
      { modeled_key, ref_key, ca_rmsd, mobile_cif, valid_pair }
    """
    target_set = set(target_valid_pair) if target_valid_pair else None
    index = _align_index_cached(_normalize_align_sources(align_results))
    bucket = index.get((str(model), str(set_name), str(cluster_id)), ())

    out: List[Dict[str, Any]] = []
    for row in bucket:
        inferred_entity = _infer_model_entity_from_row(row)
        if model_entity and inferred_entity != model_entity:
            continue

        vp = row.get("valid_pair")
        if target_set is not None:
            if vp is None or set(vp) != target_set:
                continue

        ref_cif = str(row.get("ref_cif", ""))
        ref_chain = str(row.get("ref_chain", ""))
        if not ref_cif or not ref_chain:
            continue

        ref_key = f"{Path(ref_cif).stem}_{ref_chain}"
        ca_rmsd = row.get("rmsd_ca")
        if ca_rmsd is None:
            continue

        out.append(
            {
                "modeled_key": str(row.get("output_cif", "")),  # best proxy in this repo
                "ref_key": ref_key,
                "ca_rmsd": float(ca_rmsd),
                "mobile_cif": str(row.get("mobile_cif", "")),
                "valid_pair": vp,
            }
        )
    return out


@functools.lru_cache(maxsize=4096)
def _load_cluster_ref_metrics_cached(
    ref_metrics_dir_str: str, cluster_id: str
) -> Tuple[Dict[str, Any], ...]:
    """Read every ``*_metrics.json`` under one cluster's aligned_references dir.

    Cached per (ref_metrics_dir, cluster_id) so intrinsic / induced-set
    lookups against the same cluster do not re-walk the directory tree.
    """
    base = Path(ref_metrics_dir_str) / "aligned_references" / cluster_id
    if not base.exists():
        return ()
    rows: List[Dict[str, Any]] = []
    for metrics_file in base.rglob("*_metrics.json"):
        try:
            rows.append(_load_json(metrics_file))
        except Exception:
            continue
    return tuple(rows)


def _load_reference_rmsd_from_metrics(
    ref_metrics_dir: Path, cluster_id: str, ref1_base: str, ref2_base: str
) -> Optional[float]:
    for data in _load_cluster_ref_metrics_cached(str(ref_metrics_dir), str(cluster_id)):
        ca_rmsd = data.get("ca_rmsd")
        if ca_rmsd is None:
            continue
        ref1_tag_base = parse_tag_to_base(str(data.get("ref1_tag", "")))
        ref2_tag_base = parse_tag_to_base(str(data.get("ref2_tag", "")))
        if (ref1_tag_base == ref1_base and ref2_tag_base == ref2_base) or (
            ref1_tag_base == ref2_base and ref2_tag_base == ref1_base
        ):
            return float(ca_rmsd)
    return None


def _extract_pair_for_set(pair_info: Any) -> Tuple[List[str], Dict[str, Any]]:
    if isinstance(pair_info, dict):
        pair = pair_info.get("valid_pair", pair_info)
        return list(pair), pair_info
    return list(pair_info), {"valid_pair": list(pair_info)}


def process_intrinsic(
    align_results: Any,
    ref_metrics_dir: Path,
    pair_info: Dict,
    cluster_id: str,
    models: List[str],
) -> Dict[str, Any]:
    valid_pair = (
        pair_info.get("valid_pair", pair_info) if isinstance(pair_info, dict) else pair_info
    )
    model_entity = pair_info.get("model_entity") if isinstance(pair_info, dict) else None

    result: Dict[str, Any] = {
        "set_type": "intrinsic",
        "cluster_id": cluster_id,
        "valid_pair": valid_pair,
        "model_entity": model_entity,
        "ref1": parse_tag_to_base(valid_pair[0]),
        "ref2": parse_tag_to_base(valid_pair[1]),
        "rmsd_ref1_ref2": None,
        "models": {},
    }

    ref1_base = result["ref1"]
    ref2_base = result["ref2"]
    rmsd_ref = _load_reference_rmsd_from_metrics(ref_metrics_dir, cluster_id, ref1_base, ref2_base)
    result["rmsd_ref1_ref2"] = rmsd_ref
    if rmsd_ref is None:
        return result

    set_name = "intrinsic"
    for model in models:
        model_result: Dict[str, Any] = {
            "predictions": [],
            "mean_rmsd_pred_ref1": None,
            "mean_rmsd_pred_ref2": None,
            "mean_confbench_score": None,
            "n_predictions": 0,
        }

        pred_data = _load_prediction_rmsds_from_align_results(
            align_results,
            model=model,
            set_name=set_name,
            cluster_id=cluster_id,
            model_entity=model_entity,
            target_valid_pair=valid_pair,
        )
        if not pred_data:
            result["models"][model] = model_result
            continue

        pred_by_key: Dict[Tuple[int, int], Dict[str, Any]] = defaultdict(dict)
        entry_index = 0
        for entry in pred_data:
            modeled_key = entry.get("modeled_key", "")
            ref_key = entry.get("ref_key", "")
            ca_rmsd = entry.get("ca_rmsd")
            mobile_cif = entry.get("mobile_cif", "")
            if ca_rmsd is None:
                continue
            seed_num, model_num = parse_modeled_key(modeled_key, mobile_cif)
            if seed_num == -1 and model_num == -1:
                pred_key = (entry_index // 2, entry_index % 2)
            else:
                pred_key = (seed_num, model_num)
            ref_base = parse_ref_key_to_tag(ref_key)
            if ref_base == ref1_base:
                pred_by_key[pred_key]["ref1"] = float(ca_rmsd)
                pred_by_key[pred_key]["seed"] = seed_num
                pred_by_key[pred_key]["model"] = model_num
            elif ref_base == ref2_base:
                pred_by_key[pred_key]["ref2"] = float(ca_rmsd)
            entry_index += 1

        all_rmsd_ref1: List[float] = []
        all_rmsd_ref2: List[float] = []
        all_scores: List[float] = []

        for pred_key, rmsd_dict in pred_by_key.items():
            if "ref1" not in rmsd_dict or "ref2" not in rmsd_dict:
                continue
            rmsd_pred_ref1 = float(rmsd_dict["ref1"])
            rmsd_pred_ref2 = float(rmsd_dict["ref2"])
            score = calculate_confbench_score(rmsd_pred_ref1, rmsd_pred_ref2, float(rmsd_ref))
            model_result["predictions"].append(
                {
                    "seed": rmsd_dict.get("seed", pred_key[0]),
                    "model_num": rmsd_dict.get("model", pred_key[1]),
                    "rmsd_pred_ref1": round(rmsd_pred_ref1, 4),
                    "rmsd_pred_ref2": round(rmsd_pred_ref2, 4),
                    "confbench_score": round(score, 4),
                }
            )
            all_rmsd_ref1.append(rmsd_pred_ref1)
            all_rmsd_ref2.append(rmsd_pred_ref2)
            all_scores.append(score)

        if all_scores:
            model_result["mean_rmsd_pred_ref1"] = round(float(np.mean(all_rmsd_ref1)), 4)
            model_result["mean_rmsd_pred_ref2"] = round(float(np.mean(all_rmsd_ref2)), 4)
            model_result["mean_confbench_score"] = round(float(np.mean(all_scores)), 4)
            model_result["n_predictions"] = int(len(all_scores))

        result["models"][model] = model_result

    return result


def _process_induced_set(
    set_type: str,
    align_results: Any,
    ref_metrics_dir: Path,
    pair_info: Dict,
    cluster_id: str,
    models: List[str],
) -> Dict[str, Any]:
    valid_pair = (
        pair_info.get("valid_pair", pair_info) if isinstance(pair_info, dict) else pair_info
    )
    apo_model_entity = pair_info.get("apo_model_entity") if isinstance(pair_info, dict) else None
    holo_model_entity = pair_info.get("holo_model_entity") if isinstance(pair_info, dict) else None

    apo_tag = next((t for t in valid_pair if str(t).endswith("_m")), None)
    holo_tag = next((t for t in valid_pair if str(t).endswith("_x")), None)
    if apo_tag is None or holo_tag is None:
        return {}

    if apo_model_entity is None:
        apo_model_entity = apo_tag
    if holo_model_entity is None:
        holo_model_entity = holo_tag

    apo_ref = parse_tag_to_base(apo_tag)
    holo_ref = parse_tag_to_base(holo_tag)

    result: Dict[str, Any] = {
        "set_type": set_type,
        "cluster_id": cluster_id,
        "valid_pair": valid_pair,
        "apo_model_entity": apo_model_entity,
        "holo_model_entity": holo_model_entity,
        "apo_ref": apo_ref,
        "holo_ref": holo_ref,
        "rmsd_apo_holo_ref": None,
        "models": {},
    }

    rmsd_ref = _load_reference_rmsd_from_metrics(ref_metrics_dir, cluster_id, apo_ref, holo_ref)
    result["rmsd_apo_holo_ref"] = rmsd_ref
    if rmsd_ref is None:
        return result

    models_for_induced = [m for m in models if m != "bioemu"]

    def calc_one_prediction_group(
        pred_data: List[Dict[str, Any]],
        ref_a: str,
        ref_b: str,
    ) -> Tuple[List[Dict[str, Any]], Optional[float], Optional[float], Optional[float], int]:
        pred_by_key: Dict[Any, Dict[str, Any]] = defaultdict(dict)
        entry_index = 0
        for entry in pred_data:
            modeled_key = entry.get("modeled_key", "")
            ref_key = entry.get("ref_key", "")
            ca_rmsd = entry.get("ca_rmsd")
            mobile_cif = entry.get("mobile_cif", "")
            if ca_rmsd is None:
                continue
            seed_num, model_num = parse_modeled_key(modeled_key, mobile_cif)
            pred_key: Any
            if seed_num == -1 and model_num == -1:
                pred_key = mobile_cif if mobile_cif else f"entry_{entry_index}"
            else:
                pred_key = (seed_num, model_num)
            ref_base = parse_ref_key_to_tag(ref_key)
            if ref_base == ref_a:
                pred_by_key[pred_key]["a"] = float(ca_rmsd)
                pred_by_key[pred_key]["seed"] = seed_num
                pred_by_key[pred_key]["model"] = model_num
            elif ref_base == ref_b:
                pred_by_key[pred_key]["b"] = float(ca_rmsd)
            entry_index += 1

        preds: List[Dict[str, Any]] = []
        all_a: List[float] = []
        all_b: List[float] = []
        all_scores: List[float] = []
        for pred_key, d in pred_by_key.items():
            if "a" not in d or "b" not in d:
                continue
            rmsd_a = float(d["a"])
            rmsd_b = float(d["b"])
            score = calculate_confbench_score(rmsd_a, rmsd_b, float(rmsd_ref))
            seed_val = d.get("seed", -1)
            model_val = d.get("model", -1)
            if isinstance(pred_key, tuple):
                seed_val = pred_key[0] if seed_val == -1 else seed_val
                model_val = pred_key[1] if model_val == -1 else model_val
            preds.append(
                {
                    "seed": seed_val,
                    "model_num": model_val,
                    "rmsd_pred_apo_ref": round(rmsd_a, 4),
                    "rmsd_pred_holo_ref": round(rmsd_b, 4),
                    "confbench_score": round(score, 4),
                }
            )
            all_a.append(rmsd_a)
            all_b.append(rmsd_b)
            all_scores.append(score)
        if not all_scores:
            return preds, None, None, None, 0
        return (
            preds,
            round(float(np.mean(all_a)), 4),
            round(float(np.mean(all_b)), 4),
            round(float(np.mean(all_scores)), 4),
            int(len(all_scores)),
        )

    for model in models_for_induced:
        model_result: Dict[str, Any] = {
            "apo_predictions": {
                "tag": apo_tag,
                "predictions": [],
                "mean_rmsd_pred_apo_ref": None,
                "mean_rmsd_pred_holo_ref": None,
                "mean_confbench_score": None,
                "n_predictions": 0,
            },
            "holo_predictions": {
                "tag": holo_tag,
                "predictions": [],
                "mean_rmsd_pred_apo_ref": None,
                "mean_rmsd_pred_holo_ref": None,
                "mean_confbench_score": None,
                "n_predictions": 0,
            },
        }

        pred_data_apo = _load_prediction_rmsds_from_align_results(
            align_results,
            model=model,
            set_name=set_type,
            cluster_id=cluster_id,
            model_entity=apo_model_entity,
            target_valid_pair=valid_pair,
        )
        preds, m_a, m_b, m_s, n = calc_one_prediction_group(pred_data_apo, apo_ref, holo_ref)
        model_result["apo_predictions"]["predictions"] = preds
        model_result["apo_predictions"]["mean_rmsd_pred_apo_ref"] = m_a
        model_result["apo_predictions"]["mean_rmsd_pred_holo_ref"] = m_b
        model_result["apo_predictions"]["mean_confbench_score"] = m_s
        model_result["apo_predictions"]["n_predictions"] = n

        pred_data_holo = _load_prediction_rmsds_from_align_results(
            align_results,
            model=model,
            set_name=set_type,
            cluster_id=cluster_id,
            model_entity=holo_model_entity,
            target_valid_pair=valid_pair,
        )
        preds, m_a, m_b, m_s, n = calc_one_prediction_group(pred_data_holo, apo_ref, holo_ref)
        model_result["holo_predictions"]["predictions"] = preds
        model_result["holo_predictions"]["mean_rmsd_pred_apo_ref"] = m_a
        model_result["holo_predictions"]["mean_rmsd_pred_holo_ref"] = m_b
        model_result["holo_predictions"]["mean_confbench_score"] = m_s
        model_result["holo_predictions"]["n_predictions"] = n

        result["models"][model] = model_result

    return result


def validate_results(all_results: Dict[str, Any], valid_pairs_data: Dict[str, Any], models: List[str]) -> Dict[str, Any]:
    validation: Dict[str, Any] = {
        "missing": {"intrinsic": [], "ligand-induced": [], "protein-induced": []},
        "complete": {"intrinsic": [], "ligand-induced": [], "protein-induced": []},
        "summary": {
            "intrinsic": {"total_pairs": 0, "complete_pairs": 0, "missing_by_model": {}},
            "ligand-induced": {"total_pairs": 0, "complete_pairs": 0, "missing_by_model": {}},
            "protein-induced": {"total_pairs": 0, "complete_pairs": 0, "missing_by_model": {}},
        },
    }
    for set_type in validation["summary"]:
        for model in models:
            validation["summary"][set_type]["missing_by_model"][model] = {"apo": 0, "holo": 0}

    for set_type in ["intrinsic", "ligand-induced", "protein-induced"]:
        set_pairs = valid_pairs_data.get(set_type, {})
        for cluster_id, pairs in (set_pairs or {}).items():
            for pair_info in pairs or []:
                pair, _info = _extract_pair_for_set(pair_info)
                if len(pair) != 2:
                    continue
                validation["summary"][set_type]["total_pairs"] += 1
                pair_key = f"{cluster_id}_{pair[0]}_{pair[1]}"
                result = all_results.get(set_type, {}).get(pair_key, {})
                pair_complete = True
                pair_missing = {"cluster_id": cluster_id, "pair": pair, "missing_models": []}

                for model in models:
                    model_data = result.get("models", {}).get(model, {})
                    if set_type == "intrinsic":
                        n_pred = model_data.get("n_predictions", 0)
                        score = model_data.get("mean_confbench_score")
                        if n_pred == 0 or score is None:
                            pair_complete = False
                            pair_missing["missing_models"].append(
                                {"model": model, "n_predictions": n_pred, "has_score": score is not None}
                            )
                            validation["summary"][set_type]["missing_by_model"][model]["apo"] += 1
                    else:
                        apo = model_data.get("apo_predictions", {})
                        holo = model_data.get("holo_predictions", {})
                        apo_n = apo.get("n_predictions", 0)
                        holo_n = holo.get("n_predictions", 0)
                        apo_score = apo.get("mean_confbench_score")
                        holo_score = holo.get("mean_confbench_score")
                        model_issues = {"model": model, "apo": None, "holo": None}
                        has_issue = False
                        if apo_n == 0 or apo_score is None:
                            pair_complete = False
                            has_issue = True
                            model_issues["apo"] = {"n_predictions": apo_n, "has_score": apo_score is not None}
                            validation["summary"][set_type]["missing_by_model"][model]["apo"] += 1
                        if holo_n == 0 or holo_score is None:
                            pair_complete = False
                            has_issue = True
                            model_issues["holo"] = {"n_predictions": holo_n, "has_score": holo_score is not None}
                            validation["summary"][set_type]["missing_by_model"][model]["holo"] += 1
                        if has_issue:
                            pair_missing["missing_models"].append(model_issues)

                if pair_complete:
                    validation["complete"][set_type].append({"cluster_id": cluster_id, "pair": pair})
                    validation["summary"][set_type]["complete_pairs"] += 1
                else:
                    validation["missing"][set_type].append(pair_missing)

    return validation


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--valid-pairs",
        type=Path,
        default=E.file("valid_pairs"),
        help="valid_pairs.json (default: eval.files.valid_pairs)",
    )
    p.add_argument(
        "--align-results",
        type=Path,
        required=True,
        nargs="+",
        help=(
            "One or more directories containing align_part*.json (or single "
            "json files). Multiple sources are merged in-memory, useful when "
            "different prediction methods were aligned in separate batch "
            "runs (e.g. main job_batches + a boltz-2 rerun under "
            "job_batches_boltz2)."
        ),
    )
    p.add_argument(
        "--ref-metrics-dir",
        type=Path,
        default=Path("reference_metrics"),
        help="Root directory produced by eval.struct.calc_reference_structural_metrics (default: ./reference_metrics)",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=E.file("confbench_scores"),
        help="Output JSON (default: eval.files.confbench_scores)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: parent of --output-json)",
    )
    p.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS),
        help="Comma-separated model keys to include (default: af3,boltz-1,boltz-2,chai-1,bioemu)",
    )
    args = p.parse_args()

    valid_pairs_path = Path(args.valid_pairs)
    if not valid_pairs_path.exists():
        raise FileNotFoundError(f"valid_pairs not found: {valid_pairs_path}")
    valid_pairs_data = _load_json(valid_pairs_path)

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    all_results: Dict[str, Dict[str, Any]] = {
        "intrinsic": {},
        "ligand-induced": {},
        "protein-induced": {},
    }

    out_dir = args.output_dir or Path(args.output_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # intrinsic dynamics set
    intrinsic_data = valid_pairs_data.get("intrinsic", {})
    for cluster_id, pairs in (intrinsic_data or {}).items():
        for pair_info in pairs or []:
            pair, info = _extract_pair_for_set(pair_info)
            if len(pair) != 2:
                continue
            res = process_intrinsic(args.align_results, args.ref_metrics_dir, info, str(cluster_id), models)
            pair_key = f"{cluster_id}_{pair[0]}_{pair[1]}"
            all_results["intrinsic"][pair_key] = res

    # induced sets
    for set_type in ["ligand-induced", "protein-induced"]:
        set_data = valid_pairs_data.get(set_type, {})
        for cluster_id, pairs in (set_data or {}).items():
            for pair_info in pairs or []:
                pair, info = _extract_pair_for_set(pair_info)
                if len(pair) != 2:
                    continue
                res = _process_induced_set(set_type, args.align_results, args.ref_metrics_dir, info, str(cluster_id), models)
                pair_key = f"{cluster_id}_{pair[0]}_{pair[1]}"
                all_results[set_type][pair_key] = res

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {out_json}")

    # summary CSV (match original filenames)
    summary_rows: List[Dict[str, Any]] = []
    for pair_key, result in all_results["intrinsic"].items():
        for model in models:
            model_data = result.get("models", {}).get(model, {})
            summary_rows.append(
                {
                    "set_type": "intrinsic",
                    "cluster_id": result.get("cluster_id"),
                    "model": model,
                    "prediction_type": "apo",
                    "ref1": result.get("ref1"),
                    "ref2": result.get("ref2"),
                    "rmsd_ref1_ref2": result.get("rmsd_ref1_ref2"),
                    "mean_rmsd_pred_ref1": model_data.get("mean_rmsd_pred_ref1"),
                    "mean_rmsd_pred_ref2": model_data.get("mean_rmsd_pred_ref2"),
                    "mean_confbench_score": model_data.get("mean_confbench_score"),
                    "n_predictions": model_data.get("n_predictions", 0),
                }
            )

    for set_type in ["ligand-induced", "protein-induced"]:
        for pair_key, result in all_results[set_type].items():
            for model in models:
                model_data = result.get("models", {}).get(model, {})
                apo_data = model_data.get("apo_predictions", {})
                holo_data = model_data.get("holo_predictions", {})
                summary_rows.append(
                    {
                        "set_type": set_type,
                        "cluster_id": result.get("cluster_id"),
                        "model": model,
                        "prediction_type": "apo",
                        "apo_ref": result.get("apo_ref"),
                        "holo_ref": result.get("holo_ref"),
                        "rmsd_apo_holo_ref": result.get("rmsd_apo_holo_ref"),
                        "mean_rmsd_pred_apo_ref": apo_data.get("mean_rmsd_pred_apo_ref"),
                        "mean_rmsd_pred_holo_ref": apo_data.get("mean_rmsd_pred_holo_ref"),
                        "mean_confbench_score": apo_data.get("mean_confbench_score"),
                        "n_predictions": apo_data.get("n_predictions", 0),
                    }
                )
                summary_rows.append(
                    {
                        "set_type": set_type,
                        "cluster_id": result.get("cluster_id"),
                        "model": model,
                        "prediction_type": "holo",
                        "apo_ref": result.get("apo_ref"),
                        "holo_ref": result.get("holo_ref"),
                        "rmsd_apo_holo_ref": result.get("rmsd_apo_holo_ref"),
                        "mean_rmsd_pred_apo_ref": holo_data.get("mean_rmsd_pred_apo_ref"),
                        "mean_rmsd_pred_holo_ref": holo_data.get("mean_rmsd_pred_holo_ref"),
                        "mean_confbench_score": holo_data.get("mean_confbench_score"),
                        "n_predictions": holo_data.get("n_predictions", 0),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "confbench_summary_valid_pairs.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    # validation report
    valid_pairs_for_validation = {
        "intrinsic": valid_pairs_data.get("intrinsic", {}),
        "ligand-induced": valid_pairs_data.get("ligand-induced", {}),
        "protein-induced": valid_pairs_data.get("protein-induced", {}),
    }
    validation_result = validate_results(all_results, valid_pairs_for_validation, models)
    validation_path = out_dir / "confbench_validation_report.json"
    with open(validation_path, "w") as f:
        json.dump(validation_result, f, indent=2)
    print(f"Validation report saved to: {validation_path}")


if __name__ == "__main__":
    main()


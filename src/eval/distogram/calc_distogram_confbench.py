#!/usr/bin/env python3
"""
Distogram-based ConfBench scores for valid pairs.

This file is intentionally kept as ``calc_distogram_confbench.py`` and matches the
final script used in the foundation repo (``step14_yubeen_re.py``), adapted to
ProMiSE-bench paths/config and distogram artefact layout.

Inputs
------
- ``valid_pairs.json``: by set type (apo-monomers / ligand-induced / protein-induced).
- ``distogram_tasks.json``: produced by ``collect_distograms`` (and used by loss/diff steps).
- ``reference_distogram_diff.json`` tree: produced by ``calc_reference_distogram_diff``.
- Per-prediction distogram loss JSON: ``distogram_loss_real_final.json`` written next to
  each prediction distogram directory by ``calc_distogram_loss``.

Output
------
Writes a JSON compatible with ``eval/merge_all.py`` (default: ``eval.files.confbench_distogram``).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils._config import eval_cfg as E
from eval.distogram.path_utils import reference_distogram_diff_path


# =============================================================================
# Helper Functions (ported from step14_yubeen_re.py)
# =============================================================================


def calc_confbench_score(
    dist_pred_ref1: float,
    dist_pred_ref2: float,
    dist_ref1_ref2: float,
) -> float:
    """
    Score = (dist_ref1 - dist_ref2) / sqrt(0.5 * (dist_ref1^2 + dist_ref2^2 + dist_ref1_ref2^2))
    Positive score means prediction is closer to ref2 (typically holo).
    """
    try:
        if (
            dist_pred_ref1 is None
            or dist_pred_ref2 is None
            or dist_ref1_ref2 is None
            or math.isnan(dist_pred_ref1)
            or math.isnan(dist_pred_ref2)
            or math.isnan(dist_ref1_ref2)
        ):
            return float("nan")
        denominator = math.sqrt(
            0.5 * (dist_pred_ref1**2 + dist_pred_ref2**2 + dist_ref1_ref2**2)
        )
        if denominator == 0:
            return float("nan")
        return (dist_pred_ref1 - dist_pred_ref2) / denominator
    except Exception:
        return float("nan")


def try_load_dist_diff(
    ref_distogram_file: Path, refA: str, refB: str, method: str
) -> Optional[Tuple[float, float]]:
    """
    Load (dist, dynamic_dist) from reference_distogram_diff.json by searching pairwise_comparisons.
    """
    if not ref_distogram_file.exists():
        return None

    with open(ref_distogram_file, "r") as f:
        data = json.load(f)

    for pairwise_comparison in data.get("pairwise_comparisons", []):
        a = pairwise_comparison.get("reference_A")
        b = pairwise_comparison.get("reference_B")
        if (a == refA and b == refB) or (a == refB and b == refA):
            if method in ["af3", "bioemu"]:
                dist = pairwise_comparison.get("mean_distance_difference_af3")
                dynamic_dist = pairwise_comparison.get("dynamic_mean_distance_difference_af3")
            elif method in ("boltz1", "boltz2"):
                dist = pairwise_comparison.get("mean_distance_difference_boltz")
                dynamic_dist = pairwise_comparison.get("dynamic_mean_distance_difference_boltz")
            else:
                return None
            if dist is None:
                return None
            return float(dist), (float(dynamic_dist) if dynamic_dist is not None else None)
    return None


def get_pair_type_from_path(path: Path) -> Optional[str]:
    parts = path.parts
    for method_type in ["apo-monomers", "ligand-induced", "protein-induced"]:
        if method_type in parts:
            return method_type
    return None


def distogram_loss_path_from_task(task: Dict[str, Any]) -> Optional[Path]:
    prediction_distograms = task.get("prediction_distograms", [])
    if not prediction_distograms:
        return None
    distogram_dir = Path(prediction_distograms[0]).parent
    return distogram_dir / "distogram_loss_real_final.json"


def parse_seed_results(
    distogram_loss_file: Path, ref1: str, ref2: str
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Parse seed results from distogram_loss_real_final.json."""
    per_seed_loss: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    with open(distogram_loss_file, "r") as f:
        distogram_loss_data = json.load(f)[0]

    for references in distogram_loss_data.get("references", []):
        ref_tag = references.get("reference_yaml_tag")
        if ref_tag == ref1:
            ref_key = "ref1"
        elif ref_tag == ref2:
            ref_key = "ref2"
        else:
            continue

        dynamic_key1 = f"{ref1}|{ref2}"
        dynamic_key2 = f"{ref2}|{ref1}"

        for seed_info in references.get("seed_results", []):
            try:
                seed_id = int(seed_info["seed"])
            except Exception:
                continue

            dynamic_loss = None
            dynamic_expectation_of_distances = seed_info.get("dynamic_expectation_of_distances", {})
            if dynamic_key1 in dynamic_expectation_of_distances:
                dynamic_loss = dynamic_expectation_of_distances[dynamic_key1]
            elif dynamic_key2 in dynamic_expectation_of_distances:
                dynamic_loss = dynamic_expectation_of_distances[dynamic_key2]

            dynamic_entropy = None
            dynamic_entropies = seed_info.get("dynamic_entropies", {})
            if dynamic_key1 in dynamic_entropies:
                dynamic_entropy = dynamic_entropies[dynamic_key1]
            elif dynamic_key2 in dynamic_entropies:
                dynamic_entropy = dynamic_entropies[dynamic_key2]

            total_loss = None
            total_losses = seed_info.get("total_expectation_of_distances_per_pair", {})
            if dynamic_key1 in total_losses:
                total_loss = total_losses[dynamic_key1]
            elif dynamic_key2 in total_losses:
                total_loss = total_losses[dynamic_key2]

            total_entropy = None
            total_entropies = seed_info.get("total_entropies_per_pair", {})
            if dynamic_key1 in total_entropies:
                total_entropy = total_entropies[dynamic_key1]
            elif dynamic_key2 in total_entropies:
                total_entropy = total_entropies[dynamic_key2]

            per_seed_loss[seed_id][ref_key] = {
                "total_loss": total_loss,
                "dynamic_loss": dynamic_loss,
                "dynamic_entropy": dynamic_entropy,
                "total_entropy": total_entropy,
            }

    return dict(per_seed_loss)


def build_predictions(
    per_seed_loss: dict,
    dist_ref1_ref2: float,
    dynamic_dist_ref1_ref2: Optional[float],
    ref1: str,
    ref2: str,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    for seed_id, losses in per_seed_loss.items():
        if "ref1" not in losses or "ref2" not in losses:
            continue

        dist_pred_ref1 = losses["ref1"].get("total_loss")
        dist_pred_ref2 = losses["ref2"].get("total_loss")
        dynamic_dist_pred_ref1 = losses["ref1"].get("dynamic_loss")
        dynamic_dist_pred_ref2 = losses["ref2"].get("dynamic_loss")
        dynamic_entropy_ref1 = losses["ref1"].get("dynamic_entropy")
        dynamic_entropy_ref2 = losses["ref2"].get("dynamic_entropy")
        total_entropy_ref1 = losses["ref1"].get("total_entropy")
        total_entropy_ref2 = losses["ref2"].get("total_entropy")

        confbench_score = calc_confbench_score(dist_pred_ref1, dist_pred_ref2, dist_ref1_ref2)

        if (
            dynamic_dist_pred_ref1 is not None
            and dynamic_dist_pred_ref2 is not None
            and dynamic_dist_ref1_ref2 is not None
        ):
            dynamic_confbench_score = calc_confbench_score(
                dynamic_dist_pred_ref1, dynamic_dist_pred_ref2, dynamic_dist_ref1_ref2
            )
        else:
            dynamic_confbench_score = float("nan")
            dynamic_dist_pred_ref1 = float("nan") if dynamic_dist_pred_ref1 is None else dynamic_dist_pred_ref1
            dynamic_dist_pred_ref2 = float("nan") if dynamic_dist_pred_ref2 is None else dynamic_dist_pred_ref2

        pred_dict: Dict[str, Any] = {
            "seed": seed_id,
            "dist_pred_ref1": dist_pred_ref1,
            "dist_pred_ref2": dist_pred_ref2,
            "confbench_score": confbench_score,
            "dynamic_dist_pred_ref1": dynamic_dist_pred_ref1,
            "dynamic_dist_pred_ref2": dynamic_dist_pred_ref2,
            "dynamic_confbench_score": dynamic_confbench_score,
            "dynamic_entropy_ref1": dynamic_entropy_ref1,
            "dynamic_entropy_ref2": dynamic_entropy_ref2,
        }
        if total_entropy_ref1 is not None:
            pred_dict["total_entropy_ref1"] = total_entropy_ref1
        if total_entropy_ref2 is not None:
            pred_dict["total_entropy_ref2"] = total_entropy_ref2

        predictions.append(pred_dict)
    return predictions


def calc_mean_scores(predictions: List[Dict[str, Any]]) -> Tuple[float, float]:
    valid_confbench = [p["confbench_score"] for p in predictions if not math.isnan(p["confbench_score"])]
    valid_dynamic_confbench = [
        p["dynamic_confbench_score"] for p in predictions if not math.isnan(p["dynamic_confbench_score"])
    ]
    mean_confbench = sum(valid_confbench) / len(valid_confbench) if valid_confbench else float("nan")
    mean_dynamic_confbench = (
        sum(valid_dynamic_confbench) / len(valid_dynamic_confbench)
        if valid_dynamic_confbench
        else float("nan")
    )
    return mean_confbench, mean_dynamic_confbench


# =============================================================================
# Processing Functions
# =============================================================================


def _task_index(tasks: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    idx: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for t in tasks:
        key = (t.get("method", ""), t.get("method_type", ""), t.get("cluster_id", ""), t.get("prediction_yaml_tag", ""))
        idx[key] = t
    return idx


def _find_task(
    idx: Dict[Tuple[str, str, str, str], Dict[str, Any]],
    method: str,
    pair_type: str,
    cluster_id: str,
    pred_yaml_tag: str,
) -> Optional[Dict[str, Any]]:
    return idx.get((method, pair_type, cluster_id, pred_yaml_tag))


def process_apo_monomers(
    task_idx: Dict[Tuple[str, str, str, str], Dict[str, Any]],
    ref_distogram_root: Path,
    cluster_id: str,
    pair_dict: Dict[str, Any],
    ref1: str,
    ref2: str,
    method: str,
    error_list: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    pair_type = "apo-monomers"
    model_entity = pair_dict.get("model_entity")
    if not model_entity:
        return None

    # reference_distogram_diff.json path (repo convention)
    t = _find_task(task_idx, method, pair_type, cluster_id, model_entity)
    if t is None:
        error_list.append({"pair_type": pair_type, "cluster_id": cluster_id, "method": method, "error": "task_not_found"})
        return None

    ref_diff_path = reference_distogram_diff_path(ref_distogram_root, t)
    dist_result = try_load_dist_diff(ref_diff_path, ref1, ref2, method)
    if dist_result is None:
        error_list.append(
            {
                "pair_type": pair_type,
                "cluster_id": cluster_id,
                "method": method,
                "ref1": ref1,
                "ref2": ref2,
                "error": "reference_distogram_diff.json not found or invalid",
                "path": str(ref_diff_path),
            }
        )
        return None

    dist_ref1_ref2, dynamic_dist_ref1_ref2 = dist_result

    distogram_loss_file = distogram_loss_path_from_task(t)
    if distogram_loss_file is None or not distogram_loss_file.exists():
        error_list.append(
            {
                "pair_type": pair_type,
                "cluster_id": cluster_id,
                "method": method,
                "ref1": ref1,
                "ref2": ref2,
                "model_entity": model_entity,
                "error": "distogram_loss_real_final.json not found",
            }
        )
        return None

    per_seed_loss = parse_seed_results(distogram_loss_file, ref1, ref2)
    if not per_seed_loss:
        error_list.append(
            {
                "pair_type": pair_type,
                "cluster_id": cluster_id,
                "method": method,
                "ref1": ref1,
                "ref2": ref2,
                "error": "No seed results found",
            }
        )
        return None

    # If dynamic_loss missing, attempt to swap method_type segment to apo-monomers (ported behavior)
    try:
        first_seed = list(per_seed_loss.keys())[0]
        if (
            per_seed_loss[first_seed]["ref1"].get("dynamic_loss") is None
            or per_seed_loss[first_seed]["ref2"].get("dynamic_loss") is None
        ):
            current_method_type = get_pair_type_from_path(distogram_loss_file)
            if current_method_type is not None and current_method_type != "apo-monomers":
                alt = Path(str(distogram_loss_file).replace(current_method_type, "apo-monomers"))
                if alt.exists():
                    per_seed_loss_apo = parse_seed_results(alt, ref1, ref2)
                    if per_seed_loss_apo:
                        per_seed_loss = per_seed_loss_apo
    except Exception:
        pass

    predictions = build_predictions(per_seed_loss, dist_ref1_ref2, dynamic_dist_ref1_ref2, ref1, ref2)
    mean_confbench, mean_dynamic_confbench = calc_mean_scores(predictions)
    return {
        "dist_ref1_ref2": dist_ref1_ref2,
        "dynamic_dist_ref1_ref2": dynamic_dist_ref1_ref2,
        "predictions": predictions,
        "mean_confbench_score": mean_confbench,
        "mean_dynamic_confbench_score": mean_dynamic_confbench,
    }


def process_induced(
    task_idx: Dict[Tuple[str, str, str, str], Dict[str, Any]],
    ref_distogram_root: Path,
    cluster_id: str,
    pair_dict: Dict[str, Any],
    ref1: str,
    ref2: str,
    method: str,
    pair_type: str,
    error_list: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if "_m" not in ref1 or "_x" not in ref2:
        error_list.append(
            {
                "pair_type": pair_type,
                "cluster_id": cluster_id,
                "method": method,
                "ref1": ref1,
                "ref2": ref2,
                "error": "Reference names do not follow expected naming convention",
            }
        )
        return None

    apo_model_entity = pair_dict.get("apo_model_entity")
    holo_model_entity = pair_dict.get("holo_model_entity")
    if not apo_model_entity or not holo_model_entity:
        return None

    t_apo = _find_task(task_idx, method, pair_type, cluster_id, apo_model_entity)
    if t_apo is None:
        return None
    ref_diff_path = reference_distogram_diff_path(ref_distogram_root, t_apo)
    dist_result = try_load_dist_diff(ref_diff_path, ref1, ref2, method)
    if dist_result is None:
        error_list.append(
            {
                "pair_type": pair_type,
                "cluster_id": cluster_id,
                "method": method,
                "ref1": ref1,
                "ref2": ref2,
                "error": "reference_distogram_diff.json not found or invalid",
                "path": str(ref_diff_path),
            }
        )
        return None

    dist_ref1_ref2, dynamic_dist_ref1_ref2 = dist_result

    t_holo = _find_task(task_idx, method, pair_type, cluster_id, holo_model_entity)
    if t_holo is None:
        return None

    distogram_loss_file_apo = distogram_loss_path_from_task(t_apo)
    distogram_loss_file_holo = distogram_loss_path_from_task(t_holo)
    if distogram_loss_file_apo is None or not distogram_loss_file_apo.exists():
        error_list.append(
            {"pair_type": pair_type, "cluster_id": cluster_id, "method": method, "error": "APO distogram_loss_real_final.json not found"}
        )
        return None
    if distogram_loss_file_holo is None or not distogram_loss_file_holo.exists():
        error_list.append(
            {"pair_type": pair_type, "cluster_id": cluster_id, "method": method, "error": "HOLO distogram_loss_real_final.json not found"}
        )
        return None

    # APO
    per_seed_loss_apo = parse_seed_results(distogram_loss_file_apo, ref1, ref2)
    try:
        first_seed = list(per_seed_loss_apo.keys())[0]
        if (
            per_seed_loss_apo[first_seed]["ref1"].get("dynamic_loss") is None
            or per_seed_loss_apo[first_seed]["ref2"].get("dynamic_loss") is None
        ):
            current_method_type = get_pair_type_from_path(distogram_loss_file_apo)
            if current_method_type is not None and current_method_type != "apo-monomers":
                alt = Path(str(distogram_loss_file_apo).replace(current_method_type, "apo-monomers"))
                if alt.exists():
                    per_seed_loss_apo2 = parse_seed_results(alt, ref1, ref2)
                    if per_seed_loss_apo2:
                        per_seed_loss_apo = per_seed_loss_apo2
    except Exception:
        pass

    apo_predictions = build_predictions(per_seed_loss_apo, dist_ref1_ref2, dynamic_dist_ref1_ref2, ref1, ref2)
    apo_mean_confbench, apo_mean_dynamic_confbench = calc_mean_scores(apo_predictions)

    # HOLO
    per_seed_loss_holo = parse_seed_results(distogram_loss_file_holo, ref1, ref2)
    holo_predictions = build_predictions(per_seed_loss_holo, dist_ref1_ref2, dynamic_dist_ref1_ref2, ref1, ref2)
    holo_mean_confbench, holo_mean_dynamic_confbench = calc_mean_scores(holo_predictions)

    return {
        "dist_ref1_ref2": dist_ref1_ref2,
        "dynamic_dist_ref1_ref2": dynamic_dist_ref1_ref2,
        "apo_predictions": {
            "tag": ref1,
            "predictions": apo_predictions,
            "mean_confbench_score": apo_mean_confbench,
            "mean_dynamic_confbench_score": apo_mean_dynamic_confbench,
        },
        "holo_predictions": {
            "tag": ref2,
            "predictions": holo_predictions,
            "mean_confbench_score": holo_mean_confbench,
            "mean_dynamic_confbench_score": holo_mean_dynamic_confbench,
        },
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--valid-pairs", type=str, default=None, help="valid_pairs.json (default: eval.files.valid_pairs)")
    p.add_argument("--tasks", "-t", type=str, default=None, help="distogram_tasks.json (default: eval.files.distogram_tasks or eval.dirs.distogram/distogram_tasks.json)")
    p.add_argument("--ref-distogram-dir", "-r", type=str, default=None, help="reference_distogram_diff root (default: eval.dirs.ref_distogram / eval.external.ref_distogram_dir)")
    p.add_argument("--output", "-o", type=str, default=None, help="Output JSON (default: eval.files.confbench_distogram)")
    p.add_argument("--method", "-m", type=str, default=None, help="Filter by method (af3, boltz1, boltz2, bioemu)")
    args = p.parse_args()

    valid_pairs_path = E.distogram_valid_pairs_path(args.valid_pairs)
    tasks_path = E.distogram_tasks_path(args.tasks)
    ref_distogram_root = E.distogram_ref_distogram_dir(args.ref_distogram_dir)

    if not tasks_path.exists():
        raise FileNotFoundError(f"distogram tasks not found: {tasks_path}")

    with open(valid_pairs_path, "r") as f:
        valid_pairs_data = json.load(f)

    with open(tasks_path, "r") as f:
        tasks = json.load(f)
    idx = _task_index(tasks)

    methods = ["af3", "boltz1", "boltz2", "bioemu"]
    if args.method:
        methods = [args.method]

    total_dict: Dict[str, Dict[str, Any]] = {}
    error_list: List[Dict[str, Any]] = []

    for pair_type, pairs in valid_pairs_data.items():
        dict_per_pair: Dict[str, Any] = {}
        for cluster_id, pair_info in pairs.items():
            for pair_dict in pair_info:
                ref1 = pair_dict["valid_pair"][0]
                ref2 = pair_dict["valid_pair"][1]

                dict_per_method: Dict[str, Any] = {}
                for method in methods:
                    if pair_type == "apo-monomers":
                        result = process_apo_monomers(
                            idx, ref_distogram_root, cluster_id, pair_dict, ref1, ref2, method, error_list
                        )
                    else:
                        if method == "bioemu":
                            continue
                        result = process_induced(
                            idx, ref_distogram_root, cluster_id, pair_dict, ref1, ref2, method, pair_type, error_list
                        )
                    if result is not None:
                        dict_per_method[method] = result

                result_entry = {
                    "set_type": pair_type,
                    "cluster_id": cluster_id,
                    "valid_pair": [ref1, ref2],
                    "ref1": ref1,
                    "ref2": ref2,
                    "models": dict_per_method,
                }
                dict_per_pair[f"{cluster_id}_{ref1}_{ref2}"] = result_entry

        total_dict[pair_type] = dict_per_pair

    out_path = Path(args.output) if args.output else (E.file("confbench_distogram") or (tasks_path.parent / "confbench_scores_distogram.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(total_dict, f, indent=4)
    print(f"Wrote {out_path}")

    if error_list:
        err_path = out_path.parent / f"{out_path.stem}_errors.json"
        with open(err_path, "w") as f:
            json.dump(error_list, f, indent=4)
        print(f"{len(error_list)} errors saved to {err_path}")


if __name__ == "__main__":
    main()

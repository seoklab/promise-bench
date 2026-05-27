#!/usr/bin/env python3
"""
Generate alignment tasks from ``valid_pairs.json`` + enriched seq_cluster map JSON.

This module **lives under ``eval/align``**: it writes the alignment task JSON that is
later sharded by ``eval.align.split_alignment_jobs`` and executed by
``eval.align.struct_align_batch``.

Inputs
------
- ``valid_pairs.json``: produced by ``python -m curation.make_pairs`` (list pairs
  ``[tag1, tag2]`` per cluster, or optional enriched dicts with ``valid_pair``).
- enriched seq_cluster map JSON (a.k.a. ``seq_cluster_to_answer_map``): prediction
  glob patterns, chains, holo_predictions layout; also from ``make_pairs``.

Usage
-----

  python -m eval.align.generate_alignment_tasks --help
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils._config import eval_cfg as E, pipeline_cfg as C

_INTRINSIC_SET = "intrinsic"


def _path_json_default(obj: object) -> str:
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Not JSON serializable: {type(obj)!r}")


def find_conf_label(yaml_tag: str, cluster_data: Dict) -> str:
    base_tag = yaml_tag
    if base_tag.endswith("_m"):
        base_tag = base_tag[:-2]
    elif base_tag.endswith("_x"):
        base_tag = base_tag[:-2]

    apo_list = cluster_data.get("apo", [])
    for entry in apo_list:
        if "_conf_" in entry:
            entry_base = entry.rsplit("_conf_", 1)[0]
            if entry_base == base_tag:
                return "conf_" + entry.rsplit("_conf_", 1)[1]

    holo_list = cluster_data.get("holo", [])
    for entry in holo_list:
        if "_conf_" in entry:
            entry_base = entry.rsplit("_conf_", 1)[0]
            if entry_base == base_tag:
                return "conf_" + entry.rsplit("_conf_", 1)[1]

    if yaml_tag.endswith("_m"):
        return "conf_m"
    if yaml_tag.endswith("_x"):
        return "conf_x"
    return "unknown"


def find_reference_cif(yaml_tag: str, cif_root: Path) -> Optional[Path]:
    parts = yaml_tag.split("_")
    if len(parts) < 2:
        return None
    pdb_id, asm_num = parts[0], parts[1]
    mid = pdb_id[1:3]
    cif_file = cif_root / mid.upper() / pdb_id.upper() / f"asm_{pdb_id}_{asm_num}.cif"
    return cif_file if cif_file.exists() else None


def extract_chain_from_yaml(yaml_tag: str) -> str:
    parts = yaml_tag.split("_")
    if len(parts) >= 3:
        return parts[2]
    return "A"


def extract_file_info(file_path: Path, method: str, yaml_tag: str) -> Dict[str, Any]:
    """
    Seed / sample / actual_yaml_tag for output filenames.
    Prediction chain for alignment comes from distogram_data target_chain; actual_chain
    here is only a fallback from the yaml tag (no external mapping JSON required).
    """

    actual_yaml_tag = yaml_tag
    seed = None
    sample = None

    if method in ("boltz2", "boltz1", "boltz-2", "boltz-1"):
        for part in file_path.parts:
            if part.startswith("seed_"):
                seed = part.split("_")[1] if "_" in part else part
            elif part.startswith("boltz_results_"):
                actual_yaml_tag = part[len("boltz_results_") :].replace("_with_msa", "")
        sample = file_path.stem.split("_")[-1] if "_model_" in file_path.stem else "1"
    elif method == "af3":
        for part in file_path.parts:
            if part.startswith("seed_"):
                seed = part.split("_")[1] if "_" in part else part
            elif part.startswith("sample_"):
                sample = part.split("_")[1] if "_" in part else "1"
            elif part.startswith("fold_job_"):
                sample = part.split("_")[-1] if "_" in part else "1"
            elif "seed-" in part and "sample-" in part:
                seed = part.split("_")[0].split("-")[1]
                sample = part.split("_")[1].split("-")[1]
    elif method in ("chai", "chai-1"):
        for part in file_path.parts:
            if part.startswith("seed_"):
                seed = part.split("_")[1] if "_" in part else part
                break
        sample = file_path.stem.split("_")[-1] if "model_idx_" in file_path.stem else "1"
    elif method == "bioemu":
        sample = file_path.stem.split("_")[-1] if "_" in file_path.stem else "1"

    actual_chain = extract_chain_from_yaml(yaml_tag)
    return {
        "seed": seed,
        "sample": sample,
        "actual_yaml_tag": actual_yaml_tag,
        "actual_chain": actual_chain,
    }


def _infer_model_entity(cluster_data: Dict[str, Any], valid_pair: List[str]) -> Optional[str]:
    """Pick prediction yaml tag for intrinsic pairs (from map, else first valid_pair tag)."""
    tags_in_preds: List[str] = []
    for method_info in cluster_data.get("apo_predictions", {}).values():
        if isinstance(method_info, dict):
            yt = method_info.get("yaml_tag")
            if yt:
                tags_in_preds.append(str(yt))
    for tag in valid_pair:
        if tag in tags_in_preds:
            return tag
    if tags_in_preds:
        return tags_in_preds[0]
    return valid_pair[0] if valid_pair else None


def _normalize_pair_info(
    pair_info: Any,
    pair_type: str,
    cluster_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
  Accept make_pairs list pairs ``[tag1, tag2]`` or enriched dicts.

  Returns dict with ``valid_pair`` and entity fields, or *None* if invalid.
  """
    if isinstance(pair_info, dict):
        valid_pair = pair_info.get("valid_pair")
        if not valid_pair or not isinstance(valid_pair, (list, tuple)) or len(valid_pair) != 2:
            return None
        valid_pair = list(valid_pair)
        model_entity = pair_info.get("model_entity")
        apo_entity = pair_info.get("apo_model_entity")
        holo_entity = pair_info.get("holo_model_entity")
    elif isinstance(pair_info, (list, tuple)) and len(pair_info) == 2:
        valid_pair = [str(pair_info[0]), str(pair_info[1])]
        model_entity = None
        apo_entity = None
        holo_entity = None
    else:
        return None

    if pair_type == _INTRINSIC_SET:
        if not model_entity:
            model_entity = _infer_model_entity(cluster_data, valid_pair)
        if not model_entity:
            return None
        return {
            "valid_pair": valid_pair,
            "model_entity": model_entity,
            "apo_model_entity": None,
            "holo_model_entity": None,
        }

    if not apo_entity:
        apo_entity = next((t for t in valid_pair if str(t).endswith("_m")), None)
    if not holo_entity:
        holo_entity = next((t for t in valid_pair if str(t).endswith("_x")), None)
    if not apo_entity or not holo_entity:
        return None
    return {
        "valid_pair": valid_pair,
        "model_entity": None,
        "apo_model_entity": apo_entity,
        "holo_model_entity": holo_entity,
    }


def _other_cluster_apo_refs(
    cluster_data: Dict[str, Any], apo_entity: str
) -> List[str]:
    """Cluster 의 ``apo`` entries 중 *apo_entity* 가 아닌 다른 apo references 반환.

    Used for the relaxed apo success rule on induced pairs: an apo prediction may match
    any apo conformation in the same cluster, not only the pair's designated apo.
    Returns yaml_tag (``_m`` suffix) form, e.g. ``"3d8r_1_A1_m"``.
    """
    apo_entries = cluster_data.get("apo", []) or []
    if not apo_entries or not apo_entity:
        return []
    apo_base = apo_entity[:-2] if apo_entity.endswith("_m") else apo_entity
    out: List[str] = []
    seen = {apo_base}
    for e in apo_entries:
        if "_conf_" not in e:
            continue
        base = e.rsplit("_conf_", 1)[0]
        if base in seen:
            continue
        seen.add(base)
        out.append(f"{base}_m")
    return out


def _get_holo_predictions(cluster_data: Dict[str, Any], holo_entity: str) -> Dict[str, Any]:
    """Return per-method prediction info for one holo yaml tag."""
    raw = cluster_data.get("holo_predictions", {})
    if holo_entity in raw and isinstance(raw[holo_entity], dict):
        nested = raw[holo_entity]
        if nested and isinstance(next(iter(nested.values()), None), dict):
            sample = next(iter(nested.values()))
            if "pattern" in sample or "yaml_tag" in sample:
                return nested
    for _key, methods_dict in raw.items():
        if not isinstance(methods_dict, dict):
            continue
        if _key == holo_entity:
            return methods_dict
        for method_info in methods_dict.values():
            if isinstance(method_info, dict) and method_info.get("yaml_tag") == holo_entity:
                return methods_dict
    return {}


def generate_alignment_tasks(
    valid_pairs_file: Path,
    distogram_data_file: Path,
    output_json: Path,
    output_dir: Path,
    cif_dir: Path,
) -> List[Dict]:
    with open(valid_pairs_file) as f:
        valid_pairs = json.load(f)
    with open(distogram_data_file) as f:
        distogram_data = json.load(f)

    cif_root = cif_dir
    alignment_tasks: List[Dict] = []
    error_list: List[Dict] = []

    for pair_type, clusters in valid_pairs.items():
        print(f"\n{'=' * 80}\nProcessing {pair_type}\n{'=' * 80}")

        for cluster_id, pairs_list in clusters.items():
            print(f"\nCluster: {cluster_id}")

            if pair_type not in distogram_data:
                print(f"  WARNING: {pair_type} not in distogram_data")
                continue
            if cluster_id not in distogram_data[pair_type]:
                print(f"  WARNING: {cluster_id} not in distogram_data[{pair_type}]")
                continue

            cluster_data = distogram_data[pair_type][cluster_id]
            is_intrinsic = pair_type == _INTRINSIC_SET

            for pair_info in pairs_list:
                normalized = _normalize_pair_info(pair_info, pair_type, cluster_data)
                if normalized is None:
                    raw_pair = (
                        list(pair_info)
                        if isinstance(pair_info, (list, tuple))
                        else pair_info.get("valid_pair", pair_info)
                        if isinstance(pair_info, dict)
                        else pair_info
                    )
                    error_list.append(
                        {
                            "pair_type": pair_type,
                            "cluster_id": cluster_id,
                            "valid_pair": raw_pair,
                            "error": (
                                "Invalid pair entry (expected [tag1, tag2] or dict with valid_pair)"
                                if not is_intrinsic
                                else "Invalid pair or could not infer model_entity for intrinsic"
                            ),
                        }
                    )
                    continue

                valid_pair = normalized["valid_pair"]
                mobiles = list(valid_pair)
                apo_entity = normalized["apo_model_entity"]
                holo_entity = normalized["holo_model_entity"]

                if is_intrinsic:
                    model_entity = normalized["model_entity"]
                    reference_entities = [model_entity]
                    extra_apo_mobiles: List[str] = []
                else:
                    reference_entities = [apo_entity, holo_entity]
                    extra_apo_mobiles = _other_cluster_apo_refs(cluster_data, apo_entity)

                print(f"  Pair: {valid_pair} -> align to {reference_entities}")
                if extra_apo_mobiles:
                    print(f"    + extra apo-mobile refs (relaxed apo rule): {extra_apo_mobiles}")

                holo_preds = (
                    _get_holo_predictions(cluster_data, holo_entity)
                    if not is_intrinsic and holo_entity
                    else {}
                )

                if is_intrinsic:
                    prediction_dict_by_reference = {
                        reference_entities[0]: cluster_data.get("apo_predictions", {})
                    }
                else:
                    prediction_dict_by_reference = {
                        apo_entity: cluster_data.get("apo_predictions", {}),
                        holo_entity: holo_preds,
                    }

                for reference in reference_entities:
                    prediction_dict = prediction_dict_by_reference.get(reference, {})

                    if not find_reference_cif(reference, cif_root):
                        print(reference, cif_root)
                        error_list.append(
                            {
                                "pair_type": pair_type,
                                "cluster_id": cluster_id,
                                "valid_pair": valid_pair,
                                "reference": reference,
                                "error": f"Reference CIF not found for {reference}",
                            }
                        )
                        continue

                    for method, method_info in prediction_dict.items():
                        pattern = method_info.get("pattern", "")
                        yaml_tag = method_info.get("yaml_tag", "")
                        target_chain = method_info.get("target_chain", "A")

                        if not pattern:
                            continue
                        if not yaml_tag and method != "bioemu":
                            print(f"    Skipping {method} without yaml_tag")
                            continue

                        pred_files = glob.glob(pattern)
                        if not pred_files:
                            print(f"    {method}: No files for pattern {pattern}")
                            continue

                        print(f"    {method} -> {reference}: {len(pred_files)} prediction(s)")

                        if (not is_intrinsic) and reference == apo_entity and extra_apo_mobiles:
                            per_ref_mobiles = list(mobiles) + list(extra_apo_mobiles)
                        else:
                            per_ref_mobiles = list(mobiles)

                        for mobile_yaml in per_ref_mobiles:
                            mobile_cif = find_reference_cif(mobile_yaml, cif_root)
                            if not mobile_cif:
                                error_list.append(
                                    {
                                        "pair_type": pair_type,
                                        "cluster_id": cluster_id,
                                        "valid_pair": valid_pair,
                                        "mobile": mobile_yaml,
                                        "error": f"Mobile CIF not found for {mobile_yaml}",
                                    }
                                )
                                continue

                            mobile_ref_chain = extract_chain_from_yaml(mobile_yaml)

                            for pred_file in pred_files:
                                pred_path = Path(pred_file)
                                file_info = extract_file_info(pred_path, method, yaml_tag or "")
                                mobile_chain = target_chain
                                actual_yaml_tag = file_info.get("actual_yaml_tag", yaml_tag)
                                seed_str = (
                                    f"seed_{file_info['seed']}" if file_info["seed"] else "seed_unknown"
                                )
                                sample_str = (
                                    f"model_{file_info['sample']}" if file_info["sample"] else "model_1"
                                )
                                mobile_parts = mobile_yaml.split("_")
                                if len(mobile_parts) >= 3:
                                    mobile_asm_name = (
                                        f"{mobile_parts[0]}_asm{mobile_parts[1]}_{mobile_parts[2]}"
                                    )
                                else:
                                    mobile_asm_name = mobile_yaml

                                output_subdir = (
                                    Path(output_dir) / method / pair_type / cluster_id / reference
                                )
                                if method == "bioemu":
                                    output_filename = f"{sample_str}_aligned_to_{mobile_asm_name}.cif"
                                else:
                                    output_filename = (
                                        f"{seed_str}_{actual_yaml_tag}_{sample_str}_aligned_to_{mobile_asm_name}.cif"
                                    )
                                output_path = output_subdir / output_filename

                                if is_intrinsic:
                                    method_type = "apo"
                                    target_state = "apo"
                                    reference_state = "apo"
                                else:
                                    if reference == apo_entity:
                                        method_type = "apo"
                                        target_state = "apo"
                                    else:
                                        method_type = "holo"
                                        target_state = "holo"
                                    if mobile_yaml == apo_entity:
                                        reference_state = "apo"
                                    elif mobile_yaml == holo_entity:
                                        reference_state = "holo"
                                    elif mobile_yaml in extra_apo_mobiles:
                                        reference_state = "apo"
                                    elif mobile_yaml == valid_pair[0]:
                                        reference_state = "apo"
                                    else:
                                        reference_state = "holo"

                                target_conformation = (
                                    find_conf_label(yaml_tag, cluster_data) if yaml_tag else "unknown"
                                )
                                reference_conformation = find_conf_label(mobile_yaml, cluster_data)

                                alignment_tasks.append(
                                    {
                                        "ref_cif": mobile_cif.resolve(),
                                        "mobile_cif": pred_path.resolve(),
                                        "output_cif": output_path.resolve(),
                                        "ref_chain": mobile_ref_chain,
                                        "mobile_chain": mobile_chain,
                                        "target_conformation": target_conformation,
                                        "target_state": target_state,
                                        "reference_conformation": reference_conformation,
                                        "reference_state": reference_state,
                                        "prediction_method": method,
                                        "cluster_id": cluster_id,
                                        "method_type": method_type,
                                        "pair_type": pair_type,
                                        "valid_pair": valid_pair,
                                        "model_entity": reference,
                                        "mobile_entity": mobile_yaml,
                                    }
                                )

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(alignment_tasks, f, indent=2, default=_path_json_default)
    print(f"\nGenerated {len(alignment_tasks)} tasks -> {out_path}")

    if error_list:
        err_path = out_path.parent / f"{out_path.stem}_errors.json"
        with open(err_path, "w") as f:
            json.dump(error_list, f, indent=2)
        print(f"Errors: {len(error_list)} -> {err_path}")

    return alignment_tasks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--valid-pairs", type=Path, default=E.file("valid_pairs"))
    p.add_argument(
        "--answer-map",
        type=Path,
        default=C.file("answer_map") or Path("seq_cluster_to_answer_map.json"),
        help="Path to seq_cluster_to_answer_map.json (from curation.make_pairs).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=(E.dir("output") / "alignment_tasks.json").resolve(),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=(E.dir("output") / "aligned_cif").resolve(),
    )
    p.add_argument("--cif-dir", type=Path, default=C.dir("cif_asms"))
    args = p.parse_args()
    generate_alignment_tasks(
        valid_pairs_file=args.valid_pairs,
        distogram_data_file=args.answer_map,
        output_json=args.output,
        output_dir=args.output_dir,
        cif_dir=args.cif_dir,
    )


if __name__ == "__main__":
    main()


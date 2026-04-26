#!/usr/bin/env python3
"""
Generate alignment tasks from ``valid_pairs.json`` + enriched seq_cluster map JSON.

This module **lives under ``eval/align``**: it writes the alignment task JSON that is
later sharded by ``eval.align.split_alignment_jobs`` and executed by
``eval.align.nurikit_align_batch``.

Inputs
------
- ``valid_pairs.json``: produced by ``python -m curation.make_pairs``.
- enriched seq_cluster map JSON (a.k.a. ``seq_cluster_to_answer_map``): also produced by
  ``python -m curation.make_pairs``.

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


def generate_alignment_tasks(
    valid_pairs_file: str,
    distogram_data_file: str,
    output_json: str,
    output_dir: str,
    cif_dir: str,
) -> List[Dict]:
    with open(valid_pairs_file) as f:
        valid_pairs = json.load(f)
    with open(distogram_data_file) as f:
        distogram_data = json.load(f)

    cif_root = Path(cif_dir)
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
            apo_entity: Optional[str] = None
            holo_entity: Optional[str] = None

            for pair_info in pairs_list:
                if not isinstance(pair_info, dict) or "valid_pair" not in pair_info:
                    continue
                valid_pair = pair_info["valid_pair"]
                mobiles = list(valid_pair)

                if pair_type == "intrinsic":
                    model_entity = pair_info.get("model_entity")
                    if not model_entity:
                        error_list.append(
                            {
                                "pair_type": pair_type,
                                "cluster_id": cluster_id,
                                "valid_pair": valid_pair,
                                "error": "No model_entity found",
                            }
                        )
                        continue
                    reference_entities = [model_entity]
                else:
                    apo_entity = pair_info.get("apo_model_entity")
                    holo_entity = pair_info.get("holo_model_entity")
                    if not apo_entity or not holo_entity:
                        error_list.append(
                            {
                                "pair_type": pair_type,
                                "cluster_id": cluster_id,
                                "valid_pair": valid_pair,
                                "error": "No apo/holo_model_entity found",
                            }
                        )
                        continue
                    reference_entities = [apo_entity, holo_entity]

                print(f"  Pair: {valid_pair} -> align to {reference_entities}")

                holo_preds: Dict[str, Any] = {}
                if pair_type != "intrinsic" and holo_entity is not None:
                    holo_preds_raw = cluster_data.get("holo_predictions", {})
                    for _conf_label, methods_dict in holo_preds_raw.items():
                        if not isinstance(methods_dict, dict):
                            continue
                        for _m, method_info in methods_dict.items():
                            if isinstance(method_info, dict) and method_info.get(
                                "yaml_tag"
                            ) == holo_entity:
                                holo_preds = methods_dict
                                break
                        if holo_preds:
                            break

                if pair_type == "intrinsic":
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

                        for mobile_yaml in mobiles:
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

                                if pair_type == "intrinsic":
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
                                        "ref_cif": str(mobile_cif.resolve()),
                                        "mobile_cif": str(pred_path.resolve()),
                                        "output_cif": str(output_path.resolve()),
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
        json.dump(alignment_tasks, f, indent=2)
    print(f"\nGenerated {len(alignment_tasks)} tasks -> {out_path}")

    if error_list:
        err_path = out_path.parent / f"{out_path.stem}_errors.json"
        with open(err_path, "w") as f:
            json.dump(error_list, f, indent=2)
        print(f"Errors: {len(error_list)} -> {err_path}")

    return alignment_tasks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--valid-pairs", default=str(E.file("valid_pairs")))
    p.add_argument("--distogram-data", default=str(C.file("answer_map") or Path("seq_cluster_to_answer_map.json")))
    p.add_argument("--output", "-o", default=str((E.dir("output") / "alignment_tasks.json").resolve()))
    p.add_argument("--output-dir", default=str((E.dir("output") / "aligned_cif").resolve()))
    p.add_argument("--cif-dir", default=str(C.dir("cif_asms")))
    args = p.parse_args()
    generate_alignment_tasks(
        valid_pairs_file=args.valid_pairs,
        distogram_data_file=args.distogram_data,
        output_json=args.output,
        output_dir=args.output_dir,
        cif_dir=args.cif_dir,
    )


if __name__ == "__main__":
    main()


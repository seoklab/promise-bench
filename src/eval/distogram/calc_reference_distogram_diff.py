#!/usr/bin/env python3
"""
Calculate distogram differences between reference conformations.

For each prediction task, computes pairwise distogram differences between
reference conformations using MSA alignment. Identifies positions where
references differ significantly in their distance distributions.

Usage (``PYTHONPATH=src``)::

    python -m eval.distogram.calc_reference_distogram_diff --tasks data_eval/distogram/distogram_tasks.json --threshold 3.0
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import json
import numpy as np
import itertools
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from utils._config import eval_cfg as E


# -----------------------------
# Constants (same as calc_distogram_loss.py)
# -----------------------------
# AF3 bins
AF3_BIN_LOW = 2.3125
AF3_BIN_HIGH = 21.6875
AF3_NUM_BINS_MINUS1 = 63

_af3_breaks_np = np.linspace(AF3_BIN_LOW, AF3_BIN_HIGH, AF3_NUM_BINS_MINUS1)
_af3_bin_centers_np = [0.5 * _af3_breaks_np[0]]
_af3_bin_centers_np.extend(0.5 * (_af3_breaks_np[1:] + _af3_breaks_np[:-1]))
_af3_bin_centers_np.append(_af3_breaks_np[-1])
AF3_BIN_CENTERS_NP = np.array(_af3_bin_centers_np, dtype=np.float32)

# Boltz bins
BOLTZ_BIN_LOW = 2.0
BOLTZ_BIN_HIGH = 22.0
BOLTZ_NUM_BINS_MINUS1 = 63

_boltz_breaks_np = np.linspace(BOLTZ_BIN_LOW, BOLTZ_BIN_HIGH, BOLTZ_NUM_BINS_MINUS1)
_boltz_bin_centers_np = [0.5 * _boltz_breaks_np[0]]
_boltz_bin_centers_np.extend(0.5 * (_boltz_breaks_np[1:] + _boltz_breaks_np[:-1]))
_boltz_bin_centers_np.append(_boltz_breaks_np[-1])
BOLTZ_BIN_CENTERS_NP = np.array(_boltz_bin_centers_np, dtype=np.float32)

# Keep backward compatibility - use AF3 as default
BIN_LOW = AF3_BIN_LOW
BIN_HIGH = AF3_BIN_HIGH
NUM_BINS_MINUS1 = AF3_NUM_BINS_MINUS1
_breaks_np = _af3_breaks_np
BIN_CENTERS_NP = AF3_BIN_CENTERS_NP


def parse_a3m(a3m_path: Path) -> Dict[str, str]:
    """Parse a3m file and return dict of {header: aligned_sequence}."""
    sequences = {}
    current_header = None
    current_seq: list[str] = []

    with open(a3m_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    sequences[current_header] = "".join(current_seq)
                current_header = line[1:].split()[0]
                current_seq = []
            else:
                # Remove lowercase letters (insertions in a3m format)
                filtered = "".join(c for c in line if not c.islower())
                current_seq.append(filtered)

        if current_header is not None:
            sequences[current_header] = "".join(current_seq)

    return sequences


def get_alignment_mapping(
    model_seq: str,
    ref_seq: str,
) -> Dict[int, int]:
    """
    Given two aligned sequences (with gaps), return mapping from
    model_idx (0-based) -> ref_idx (0-based) for aligned positions.
    """
    mapping = {}
    model_idx = 0
    ref_idx = 0

    for i in range(len(model_seq)):
        model_char = model_seq[i]
        ref_char = ref_seq[i]

        if model_char != "-" and ref_char != "-":
            mapping[model_idx] = ref_idx

        if model_char != "-":
            model_idx += 1
        if ref_char != "-":
            ref_idx += 1

    return mapping


def distogram_from_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Compute distogram (one-hot encoded) from CB coordinates.

    Returns:
        distogram: (L, L, 64) one-hot encoded distance bins
    """
    L = len(coords)

    # Compute pairwise distances
    distance_matrix = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

    # Convert to bin indices (using AF3 bins for reference distograms)
    bin_indices = np.searchsorted(_af3_breaks_np, distance_matrix)
    bin_indices = np.clip(bin_indices, 0, 63)
    # One-hot encode
    distogram = np.zeros((L, L, 64))
    for i in range(L):
        for j in range(L):
            distogram[i, j, bin_indices[i, j]] = 1.0

    return distogram


def get_ref_id_from_cif(ref_cif: str, ref_chain: str) -> Optional[str]:
    """
    Extract reference ID from ref_cif path and chain.
    e.g., .../2WRZ/asm_2wrz_2.cif, chain B1 -> 2wrz_B1
    """
    path = Path(ref_cif)
    name = path.stem  # asm_2wrz_2
    parts = name.split("_")
    if len(parts) >= 2:
        pdb_id = parts[1]  # 2wrz
        return f"{pdb_id}_{ref_chain}"
    return None


def load_reference_distogram(
    ref_info: dict,
    alignments: Dict[str, str],
    model_aligned_seq: str,
    errors: list,
) -> Optional[Tuple[np.ndarray, Dict[int, int], str]]:
    """
    Load reference distogram and create alignment mapping to model sequence.

    Args:
        ref_info: Reference info from task
        alignments: MSA alignments
        model_aligned_seq: Model's aligned sequence
        errors: Error list to append to

    Returns:
        Tuple of (distogram, model_idx_to_ref_idx, ref_id) or None if failed
    """
    ref_cif = ref_info.get("ref_cif", "")
    ref_chain = ref_info.get("ref_chain", "")
    reference_yaml_tag = ref_info.get("reference_yaml_tag", "")

    # Get reference ID and aligned sequence
    ref_id = get_ref_id_from_cif(ref_cif, ref_chain)

    if not ref_id or ref_id not in alignments:
        # Try variations
        possible_ids = [
            ref_id,
            ref_id.lower() if ref_id else None,
            f"{Path(ref_cif).stem.split('_')[1]}_{ref_chain}",
            ref_id[:-1] if ref_id else None,
        ]
        ref_aligned_seq = None
        for pid in possible_ids:
            if pid and pid in alignments:
                ref_aligned_seq = alignments[pid]
                ref_id = pid
                break

        if ref_aligned_seq is None:
            errors.append(
                {
                    "reference": reference_yaml_tag,
                    "error": f"ref_id {ref_id} not found in alignments",
                }
            )
            return None
    else:
        ref_aligned_seq = alignments[ref_id]

    # Get alignment mapping: model_seq_idx -> ref_seq_idx
    alignment_mapping = get_alignment_mapping(model_aligned_seq, ref_aligned_seq)

    if len(alignment_mapping) < 3:
        errors.append(
            {
                "reference": reference_yaml_tag,
                "error": f"too few aligned positions: {len(alignment_mapping)}",
            }
        )
        return None

    # Load reference CB coordinates
    ref_cb_json = ref_info.get("reference_cb_json", "")
    if not ref_cb_json or not Path(ref_cb_json).exists():
        errors.append(
            {
                "reference": reference_yaml_tag,
                "error": f"reference_cb_json not found: {ref_cb_json}",
            }
        )
        return None

    with open(ref_cb_json) as f:
        ref_cb_data = json.load(f)

    # Get reference chain data
    if ref_chain not in ref_cb_data.get("all_chains", {}):
        errors.append(
            {
                "reference": reference_yaml_tag,
                "error": f"ref chain {ref_chain} not found in CB data",
            }
        )
        return None

    ref_chain_data = ref_cb_data["all_chains"][ref_chain]
    ref_seq_id_to_coord = {
        int(k): v for k, v in ref_chain_data["seq_id_to_coord"].items()
    }

    # Convert coordinates to numpy array and create index mapping
    ref_seq_ids_sorted = sorted(ref_seq_id_to_coord.keys())
    ref_coords_list = [ref_seq_id_to_coord[seq_id] for seq_id in ref_seq_ids_sorted]
    ref_coords = np.array(ref_coords_list)  # (L_ref, 3)

    # Create mapping: ref_seq_id -> CB array index
    ref_seq_id_to_cb_idx = {
        seq_id: idx for idx, seq_id in enumerate(ref_seq_ids_sorted)
    }

    # Compute reference distogram from CB coordinates
    ref_distogram = distogram_from_coordinates(ref_coords)  # (L_ref, L_ref, 64)

    # Create mapping: model_idx -> ref_cb_idx
    model_idx_to_ref_cb_idx = {}
    for model_idx, ref_seq_id in alignment_mapping.items():
        if ref_seq_id in ref_seq_id_to_cb_idx:
            model_idx_to_ref_cb_idx[model_idx] = ref_seq_id_to_cb_idx[ref_seq_id]

    return ref_distogram, model_idx_to_ref_cb_idx, ref_id


def calc_distogram_difference(
    ref_A_distogram: np.ndarray,
    ref_B_distogram: np.ndarray,
    model_idx_to_refA_idx: Dict[int, int],
    model_idx_to_refB_idx: Dict[int, int],
    threshold: float = 3.0,
    bin_centers: Optional[np.ndarray] = None,
) -> Tuple[float, int, List[Tuple[int, int, float]], float]:
    """
    Calculate distogram difference between two references.

    Args:
        ref_A_distogram: Reference A distogram (L_A, L_A, 64) - one-hot
        ref_B_distogram: Reference B distogram (L_B, L_B, 64) - one-hot
        model_idx_to_refA_idx: Mapping from model index to ref A CB index
        model_idx_to_refB_idx: Mapping from model index to ref B CB index
        threshold: Distance threshold (Angstroms) to report differences
        bin_centers: Bin centers to use for distance calculation (default: AF3_BIN_CENTERS_NP)

    Returns:
        mean_distance_diff: Mean absolute distance difference (all positions)
        n_aligned: Number of aligned position pairs
        diff_positions: List of (model_i, model_j, distance_diff) exceeding threshold
        dynamic_mean_distance_diff: Mean distance difference for dynamic positions only
    """
    if bin_centers is None:
        bin_centers = AF3_BIN_CENTERS_NP
    # Find common model indices that exist in both mappings
    common_model_indices = sorted(
        set(model_idx_to_refA_idx.keys()) & set(model_idx_to_refB_idx.keys())
    )

    if len(common_model_indices) < 3:
        print("    Not enough common aligned residues to compare.")
        return float("nan"), 0, []

    # Extract bin indices from one-hot encoded distograms
    distance_diffs = []
    diff_positions = []

    for i, model_i in enumerate(common_model_indices):
        for j, model_j in enumerate(common_model_indices):
            if i >= j:  # Only compute upper triangle (including diagonal)
                continue

            refA_i = model_idx_to_refA_idx[model_i]
            refA_j = model_idx_to_refA_idx[model_j]
            refB_i = model_idx_to_refB_idx[model_i]
            refB_j = model_idx_to_refB_idx[model_j]

            # Get bin indices (argmax of one-hot)
            bin_A = ref_A_distogram[refA_i, refA_j].argmax()
            bin_B = ref_B_distogram[refB_i, refB_j].argmax()

            # Get bin centers
            dist_A = bin_centers[bin_A]
            dist_B = bin_centers[bin_B]

            # Calculate distance difference
            dist_diff = abs(dist_A - dist_B)
            distance_diffs.append(dist_diff)

            # Record if exceeds threshold
            if dist_diff >= threshold:
                diff_positions.append((model_i, model_j, float(dist_diff)))

    if not distance_diffs:
        return float("nan"), 0, [], float("nan")

    mean_diff = float(np.mean(distance_diffs))
    n_aligned = len(common_model_indices)

    # Calculate dynamic region mean (positions exceeding threshold)
    if diff_positions:
        dynamic_diffs = [
            pos[2] for pos in diff_positions
        ]  # Extract distance_diff values
        dynamic_mean_diff = float(np.mean(dynamic_diffs))
    else:
        dynamic_mean_diff = float("nan")

    return mean_diff, n_aligned, diff_positions, dynamic_mean_diff


def visualize_distogram_difference(
    ref_A_distogram: np.ndarray,
    ref_B_distogram: np.ndarray,
    model_idx_to_refA_idx: Dict[int, int],
    model_idx_to_refB_idx: Dict[int, int],
    ref_A_name: str,
    ref_B_name: str,
    output_path: Path,
):
    """
    Visualize distogram difference between two references.

    Creates a figure with 3 subplots:
    1. Reference A distance matrix
    2. Reference B distance matrix
    3. Absolute difference matrix

    Args:
        ref_A_distogram: Reference A distogram (L_A, L_A, 64) - one-hot
        ref_B_distogram: Reference B distogram (L_B, L_B, 64) - one-hot
        model_idx_to_refA_idx: Mapping from model index to ref A CB index
        model_idx_to_refB_idx: Mapping from model index to ref B CB index
        ref_A_name: Name for reference A (for title)
        ref_B_name: Name for reference B (for title)
        output_path: Path to save the figure
    """
    # Find common model indices
    common_model_indices = sorted(
        set(model_idx_to_refA_idx.keys()) & set(model_idx_to_refB_idx.keys())
    )

    if len(common_model_indices) < 3:
        return

    N = len(common_model_indices)

    # Extract aligned distance matrices
    dist_matrix_A = np.zeros((N, N))
    dist_matrix_B = np.zeros((N, N))

    for i, model_i in enumerate(common_model_indices):
        for j, model_j in enumerate(common_model_indices):
            refA_i = model_idx_to_refA_idx[model_i]
            refA_j = model_idx_to_refA_idx[model_j]
            refB_i = model_idx_to_refB_idx[model_i]
            refB_j = model_idx_to_refB_idx[model_j]

            # Get bin indices and convert to distances (using AF3 bins for visualization)
            bin_A = ref_A_distogram[refA_i, refA_j].argmax()
            bin_B = ref_B_distogram[refB_i, refB_j].argmax()

            dist_matrix_A[i, j] = AF3_BIN_CENTERS_NP[bin_A]
            dist_matrix_B[i, j] = AF3_BIN_CENTERS_NP[bin_B]

    # Calculate difference matrix
    diff_matrix = np.abs(dist_matrix_A - dist_matrix_B)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Ref A
    im1 = axes[0].imshow(dist_matrix_A, cmap="viridis", aspect="auto", vmin=0, vmax=22)
    axes[0].set_title(f"Reference A: {ref_A_name}", fontsize=12)
    axes[0].set_xlabel("Model sequence index")
    axes[0].set_ylabel("Model sequence index")
    plt.colorbar(im1, ax=axes[0], label="Distance (Å)")

    # Ref B
    im2 = axes[1].imshow(dist_matrix_B, cmap="viridis", aspect="auto", vmin=0, vmax=22)
    axes[1].set_title(f"Reference B: {ref_B_name}", fontsize=12)
    axes[1].set_xlabel("Model sequence index")
    axes[1].set_ylabel("Model sequence index")
    plt.colorbar(im2, ax=axes[1], label="Distance (Å)")

    # Difference
    im3 = axes[2].imshow(diff_matrix, cmap="Reds", aspect="auto", vmin=0)
    axes[2].set_title("Absolute Difference", fontsize=12)
    axes[2].set_xlabel("Model sequence index")
    axes[2].set_ylabel("Model sequence index")
    plt.colorbar(im3, ax=axes[2], label="Distance Difference (Å)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    #print(f"    Saved visualization to {output_path}")


def process_task(
    task: dict,
    rep_seqs: dict,
    msa_dir: Path,
    threshold: float,
    errors: list,
    save_visualizations: bool = False,
    output_dir: Optional[Path] = None,
    skip_existing: bool = True,
) -> Optional[dict]:
    """
    Process a single task: compute pairwise reference distogram differences.

    Args:
        task: Task dictionary
        rep_seqs: Representative sequences
        msa_dir: MSA directory
        threshold: Distance threshold
        errors: Error list
        save_visualizations: Whether to save visualizations
        output_dir: Base output directory (e.g., ref_distogram)
        skip_existing: Whether to skip if output files already exist

    Returns:
        Result dict or None if processing failed
    """
    cluster_id = task.get("cluster_id")
    prediction_yaml_tag = task.get("prediction_yaml_tag")
    method_type = task.get("method_type", "")

    if not cluster_id or cluster_id not in rep_seqs:
        errors.append(
            {
                "task": prediction_yaml_tag,
                "error": f"cluster_id {cluster_id} not in representative_sequences",
            }
        )
        return None

    # Get representative header for modeled sequence
    model_header = rep_seqs[cluster_id]["header"]

    # Find a3m file
    cluster_prefix = cluster_id[1:3].upper()
    a3m_path = msa_dir / cluster_prefix / f"{cluster_id.upper()}.a3m"

    if not a3m_path.exists():
        errors.append(
            {"task": prediction_yaml_tag, "error": f"a3m not found: {a3m_path}"}
        )
        return None

    # Parse a3m
    alignments = parse_a3m(a3m_path)

    if model_header not in alignments:
        errors.append(
            {
                "task": prediction_yaml_tag,
                "error": f"model header {model_header} not in a3m",
            }
        )
        return None

    model_aligned_seq = alignments[model_header]

    # Get all references
    references = task.get("references", [])
    if len(references) < 2:
        errors.append(
            {
                "task": prediction_yaml_tag,
                "error": f"need at least 2 references, got {len(references)}",
            }
        )
        return None

    # Load all reference distograms
    ref_data = []
    for ref_info in references:
        result = load_reference_distogram(
            ref_info, alignments, model_aligned_seq, errors
        )
        if result is not None:
            distogram, mapping, ref_id = result
            ref_data.append(
                {
                    "reference_yaml_tag": ref_info.get("reference_yaml_tag"),
                    "reference_conformation": ref_info.get(
                        "reference_conformation", ""
                    ),
                    "reference_state": ref_info.get("reference_state", ""),
                    "distogram": distogram,
                    "mapping": mapping,
                    "ref_id": ref_id,
                }
            )

    # Print reference count summary
    n_original_refs = len(references)
    n_loaded_refs = len(ref_data)
    n_expected_pairs = n_loaded_refs * (n_loaded_refs - 1) // 2
    print(f"    References: {n_loaded_refs}/{n_original_refs} loaded successfully")
    print(f"    Expected pairwise combinations: {n_expected_pairs}")

    if len(ref_data) < 2:
        errors.append(
            {
                "task": prediction_yaml_tag,
                "error": f"successfully loaded only {len(ref_data)} references, need at least 2",
            }
        )
        return None

    # Compute pairwise differences
    pairwise_results = []

    # Create output directory structure
    individual_results_dir = None
    if output_dir:
        individual_results_dir = (
            output_dir / method_type / cluster_id / prediction_yaml_tag
        )
        individual_results_dir.mkdir(parents=True, exist_ok=True)

    # --- SMART SKIP LOGIC FOR AGGREGATED FILE (BEFORE individual loop) ---
    # Check if all required pairs already exist in aggregated file
    already_computed_pairs = set()
    if skip_existing and individual_results_dir:
        result_file_all = individual_results_dir / "reference_distogram_diff.json"
        if result_file_all.exists():
            try:
                with open(result_file_all, "r") as f:
                    existing_agg = json.load(f)
                if existing_agg and "pairwise_comparisons" in existing_agg:
                    # Load existing pairs into pairwise_results
                    for comp in existing_agg["pairwise_comparisons"]:
                        tagA = comp.get("reference_A")
                        confA = comp.get("reference_A_conformation")
                        tagB = comp.get("reference_B")
                        confB = comp.get("reference_B_conformation")
                        # Mark as already computed
                        already_computed_pairs.add((tagA, confA, tagB, confB))
                    print(f"  Found {len(already_computed_pairs)} existing pairs in aggregated file")
            except Exception as e:
                print(f"  Error reading existing aggregated result: {e}")
    # --- END SMART SKIP LOGIC ---

    for pair_idx, ((idx_A, ref_A), (idx_B, ref_B)) in enumerate(
        itertools.combinations(enumerate(ref_data), 2)
    ):
        # Create safe filename components
        ref_A_tag = ref_A["reference_yaml_tag"]
        ref_A_conf = ref_A["reference_conformation"]
        ref_B_tag = ref_B["reference_yaml_tag"]
        ref_B_conf = ref_B["reference_conformation"]

        # Check if this pair was already in aggregated file
        pair_key = (ref_A_tag, ref_A_conf, ref_B_tag, ref_B_conf)
        if pair_key in already_computed_pairs:
            # Skip - already in aggregated file, will be preserved in merge
            print(f"    Pair already in aggregated file: {ref_A_tag}_{ref_A_conf} vs {ref_B_tag}_{ref_B_conf}")
            continue

        # Check if individual result file already exists
        if skip_existing and individual_results_dir:
            result_filename = f"{ref_A_tag}_{ref_A_conf}_vs_{ref_B_tag}_{ref_B_conf}_distogram_diff.json"
            result_file = individual_results_dir / result_filename
            if result_file.exists():
                print(
                    f"    Skipping {ref_A_tag}_{ref_A_conf} vs {ref_B_tag}_{ref_B_conf} (already exists)"
                )
                # Still need to load for aggregated result
                try:
                    with open(result_file, "r") as f:
                        existing_result = json.load(f)
                    pairwise_results.append(existing_result)
                except Exception as e:
                    errors.append(
                        {
                            "task": prediction_yaml_tag,
                            "pair": f"{ref_A_tag}_{ref_A_conf} vs {ref_B_tag}_{ref_B_conf}",
                            "error": f"Failed to load existing result: {str(e)}",
                        }
                    )
                continue

        # Calculate with AF3 bins
        mean_diff_af3, n_aligned, diff_positions_af3, dynamic_mean_diff_af3 = (
            calc_distogram_difference(
                ref_A["distogram"],
                ref_B["distogram"],
                ref_A["mapping"],
                ref_B["mapping"],
                threshold=threshold,
                bin_centers=AF3_BIN_CENTERS_NP,
            )
        )

        # Calculate with Boltz bins
        mean_diff_boltz, _, diff_positions_boltz, dynamic_mean_diff_boltz = (
            calc_distogram_difference(
                ref_A["distogram"],
                ref_B["distogram"],
                ref_A["mapping"],
                ref_B["mapping"],
                threshold=threshold,
                bin_centers=BOLTZ_BIN_CENTERS_NP,
            )
        )

        # Generate visualization if requested (using AF3 bins)
        if save_visualizations and individual_results_dir:
            ref_A_name = f"{ref_A['reference_conformation']}_{ref_A['reference_state']}"
            ref_B_name = f"{ref_B['reference_conformation']}_{ref_B['reference_state']}"

            # Create visualization filename
            viz_filename = f"{ref_A_tag}_{ref_A_conf}_vs_{ref_B_tag}_{ref_B_conf}_distogram_diff.png"
            viz_path = individual_results_dir / viz_filename

            try:
                visualize_distogram_difference(
                    ref_A["distogram"],
                    ref_B["distogram"],
                    ref_A["mapping"],
                    ref_B["mapping"],
                    ref_A_name,
                    ref_B_name,
                    viz_path,
                )
            except Exception as e:
                errors.append(
                    {
                        "task": prediction_yaml_tag,
                        "pair": f"{ref_A_name} vs {ref_B_name}",
                        "error": f"visualization failed: {str(e)}",
                    }
                )

        # Create result dict
        comparison_result = {
            "cluster_id": cluster_id,
            "prediction_yaml_tag": prediction_yaml_tag,
            "reference_A": ref_A["reference_yaml_tag"],
            "reference_A_conformation": ref_A["reference_conformation"],
            "reference_A_state": ref_A["reference_state"],
            "reference_B": ref_B["reference_yaml_tag"],
            "reference_B_conformation": ref_B["reference_conformation"],
            "reference_B_state": ref_B["reference_state"],
            # AF3 metrics
            "mean_distance_difference_af3": mean_diff_af3,
            "dynamic_mean_distance_difference_af3": dynamic_mean_diff_af3,
            "n_diff_positions_af3": len(diff_positions_af3),
            "diff_positions_af3": diff_positions_af3,  # List of (model_i, model_j, distance_diff)
            # Boltz metrics
            "mean_distance_difference_boltz": mean_diff_boltz,
            "dynamic_mean_distance_difference_boltz": dynamic_mean_diff_boltz,
            "n_diff_positions_boltz": len(diff_positions_boltz),
            "diff_positions_boltz": diff_positions_boltz,
            # Common
            "n_aligned_residues": n_aligned,
            # Legacy fields (use AF3 for backward compatibility)
            "mean_distance_difference": mean_diff_af3,
            "dynamic_mean_distance_difference": dynamic_mean_diff_af3,
        }
        pairwise_results.append(comparison_result)

        # Save individual comparison result
        if individual_results_dir:
            result_filename = f"{ref_A_tag}_{ref_A_conf}_vs_{ref_B_tag}_{ref_B_conf}_distogram_diff.json"
            result_file = individual_results_dir / result_filename
            try:
                with open(result_file, "w") as f:
                    json.dump(comparison_result, f, indent=2)
                #print(f"    Saved comparison result to: {result_file}")
            except Exception as e:
                errors.append(
                    {
                        "task": prediction_yaml_tag,
                        "pair": f"{ref_A_tag}_{ref_A_conf} vs {ref_B_tag}_{ref_B_conf}",
                        "error": f"Failed to save result: {str(e)}",
                    }
                )

    result = {
        "cluster_id": cluster_id,
        "prediction_yaml_tag": prediction_yaml_tag,
        "method": task.get("method", ""),
        "method_type": task.get("method_type", ""),
        "n_references": len(ref_data),
        "n_comparisons": len(pairwise_results),
        "pairwise_comparisons": pairwise_results,
    }

    result_filename_all = "reference_distogram_diff.json"
    if individual_results_dir:
        result_file_all = individual_results_dir / result_filename_all
        if skip_existing and result_file_all.exists():
            try:
                with open(result_file_all, "r") as f:
                    existing_result = json.load(f)
                # Merge pairwise_comparisons and update metadata
                merged = dict(result)
                old_pairs = existing_result.get("pairwise_comparisons", [])
                new_pairs = result.get("pairwise_comparisons", [])

                # Avoid duplicates and collect all unique references
                seen = set()
                merged_pairs = []
                all_references = set()

                for comp in old_pairs + new_pairs:
                    tagA = comp.get(
                        "reference_A"
                    )  # reference_A is the yaml_tag directly
                    confA = comp.get("reference_A_conformation")
                    tagB = comp.get(
                        "reference_B"
                    )  # reference_B is the yaml_tag directly
                    confB = comp.get("reference_B_conformation")
                    key = (tagA, confA, tagB, confB)
                    if key not in seen:
                        merged_pairs.append(comp)
                        seen.add(key)
                        # Collect all unique references
                        all_references.add((tagA, confA))
                        all_references.add((tagB, confB))

                # Update metadata based on merged data
                merged["pairwise_comparisons"] = merged_pairs
                merged["n_comparisons"] = len(merged_pairs)
                merged["n_references"] = len(all_references)

                print(
                    f"  Merged: {len(old_pairs)} existing + {len(new_pairs)} new = {len(merged_pairs)} total pairs"
                )
                print(f"  Total unique references: {len(all_references)}")

                with open(result_file_all, "w") as f:
                    json.dump(merged, f, indent=2)
                print(f"  Updated aggregated result to: {result_file_all}")
            except Exception as e:
                errors.append(
                    {
                        "task": prediction_yaml_tag,
                        "error": f"Failed to merge/save aggregated result: {str(e)}",
                    }
                )
        else:
            with open(result_file_all, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved aggregated result to: {result_file_all}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Calculate distogram differences between reference conformations"
    )
    parser.add_argument(
        "--tasks", "-t", type=str, required=True, help="Path to distogram_tasks.json"
    )
    parser.add_argument(
        "--rep-seq",
        type=str,
        default=None,
        help="representative_sequences JSON (default: pipeline.files.rep_seq)",
    )
    parser.add_argument(
        "--msa-dir",
        type=str,
        default=None,
        help="a3m root (default: pipeline.dirs.msas)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Distance threshold (Angstroms) to report as significant difference",
    )
    parser.add_argument(
        "--start", "-s", type=int, default=0, help="Start index for parallel processing"
    )
    parser.add_argument(
        "--end", "-e", type=int, default=None, help="End index for parallel processing"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file for aggregated results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="reference_distogram_diff root (default: eval.dirs.ref_distogram / external.ref_distogram_dir)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Recalculate even if individual result files already exist",
    )

    args = parser.parse_args()

    tasks_json = Path(args.tasks)
    rep_seq_json = E.distogram_rep_seq_json(args.rep_seq)
    msa_dir = E.distogram_msa_dir(args.msa_dir)

    if not tasks_json.exists():
        raise FileNotFoundError(f"Tasks JSON not found: {tasks_json}")

    # Load tasks
    with open(tasks_json, "r") as f:
        tasks = json.load(f)

    # Load representative sequences
    with open(rep_seq_json, "r") as f:
        rep_seqs = json.load(f)

    print(f"Loaded {len(tasks)} tasks")
    print(f"Loaded {len(rep_seqs)} representative sequences")
    print(f"Distance threshold: {args.threshold} Angstroms")

    # Apply start/end indices
    if args.end is None:
        args.end = len(tasks)

    tasks_to_process = tasks[args.start : args.end]
    print(f"Processing tasks [{args.start}:{args.end}] ({len(tasks_to_process)} tasks)")

    # Set up output directory
    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        output_base_dir = E.distogram_ref_distogram_dir(None)

    print(f"Output directory: {output_base_dir}")

    results = []
    errors = []

    for i, task in enumerate(tasks_to_process):
        if (i + 1) % 1 == 0 or i == 0:
            n_refs = len(task.get("references", []))
            print(
                f"  [{args.start + i + 1}/{args.end}] Processing {task.get('cluster_id')}/{task.get('prediction_yaml_tag')} ({n_refs} references)..."
            )

        result = process_task(
            task,
            rep_seqs,
            msa_dir,
            args.threshold,
            errors,
            save_visualizations=True,
            output_dir=output_base_dir,
            skip_existing=not args.no_skip,
        )
        if result:
            results.append(result)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            tasks_json.parent / f"reference_distogram_diff_{args.start}_{args.end}.json"
        )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {output_path}")

    # Save errors
    if errors:
        error_path = (
            tasks_json.parent
            / f"reference_distogram_diff_errors_{args.start}_{args.end}.json"
        )
        with open(error_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Saved {len(errors)} errors to {error_path}")

    # Print summary
    print("\nSummary:")
    print(f"  Total tasks processed: {len(tasks_to_process)}")
    print(f"  Successful: {len(results)}")
    print(f"  Errors: {len(errors)}")

    if results:
        # Collect statistics
        all_mean_diffs = []
        all_dynamic_mean_diffs = []
        all_n_diff_positions = []
        total_comparisons = 0

        for r in results:
            for comp in r.get("pairwise_comparisons", []):
                if not np.isnan(comp["mean_distance_difference"]):
                    all_mean_diffs.append(comp["mean_distance_difference"])
                    all_n_diff_positions.append(comp["n_diff_positions_af3"])
                    total_comparisons += 1
                if not np.isnan(comp["dynamic_mean_distance_difference_af3"]):
                    all_dynamic_mean_diffs.append(
                        comp["dynamic_mean_distance_difference_af3"]
                    )

        if all_mean_diffs:
            print(f"  Total pairwise comparisons: {total_comparisons}")
            print(
                f"  Mean distance difference (all): {np.mean(all_mean_diffs):.3f} ± {np.std(all_mean_diffs):.3f} Å"
            )
            print(
                f"  Min/Max difference (all): {np.min(all_mean_diffs):.3f} / {np.max(all_mean_diffs):.3f} Å"
            )

        if all_dynamic_mean_diffs:
            print(
                f"  Mean distance difference (dynamic only): {np.mean(all_dynamic_mean_diffs):.3f} ± {np.std(all_dynamic_mean_diffs):.3f} Å"
            )
            print(
                f"  Min/Max difference (dynamic): {np.min(all_dynamic_mean_diffs):.3f} / {np.max(all_dynamic_mean_diffs):.3f} Å"
            )

        if all_n_diff_positions:
            print(
                f"  Mean positions exceeding threshold: {np.mean(all_n_diff_positions):.1f}"
            )
            print(
                f"  Max positions exceeding threshold: {np.max(all_n_diff_positions)}"
            )


if __name__ == "__main__":
    main()

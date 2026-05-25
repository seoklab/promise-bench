#!/usr/bin/env python3
"""
Calculate distogram loss between predictions and references.

Uses:
1. distogram_tasks.json - task definitions with paths
2. prediction_distograms (npz files) - predicted distograms
3. reference_cb_json - reference CB coordinates
4. chain_mapping_file - chain index mapping
5. MSAs and representative_sequences.json - for alignment

Usage (``PYTHONPATH=src``)::

    python -m eval.distogram.calc_distogram_loss --tasks data_eval/distogram/distogram_tasks.json --start 0 --end 100
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse
import json
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F
import shutil
import psutil

from utils._config import eval_cfg as E
from eval.distogram.path_utils import (
    flatten_valid_pair_edges,
    reference_distogram_diff_path,
)


# -----------------------------
# Constants for different methods
# -----------------------------
# AF3 bins (default)
AF3_BIN_LOW = 2.3125
AF3_BIN_HIGH = 21.6875
AF3_NUM_BINS_MINUS1 = 63  # boundaries (=> 64 bins)

# Boltz bins
BOLTZ_BIN_LOW = 2.0
BOLTZ_BIN_HIGH = 22.0
BOLTZ_NUM_BINS_MINUS1 = 63  # boundaries (=> 64 bins)


def get_distance_bins(method: str):
    """Get distance bin parameters for different methods."""
    if method.lower() in ["boltz", "boltz2"]:
        bin_low, bin_high, num_bins_minus1 = (
            BOLTZ_BIN_LOW,
            BOLTZ_BIN_HIGH,
            BOLTZ_NUM_BINS_MINUS1,
        )
    else:  # af3 and others
        bin_low, bin_high, num_bins_minus1 = (
            AF3_BIN_LOW,
            AF3_BIN_HIGH,
            AF3_NUM_BINS_MINUS1,
        )

    boundaries_cpu = torch.linspace(bin_low, bin_high, num_bins_minus1)
    breaks_np = boundaries_cpu.numpy()
    bin_centers_np = [0.5 * breaks_np[0]]
    bin_centers_np.extend(0.5 * (breaks_np[1:] + breaks_np[:-1]))
    bin_centers_np.append(breaks_np[-1])
    bin_centers_cpu = torch.from_numpy(np.array(bin_centers_np, dtype=np.float32))

    return boundaries_cpu, bin_centers_cpu, np.array(bin_centers_np, dtype=np.float32)


# Default bins (for backward compatibility)
BIN_LOW = AF3_BIN_LOW
BIN_HIGH = AF3_BIN_HIGH
NUM_BINS_MINUS1 = AF3_NUM_BINS_MINUS1

BOUNDARIES_CPU = torch.linspace(BIN_LOW, BIN_HIGH, NUM_BINS_MINUS1)
_breaks_np = BOUNDARIES_CPU.numpy()
_bin_centers_np = [0.5 * _breaks_np[0]]
_bin_centers_np.extend(0.5 * (_breaks_np[1:] + _breaks_np[:-1]))
_bin_centers_np.append(_breaks_np[-1])
BIN_CENTERS_CPU = torch.from_numpy(np.array(_bin_centers_np, dtype=np.float32))  # [64]
BIN_CENTERS_NP = np.array(
    _bin_centers_np, dtype=np.float32
)  # numpy version for convenience


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


def distance_to_bin_index(distance_matrix: torch.Tensor, boundaries: torch.Tensor):
    """Convert distance matrix to bin indices."""
    return (distance_matrix.unsqueeze(-1) > boundaries.view(1, 1, -1)).sum(dim=-1)


def load_all_pairwise_dynamic_regions(
    ref_distogram_diff_path: Path,
    method: str,
    valid_pair_edges: set[Tuple[str, str]],
) -> Dict[Tuple[str, str], set[Tuple[int, int]]]:
    """
    Load all pairwise dynamic regions from reference_distogram_diff.json.

    Args:
        ref_distogram_diff_path: Path to reference_distogram_diff.json
        method: Method name (af3, boltz1, boltz2, etc.) for bin selection

    Returns:
        Dictionary mapping (ref_A, ref_B) -> set of (idx1, idx2) pairs for dynamic regions
    """
    if not ref_distogram_diff_path.exists():
        print(f"    Warning: {ref_distogram_diff_path} does not exist.")
        return {}

    with open(ref_distogram_diff_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict) or len(data) == 0:
        print(f"    Warning: {ref_distogram_diff_path} is empty or invalid.")
        return {}

    # Determine which diff_positions key to use based on method
    if method.lower() in ["boltz", "boltz1", "boltz2"]:
        diff_key = "diff_positions_boltz"
    else:  # af3 and others
        print("    Using af3 diff_positions key for dynamic region extraction.")
        diff_key = "diff_positions_af3"
    fallback_key = "diff_positions"

    all_dynamic_regions = {}

    if "pairwise_comparisons" in data:
        for comparison in data["pairwise_comparisons"]:
            ref_A = comparison.get("reference_A", "")
            ref_B = comparison.get("reference_B", "")
            print(ref_A, ref_B)
            if not ref_A or not ref_B:
                continue
            if (ref_A, ref_B) not in valid_pair_edges:
                print("    Skipping pair not in valid pairs list.")
                continue
            # Try method-specific key first, then fallback
            if diff_key in comparison:
                diff_positions = comparison[diff_key]
            elif fallback_key in comparison:
                diff_positions = comparison[fallback_key]
            else:
                continue

            # Convert [idx1, idx2, diff] to set of (idx1, idx2) pairs
            positions = set()
            for pos in diff_positions:
                if len(pos) >= 2:
                    idx1, idx2 = int(pos[0]), int(pos[1])
                    # Add both (idx1, idx2) and (idx2, idx1) for symmetr
                    positions.add((idx1, idx2))
                    positions.add((idx2, idx1))

            # Store with canonical key (sorted alphabetically)
            key = tuple(sorted([ref_A, ref_B]))
            all_dynamic_regions[key] = positions

    return all_dynamic_regions


def load_dynamic_regions(
    ref_distogram_diff_path: Path,
    method: str,
    prediction_yaml_tag: str,
    reference_yaml_tag: str,
) -> set[Tuple[int, int]]:
    """
    Load dynamic regions from reference_distogram_diff.json for a specific pair.

    Uses method-specific diff_positions:
    - AF3: diff_positions_af3
    - Boltz: diff_positions_boltz
    - Fallback to diff_positions if method-specific not found

    Args:
        ref_distogram_diff_path: Path to reference_distogram_diff.json
        method: Method name (af3, boltz1, boltz2, etc.) for bin selection
        prediction_yaml_tag: The prediction's yaml_tag (required)
        reference_yaml_tag: The reference's yaml_tag (required)

    Returns set of (idx1, idx2) pairs for dynamic regions between this specific pair.
    """
    if not ref_distogram_diff_path.exists():
        print(f"    Warning: {ref_distogram_diff_path} does not exist.")
        return set()

    with open(ref_distogram_diff_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict) or len(data) == 0:
        print(f"    Warning: {ref_distogram_diff_path} is empty or invalid.")
        return set()

    # Determine which diff_positions key to use based on method
    if method.lower() in ["boltz", "boltz1", "boltz2"]:
        diff_key = "diff_positions_boltz"
    else:  # af3 and others
        diff_key = "diff_positions_af3"
    fallback_key = "diff_positions"

    # Find the exact pairwise comparison matching prediction and reference
    if "pairwise_comparisons" in data:
        for comparison in data["pairwise_comparisons"]:
            ref_A = comparison.get("reference_A", "")
            ref_B = comparison.get("reference_B", "")

            # Check if this comparison is between prediction and reference
            is_match = (
                ref_A == prediction_yaml_tag and ref_B == reference_yaml_tag
            ) or (ref_A == reference_yaml_tag and ref_B == prediction_yaml_tag)

            if not is_match:
                continue

            # Try method-specific key first, then fallback
            if diff_key in comparison:
                diff_positions = comparison[diff_key]
            elif fallback_key in comparison:
                diff_positions = comparison[fallback_key]
            else:
                print(
                    f"    Warning: No diff_positions found for {prediction_yaml_tag} vs {reference_yaml_tag}"
                )
                return set()

            # Convert [idx1, idx2, diff] to set of (idx1, idx2) pairs
            positions = set()
            for pos in diff_positions:
                if len(pos) >= 2:
                    idx1, idx2 = int(pos[0]), int(pos[1])
                    # Add both (idx1, idx2) and (idx2, idx1) for symmetry
                    positions.add((idx1, idx2))
                    positions.add((idx2, idx1))
            return positions

    print(
        f"    Warning: No matching pairwise comparison found for {prediction_yaml_tag} vs {reference_yaml_tag}"
    )
    return set()


def distogram_from_coordinates(coords: np.ndarray, method: str = "af3") -> np.ndarray:
    """Compute distogram from CB coordinates using method-specific bins."""
    boundaries_cpu, _, _ = get_distance_bins(method)

    coords_tensor = torch.from_numpy(coords)  # (L, 3)
    distance_matrix = torch.cdist(coords_tensor, coords_tensor)  # (L, L)
    bin_indices = distance_to_bin_index(distance_matrix, boundaries_cpu)  # (L, L)
    distogram = F.one_hot(bin_indices, num_classes=64).float()  # (L, L, 64)
    return distogram.numpy()


def calc_distogram_loss_multi_dynamic(
    method: str,
    reference_yaml_tag: str,
    prediction_yaml_tag: str,
    disto_target: np.ndarray,
    disto_pred: np.ndarray,
    ref_seq_id_to_ref_cb_idx: Dict[int, int],
    model_seq_id_to_model_disto_idx: Dict[int, int],
    all_dynamic_regions: Dict[Tuple[str, str], set[Tuple[int, int]]],
    pair_to_common_seq_ids: Dict[Tuple[str, str], set[int]] = None,
) -> Tuple[
    float,
    int,
    Dict[Tuple[str, str], float],
    Dict[Tuple[str, str], int],
    Dict[Tuple[str, str], float],
    float,
    Dict[Tuple[str, str], float],
    Dict[Tuple[str, str], int],
    Dict[Tuple[str, str], float],
]:
    """
    Calculate distogram loss with multiple dynamic region definitions.

    Args:
        disto_target: Reference distogram (L_ref, L_ref, 64)
        disto_pred: Predicted distogram (L_model, L_model, 64)
        ref_seq_id_to_ref_cb_idx: Mapping from reference seq_id to CB coord index
        model_seq_id_to_model_disto_idx: Mapping from model seq_id to distogram index
        all_dynamic_regions: Dict mapping (ref_A, ref_B) -> set of (idx1, idx2) dynamic positions
        pair_to_common_seq_ids: Dict mapping (ref_A, ref_B) -> set of common seq_ids between the pair

    Returns:
        total_loss: Total negative log probability (lower is better) - using all common residues
        n_aligned_total: Number of aligned residue pairs (total)
        dynamic_losses: Dict mapping (ref_A, ref_B) -> dynamic loss for that region definition
        n_aligned_dynamics: Dict mapping (ref_A, ref_B) -> n_aligned for that region definition
        dynamic_entropies: Dict mapping (ref_A, ref_B) -> entropy for that region definition
        total_entropy: Average entropy of predicted distribution in all aligned regions
        total_losses_per_pair: Dict mapping (ref_A, ref_B) -> total loss using only residues common to pair
        n_aligned_totals_per_pair: Dict mapping (ref_A, ref_B) -> n_aligned total for pair
        total_entropies_per_pair: Dict mapping (ref_A, ref_B) -> total entropy for pair
    """
    # Find common sequence IDs that exist in both mappings
    common_seq_ids = sorted(
        set(ref_seq_id_to_ref_cb_idx.keys())
        & set(model_seq_id_to_model_disto_idx.keys())
    )

    if len(common_seq_ids) < 3:
        print("  [ERROR] Not enough common residues for distogram loss calculation.")
        # Full 9-tuple to match the normal return signature.
        return (
            float("nan"),  # total_loss
            0,             # n_aligned_total
            {},            # dynamic_losses
            {},            # n_aligned_dynamics
            {},            # dynamic_entropies
            float("nan"),  # total_entropy
            {},            # total_losses_per_pair
            {},            # n_aligned_totals_per_pair
            {},            # total_entropies_per_pair
        )

    # Get indices for both distograms
    ref_cb_indices = [ref_seq_id_to_ref_cb_idx[seq_id] for seq_id in common_seq_ids]
    model_disto_indices = [
        model_seq_id_to_model_disto_idx[seq_id] for seq_id in common_seq_ids
    ]

    # Extract aligned regions
    disto_target_filtered = disto_target[
        np.ix_(ref_cb_indices, ref_cb_indices)
    ]  # (N, N, 64)
    disto_pred_filtered = disto_pred[
        np.ix_(model_disto_indices, model_disto_indices)
    ]  # (N, N, 64)

    # Calculate total loss
    eps = 1e-8
    disto_pred_filtered_clipped = np.clip(disto_pred_filtered, eps, 1.0)

    # Cross-entropy loss: -sum(target * log(pred))
    log_pred = np.log(disto_pred_filtered_clipped)
    loss_matrix = -(disto_target_filtered * log_pred).sum(axis=-1)
    total_loss = loss_matrix.sum() / (len(common_seq_ids) ** 2)
    n_aligned_total = int(len(common_seq_ids) * len(common_seq_ids))

    # Calculate total entropy (all aligned positions)
    total_entropy_per_position = -np.sum(
        disto_pred_filtered_clipped * np.log(disto_pred_filtered_clipped + eps), axis=-1
    )
    total_entropy = float(np.mean(total_entropy_per_position))

    # Calculate dynamic region losses for each pairwise definition
    dynamic_losses = {}
    n_aligned_dynamics = {}
    dynamic_entropies = {}

    for pair_key, dynamic_positions in all_dynamic_regions.items():
        if method != 'apo-monomers':
            if reference_yaml_tag != prediction_yaml_tag:
                if '_x' in reference_yaml_tag:
                    if pair_key[0] != reference_yaml_tag and pair_key[1] != reference_yaml_tag:
                        continue
                if '_x' in prediction_yaml_tag:
                    if pair_key[0] != prediction_yaml_tag and pair_key[1] != prediction_yaml_tag:
                        continue
        if not dynamic_positions:
            dynamic_losses[pair_key] = float("nan")
            n_aligned_dynamics[pair_key] = 0
            dynamic_entropies[pair_key] = float("nan")
            continue

        # Create mask for this dynamic region definition
        # Only include positions where BOTH seq_ids are in common_seq_ids (intersection)
        dynamic_mask = np.zeros((len(common_seq_ids), len(common_seq_ids)), dtype=bool)
        common_seq_ids_set = set(common_seq_ids)

        for i, seq_id_i in enumerate(common_seq_ids):
            for j, seq_id_j in enumerate(common_seq_ids):
                # Check if this position is in dynamic_positions AND both seq_ids are in current alignment
                if (seq_id_i, seq_id_j) in dynamic_positions:
                    # Both seq_ids are guaranteed to be in common_seq_ids by construction
                    dynamic_mask[i, j] = True

        if not np.any(dynamic_mask):
            dynamic_losses[pair_key] = float("nan")
            n_aligned_dynamics[pair_key] = 0
            dynamic_entropies[pair_key] = float("nan")
            continue

        # Apply mask to get dynamic region only
        disto_target_dynamic = disto_target_filtered[dynamic_mask]  # (N_dynamic, 64)
        disto_pred_dynamic = disto_pred_filtered_clipped[
            dynamic_mask
        ]  # (N_dynamic, 64)

        # Calculate loss for dynamic region
        log_pred_dynamic = np.log(disto_pred_dynamic)
        loss_dynamic_total = -np.sum(disto_target_dynamic * log_pred_dynamic)
        n_dynamic = int(np.sum(dynamic_mask))
        dynamic_loss = loss_dynamic_total / n_dynamic if n_dynamic > 0 else float("nan")

        # Calculate entropy for dynamic region
        entropy_per_position = -np.sum(
            disto_pred_dynamic * np.log(disto_pred_dynamic + eps), axis=-1
        )
        dynamic_entropy = float(np.mean(entropy_per_position))

        dynamic_losses[pair_key] = dynamic_loss
        n_aligned_dynamics[pair_key] = n_dynamic
        dynamic_entropies[pair_key] = dynamic_entropy

    # Calculate total loss per pair (using only residues common to both references in the pair)
    total_losses_per_pair = {}
    n_aligned_totals_per_pair = {}
    total_entropies_per_pair = {}

    if pair_to_common_seq_ids:
        for pair_key, pair_common_seq_ids in pair_to_common_seq_ids.items():
            if method != 'apo-monomers':
                if reference_yaml_tag != prediction_yaml_tag:
                    if '_x' in reference_yaml_tag:
                        if pair_key[0] != reference_yaml_tag and pair_key[1] != reference_yaml_tag:
                            print("not matching reference_yaml_tag")
                            continue
                    if '_x' in prediction_yaml_tag:
                        if pair_key[0] != prediction_yaml_tag and pair_key[1] != prediction_yaml_tag:
                            print("not matching prediction_yaml_tag")
                            continue

            if not pair_common_seq_ids or len(pair_common_seq_ids) < 3:
                total_losses_per_pair[pair_key] = float("nan")
                n_aligned_totals_per_pair[pair_key] = 0
                total_entropies_per_pair[pair_key] = float("nan")
                continue

            # Intersect pair's common seq_ids with current alignment's common_seq_ids
            common_for_pair = sorted(set(common_seq_ids) & pair_common_seq_ids)

            if len(common_for_pair) < 3:
                total_losses_per_pair[pair_key] = float("nan")
                n_aligned_totals_per_pair[pair_key] = 0
                total_entropies_per_pair[pair_key] = float("nan")
                continue

            # Get indices for this pair's common residues
            pair_ref_indices = [
                ref_seq_id_to_ref_cb_idx[seq_id] for seq_id in common_for_pair
            ]
            pair_model_indices = [
                model_seq_id_to_model_disto_idx[seq_id] for seq_id in common_for_pair
            ]

            # Extract regions for this pair
            disto_target_pair = disto_target[np.ix_(pair_ref_indices, pair_ref_indices)]
            disto_pred_pair = disto_pred[np.ix_(pair_model_indices, pair_model_indices)]
            disto_pred_pair_clipped = np.clip(disto_pred_pair, eps, 1.0)

            # Calculate total loss for this pair
            log_pred_pair = np.log(disto_pred_pair_clipped)
            loss_matrix_pair = -(disto_target_pair * log_pred_pair).sum(axis=-1)
            pair_total_loss = loss_matrix_pair.sum() / (len(common_for_pair) ** 2)
            pair_n_aligned = int(len(common_for_pair) * len(common_for_pair))

            # Calculate entropy for this pair
            entropy_per_position_pair = -np.sum(
                disto_pred_pair_clipped * np.log(disto_pred_pair_clipped + eps), axis=-1
            )
            pair_total_entropy = float(np.mean(entropy_per_position_pair))

            total_losses_per_pair[pair_key] = float(pair_total_loss)
            n_aligned_totals_per_pair[pair_key] = pair_n_aligned
            total_entropies_per_pair[pair_key] = pair_total_entropy

    return (
        total_loss,
        n_aligned_total,
        dynamic_losses,
        n_aligned_dynamics,
        dynamic_entropies,
        total_entropy,
        total_losses_per_pair,
        n_aligned_totals_per_pair,
        total_entropies_per_pair,
    )


def calc_expected_distance_error(
    disto_target: np.ndarray,
    disto_pred: np.ndarray,
    ref_seq_id_to_ref_cb_idx: Dict[int, int],
    model_seq_id_to_model_disto_idx: Dict[int, int],
    dynamic_positions: set[Tuple[int, int]] = None,
    method: str = "af3",
) -> Tuple[float, float, int, float, float, int]:
    """
    Calculate expected distance error using two methods for both total and dynamic regions.

    Method 1 (expectation_of_distance): E[|d - d_true|] - expectation of distance (more accurate)
    Method 2 (distance_of_expectation): |E[d] - d_true| - distance of expectation (underestimates by Jensen's inequality)

    Args:
        disto_target: Reference distogram (L_ref, L_ref, 64) - one-hot encoded
        disto_pred: Predicted distogram (L_model, L_model, 64) - probabilities
        ref_seq_id_to_ref_cb_idx: Mapping from reference seq_id to CB coord index
        model_seq_id_to_model_disto_idx: Mapping from model seq_id to distogram index
        dynamic_positions: Set of (seq_id1, seq_id2) pairs for dynamic regions
        method: Method name (af3, boltz1, boltz2, etc.) for selecting bin centers

    Returns:
        expectation_of_distance: E[|d - d_true|] in Angstroms (total)
        distance_of_expectation: |E[d] - d_true| in Angstroms (total)
        n_aligned: Number of aligned residues (total)
        dynamic_expectation_of_distance: E[|d - d_true|] in Angstroms (dynamic only)
        dynamic_distance_of_expectation: |E[d] - d_true| in Angstroms (dynamic only)
        n_aligned_dynamic: Number of aligned residues (dynamic only)
    """
    # Find common sequence IDs
    common_seq_ids = sorted(
        set(ref_seq_id_to_ref_cb_idx.keys())
        & set(model_seq_id_to_model_disto_idx.keys())
    )

    if len(common_seq_ids) < 3:
        return float("nan"), float("nan"), 0, float("nan"), float("nan"), 0

    # Get indices for both distograms
    ref_cb_indices = [ref_seq_id_to_ref_cb_idx[seq_id] for seq_id in common_seq_ids]
    model_disto_indices = [
        model_seq_id_to_model_disto_idx[seq_id] for seq_id in common_seq_ids
    ]

    # Extract aligned regions
    disto_target_filtered = disto_target[
        np.ix_(ref_cb_indices, ref_cb_indices)
    ]  # (N, N, 64)
    disto_pred_filtered = disto_pred[
        np.ix_(model_disto_indices, model_disto_indices)
    ]  # (N, N, 64)

    # Get method-specific bin centers
    _, _, bin_centers_np = get_distance_bins(method)

    # Get true bin indices from one-hot encoded target
    target_bins = disto_target_filtered.argmax(axis=-1)  # (N, N)

    # Get true distances (bin centers)
    true_distances = bin_centers_np[target_bins]  # (N, N)

    # Method 1: E[|d - d_true|] - expectation of distance
    # (N, N, 64): distance from each bin center to true distance
    dist_to_true = np.abs(bin_centers_np[None, None, :] - true_distances[:, :, None])
    # (N, N): weighted sum
    expectation_of_distance_matrix = (disto_pred_filtered * dist_to_true).sum(axis=-1)
    expectation_of_distance = expectation_of_distance_matrix.mean()

    # Method 2: |E[d] - d_true| - distance of expectation
    # Compute expected predicted distance: sum(p_i * bin_center_i)
    predicted_distances = (
        disto_pred_filtered * bin_centers_np[np.newaxis, np.newaxis, :]
    ).sum(axis=-1)  # (N, N)
    # Compute distance error
    distance_of_expectation_matrix = np.abs(
        predicted_distances - true_distances
    )  # (N, N)
    distance_of_expectation = distance_of_expectation_matrix.mean()

    # Calculate dynamic region distance errors
    dynamic_expectation_of_distance = float("nan")
    dynamic_distance_of_expectation = float("nan")
    n_aligned_dynamic = 0

    if dynamic_positions:
        # Create mask for dynamic positions
        dynamic_mask = np.zeros((len(common_seq_ids), len(common_seq_ids)), dtype=bool)

        for i, seq_id_i in enumerate(common_seq_ids):
            for j, seq_id_j in enumerate(common_seq_ids):
                if (seq_id_i, seq_id_j) in dynamic_positions:
                    dynamic_mask[i, j] = True

        if np.any(dynamic_mask):
            # Apply mask to get dynamic region only
            expectation_of_distance_dynamic = expectation_of_distance_matrix[
                dynamic_mask
            ]
            distance_of_expectation_dynamic = distance_of_expectation_matrix[
                dynamic_mask
            ]

            n_aligned_dynamic = int(np.sum(dynamic_mask))
            dynamic_expectation_of_distance = float(
                expectation_of_distance_dynamic.mean()
            )
            dynamic_distance_of_expectation = float(
                distance_of_expectation_dynamic.mean()
            )

    return (
        float(expectation_of_distance),
        float(distance_of_expectation),
        len(common_seq_ids),
        dynamic_expectation_of_distance,
        dynamic_distance_of_expectation,
        n_aligned_dynamic,
    )


def calc_expected_distance_error_multi_dynamic(
    method_type: str,
    reference_yaml_tag: str,
    prediction_yaml_tag: str,
    disto_target: np.ndarray,
    disto_pred: np.ndarray,
    ref_seq_id_to_ref_cb_idx: Dict[int, int],
    model_seq_id_to_model_disto_idx: Dict[int, int],
    all_dynamic_regions: Dict[Tuple[str, str], set[Tuple[int, int]]],
    method: str = "af3",
    pair_to_common_seq_ids: Dict[Tuple[str, str], set[int]] = None,
) -> Tuple[
    float,
    float,
    int,
    Dict[Tuple[str, str], float],
    Dict[Tuple[str, str], float],
    Dict[Tuple[str, str], int],
    Dict[Tuple[str, str], float],
    Dict[Tuple[str, str], float],
    Dict[Tuple[str, str], int],
]:
    """
    Calculate expected distance error with multiple dynamic region definitions.

    Args:
        disto_target: Reference distogram (L_ref, L_ref, 64) - one-hot encoded
        disto_pred: Predicted distogram (L_model, L_model, 64) - probabilities
        ref_seq_id_to_ref_cb_idx: Mapping from reference seq_id to CB coord index
        model_seq_id_to_model_disto_idx: Mapping from model seq_id to distogram index
        all_dynamic_regions: Dict mapping (ref_A, ref_B) -> set of (idx1, idx2) dynamic positions
        method: Method name (af3, boltz1, boltz2, etc.) for selecting bin centers
        pair_to_common_seq_ids: Dict mapping (ref_A, ref_B) -> set of common seq_ids between the pair

    Returns:
        expectation_of_distance: E[|d - d_true|] in Angstroms (total, all common residues)
        distance_of_expectation: |E[d] - d_true| in Angstroms (total, all common residues)
        n_aligned: Number of aligned residues (total)
        dynamic_expectation_of_distances: Dict (ref_A, ref_B) -> E[|d - d_true|] for that region
        dynamic_distance_of_expectations: Dict (ref_A, ref_B) -> |E[d] - d_true| for that region
        n_aligned_dynamics: Dict (ref_A, ref_B) -> n_aligned for that region
        total_expectation_of_distances_per_pair: Dict (ref_A, ref_B) -> E[|d - d_true|] using only residues common to pair
        total_distance_of_expectations_per_pair: Dict (ref_A, ref_B) -> |E[d] - d_true| using only residues common to pair
        n_aligned_totals_per_pair: Dict (ref_A, ref_B) -> n_aligned total for pair
    """
    # Find common sequence IDs
    common_seq_ids = sorted(
        set(ref_seq_id_to_ref_cb_idx.keys())
        & set(model_seq_id_to_model_disto_idx.keys())
    )

    if len(common_seq_ids) < 3:
        # Full 9-tuple to match the normal return signature.
        return (
            float("nan"),  # expectation_of_distance
            float("nan"),  # distance_of_expectation
            0,             # n_aligned (= len(common_seq_ids))
            {},            # dynamic_expectation_of_distances
            {},            # dynamic_distance_of_expectations
            {},            # n_aligned_dynamics
            {},            # total_expectation_of_distances_per_pair
            {},            # total_distance_of_expectations_per_pair
            {},            # n_aligned_totals_per_pair
        )

    # Get indices for both distograms
    ref_cb_indices = [ref_seq_id_to_ref_cb_idx[seq_id] for seq_id in common_seq_ids]
    model_disto_indices = [
        model_seq_id_to_model_disto_idx[seq_id] for seq_id in common_seq_ids
    ]

    # Extract aligned regions
    disto_target_filtered = disto_target[
        np.ix_(ref_cb_indices, ref_cb_indices)
    ]  # (N, N, 64)
    disto_pred_filtered = disto_pred[
        np.ix_(model_disto_indices, model_disto_indices)
    ]  # (N, N, 64)

    # Get method-specific bin centers
    _, _, bin_centers_np = get_distance_bins(method)

    # Get true bin indices from one-hot encoded target
    target_bins = disto_target_filtered.argmax(axis=-1)  # (N, N)

    # Get true distances (bin centers)
    true_distances = bin_centers_np[target_bins]  # (N, N)

    # Method 1: E[|d - d_true|] - expectation of distance
    dist_to_true = np.abs(bin_centers_np[None, None, :] - true_distances[:, :, None])
    expectation_of_distance_matrix = (disto_pred_filtered * dist_to_true).sum(axis=-1)
    expectation_of_distance = expectation_of_distance_matrix.mean()

    # Method 2: |E[d] - d_true| - distance of expectation
    predicted_distances = (
        disto_pred_filtered * bin_centers_np[np.newaxis, np.newaxis, :]
    ).sum(axis=-1)  # (N, N)
    distance_of_expectation_matrix = np.abs(predicted_distances - true_distances)
    distance_of_expectation = distance_of_expectation_matrix.mean()

    # Calculate dynamic region distance errors for each pairwise definition
    dynamic_expectation_of_distances = {}
    dynamic_distance_of_expectations = {}
    n_aligned_dynamics = {}

    for pair_key, dynamic_positions in all_dynamic_regions.items():
        if method_type != 'apo-monomers':
            if reference_yaml_tag != prediction_yaml_tag:
                if '_x' in reference_yaml_tag:
                    if pair_key[0] != reference_yaml_tag and pair_key[1] != reference_yaml_tag:
                        print("not matching reference_yaml_tag")
                        continue
                if '_x' in prediction_yaml_tag:
                    if pair_key[0] != prediction_yaml_tag and pair_key[1] != prediction_yaml_tag:
                        print("not matching prediction_yaml_tag")
                        continue

        if not dynamic_positions:
            dynamic_expectation_of_distances[pair_key] = float("nan")
            dynamic_distance_of_expectations[pair_key] = float("nan")
            n_aligned_dynamics[pair_key] = 0
            continue

        # Create mask for this dynamic region definition
        # Only include positions where BOTH seq_ids are in common_seq_ids (intersection)
        dynamic_mask = np.zeros((len(common_seq_ids), len(common_seq_ids)), dtype=bool)
        common_seq_ids_set = set(common_seq_ids)

        for i, seq_id_i in enumerate(common_seq_ids):
            for j, seq_id_j in enumerate(common_seq_ids):
                # Check if this position is in dynamic_positions AND both seq_ids are in current alignment
                if (seq_id_i, seq_id_j) in dynamic_positions:
                    # Both seq_ids are guaranteed to be in common_seq_ids by construction
                    dynamic_mask[i, j] = True

        if not np.any(dynamic_mask):
            dynamic_expectation_of_distances[pair_key] = float("nan")
            dynamic_distance_of_expectations[pair_key] = float("nan")
            n_aligned_dynamics[pair_key] = 0
            continue

        # Apply mask to get dynamic region only
        expectation_of_distance_dynamic = expectation_of_distance_matrix[dynamic_mask]
        distance_of_expectation_dynamic = distance_of_expectation_matrix[dynamic_mask]

        n_aligned_dynamics[pair_key] = int(np.sum(dynamic_mask))
        dynamic_expectation_of_distances[pair_key] = float(
            expectation_of_distance_dynamic.mean()
        )
        dynamic_distance_of_expectations[pair_key] = float(
            distance_of_expectation_dynamic.mean()
        )

    # Calculate total distance error per pair (using only residues common to both references in the pair)
    total_expectation_of_distances_per_pair = {}
    total_distance_of_expectations_per_pair = {}
    n_aligned_totals_per_pair = {}

    if pair_to_common_seq_ids:
        for pair_key, pair_common_seq_ids in pair_to_common_seq_ids.items():

            if method_type != 'apo-monomers':
                if reference_yaml_tag != prediction_yaml_tag:
                    if '_x' in reference_yaml_tag:
                        if pair_key[0] != reference_yaml_tag and pair_key[1] != reference_yaml_tag:
                            print("not matching reference_yaml_tag in total per pair")
                            continue
                    if '_x' in prediction_yaml_tag:
                        if pair_key[0] != prediction_yaml_tag and pair_key[1] != prediction_yaml_tag:
                            print("not matching prediction_yaml_tag in total per pair")
                            continue

            if not pair_common_seq_ids or len(pair_common_seq_ids) < 3:
                total_expectation_of_distances_per_pair[pair_key] = float("nan")
                total_distance_of_expectations_per_pair[pair_key] = float("nan")
                n_aligned_totals_per_pair[pair_key] = 0
                continue

            # Intersect pair's common seq_ids with current alignment's common_seq_ids
            common_for_pair = sorted(set(common_seq_ids) & pair_common_seq_ids)

            if len(common_for_pair) < 3:
                total_expectation_of_distances_per_pair[pair_key] = float("nan")
                total_distance_of_expectations_per_pair[pair_key] = float("nan")
                n_aligned_totals_per_pair[pair_key] = 0
                continue

            # Get indices for this pair's common residues
            pair_ref_indices = [
                ref_seq_id_to_ref_cb_idx[seq_id] for seq_id in common_for_pair
            ]
            pair_model_indices = [
                model_seq_id_to_model_disto_idx[seq_id] for seq_id in common_for_pair
            ]

            # Extract regions for this pair
            disto_target_pair = disto_target[np.ix_(pair_ref_indices, pair_ref_indices)]
            disto_pred_pair = disto_pred[np.ix_(pair_model_indices, pair_model_indices)]

            # Get true distances for this pair
            target_bins_pair = disto_target_pair.argmax(axis=-1)
            true_distances_pair = bin_centers_np[target_bins_pair]

            # Method 1: E[|d - d_true|]
            dist_to_true_pair = np.abs(
                bin_centers_np[None, None, :] - true_distances_pair[:, :, None]
            )
            expectation_of_distance_pair = (disto_pred_pair * dist_to_true_pair).sum(
                axis=-1
            )
            pair_expectation = float(expectation_of_distance_pair.mean())

            # Method 2: |E[d] - d_true|
            predicted_distances_pair = (
                disto_pred_pair * bin_centers_np[np.newaxis, np.newaxis, :]
            ).sum(axis=-1)
            distance_of_expectation_pair = np.abs(
                predicted_distances_pair - true_distances_pair
            )
            pair_distance = float(distance_of_expectation_pair.mean())

            total_expectation_of_distances_per_pair[pair_key] = pair_expectation
            total_distance_of_expectations_per_pair[pair_key] = pair_distance
            n_aligned_totals_per_pair[pair_key] = len(common_for_pair)

    return (
        float(expectation_of_distance),
        float(distance_of_expectation),
        len(common_seq_ids),
        dynamic_expectation_of_distances,
        dynamic_distance_of_expectations,
        n_aligned_dynamics,
        total_expectation_of_distances_per_pair,
        total_distance_of_expectations_per_pair,
        n_aligned_totals_per_pair,
    )


def calc_correct_ratio(
    disto_target: np.ndarray,
    disto_pred: np.ndarray,
    ref_seq_id_to_ref_cb_idx: Dict[int, int],
    model_seq_id_to_model_disto_idx: Dict[int, int],
) -> Tuple[float, int]:
    """Calculate fraction of correctly predicted distance bins."""
    common_seq_ids = sorted(
        set(ref_seq_id_to_ref_cb_idx.keys())
        & set(model_seq_id_to_model_disto_idx.keys())
    )

    if len(common_seq_ids) < 3:
        return float("nan"), 0

    ref_cb_indices = [ref_seq_id_to_ref_cb_idx[seq_id] for seq_id in common_seq_ids]
    model_disto_indices = [
        model_seq_id_to_model_disto_idx[seq_id] for seq_id in common_seq_ids
    ]

    disto_target_filtered = disto_target[np.ix_(ref_cb_indices, ref_cb_indices)]
    disto_pred_filtered = disto_pred[np.ix_(model_disto_indices, model_disto_indices)]

    # Compare argmax (most likely bin)
    target_bins = disto_target_filtered.argmax(axis=-1)
    pred_bins = disto_pred_filtered.argmax(axis=-1)

    correct = (target_bins == pred_bins).sum()
    total = len(common_seq_ids) ** 2

    return float(correct / total), len(common_seq_ids)


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


def process_task(
    task: dict,
    rep_seqs: dict,
    msa_dir: Path,
    errors: list,
    ref_distogram_dir: Path,
    valid_pair_edges: set[Tuple[str, str]],
) -> Optional[dict]:
    """
    Process a single distogram task with multiple references.

    Returns:
        Result dict or None if processing failed
    """
    cluster_id = task.get("cluster_id")
    prediction_yaml_tag = task.get("prediction_yaml_tag")
    target_conformation = task.get("target_conformation", "default")

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

    # Load chain mapping file (common for all references)
    chain_mapping_file = task.get("chain_mapping_file", "")
    if not chain_mapping_file or not Path(chain_mapping_file).exists():
        errors.append(
            {
                "task": prediction_yaml_tag,
                "error": f"chain_mapping_file not found: {chain_mapping_file}",
            }
        )
        return None

    with open(chain_mapping_file) as f:
        chain_mapping = json.load(f)

    mobile_chain = task.get("mobile_chain", "")
    if mobile_chain not in chain_mapping.get("chains", {}):
        errors.append(
            {
                "task": prediction_yaml_tag,
                "error": f"mobile_chain {mobile_chain} not in chain_mapping. Available: {list(chain_mapping.get('chains', {}).keys())}",
            }
        )
        return None

    chain_info = chain_mapping["chains"][mobile_chain]
    chain_start = chain_info["start"]
    chain_end = chain_info["end"]
    chain_length = chain_info["length"]

    # Process each prediction distogram (common for all references)
    prediction_distograms = task.get("prediction_distograms", [])
    if not prediction_distograms:
        errors.append(
            {"task": prediction_yaml_tag, "error": "no prediction_distograms"}
        )
        return None

    # Process each reference
    references = task.get("references", [])
    if not references:
        errors.append({"task": prediction_yaml_tag, "error": "no references"})
        return None

    # Load all pairwise dynamic regions upfront (before reference loop)
    distogram_dir = (
        Path(prediction_distograms[0]).parent if prediction_distograms else None
    )
    all_dynamic_regions = {}
    if distogram_dir:
        ref_distogram_diff_path = reference_distogram_diff_path(
            ref_distogram_dir, task
        )
        print(f"Loading pairwise dynamic regions from {ref_distogram_diff_path}")
        all_dynamic_regions = load_all_pairwise_dynamic_regions(
            ref_distogram_diff_path,
            method=task["method"],
            valid_pair_edges=valid_pair_edges,
        )
        print(
            f"  Loaded {len(all_dynamic_regions)} pairwise dynamic region definitions"
        )
        for pair_key, positions in all_dynamic_regions.items():
            print(f"    {pair_key}: {len(positions)} dynamic positions")

    # Pre-load all reference CB data to find common residues per pair
    ref_tag_to_cb_seq_ids = {}  # reference_yaml_tag -> set of seq_ids
    for ref_info in references:
        reference_yaml_tag = ref_info.get("reference_yaml_tag")
        ref_cb_json = ref_info.get("reference_cb_json", "")
        ref_chain = ref_info.get("ref_chain", "")
        print(ref_cb_json)
        if ref_cb_json and Path(ref_cb_json).exists():
            try:
                with open(ref_cb_json) as f:
                    ref_cb_data = json.load(f)

                if ref_chain in ref_cb_data.get("all_chains", {}):
                    ref_chain_data = ref_cb_data["all_chains"][ref_chain]
                    seq_ids = set(
                        int(k) for k in ref_chain_data["seq_id_to_coord"].keys()
                    )
                    ref_tag_to_cb_seq_ids[reference_yaml_tag] = seq_ids
                    print(f"  Loaded {len(seq_ids)} seq_ids from {reference_yaml_tag}")
            except Exception as e:
                print(
                    f"  Warning: Failed to load CB data for {reference_yaml_tag}: {e}"
                )

    # Build pair-to-common-seq-ids mapping
    pair_to_common_seq_ids = {}  # (ref_A, ref_B) -> set of common seq_ids
    for pair_key in all_dynamic_regions.keys():
        ref_A, ref_B = pair_key
        seq_ids_A = ref_tag_to_cb_seq_ids.get(ref_A, set())
        seq_ids_B = ref_tag_to_cb_seq_ids.get(ref_B, set())
        common_seq_ids = seq_ids_A & seq_ids_B
        pair_to_common_seq_ids[pair_key] = common_seq_ids
        print(
            f"  Pair {pair_key}: {len(common_seq_ids)} common seq_ids (A={len(seq_ids_A)}, B={len(seq_ids_B)})"
        )

    reference_results = []

    for ref_info in references:
        reference_conformation = ref_info.get("reference_conformation", "")
        reference_yaml_tag = ref_info.get("reference_yaml_tag")
        ref_cif = ref_info.get("ref_cif", "")
        ref_chain = ref_info.get("ref_chain", "")

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
                print(f"  [ERROR] Skipping reference {reference_yaml_tag}: ref_id '{ref_id}' not found in MSA")
                print(f"     Tried: {[p for p in possible_ids if p]}")
                print(f"     Available in MSA: {list(alignments.keys())[:10]}... (showing first 10)")
                errors.append(
                    {
                        "task": prediction_yaml_tag,
                        "reference": reference_yaml_tag,
                        "error": f"ref_id {ref_id} not found in alignments",
                    }
                )
                continue
        else:
            ref_aligned_seq = alignments[ref_id]

        # Get alignment mapping: model_seq_idx -> ref_seq_idx
        alignment_mapping = get_alignment_mapping(model_aligned_seq, ref_aligned_seq)

        if len(alignment_mapping) < 3:
            errors.append(
                {
                    "task": prediction_yaml_tag,
                    "reference": reference_yaml_tag,
                    "error": f"too few aligned positions: {len(alignment_mapping)}",
                }
            )
            continue

        # Load reference CB coordinates
        ref_cb_json = ref_info.get("reference_cb_json", "")
        if not ref_cb_json or not Path(ref_cb_json).exists():
            errors.append(
                {
                    "task": prediction_yaml_tag,
                    "reference": reference_yaml_tag,
                    "error": f"reference_cb_json not found: {ref_cb_json}",
                }
            )
            continue

        with open(ref_cb_json) as f:
            ref_cb_data = json.load(f)

        # Get reference chain data - simply use ref_chain as key
        if ref_chain not in ref_cb_data.get("all_chains", {}):
            errors.append(
                {
                    "task": prediction_yaml_tag,
                    "reference": reference_yaml_tag,
                    "error": f"ref chain {ref_chain} not found in CB data. Available: {list(ref_cb_data.get('all_chains', {}).keys())}",
                }
            )
            continue

        ref_chain_data = ref_cb_data["all_chains"][ref_chain]

        # ref_chain_data["seq_id_to_coord"] maps seq_id (0-based) -> [x, y, z]
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

        # Process prediction distograms for this reference
        seed_results = []
        print(f"    Processing {len(prediction_distograms)} distogram files...")

        for distogram_path in prediction_distograms:
            distogram_path = Path(distogram_path)
            if not distogram_path.exists():
                print(f"    Skipping missing file: {distogram_path}")
                continue

            # Extract seed from filename
            seed_name = distogram_path.stem.split("_distogram")[0]
            seed_parts = seed_name.split("_")
            if len(seed_parts) >= 2 and seed_parts[0] == "seed":
                seed_id = seed_parts[1]
            else:
                seed_id = seed_name

            print(f"    Processing seed: {seed_id}")

            try:
                # Load predicted distogram
                dist_data = np.load(distogram_path, allow_pickle=True)
                if task["method"] != 'bioemu':
                    pred_distogram_full = dist_data["distogram"]
                else:
                    pred_distogram_full = dist_data["distogram"]["logits"]
                # Handle different possible shapes
                if pred_distogram_full.ndim == 5:
                    # [1, L, L, 1, 64] -> [L, L, 64]
                    pred_distogram_full = pred_distogram_full[0, :, :, 0, :]
                elif pred_distogram_full.ndim == 4:
                    pred_distogram_full = pred_distogram_full[0]
                elif pred_distogram_full.ndim == 3:
                    pred_distogram_full = pred_distogram_full
                else:
                    errors.append(
                        {
                            "task": prediction_yaml_tag,
                            "reference": reference_yaml_tag,
                            "seed": seed_id,
                            "error": f"unexpected distogram shape: {pred_distogram_full.shape}",
                        }
                    )
                    continue

                # Extract chain region
                pred_distogram_chain = pred_distogram_full[
                    chain_start : chain_end + 1, chain_start : chain_end + 1 :
                ]

                # Normalize to probabilities
                pred_distogram_probs_tensor = torch.from_numpy(
                    pred_distogram_chain
                ).float()
                pred_distogram_probs = F.softmax(
                    pred_distogram_probs_tensor, dim=-1
                ).numpy()

                # Create alignment mappings
                ref_seq_id_to_ref_cb_idx_mapped = {}
                model_seq_id_to_model_disto_idx_mapped = {}

                for model_idx, ref_idx in alignment_mapping.items():
                    ref_seq_id = ref_idx

                    if ref_seq_id in ref_seq_id_to_cb_idx:
                        if model_idx < chain_length:
                            ref_seq_id_to_ref_cb_idx_mapped[model_idx] = (
                                ref_seq_id_to_cb_idx[ref_seq_id]
                            )
                            model_seq_id_to_model_disto_idx_mapped[model_idx] = (
                                model_idx
                            )

                # Calculate loss (both total and per-pair dynamic)
                (
                    total_loss,
                    n_aligned_total,
                    dynamic_losses,
                    n_aligned_dynamics,
                    dynamic_entropies,
                    total_entropy,
                    total_losses_per_pair,
                    n_aligned_totals_per_pair,
                    total_entropies_per_pair,
                ) = calc_distogram_loss_multi_dynamic(
                    task["method_type"],
                    reference_yaml_tag,
                    prediction_yaml_tag,
                    ref_distogram,
                    pred_distogram_probs,
                    ref_seq_id_to_ref_cb_idx_mapped,
                    model_seq_id_to_model_disto_idx_mapped,
                    all_dynamic_regions,
                    pair_to_common_seq_ids,
                )

                # Calculate expected distance error (both total and per-pair dynamic)
                (
                    expectation_of_distance,
                    distance_of_expectation,
                    _,
                    dynamic_expectation_of_distances,
                    dynamic_distance_of_expectations,
                    _,
                    total_expectation_of_distances_per_pair,
                    total_distance_of_expectations_per_pair,
                    _,
                ) = calc_expected_distance_error_multi_dynamic(
                    task["method_type"],
                    reference_yaml_tag,
                    prediction_yaml_tag,
                    ref_distogram,
                    pred_distogram_probs,
                    ref_seq_id_to_ref_cb_idx_mapped,
                    model_seq_id_to_model_disto_idx_mapped,
                    all_dynamic_regions,
                    task["method"],
                    pair_to_common_seq_ids=pair_to_common_seq_ids,
                )

                correct_ratio, _ = calc_correct_ratio(
                    ref_distogram,
                    pred_distogram_probs,
                    ref_seq_id_to_ref_cb_idx_mapped,
                    model_seq_id_to_model_disto_idx_mapped,
                )

                # Convert tuple keys to string for JSON serialization
                dynamic_losses_json = (
                    {f"{k[0]}|{k[1]}": v for k, v in dynamic_losses.items()}
                    if dynamic_losses
                    else {}
                )
                n_aligned_dynamics_json = (
                    {f"{k[0]}|{k[1]}": v for k, v in n_aligned_dynamics.items()}
                    if n_aligned_dynamics
                    else {}
                )
                dynamic_entropies_json = (
                    {f"{k[0]}|{k[1]}": v for k, v in dynamic_entropies.items()}
                    if dynamic_entropies
                    else {}
                )
                dynamic_expectation_of_distances_json = (
                    {
                        f"{k[0]}|{k[1]}": v
                        for k, v in dynamic_expectation_of_distances.items()
                    }
                    if dynamic_expectation_of_distances
                    else {}
                )
                dynamic_distance_of_expectations_json = (
                    {
                        f"{k[0]}|{k[1]}": v
                        for k, v in dynamic_distance_of_expectations.items()
                    }
                    if dynamic_distance_of_expectations
                    else {}
                )
                total_losses_per_pair_json = (
                    {f"{k[0]}|{k[1]}": v for k, v in total_losses_per_pair.items()}
                    if total_losses_per_pair
                    else {}
                )
                n_aligned_totals_per_pair_json = (
                    {f"{k[0]}|{k[1]}": v for k, v in n_aligned_totals_per_pair.items()}
                    if n_aligned_totals_per_pair
                    else {}
                )
                total_entropies_per_pair_json = (
                    {f"{k[0]}|{k[1]}": v for k, v in total_entropies_per_pair.items()}
                    if total_entropies_per_pair
                    else {}
                )
                total_expectation_of_distances_per_pair_json = (
                    {
                        f"{k[0]}|{k[1]}": v
                        for k, v in total_expectation_of_distances_per_pair.items()
                    }
                    if total_expectation_of_distances_per_pair
                    else {}
                )
                total_distance_of_expectations_per_pair_json = (
                    {
                        f"{k[0]}|{k[1]}": v
                        for k, v in total_distance_of_expectations_per_pair.items()
                    }
                    if total_distance_of_expectations_per_pair
                    else {}
                )

                seed_results.append(
                    {
                        "seed": seed_id,
                        "distogram_path": str(distogram_path),
                        "total_loss": float(total_loss)
                        if not np.isnan(total_loss)
                        else None,
                        "dynamic_losses": {
                            k: (float(v) if not np.isnan(v) else None)
                            for k, v in dynamic_losses_json.items()
                        },
                        "dynamic_entropies": {
                            k: (float(v) if not np.isnan(v) else None)
                            for k, v in dynamic_entropies_json.items()
                        },
                        "total_entropy": float(total_entropy)
                        if not np.isnan(total_entropy)
                        else None,
                        "expectation_of_distance": float(expectation_of_distance)
                        if not np.isnan(expectation_of_distance)
                        else None,
                        "distance_of_expectation": float(distance_of_expectation)
                        if not np.isnan(distance_of_expectation)
                        else None,
                        "dynamic_expectation_of_distances": {
                            k: (float(v) if not np.isnan(v) else None)
                            for k, v in dynamic_expectation_of_distances_json.items()
                        },
                        "dynamic_distance_of_expectations": {
                            k: (float(v) if not np.isnan(v) else None)
                            for k, v in dynamic_distance_of_expectations_json.items()
                        },
                        "correct_ratio": float(correct_ratio)
                        if not np.isnan(correct_ratio)
                        else None,
                        "n_aligned_total": int(n_aligned_total),
                        "n_aligned_dynamics": {
                            k: int(v) for k, v in n_aligned_dynamics_json.items()
                        },
                        "total_losses_per_pair": {
                            k: (float(v) if not np.isnan(v) else None)
                            for k, v in total_losses_per_pair_json.items()
                        },
                        "n_aligned_totals_per_pair": {
                            k: int(v) for k, v in n_aligned_totals_per_pair_json.items()
                        },
                        "total_entropies_per_pair": {
                            k: (float(v) if not np.isnan(v) else None)
                            for k, v in total_entropies_per_pair_json.items()
                        },
                        "total_expectation_of_distances_per_pair": {
                            k: (float(v) if not np.isnan(v) else None)
                            for k, v in total_expectation_of_distances_per_pair_json.items()
                        },
                        "total_distance_of_expectations_per_pair": {
                            k: (float(v) if not np.isnan(v) else None)
                            for k, v in total_distance_of_expectations_per_pair_json.items()
                        },
                    }
                )

                # Print summary (show first pair's dynamic loss as example)
                first_pair_key = (
                    list(dynamic_losses.keys())[0] if dynamic_losses else None
                )
                dynamic_loss_example = (
                    dynamic_losses.get(first_pair_key, float("nan"))
                    if first_pair_key
                    else float("nan")
                )
                dynamic_entropy_example = (
                    dynamic_entropies.get(first_pair_key, float("nan"))
                    if first_pair_key
                    else float("nan")
                )
                print(
                    f"    Completed seed: {seed_id}, total_loss: {total_loss:.4f}, "
                    f"dynamic_loss[{first_pair_key}]: {dynamic_loss_example:.4f}, "
                    f"dynamic_entropy[{first_pair_key}]: {dynamic_entropy_example:.4f}, "
                    f"total_entropy: {total_entropy:.4f}, correct_ratio: {correct_ratio:.4f}"
                )

            except Exception as e:
                errors.append(
                    {
                        "task": prediction_yaml_tag,
                        "reference": reference_yaml_tag,
                        "seed": seed_id,
                        "error": f"error processing: {str(e)}",
                    }
                )
                print(f"    Error processing seed {seed_id}: {str(e)}")
                continue

        print(
            f"    Completed {len(seed_results)} seeds for reference {reference_yaml_tag}"
        )
        if not seed_results:
            continue

        # Compute mean across seeds for this reference
        valid_total_losses = [
            r["total_loss"] for r in seed_results if r["total_loss"] is not None
        ]
        valid_ratios = [
            r["correct_ratio"] for r in seed_results if r["correct_ratio"] is not None
        ]
        valid_expectation_of_distance = [
            r["expectation_of_distance"]
            for r in seed_results
            if r["expectation_of_distance"] is not None
        ]
        valid_distance_of_expectation = [
            r["distance_of_expectation"]
            for r in seed_results
            if r["distance_of_expectation"] is not None
        ]

        # Compute mean for per-pair dynamic metrics
        all_pair_keys = set()
        for r in seed_results:
            if r.get("dynamic_losses"):
                all_pair_keys.update(r["dynamic_losses"].keys())

        mean_dynamic_losses = {}
        std_dynamic_losses = {}
        mean_dynamic_entropies = {}
        std_dynamic_entropies = {}
        mean_dynamic_expectation_of_distances = {}
        std_dynamic_expectation_of_distances = {}
        mean_dynamic_distance_of_expectations = {}
        std_dynamic_distance_of_expectations = {}
        mean_n_aligned_dynamics = {}

        for pair_key in all_pair_keys:
            # Dynamic losses
            valid_losses = [
                r["dynamic_losses"].get(pair_key)
                for r in seed_results
                if r.get("dynamic_losses")
                and r["dynamic_losses"].get(pair_key) is not None
            ]
            mean_dynamic_losses[pair_key] = (
                float(np.mean(valid_losses)) if valid_losses else None
            )
            std_dynamic_losses[pair_key] = (
                float(np.std(valid_losses)) if valid_losses else None
            )

            # Dynamic entropies
            valid_entropies = [
                r["dynamic_entropies"].get(pair_key)
                for r in seed_results
                if r.get("dynamic_entropies")
                and r["dynamic_entropies"].get(pair_key) is not None
            ]
            mean_dynamic_entropies[pair_key] = (
                float(np.mean(valid_entropies)) if valid_entropies else None
            )
            std_dynamic_entropies[pair_key] = (
                float(np.std(valid_entropies)) if valid_entropies else None
            )

            # Dynamic expectation of distances
            valid_exp_dist = [
                r["dynamic_expectation_of_distances"].get(pair_key)
                for r in seed_results
                if r.get("dynamic_expectation_of_distances")
                and r["dynamic_expectation_of_distances"].get(pair_key) is not None
            ]
            mean_dynamic_expectation_of_distances[pair_key] = (
                float(np.mean(valid_exp_dist)) if valid_exp_dist else None
            )
            std_dynamic_expectation_of_distances[pair_key] = (
                float(np.std(valid_exp_dist)) if valid_exp_dist else None
            )

            # Dynamic distance of expectations
            valid_dist_exp = [
                r["dynamic_distance_of_expectations"].get(pair_key)
                for r in seed_results
                if r.get("dynamic_distance_of_expectations")
                and r["dynamic_distance_of_expectations"].get(pair_key) is not None
            ]
            mean_dynamic_distance_of_expectations[pair_key] = (
                float(np.mean(valid_dist_exp)) if valid_dist_exp else None
            )
            std_dynamic_distance_of_expectations[pair_key] = (
                float(np.std(valid_dist_exp)) if valid_dist_exp else None
            )

            # n_aligned_dynamics
            valid_n_aligned = [
                r["n_aligned_dynamics"].get(pair_key)
                for r in seed_results
                if r.get("n_aligned_dynamics")
                and r["n_aligned_dynamics"].get(pair_key) is not None
            ]
            mean_n_aligned_dynamics[pair_key] = (
                float(np.mean(valid_n_aligned)) if valid_n_aligned else None
            )

        # Compute mean for per-pair total metrics
        mean_total_losses_per_pair = {}
        std_total_losses_per_pair = {}
        mean_total_entropies_per_pair = {}
        std_total_entropies_per_pair = {}
        mean_total_expectation_of_distances_per_pair = {}
        std_total_expectation_of_distances_per_pair = {}
        mean_total_distance_of_expectations_per_pair = {}
        std_total_distance_of_expectations_per_pair = {}
        mean_n_aligned_totals_per_pair = {}

        for pair_key in all_pair_keys:
            # Total losses per pair
            valid_total_losses_pair = [
                r["total_losses_per_pair"].get(pair_key)
                for r in seed_results
                if r.get("total_losses_per_pair")
                and r["total_losses_per_pair"].get(pair_key) is not None
            ]
            mean_total_losses_per_pair[pair_key] = (
                float(np.mean(valid_total_losses_pair))
                if valid_total_losses_pair
                else None
            )
            std_total_losses_per_pair[pair_key] = (
                float(np.std(valid_total_losses_pair))
                if valid_total_losses_pair
                else None
            )

            # Total entropies per pair
            valid_total_entropies_pair = [
                r["total_entropies_per_pair"].get(pair_key)
                for r in seed_results
                if r.get("total_entropies_per_pair")
                and r["total_entropies_per_pair"].get(pair_key) is not None
            ]
            mean_total_entropies_per_pair[pair_key] = (
                float(np.mean(valid_total_entropies_pair))
                if valid_total_entropies_pair
                else None
            )
            std_total_entropies_per_pair[pair_key] = (
                float(np.std(valid_total_entropies_pair))
                if valid_total_entropies_pair
                else None
            )

            # Total expectation of distances per pair
            valid_total_exp_dist_pair = [
                r["total_expectation_of_distances_per_pair"].get(pair_key)
                for r in seed_results
                if r.get("total_expectation_of_distances_per_pair")
                and r["total_expectation_of_distances_per_pair"].get(pair_key)
                is not None
            ]
            mean_total_expectation_of_distances_per_pair[pair_key] = (
                float(np.mean(valid_total_exp_dist_pair))
                if valid_total_exp_dist_pair
                else None
            )
            std_total_expectation_of_distances_per_pair[pair_key] = (
                float(np.std(valid_total_exp_dist_pair))
                if valid_total_exp_dist_pair
                else None
            )

            # Total distance of expectations per pair
            valid_total_dist_exp_pair = [
                r["total_distance_of_expectations_per_pair"].get(pair_key)
                for r in seed_results
                if r.get("total_distance_of_expectations_per_pair")
                and r["total_distance_of_expectations_per_pair"].get(pair_key)
                is not None
            ]
            mean_total_distance_of_expectations_per_pair[pair_key] = (
                float(np.mean(valid_total_dist_exp_pair))
                if valid_total_dist_exp_pair
                else None
            )
            std_total_distance_of_expectations_per_pair[pair_key] = (
                float(np.std(valid_total_dist_exp_pair))
                if valid_total_dist_exp_pair
                else None
            )

            # n_aligned_totals_per_pair
            valid_n_aligned_total_pair = [
                r["n_aligned_totals_per_pair"].get(pair_key)
                for r in seed_results
                if r.get("n_aligned_totals_per_pair")
                and r["n_aligned_totals_per_pair"].get(pair_key) is not None
            ]
            mean_n_aligned_totals_per_pair[pair_key] = (
                float(np.mean(valid_n_aligned_total_pair))
                if valid_n_aligned_total_pair
                else None
            )

        ref_result = {
            "reference_yaml_tag": reference_yaml_tag,
            "target_conformation": task.get("target_conformation", ""),
            "target_state": task.get("target_state", ""),
            "reference_conformation": ref_info.get("reference_conformation", ""),
            "reference_state": ref_info.get("reference_state", ""),
            "ref_cif": ref_cif,
            "ref_chain": ref_chain,
            "n_seeds": len(seed_results),
            "mean_total_loss": float(np.mean(valid_total_losses))
            if valid_total_losses
            else None,
            "std_total_loss": float(np.std(valid_total_losses))
            if valid_total_losses
            else None,
            "mean_dynamic_losses": mean_dynamic_losses,
            "std_dynamic_losses": std_dynamic_losses,
            "mean_dynamic_entropies": mean_dynamic_entropies,
            "std_dynamic_entropies": std_dynamic_entropies,
            "mean_expectation_of_distance": float(
                np.mean(valid_expectation_of_distance)
            )
            if valid_expectation_of_distance
            else None,
            "std_expectation_of_distance": float(np.std(valid_expectation_of_distance))
            if valid_expectation_of_distance
            else None,
            "mean_distance_of_expectation": float(
                np.mean(valid_distance_of_expectation)
            )
            if valid_distance_of_expectation
            else None,
            "std_distance_of_expectation": float(np.std(valid_distance_of_expectation))
            if valid_distance_of_expectation
            else None,
            "mean_dynamic_expectation_of_distances": mean_dynamic_expectation_of_distances,
            "std_dynamic_expectation_of_distances": std_dynamic_expectation_of_distances,
            "mean_dynamic_distance_of_expectations": mean_dynamic_distance_of_expectations,
            "std_dynamic_distance_of_expectations": std_dynamic_distance_of_expectations,
            "mean_correct_ratio": float(np.mean(valid_ratios))
            if valid_ratios
            else None,
            "std_correct_ratio": float(np.std(valid_ratios)) if valid_ratios else None,
            "mean_n_aligned_total": float(
                np.mean([r["n_aligned_total"] for r in seed_results])
            ),
            "mean_n_aligned_dynamics": mean_n_aligned_dynamics,
            "mean_total_losses_per_pair": mean_total_losses_per_pair,
            "std_total_losses_per_pair": std_total_losses_per_pair,
            "mean_total_entropies_per_pair": mean_total_entropies_per_pair,
            "std_total_entropies_per_pair": std_total_entropies_per_pair,
            "mean_total_expectation_of_distances_per_pair": mean_total_expectation_of_distances_per_pair,
            "std_total_expectation_of_distances_per_pair": std_total_expectation_of_distances_per_pair,
            "mean_total_distance_of_expectations_per_pair": mean_total_distance_of_expectations_per_pair,
            "std_total_distance_of_expectations_per_pair": std_total_distance_of_expectations_per_pair,
            "mean_n_aligned_totals_per_pair": mean_n_aligned_totals_per_pair,
            "seed_results": seed_results,
        }
        reference_results.append(ref_result)

    if not reference_results:
        return None

    # Create result for this prediction with all references
    result = {
        "cluster_id": cluster_id,
        "prediction_yaml_tag": prediction_yaml_tag,
        "method": task.get("method", ""),
        "method_type": task.get("method_type", ""),
        "mobile_chain": mobile_chain,
        "references": reference_results,
    }

    # Add holo_conformation if present
    if "holo_conformation" in task:
        result["holo_conformation"] = task["holo_conformation"]

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Calculate distogram loss for prediction tasks"
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
        "--ref-distogram-dir",
        type=str,
        default=None,
        help="reference_distogram_diff root (default: eval.external.ref_distogram_dir or eval.dirs.ref_distogram)",
    )
    parser.add_argument(
        "--valid-pairs",
        type=str,
        default=None,
        help="valid_pairs.json (default: eval.files.valid_pairs)",
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
        help="Output JSON file (default: distogram_loss_results_{start}_{end}.json)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Recalculate even if distogram_loss.json already exists",
    )

    args = parser.parse_args()

    tasks_json = Path(args.tasks)
    rep_seq_json = E.distogram_rep_seq_json(args.rep_seq)
    msa_dir = E.distogram_msa_dir(args.msa_dir)
    ref_distogram_dir = E.distogram_ref_distogram_dir(args.ref_distogram_dir)
    valid_pairs_path = E.distogram_valid_pairs_path(args.valid_pairs)

    if not tasks_json.exists():
        raise FileNotFoundError(f"Tasks JSON not found: {tasks_json}")
    if not valid_pairs_path.exists():
        raise FileNotFoundError(f"valid_pairs not found: {valid_pairs_path}")

    # Load tasks
    with open(tasks_json, "r") as f:
        tasks = json.load(f)

    # Load representative sequences
    with open(rep_seq_json, "r") as f:
        rep_seqs = json.load(f)

    with open(valid_pairs_path, "r") as f:
        valid_pairs_raw = json.load(f)
    valid_pair_edges = set(flatten_valid_pair_edges(valid_pairs_raw))

    print(f"Loaded {len(tasks)} tasks")
    print(f"Loaded {len(rep_seqs)} representative sequences")
    print(
        f"Loaded {len(valid_pair_edges)} directed valid-pair edges from {valid_pairs_path}"
    )

    # Check system resources
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(str(tasks_json.parent))
    print(
        f"Available memory: {memory.available / (1024**3):.1f} GB ({memory.percent}% used)"
    )
    print(f"Available disk space: {disk.free / (1024**3):.1f} GB")

    # Apply start/end indices
    if args.end is None:
        args.end = len(tasks)

    tasks_to_process = tasks[args.start : args.end]
    print(f"Processing tasks [{args.start}:{args.end}] ({len(tasks_to_process)} tasks)")

    results = []
    errors = []
    skipped = 0

    for i, task in enumerate(tasks_to_process):
        if (i + 1) % 1 == 0 or i == 0:
            method_name = (
                task.get("prediction_method") or task.get("method") or "unknown"
            )
            method_type = task.get("method_type", "")
            n_refs = len(task.get("references", []))
            print(
                f"  [{args.start + i + 1}/{args.end}] Processing {task.get('cluster_id')}/{task.get('prediction_yaml_tag')} ({n_refs} references, method={method_name}, type={method_type})..."
            )

        prediction_distograms = task.get("prediction_distograms", [])
        distogram_dir = (
            Path(prediction_distograms[0]).parent if prediction_distograms else None
        )
        output_file = (
            distogram_dir / "distogram_loss_final.json" if distogram_dir else None
        )

        # Smart skip: verify per-reference tags and per-dynamic-pair keys
        skip_task = False
        missing_reference_tags = set()
        missing_dynamic_pairs = set()
        existing_results = []
        if not args.no_skip and output_file and output_file.exists():
            try:
                with open(output_file, "r") as f:
                    existing_json = json.load(f)
                # Collect existing reference tags and dynamic-pair keys
                existing_ref_tags = set()
                existing_dynamic_pairs = set()
                if existing_json and isinstance(existing_json, list):
                    for entry in existing_json:
                        for ref in entry.get("references", []):
                            tag = ref.get("reference_yaml_tag")
                            if tag:
                                existing_ref_tags.add(tag)
                            # Collect existing dynamic pairs from mean_dynamic_losses
                            mean_dynamic_losses = ref.get("mean_dynamic_losses", {})
                            for pair_key in mean_dynamic_losses.keys():
                                existing_dynamic_pairs.add(pair_key)

                # Reference tags required for this task
                needed_ref_tags = set(
                    ref.get("reference_yaml_tag")
                    for ref in task.get("references", [])
                    if ref.get("reference_yaml_tag")
                )
                missing_reference_tags = needed_ref_tags - existing_ref_tags

                # Dynamic pairs required for this task (from reference_distogram_diff.json)
                ref_distogram_diff_path = reference_distogram_diff_path(
                    ref_distogram_dir, task
                )
                needed_dynamic_pairs = set()
                if ref_distogram_diff_path.exists():
                    try:
                        with open(ref_distogram_diff_path, "r") as f:
                            diff_data = json.load(f)
                        if (
                            isinstance(diff_data, dict)
                            and "pairwise_comparisons" in diff_data
                        ):
                            for comparison in diff_data["pairwise_comparisons"]:
                                ref_A = comparison.get("reference_A", "")
                                ref_B = comparison.get("reference_B", "")
                                if ref_A and ref_B:
                                    pair_key = "|".join(sorted([ref_A, ref_B]))
                                    needed_dynamic_pairs.add(pair_key)
                    except Exception:
                        pass
                print(needed_dynamic_pairs)
                print(existing_dynamic_pairs)
                missing_dynamic_pairs = needed_dynamic_pairs - existing_dynamic_pairs

                if not missing_reference_tags and not missing_dynamic_pairs:
                    print(
                        f"    Skipping (all references and dynamic pairs already computed): {output_file}"
                    )
                    skipped += 1
                    continue
                else:
                    if missing_reference_tags:
                        print(f"    Found missing references: {missing_reference_tags}")
                    if missing_dynamic_pairs:
                        print(
                            f"    Found missing dynamic pairs: {missing_dynamic_pairs}"
                        )
                    # Preserve existing JSON entries for merge/recompute
                    existing_results = (
                        existing_json if isinstance(existing_json, list) else []
                    )
            except Exception as e:
                print(f"    Error reading existing distogram_loss_final.json: {e}")
                print(f"    Will recompute all references for this task")
                existing_results = []

        # Gaps vs on-disk JSON -> full recompute for this task (incremental merge lives in elif False, currently off).
        if missing_dynamic_pairs or missing_reference_tags:
            # Missing dynamic pairs or new reference tags require recomputing all references together
            if missing_dynamic_pairs:
                print(f"    Recomputing all references due to missing dynamic pairs")
            if missing_reference_tags:
                print(
                    f"    Recomputing all references due to missing references (new references need dynamic pairs with existing ones)"
                )
            result = process_task(
                task,
                rep_seqs,
                msa_dir,
                errors,
                ref_distogram_dir,
                valid_pair_edges,
            )
            if result:
                results.append(result)
                if distogram_dir:
                    try:
                        distogram_dir.mkdir(parents=True, exist_ok=True)
                        with open(output_file, "w") as f:
                            json.dump([result], f, indent=2)
                        print(f"    Saved to {output_file}")
                    except Exception as e:
                        errors.append(
                            {
                                "dir": str(distogram_dir),
                                "error": f"Failed to save per-directory file: {str(e)}",
                            }
                        )
                        print(f"    Error saving to {distogram_dir}: {str(e)}")
        elif False:  # Disabled incremental reference addition
            # Keep only references still marked missing
            filtered_task = dict(task)
            filtered_task["references"] = [
                ref
                for ref in task.get("references", [])
                if ref.get("reference_yaml_tag") in missing_reference_tags
            ]
            result = process_task(
                filtered_task,
                rep_seqs,
                msa_dir,
                errors,
                ref_distogram_dir,
                valid_pair_edges,
            )
            # Merge with existing on-disk results
            if result:
                # Merge old and new reference dicts
                merged_references = []
                # Existing reference dicts
                old_refs = []
                if existing_results and isinstance(existing_results, list):
                    for entry in existing_results:
                        old_refs.extend(entry.get("references", []))
                # Newly computed reference dicts
                new_refs = result.get("references", [])
                # Deduplicate by reference_yaml_tag
                seen_tags = set()
                for ref in old_refs + new_refs:
                    tag = ref.get("reference_yaml_tag")
                    if tag and tag not in seen_tags:
                        merged_references.append(ref)
                        seen_tags.add(tag)
                # Build final merged result dict
                merged_result = dict(result)
                merged_result["references"] = merged_references
                # Save
                if distogram_dir:
                    try:
                        distogram_dir.mkdir(parents=True, exist_ok=True)
                        with open(output_file, "w") as f:
                            json.dump([merged_result], f, indent=2)
                        print(
                            f"    Updated {output_file} with {len(new_refs)} new references (total {len(merged_references)})"
                        )
                    except Exception as e:
                        errors.append(
                            {
                                "dir": str(distogram_dir),
                                "error": f"Failed to save per-directory file: {str(e)}",
                            }
                        )
                        print(f"    Error saving to {distogram_dir}: {str(e)}")
                results.append(merged_result)
        else:
            # Default path: process all references for the task
            result = process_task(
                task,
                rep_seqs,
                msa_dir,
                errors,
                ref_distogram_dir,
                valid_pair_edges,
            )
            if result:
                results.append(result)
                if distogram_dir:
                    try:
                        distogram_dir.mkdir(parents=True, exist_ok=True)
                        with open(output_file, "w") as f:
                            json.dump([result], f, indent=2)
                        print(f"    Saved to {output_file}")
                    except Exception as e:
                        errors.append(
                            {
                                "dir": str(distogram_dir),
                                "error": f"Failed to save per-directory file: {str(e)}",
                            }
                        )
                        print(f"    Error saving to {distogram_dir}: {str(e)}")

    # No need for batch saving at the end since we save immediately

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            tasks_json.parent / f"distogram_loss_results_{args.start}_{args.end}.json"
        )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {output_path}")

    # Save errors
    if errors:
        error_path = (
            tasks_json.parent / f"distogram_loss_errors_{args.start}_{args.end}.json"
        )
        with open(error_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Saved {len(errors)} errors to {error_path}")

    # Print summary
    print("\nSummary:")
    print(f"  Total tasks processed: {len(tasks_to_process)}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Successful: {len(results)}")
    print(f"  Errors: {len(errors)}")

    if results:
        # Collect all losses and distance errors from all references in all results
        valid_total_losses = []
        valid_dynamic_losses = []
        valid_expectation_of_distance = []
        valid_dynamic_expectation_of_distance = []
        valid_distance_of_expectation = []
        valid_dynamic_distance_of_expectation = []

        for r in results:
            for ref in r.get("references", []):
                if "mean_total_loss" in ref and not np.isnan(ref["mean_total_loss"]):
                    valid_total_losses.append(ref["mean_total_loss"])
                if "mean_dynamic_loss" in ref and not np.isnan(
                    ref["mean_dynamic_loss"]
                ):
                    valid_dynamic_losses.append(ref["mean_dynamic_loss"])
                if "mean_expectation_of_distance" in ref and not np.isnan(
                    ref["mean_expectation_of_distance"]
                ):
                    valid_expectation_of_distance.append(
                        ref["mean_expectation_of_distance"]
                    )
                if "mean_dynamic_expectation_of_distance" in ref and not np.isnan(
                    ref["mean_dynamic_expectation_of_distance"]
                ):
                    valid_dynamic_expectation_of_distance.append(
                        ref["mean_dynamic_expectation_of_distance"]
                    )
                if "mean_distance_of_expectation" in ref and not np.isnan(
                    ref["mean_distance_of_expectation"]
                ):
                    valid_distance_of_expectation.append(
                        ref["mean_distance_of_expectation"]
                    )
                if "mean_dynamic_distance_of_expectation" in ref and not np.isnan(
                    ref["mean_dynamic_distance_of_expectation"]
                ):
                    valid_dynamic_distance_of_expectation.append(
                        ref["mean_dynamic_distance_of_expectation"]
                    )

        if valid_total_losses:
            print(
                f"  Mean total loss: {np.mean(valid_total_losses):.4f} +/- {np.std(valid_total_losses):.4f}"
            )
            print(f"  Min total loss: {np.min(valid_total_losses):.4f}")
            print(f"  Max total loss: {np.max(valid_total_losses):.4f}")
            print(f"  Total reference comparisons: {len(valid_total_losses)}")

        if valid_dynamic_losses:
            print(
                f"  Mean dynamic loss: {np.mean(valid_dynamic_losses):.4f} +/- {np.std(valid_dynamic_losses):.4f}"
            )
            print(f"  Min dynamic loss: {np.min(valid_dynamic_losses):.4f}")
            print(f"  Max dynamic loss: {np.max(valid_dynamic_losses):.4f}")
            print(f"  Dynamic reference comparisons: {len(valid_dynamic_losses)}")

        if valid_expectation_of_distance:
            print(
                f"  Mean expectation of distance: {np.mean(valid_expectation_of_distance):.4f} +/- {np.std(valid_expectation_of_distance):.4f} A"
            )

        if valid_dynamic_expectation_of_distance:
            print(
                f"  Mean dynamic expectation of distance: {np.mean(valid_dynamic_expectation_of_distance):.4f} +/- {np.std(valid_dynamic_expectation_of_distance):.4f} A"
            )

        if valid_distance_of_expectation:
            print(
                f"  Mean distance of expectation: {np.mean(valid_distance_of_expectation):.4f} +/- {np.std(valid_distance_of_expectation):.4f} A"
            )

        if valid_dynamic_distance_of_expectation:
            print(
                f"  Mean dynamic distance of expectation: {np.mean(valid_dynamic_distance_of_expectation):.4f} +/- {np.std(valid_dynamic_distance_of_expectation):.4f} A"
            )


if __name__ == "__main__":
    main()

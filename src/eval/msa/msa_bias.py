#!/usr/bin/env python3
"""
MSA Bias Analysis for ProMiSE-bench.

Quantify MSA coevolution bias between conformational pairs.

For each pair of structures, computes MSA preference score:
  MSA_pref = (overlap_conf2 - overlap_conf1) / (overlap_conf2 + overlap_conf1)

  Positive (+1) → MSA coevolution favors conf2 (holo) unique contacts
  Negative (-1) → MSA coevolution favors conf1 (apo) unique contacts

Inputs:
  - Renumbered PDBs from cif_to_renumbered_pdb.py
  - ESM-MSA-1b contact predictions from esm_run.py
  - valid_pairs.json defining conformational pairs

Usage:
    python -m src.eval.msa_bias \\
        --valid-pairs data/dataset/valid_pairs.json \\
        --pdb-dir data/eval/renumbered_pdbs \\
        --esm-dir data/eval/msas \\
        -o data/eval/msa_bias_results.csv
"""

import os
import re
import csv
import glob
import json
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
from Bio.PDB import PDBParser, Selection
from scipy.spatial import distance_matrix

from utils._config import eval_cfg as E

CONTACT_CUTOFF = 8.0  # Angstroms
DIAG_EXCLUSION = 3    # Exclude contacts within ±3 residues

SET_NAMES = ("intrinsic", "ligand-induced", "protein-induced")


# ============================================================================
# Contact Map
# ============================================================================

def compute_contact_map(
    pdb_path: str,
    target_size: int,
    chain_id: str = "A",
    cutoff: float = CONTACT_CUTOFF,
) -> np.ndarray:
    """
    Compute all-atom contact map from a renumbered PDB.

    Renumbered PDBs (from cif_to_renumbered_pdb) always use chain 'A'
    with residue numbers aligned to the representative MSA position.

    Args:
        pdb_path: Path to renumbered PDB file.
        target_size: Size of output matrix (L × L), matching ESM output.
        chain_id: Chain to use (default 'A').
        cutoff: Distance cutoff in Angstroms.

    Returns:
        Binary contact map of shape (target_size, target_size).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)

    # Collect residues from target chain
    residues = []
    for chain in structure[0].get_chains():
        if chain.id != chain_id:
            continue
        for res in Selection.unfold_entities(chain, "R"):
            if "CA" in res.child_dict:
                residues.append((res.get_id()[1], res))

    if not residues:
        raise ValueError(f"No CA atoms in chain {chain_id}")

    residues.sort(key=lambda x: x[0])

    # Keep only residues within matrix bounds
    valid = [(rn, res) for rn, res in residues if 0 <= rn - 1 < target_size]
    if not valid:
        return np.zeros((target_size, target_size))

    # All-atom contact calculation
    cmap = np.zeros((target_size, target_size))

    for i, (rn_i, res_i) in enumerate(valid):
        idx_i = rn_i - 1
        for j in range(i + 1, len(valid)):
            rn_j, res_j = valid[j]
            idx_j = rn_j - 1

            try:
                coords_i = np.array([a.coord for a in res_i.get_atoms()])
                coords_j = np.array([a.coord for a in res_j.get_atoms()])
                if np.min(distance_matrix(coords_i, coords_j, p=2)) < cutoff:
                    cmap[idx_i, idx_j] = 1
                    cmap[idx_j, idx_i] = 1
            except Exception:
                pass

    # Exclude diagonal (±3)
    for i in range(target_size):
        lo = max(0, i - DIAG_EXCLUSION)
        hi = min(target_size, i + DIAG_EXCLUSION + 1)
        cmap[i, lo:hi] = 0

    # Ensure symmetry
    tu = np.triu_indices(target_size)
    cmap[tu[::-1]] = cmap[tu]

    return cmap


# ============================================================================
# MSA Preference
# ============================================================================

def compute_msa_preference(
    msa_matrix: np.ndarray,
    cmap1: np.ndarray,
    cmap2: np.ndarray,
) -> Dict[str, float]:
    """
    Compute MSA preference between two conformations.

    Classifies contacts as common (both), unique-to-conf1, unique-to-conf2,
    then measures how much the MSA coevolution signal overlaps each category.

    Returns:
        Dict with msa_pref and supporting metrics.
    """
    upper = np.triu(np.ones_like(cmap1), k=1)

    common = (cmap1 == 1) & (cmap2 == 1)
    unique1 = (cmap1 == 1) & (cmap2 == 0)
    unique2 = (cmap1 == 0) & (cmap2 == 1)

    overlap1 = float(np.sum(msa_matrix * unique1 * upper))
    overlap2 = float(np.sum(msa_matrix * unique2 * upper))

    total = overlap1 + overlap2
    msa_pref = (overlap2 - overlap1) / total if total > 0 else 0.0

    return {
        "msa_pref": round(msa_pref, 8),
        "common_count": int(np.sum(common * upper)),
        "conf1_unique_count": int(np.sum(unique1 * upper)),
        "conf2_unique_count": int(np.sum(unique2 * upper)),
    }


# ============================================================================
# File Discovery
# ============================================================================

def parse_tag(tag: str) -> Tuple[str, str, str]:
    """
    Parse conformer tag, stripping the apo/holo suffix.
    e.g., '6yeb_1_A1_m' → ('6yeb', '1', 'A1')
    """
    if tag.endswith("_m") or tag.endswith("_x"):
        tag = tag[:-2]
    parts = tag.split("_")
    return parts[0], parts[1], parts[2]


def find_pdb(
    tag: str,
    pdb_dir: str,
    set_name: str,
    cluster_id: str,
) -> Optional[str]:
    """Find renumbered PDB for a tag."""
    pdb_id, asm_num, auth_asym = parse_tag(tag)

    # Exact match
    path = os.path.join(
        pdb_dir, set_name, cluster_id,
        f"{pdb_id}_{asm_num}_{auth_asym}_renumbered.pdb",
    )
    if os.path.exists(path):
        return path

    # Glob (asm_num may differ between tag and actual file)
    matches = glob.glob(os.path.join(
        pdb_dir, set_name, cluster_id,
        f"{pdb_id}_*_{auth_asym}_renumbered.pdb",
    ))
    return matches[0] if matches else None


def find_esm_files(esm_dir: str, cluster_id: str) -> List[Tuple[int, str]]:
    """
    Find ESM contact .npy files for a cluster.
    Returns [(seed, filepath), ...] sorted by seed.
    """
    pattern = os.path.join(esm_dir, f"{cluster_id}_seed*_n*_contacts.npy")
    files = glob.glob(pattern)

    if not files:
        # Fallback without _n* part
        pattern = os.path.join(esm_dir, f"{cluster_id}_seed*_contacts.npy")
        files = glob.glob(pattern)

    results = []
    for f in files:
        m = re.search(r"seed(\d+)", os.path.basename(f))
        results.append((int(m.group(1)) if m else 0, f))
    return sorted(results)


def order_pair(tag1: str, tag2: str, set_name: str) -> Tuple[str, str]:
    """
    Order pair: conf1 = apo (_m), conf2 = holo (_x) for induced sets.
    For intrinsic, keep original order.
    """
    if set_name in ("ligand-induced", "protein-induced"):
        if tag1.endswith("_x") and tag2.endswith("_m"):
            return tag2, tag1
    return tag1, tag2


# ============================================================================
# CLI
# ============================================================================

@click.command()
@click.option('--valid-pairs', type=click.Path(exists=True),
              default=str(E.file('valid_pairs')),
              show_default=True, help='Path to valid_pairs.json')
@click.option('--pdb-dir', type=click.Path(exists=True),
              default=str(E.dir('renumbered_pdbs')),
              show_default=True, help='Renumbered PDB directory')
@click.option('--esm-dir', type=click.Path(exists=True),
              default=str(E.dir('esm_contacts')),
              show_default=True, help='ESM contact prediction directory')
@click.option('-o', '--output', type=click.Path(),
              default=str(E.file('msa_bias_csv')),
              show_default=True, help='Output CSV path')
@click.option('--set-name', type=str, default=None,
              help='Process only this set')
@click.option('--cluster', type=str, default=None,
              help='Process only this cluster')
def main(valid_pairs, pdb_dir, esm_dir, output, set_name, cluster):
    """MSA Bias Analysis for ProMiSE-bench."""
    with open(valid_pairs) as f:
        valid_pairs_data = json.load(f)

    print(f"{'#'*60}")
    print("MSA Bias Analysis")
    print(f"{'#'*60}")
    print(f"PDB dir : {pdb_dir}")
    print(f"ESM dir : {esm_dir}")
    print(f"Output  : {output}")

    rows = []

    for sn in SET_NAMES:
        if set_name and set_name != sn:
            continue

        set_data = valid_pairs_data.get(sn, {})
        if not set_data:
            continue

        print(f"\n{'='*60}")
        print(f"{sn}  ({len(set_data)} clusters)")
        print(f"{'='*60}")

        for cluster_id, pairs in set_data.items():
            if cluster and cluster != cluster_id:
                continue

            # Normalize: single pair may not be double-nested
            if isinstance(pairs[0], str):
                pairs = [pairs]

            esm_files = find_esm_files(esm_dir, cluster_id)
            if not esm_files:
                print(f"  {cluster_id}: no ESM files, skipping")
                continue

            for pair in pairs:
                conf1, conf2 = order_pair(pair[0], pair[1], sn)

                pdb1 = find_pdb(conf1, pdb_dir, sn, cluster_id)
                pdb2 = find_pdb(conf2, pdb_dir, sn, cluster_id)

                if not pdb1 or not pdb2:
                    missing = []
                    if not pdb1:
                        missing.append(conf1)
                    if not pdb2:
                        missing.append(conf2)
                    print(f"  {cluster_id}: PDB missing for {', '.join(missing)}")
                    continue

                for seed, esm_path in esm_files:
                    msa_matrix = np.load(esm_path)
                    msa_size = msa_matrix.shape[0]

                    try:
                        cmap1 = compute_contact_map(pdb1, msa_size)
                        cmap2 = compute_contact_map(pdb2, msa_size)
                    except Exception as e:
                        print(f"  {cluster_id}: contact map error – {e}")
                        continue

                    result = compute_msa_preference(msa_matrix, cmap1, cmap2)
                    rows.append({
                        "conf1_name": conf1,
                        "conf2_name": conf2,
                        "msa_pref": result["msa_pref"],
                        "common_count": result["common_count"],
                        "conf1_unique_count": result["conf1_unique_count"],
                        "conf2_unique_count": result["conf2_unique_count"],
                        "seed": seed,
                        "cluster_id": cluster_id,
                        "set_name": sn,
                    })

                print(f"  {cluster_id}: {conf1} vs {conf2}  "
                      f"({len(esm_files)} seed{'s' if len(esm_files) != 1 else ''})")

    # ---- Save CSV ----
    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fieldnames = [
            "conf1_name", "conf2_name", "msa_pref",
            "common_count", "conf1_unique_count", "conf2_unique_count",
            "seed", "cluster_id", "set_name",
        ]
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} rows → {output}")
    else:
        print("\nNo results produced.")

    # ---- Summary ----
    print(f"\n{'#'*60}")
    for sn in SET_NAMES:
        s_rows = [r for r in rows if r["set_name"] == sn]
        if not s_rows:
            continue
        prefs = [r["msa_pref"] for r in s_rows]
        clusters = len(set(r["cluster_id"] for r in s_rows))
        print(f"  {sn:20s}  {clusters:3d} clusters  "
              f"mean_pref={np.mean(prefs):+.3f} ± {np.std(prefs):.3f}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()

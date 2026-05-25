#!/usr/bin/env python3
"""
Collect distogram files via symlinks using patterns from distogram_analysis_data.

This script reads distogram_pattern entries from the distogram analysis JSON
and creates symlinks to organize distograms by method, set, cluster, and yaml tag.

Output structure:
- distogram/boltz/{set}/{cluster}/{yaml}/seed_{seed}_{yaml}_distogram.npz
- distogram/af3/{set}/{cluster}/{yaml}/seed_{seed}_{yaml}_distogram.npz

Usage (``PYTHONPATH=src``)::

    python -m eval.distogram.collect_distograms --json .../seq_cluster_to_answer_map_with_cb.json --method boltz --output-dir data_eval/distogram
    python -m eval.distogram.collect_distograms --json ... --method af3 --output-dir data_eval/distogram
    python -m eval.distogram.collect_distograms --json ... --all --output-dir data_eval/distogram
"""

import argparse
import functools
from pathlib import Path
import re
import json
import gzip
import gemmi
import numpy as np
import glob
from typing import Dict, Iterable, Optional, Tuple

from utils._config import eval_cfg as E
from utils._config import pipeline_cfg as P


def _structure_npz_candidates_from_config(
    method: str, method_type: str, cluster_id: str, yaml_tag: str
) -> list[Path]:
    """Expand ``pipeline.distogram_enrich.structure_npz.<method>`` templates."""
    templates = P.distogram_enrich_structure_templates(method)
    return _expand_companion_templates(
        templates, method_type=method_type, cluster_id=cluster_id, yaml_tag=yaml_tag
    )


def _confidences_json_candidates_from_config(
    method: str, method_type: str, cluster_id: str, yaml_tag: str
) -> list[Path]:
    """Expand ``pipeline.distogram_enrich.confidences_json.<method>`` templates."""
    templates = P.distogram_enrich_confidences_templates(method)
    return _expand_companion_templates(
        templates, method_type=method_type, cluster_id=cluster_id, yaml_tag=yaml_tag
    )


def _expand_companion_templates(
    templates: list[str], *, method_type: str, cluster_id: str, yaml_tag: str
) -> list[Path]:
    if not templates:
        return []
    seen: set[str] = set()
    hits: list[Path] = []
    for tmpl in templates:
        expanded = tmpl.format(
            method_type=method_type, cluster_id=cluster_id, yaml_tag=yaml_tag
        )
        for m in glob.glob(expanded):
            if m not in seen:
                seen.add(m)
                hits.append(Path(m))
    return hits


def load_patterns_from_distogram_json(json_path: Path) -> list[str]:
    """Load all `distogram_pattern` entries from distogram JSON (recursive)."""
    try:
        j = json.loads(json_path.read_text())
    except Exception as e:
        print(f"Failed to load JSON '{json_path}': {e}")
        return []
    patterns = set()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "distogram_pattern" and isinstance(v, str):
                    patterns.add(v)
                else:
                    walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(j)
    return sorted(patterns)


# Method-name tokens accepted in distogram pattern strings (dash/underscore variants).
_METHOD_TOKENS: dict[str, tuple[str, ...]] = {
    "boltz1": ("boltz1", "boltz-1", "boltz_1"),
    "boltz2": ("boltz2", "boltz-2", "boltz_2"),
    "af3": ("af3",),
    "bioemu": ("bioemu",),
}


def _pattern_matches_method(pat: str, method: str) -> bool:
    """True iff ``pat`` unambiguously identifies ``method`` (boltz1/boltz2 must not collide)."""
    p = pat.lower()
    aliases = _METHOD_TOKENS.get(method, (method.lower(),))
    if not any(tok in p for tok in aliases):
        return False
    if method == "boltz1" and any(tok in p for tok in _METHOD_TOKENS["boltz2"]):
        return False
    if method == "boltz2" and any(tok in p for tok in _METHOD_TOKENS["boltz1"]):
        return False
    return True


def find_distogram_files_from_patterns(
    patterns: list[str],
    method_filter: Optional[str] = None,
    suffixes: tuple[str, ...] = (".npz",),
) -> list[Path]:
    """Expand each glob in ``patterns`` and keep files whose suffix is in ``suffixes``."""
    files = []
    for pat in patterns:
        if method_filter and not _pattern_matches_method(pat, method_filter):
            continue
        for m in glob.glob(pat):
            p = Path(m)
            if p.is_file() and p.suffix in suffixes:
                files.append(p)
    return files


def find_yaml_stem_in_parts(parts: Iterable[str]) -> Optional[str]:
    """Find yaml tag (e.g., '3d8t_1_A1_m') in path parts.

    Looks for a segment that ends with _m or _x, which is how yaml tags are named.
    Falls back to None if not found.
    """
    import re

    yaml_re = re.compile(r".+_[mx]$", re.IGNORECASE)
    for p in parts:
        if yaml_re.match(p):
            return p
    return None


def _get_rel_path_for_method(npz: Path, method_name: str, base_dir: Path) -> Optional[Path]:
    """Return a relative path under the method root (e.g., 'boltz/...') for parsing.

    Tries, in order:
      1) npz.relative_to(base_dir / method_name)
      2) npz.relative_to(resolved symlink target of base_dir/method_name)
      3) locate the first occurrence of a part that contains method_name (substring) or a known variant (eg 'boltz_final') and return the remainder
      4) return None if not found
    """
    method_root = base_dir / method_name

    # 1) direct relative
    try:
        return npz.relative_to(method_root)
    except Exception:
        pass

    # 2) try resolved symlink target
    try:
        resolved = method_root.resolve()
        if resolved != method_root:
            try:
                return npz.relative_to(resolved)
            except Exception:
                pass
    except Exception:
        pass

    # 3) search for a part that contains method_name or typical variants
    parts = list(npz.parts)
    for i, part in enumerate(parts):
        lname = part.lower()
        if method_name.lower() in lname or method_name.lower() + "_final" in lname:
            # return path after the method-like segment
            return Path(*parts[i + 1 :]) if i + 1 < len(parts) else Path()

    return None


def extract_chain_seq_mapping_from_boltz_npz(
    npz_path: Path, distogram_path: Optional[Path] = None
) -> dict:
    """
    Extract chain to distogram index mapping from Boltz processed npz file.

    For polymer chains: residue-based indexing (res_idx)
    For ligand chains: atom-based indexing (each atom is one distogram entry)

    Args:
        npz_path: Path to Boltz processed npz (e.g., 1kdo_2_B1_x.npz)
        distogram_path: Optional path to distogram npz to verify size

    Returns:
        {
            "chains": {
                "B1": {
                    "start": 0, "end": 226, "length": 227,
                    "mol_type": "polymer",
                    "residues": [{"idx": 0, "name": "MET"}, ...]  # for polymer
                },
                "A": {
                    "start": 227, "end": 247, "length": 21,
                    "mol_type": "ligand",
                    "comp_id": "C5P",
                    "atoms": [{"idx": 227, "name": "O3P"}, ...]  # for ligand
                }
            },
            "total_length": 248,
            "distogram_size": 248,  # if distogram_path provided
            "size_match": True      # if distogram_path provided
        }
    """
    assert npz_path.exists(), f"Boltz npz not found: {npz_path}"

    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"  Failed to load npz: {e}")
        return {}

    chains_arr = data["chains"]
    residues_arr = data["residues"]
    atoms_arr = data["atoms"]

    # mol_type: 0=protein, 1=rna, 2=dna, 3=ligand (small molecule)
    MOL_TYPE_NAMES = {0: "polymer", 1: "rna", 2: "dna", 3: "ligand"}

    chains = {}
    current_distogram_idx = 0

    for chain_row in chains_arr:
        chain_name = str(chain_row["name"])
        mol_type = int(chain_row["mol_type"])
        mol_type_name = MOL_TYPE_NAMES.get(mol_type, "unknown")

        res_idx_start = int(chain_row["res_idx"])  # start index in residues array
        res_num = int(chain_row["res_num"])  # number of residues
        atom_idx_start = int(chain_row["atom_idx"])
        atom_num = int(chain_row["atom_num"])

        if mol_type == 3:  # ligand - atom-based
            # Each atom is one distogram entry
            chain_start = current_distogram_idx
            chain_length = atom_num
            chain_end = chain_start + chain_length - 1

            # Get ligand residue info (there's usually 1 residue for the whole ligand)
            ligand_res = residues_arr[res_idx_start]
            comp_id = str(ligand_res["name"])

            # Get atom names for this chain
            atom_entries = []
            for i in range(atom_num):
                atom_row = atoms_arr[atom_idx_start + i]
                atom_name = str(atom_row["name"])
                atom_entries.append({"idx": chain_start + i, "name": atom_name})

            chains[chain_name] = {
                "start": chain_start,
                "end": chain_end,
                "length": chain_length,
                "mol_type": mol_type_name,
                "comp_id": comp_id,
                "atoms": atom_entries,
            }

            current_distogram_idx += chain_length
        else:  # polymer (protein, rna, dna) - residue-based
            chain_start = current_distogram_idx
            chain_length = res_num
            chain_end = chain_start + chain_length - 1

            # Get residue names
            residue_entries = []
            for i in range(res_num):
                res_row = residues_arr[res_idx_start + i]
                res_name = str(res_row["name"])
                residue_entries.append({"idx": chain_start + i, "name": res_name})

            chains[chain_name] = {
                "start": chain_start,
                "end": chain_end,
                "length": chain_length,
                "mol_type": mol_type_name,
                "residues": residue_entries,
            }

            current_distogram_idx += chain_length

    result = {
        "chains": chains,
        "total_length": current_distogram_idx,
    }

    # Verify against distogram size if provided
    if distogram_path and distogram_path.exists():
        try:
            with np.load(distogram_path) as disto_data:
                distogram_size = disto_data["distogram"].shape[1]
                result["distogram_size"] = distogram_size
                result["size_match"] = distogram_size == current_distogram_idx
                if not result["size_match"]:
                    print(
                        f"  Warning: Size mismatch! npz={current_distogram_idx}, distogram={distogram_size}"
                    )
        except Exception as e:
            print(f"  Warning: Failed to read distogram: {e}")

    return result


# ``intrinsic`` (pipeline) <-> ``apo-monomers`` (legacy JSON) alias table.
_AF3_METHOD_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "intrinsic": ("intrinsic", "apo-monomers", "ligand-induced", "protein-induced"),
    "ligand-induced": ("ligand-induced", "intrinsic", "apo-monomers", "protein-induced"),
    "protein-induced": ("protein-induced", "intrinsic", "apo-monomers", "ligand-induced"),
}


@functools.lru_cache(maxsize=8)
def _load_af3_chain_mapping_json(path_str: str) -> dict:
    """Load + cache the consolidated AF3 chain-mapping JSON.

    Flat JSON keyed by ``"{method_type}/{cluster_id}/{yaml_tag}"``; each
    entry has a ``"mapping"`` field of ``{cif_chain_id: target_chain_id}``.
    """
    p = Path(path_str)
    if not p.exists():
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Failed to load AF3 chain mapping JSON {p}: {e}")
        return {}


def _candidate_af3_keys(method_type: str, cluster_id: str, yaml_tag: str) -> list[str]:
    """JSON keys to probe; handles method-type aliasing and ``_m`` <-> ``_x`` toggle."""
    method_aliases = _AF3_METHOD_TYPE_ALIASES.get(method_type, (method_type,))
    yaml_variants = [yaml_tag]
    if yaml_tag.endswith("_x"):
        yaml_variants.append(yaml_tag[:-2] + "_m")
    elif yaml_tag.endswith("_m"):
        yaml_variants.append(yaml_tag[:-2] + "_x")
    keys: list[str] = []
    seen: set[str] = set()
    for mt in method_aliases:
        for yt in yaml_variants:
            k = f"{mt}/{cluster_id}/{yt}"
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def lookup_af3_cif_to_target(
    mapping_root: Optional[Path],
    cluster_id: str,
    yaml_tag: str,
    method_type: str,
) -> Optional[dict]:
    """Return ``{cif_chain_id: target_chain_id}`` for one task.

    ``None`` if the JSON or the key is missing; ``{}`` if the entry exists
    but has no ``mapping`` field.
    """
    if mapping_root is None:
        return None
    data = _load_af3_chain_mapping_json(str(mapping_root))
    if not data:
        return None
    for key in _candidate_af3_keys(method_type, cluster_id, yaml_tag):
        entry = data.get(key)
        if isinstance(entry, dict):
            mapping = entry.get("mapping")
            if isinstance(mapping, dict) and mapping:
                return {str(k): str(v) for k, v in mapping.items()}
            return {}
    return None


def extract_af3_chain_mapping(
    confidences_path: Path,
    cif_to_target: dict,
    distogram_path: Optional[Path] = None,
) -> dict:
    """Extract chain -> distogram-index mapping from AF3 confidences.json.

    Args:
        confidences_path: Path to confidences.json.
        cif_to_target: Resolved ``{cif_chain_id: target_chain_id}`` mapping.
        distogram_path: Optional distogram npz for size verification.

    A chain whose ``res_ids`` are all identical is treated as a ligand
    (atom-level tokens); otherwise it's a polymer (residue-level tokens).

    Returns Boltz-compatible structure: ``"chains"`` keyed by cif_chain_id
    plus ``"cif_to_target_chain"`` carrying the supplied mapping.
    """
    if not confidences_path.exists():
        return {}

    if cif_to_target is None:
        return {}

    try:
        with open(confidences_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Failed to load confidences.json: {e}")
        return {}

    token_chain_ids = data.get("token_chain_ids", [])
    token_res_ids = data.get("token_res_ids", [])

    if not token_chain_ids or len(token_chain_ids) != len(token_res_ids):
        return {}

    # Group tokens by chain (using cif_chain_id from confidences.json)
    cif_chains = {}  # cif_chain_id -> chain info
    current_chain = None
    chain_start = 0
    chain_res_ids = []

    for i, (chain_id, res_id) in enumerate(zip(token_chain_ids, token_res_ids)):
        if chain_id != current_chain:
            # Save previous chain
            if current_chain is not None:
                unique_res_ids = set(chain_res_ids)
                # If all tokens have the same res_id, it's a ligand (atom-level)
                if len(unique_res_ids) == 1:
                    mol_type = "ligand"
                    cif_chains[current_chain] = {
                        "start": chain_start,
                        "end": i - 1,
                        "length": i - chain_start,
                        "mol_type": mol_type,
                        "atoms": [],  # Empty - we don't know atom names from AF3
                    }
                else:
                    mol_type = "polymer"
                    cif_chains[current_chain] = {
                        "start": chain_start,
                        "end": i - 1,
                        "length": i - chain_start,
                        "mol_type": mol_type,
                        "residues": [],  # Empty - we don't know residue names from AF3
                    }
            # Start new chain
            current_chain = chain_id
            chain_start = i
            chain_res_ids = [res_id]
        else:
            chain_res_ids.append(res_id)

    # Save last chain
    if current_chain is not None:
        unique_res_ids = set(chain_res_ids)
        if len(unique_res_ids) == 1:
            mol_type = "ligand"
            cif_chains[current_chain] = {
                "start": chain_start,
                "end": len(token_chain_ids) - 1,
                "length": len(token_chain_ids) - chain_start,
                "mol_type": mol_type,
                "atoms": [],
            }
        else:
            mol_type = "polymer"
            cif_chains[current_chain] = {
                "start": chain_start,
                "end": len(token_chain_ids) - 1,
                "length": len(token_chain_ids) - chain_start,
                "mol_type": mol_type,
                "residues": [],
            }

    # Convert cif_chain_id to target_chain_id
    chains = {}
    for cif_id, chain_info in cif_chains.items():
#        target_id = cif_to_target.get(cif_id, cif_id)  # fallback to cif_id if not in mapping
        chains[cif_id] = chain_info

    total_length = len(token_chain_ids)

    result = {
        "chains": chains,
        "total_length": total_length,
        "cif_to_target_chain": cif_to_target
    }

    # Verify against distogram size if provided
    if distogram_path and distogram_path.exists():
        try:
            with np.load(distogram_path) as disto_data:
                distogram_size = disto_data["distogram"].shape[1]
                result["distogram_size"] = distogram_size
                result["size_match"] = distogram_size == total_length
                if not result["size_match"]:
                    print(
                        f"  Warning: Size mismatch! json={total_length}, distogram={distogram_size}"
                    )
        except Exception as e:
            print(f"  Warning: Failed to read distogram: {e}")

    return result


def extract_chain_seq_mapping(cif_path: Path, distogram_path: Optional[Path] = None) -> dict:
    """
    Extract chain to distogram index mapping from CIF file.
    Residues are numbered sequentially from 0 across all chains.

    Args:
        cif_path: Path to CIF file
        distogram_path: Optional path to distogram npz to verify size

    Returns:
        {
            "chains": {
                "A": {"start": 0, "end": 100, "length": 101},
                "B": {"start": 101, "end": 200, "length": 100}
            },
            "total_length": 201,
            "distogram_size": 201,  # if distogram_path provided
            "size_match": True      # if distogram_path provided
        }
    """
    # Read CIF
    if cif_path.suffix == ".gz":
        with gzip.open(cif_path, "rt", encoding="utf-8") as f:
            content = f.read()
        doc = gemmi.cif.read_string(content)
    else:
        doc = gemmi.cif.read_file(str(cif_path))

    block = doc.sole_block()

    # Read _atom_site to get residues in order
    atom_loop = block.find(
        "_atom_site.",
        [
            "label_atom_id",  # 0: atom name (CA)
            "label_asym_id",  # 1: label chain id
            "label_seq_id",  # 2: sequence id (1-based)
        ],
    )

    if atom_loop.width() == 0:
        return {}

    # Collect unique (chain, seq_id) pairs in order of appearance
    seen_residues = set()
    residue_order = []  # [(chain_id, seq_id), ...]

    for row in atom_loop:
        atom_name = str(row[0])
        chain_id = str(row[1])
        seq_id_str = str(row[2])

        # Only consider CA atoms for protein residues
        if atom_name != "CA":
            continue

        if seq_id_str == "." or seq_id_str == "?":
            continue

        label_seq_id = int(seq_id_str)
        key = (chain_id, label_seq_id)

        if key not in seen_residues:
            seen_residues.add(key)
            residue_order.append(key)

    if not residue_order:
        return {}

    # Build chain ranges based on residue order
    chains = {}
    current_chain = None
    chain_start = 0

    for idx, (chain_id, seq_id) in enumerate(residue_order):
        if chain_id != current_chain:
            # Save previous chain
            if current_chain is not None:
                chains[current_chain] = {
                    "start": chain_start,
                    "end": idx - 1,
                    "length": idx - chain_start,
                }
            # Start new chain
            current_chain = chain_id
            chain_start = idx

    # Save last chain
    if current_chain is not None:
        chains[current_chain] = {
            "start": chain_start,
            "end": len(residue_order) - 1,
            "length": len(residue_order) - chain_start,
        }

    total_length = len(residue_order)

    result = {
        "chains": chains,
        "total_length": total_length,
    }

    # Verify against distogram size if provided
    if distogram_path and distogram_path.exists():
        try:
            with np.load(distogram_path) as data:
                # distogram shape is usually (N, N, num_bins) or (N, N)
                distogram_size = data["distogram"].shape[1]
                result["distogram_size"] = distogram_size
                result["size_match"] = distogram_size == total_length
                if not result["size_match"]:
                    print(
                        f"  Warning: Size mismatch! CIF={total_length}, distogram={distogram_size}"
                    )
                else:
                    print(f"  Size match: {total_length} residues")
        except Exception as e:
            print(f"  Warning: Failed to read distogram {distogram_path}: {e}")

    return result


def collect_boltz_distograms(output_base: Path, force: bool = False, patterns: list[str] | None = None, boltz1_or_2: str = "boltz2") -> tuple[int, list]:
    """
    Collect boltz distograms using patterns from distogram_analysis_data.

    Source: {full_path_from_pattern}/*_distogram.npz
    Target: distogram/boltz/{set}/{cluster}/{yaml}/seed_{seed}_{model}.npz

    Also creates chain_seq_mapping.json with chain -> seq_id ranges.

    Args:
        output_base: Base output directory for symlinks
        force: If True, overwrite existing chain_seq_mapping.json
        patterns: List of distogram patterns from distogram_analysis_data
    """
    if boltz1_or_2 not in ["boltz1", "boltz2"]:
        raise ValueError("boltz1_or_2 must be 'boltz1' or 'boltz2'")
    if boltz1_or_2 == "boltz1":
        output_dir = output_base / "boltz1"
    else:
        output_dir = output_base / "boltz2"
    created = 0
    skipped = 0
    mappings_created = 0
    errors: list = []

    # Track which directories we've processed for chain mapping
    processed_dirs = set()

    # Find all distogram files from patterns
    if not patterns:
        print(f"No patterns provided for {boltz1_or_2}")
        return 0, []
    
    all_npz_files = find_distogram_files_from_patterns(patterns, method_filter=boltz1_or_2)
    total_files = len(all_npz_files)
    print(f"{boltz1_or_2}: Found {total_files} distogram files")

    for i, npz in enumerate(all_npz_files):
        # Progress log every 100 files
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i + 1}/{total_files}] Processing...")

        # Parse path directly from file path
        # Expected: .../boltz*/.../{set}/{cluster}/{yaml}/seed_{i}/boltz_results_{...}/predictions/{...}/{filename}
        parts = npz.parts

        # Find yaml_stem directly in parts
        yaml_stem = find_yaml_stem_in_parts(parts)
        if not yaml_stem:
            print(f"  Warning: Could not find yaml tag in {npz}, skipping")
            continue
        
        yaml_idx = parts.index(yaml_stem)
        
        # Expect: {set}/{cluster}/{yaml}
        if yaml_idx >= 2:
            set_name = parts[yaml_idx - 2]
            cluster = parts[yaml_idx - 1]
        else:
            print(f"  Warning: Could not infer set/cluster for {npz}: {parts}")
            continue

        # Extract seed number
        seed_match = re.search(r"seed_(\d+)", str(npz))
        if not seed_match:
            continue
        seed = seed_match.group(1)

        # Use yaml_stem in filename like: seed_4209_5wh1_3_C1_x_distogram.npz
        target_name = f"seed_{seed}_{yaml_stem}_distogram.npz"

        # Create target directory
        target_dir = output_dir / set_name / cluster / yaml_stem
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / target_name

        if target_file.exists() or target_file.is_symlink():
            skipped += 1
        else:
            # Create absolute symlink
            try:
                target_file.symlink_to(npz)
                created += 1
            except Exception as e:
                print(f"  Warning: Failed to create symlink {target_file} -> {npz}: {e}")
                errors.append({"type": "SymlinkError", "target": str(target_file), "source": str(npz), "message": str(e)})

        # Create chain_seq_mapping.json once per target directory
        dir_key = str(target_dir)
        mapping_file = target_dir / "chain_seq_mapping.json"

        if dir_key not in processed_dirs and (force or not mapping_file.exists()):
            processed_dirs.add(dir_key)

            # Locate Boltz processed NPZ via config templates (no fallback;
            # run ``link_boltz_structure_npz.py`` if missing).
            config_candidates = _structure_npz_candidates_from_config(
                boltz1_or_2, set_name, cluster, yaml_stem
            )
            processed_npz = None
            for cand in config_candidates:
                if cand.exists():
                    processed_npz = cand
                    break

            if processed_npz is None:
                msg = (
                    f"Processed NPZ not found for {yaml_stem} "
                    f"(method={boltz1_or_2}, set={set_name}, cluster={cluster}); "
                    f"checked pipeline.distogram_enrich.structure_npz.{boltz1_or_2} templates"
                )
                print(f"  Warning: {msg}")
                errors.append({"type": "MissingProcessedNPZ", "yaml_stem": yaml_stem, "method": boltz1_or_2, "method_type": set_name, "cluster": cluster, "distogram": str(npz)})
            else:
                try:
                    chain_mapping = extract_chain_seq_mapping_from_boltz_npz(processed_npz, npz)
                    if chain_mapping:
                        with open(mapping_file, "w") as f:
                            json.dump(chain_mapping, f, indent=2)
                        mappings_created += 1
                    else:
                        msg = f"Empty chain mapping from {processed_npz}"
                        print(f"  Warning: {msg}")
                        errors.append({"type": "EmptyChainMapping", "processed_npz": str(processed_npz), "distogram": str(npz)})
                except Exception as e:
                    print(f"  Warning: Failed to extract chain mapping from {processed_npz}: {e}")
                    errors.append({"type": "ChainMappingError", "processed_npz": str(processed_npz), "distogram": str(npz), "message": str(e)})

    print(
        f"{boltz1_or_2}: Created {created} symlinks, skipped {skipped} existing, {mappings_created} chain mappings"
    )

    if errors:
        # Save errors for boltz
        err_path = output_dir / f"mapping_errors_{boltz1_or_2}.json"
        try:
            with open(err_path, "w") as ef:
                json.dump(errors, ef, indent=2)
            print(f"  Saved {len(errors)} errors to {err_path}")
        except Exception as e:
            print(f"  Warning: Failed to save errors to {err_path}: {e}")

    return created, errors


def collect_af3_distograms(
    output_base: Path,
    force: bool = False,
    patterns: list[str] | None = None,
    af3_chain_mapping_root: Optional[Path] = None,
):
    """
    Collect AF3 distograms using patterns from distogram_analysis_data.

    Source: {full_path_from_pattern}/distogram.npz
    Target: distogram/af3/{set}/{cluster}/{yaml}/seed_{seed}_{sample}.npz

    Also creates chain_seq_mapping.json with chain -> seq_id ranges.

    Args:
        output_base: Base output directory for symlinks
        force: If True, overwrite existing chain_seq_mapping.json
        patterns: List of distogram patterns from distogram_analysis_data
    """
    output_dir = output_base / "af3"
    created = 0
    skipped = 0
    mappings_created = 0
    errors: list = []

    # Track which directories we've processed for chain mapping
    processed_dirs = set()

    # Find all distogram files from patterns
    if not patterns:
        print("No patterns provided for af3")
        return 0, []
    
    all_npz_files = find_distogram_files_from_patterns(patterns, method_filter="af3")
    total_files = len(all_npz_files)
    print(f"af3: Found {total_files} distogram files")

    for i, npz in enumerate(all_npz_files):
        # Progress log every 100 files
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i + 1}/{total_files}] Processing...")
        
        # Parse path directly from file path
        # Expected: .../af3*/.../{set}/{cluster}/{yaml}/seed_{i}/{cluster}_{yaml}/distogram.npz
        parts = npz.parts

        yaml_stem = find_yaml_stem_in_parts(parts)
        if not yaml_stem:
            print(f"  Warning: Could not find yaml tag in {npz}, skipping")
            continue
        
        yaml_idx = parts.index(yaml_stem)
        
        # Expect: {set}/{cluster}/{yaml}
        if yaml_idx >= 2:
            set_name = parts[yaml_idx - 2]
            cluster = parts[yaml_idx - 1]
        else:
            print(f"  Warning: Could not infer set/cluster for {npz}: {parts}")
            continue

        # Extract seed number from path
        seed_match = re.search(r"seed_(\d+)", str(npz))
        if not seed_match:
            continue
        seed = seed_match.group(1)

        # Use yaml_stem in filename like: seed_1_5wh1_3_C1_x_distogram.npz
        target_name = f"seed_{seed}_{yaml_stem}_distogram.npz"

        # Create target directory
        target_dir = output_dir / set_name / cluster / yaml_stem
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / target_name

        if target_file.exists() or target_file.is_symlink():
            skipped += 1
        else:
            # Create absolute symlink
            try:
                target_file.symlink_to(npz)
                created += 1
            except Exception as e:
                print(f"  Warning: Failed to create symlink {target_file} -> {npz}: {e}")
                errors.append({"type": "SymlinkError", "target": str(target_file), "source": str(npz), "message": str(e)})
        # Create chain_seq_mapping.json once per target directory
        dir_key = str(target_dir)
        mapping_file = target_dir / "chain_seq_mapping.json"

        if dir_key not in processed_dirs and (force or not mapping_file.exists()):
            processed_dirs.add(dir_key)

            # Locate AF3 confidences.json via config templates (no fallback;
            # run ``link_boltz_structure_npz.py --skip-boltz`` if missing).
            conf_file: Optional[Path] = None
            for cand in _confidences_json_candidates_from_config(
                "af3", set_name, cluster, yaml_stem
            ):
                if cand.exists():
                    conf_file = cand
                    break
            # Resolve cif -> target chain mapping from the consolidated JSON.
            cif_to_target = lookup_af3_cif_to_target(
                af3_chain_mapping_root, cluster, yaml_stem, set_name
            )

            if conf_file is None:
                msg = (
                    f"confidences.json not found for {cluster}/{yaml_stem} "
                    f"(method_type={set_name}); checked "
                    f"pipeline.distogram_enrich.confidences_json.af3 templates"
                )
                print(f"  Warning: {msg}")
                errors.append({"type": "MissingConfFile", "cluster": cluster, "yaml_stem": yaml_stem, "method_type": set_name, "distogram": str(npz)})
            elif cif_to_target is None:
                msg = (
                    f"AF3 chain mapping entry not found for {cluster}/{yaml_stem} "
                    f"(method_type={set_name}) in {af3_chain_mapping_root}"
                )
                print(f"  Warning: {msg}")
                errors.append({"type": "MissingAF3ChainMappingEntry", "cluster": cluster, "yaml_stem": yaml_stem, "method_type": set_name, "mapping_root": str(af3_chain_mapping_root), "distogram": str(npz)})
            else:
                try:
                    chain_mapping = extract_af3_chain_mapping(conf_file, cif_to_target, npz)
                    if chain_mapping:
                        with open(mapping_file, "w") as f:
                            json.dump(chain_mapping, f, indent=2)
                        mappings_created += 1
                    else:
                        msg = f"Empty AF3 chain mapping from {conf_file}"
                        print(f"  Warning: {msg}")
                        errors.append({"type": "EmptyAF3ChainMapping", "conf_file": str(conf_file), "mapping_root": str(af3_chain_mapping_root), "distogram": str(npz)})
                except Exception as e:
                    print(f"  Warning: Failed to extract chain mapping from {conf_file}: {e}")
                    errors.append({"type": "AF3ChainMappingError", "conf_file": str(conf_file), "mapping_root": str(af3_chain_mapping_root), "distogram": str(npz), "message": str(e)})

    print(
        f"af3: Created {created} symlinks, skipped {skipped} existing, {mappings_created} chain mappings"
    )

    if errors:
        # Save errors for af3
        err_path = output_dir / "mapping_errors_af3.json"
        try:
            with open(err_path, "w") as ef:
                json.dump(errors, ef, indent=2)
            print(f"  Saved {len(errors)} errors to {err_path}")
        except Exception as e:
            print(f"  Warning: Failed to save errors to {err_path}: {e}")

    return created, errors


def collect_bioemu_distograms(output_base: Path, force: bool = False, patterns: list[str] | None = None):
    """
    Collect BioEmu distograms using patterns from distogram_analysis_data.

    Source: {pattern}/*_rank_*_alphafold2_model_*_seed_*.pickle
    Target: distogram/bioemu/{set}/{cluster}/{yaml}/sample_{sample}.npz
    
    Args:
        output_base: Base output directory for symlinks
        force: If True, overwrite existing chain_seq_mapping.json
        patterns: List of distogram patterns from distogram_analysis_data
    """
    output_dir = output_base / "bioemu"
    created = 0
    skipped = 0
    mappings_created = 0
    errors: list = []
    
    # Track which directories we've processed for chain mapping
    processed_dirs = set()
    
    if not patterns:
        print("No patterns provided for bioemu")
        return 0, []

    # Patterns from make_pairs are globs -- expand them to real ``.pickle`` files.
    all_pickle_files = find_distogram_files_from_patterns(
        patterns, method_filter="bioemu", suffixes=(".pickle",)
    )
    total_files = len(all_pickle_files)
    bioemu_pattern_count = sum(1 for pat in patterns if "bioemu" in pat.lower())
    print(
        f"bioemu: Found {total_files} distogram files "
        f"from {bioemu_pattern_count} patterns"
    )

    for i, pickle_path in enumerate(all_pickle_files):
        # Progress log every 100 files
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i + 1}/{total_files}] Processing...")

        # Parse path: .../bioemu_disto/{cluster_id}/colabfold/{cluster_id}_all_rank_*_alphafold2_model_*_seed_*.pickle
        parts = pickle_path.parts
        
        # Find cluster_id (should be parent of 'colabfold')
        try:
            colabfold_idx = parts.index("colabfold")
            cluster_id = parts[colabfold_idx - 1]
        except (ValueError, IndexError):
            print(f"  Warning: Could not find cluster_id in {pickle_path}")
            continue
        
        # Extract seed and model from filename
        # Format: {cluster_id}_all_rank_001_alphafold2_model_3_seed_000.pickle
        filename = pickle_path.stem
        seed_match = re.search(r"seed_(\d+)", filename)
        model_match = re.search(r"model_(\d+)", filename)
        
        if not seed_match or not model_match:
            print(f"  Warning: Could not extract seed/model from {filename}")
            continue
        
        
        # Find corresponding sequence.fasta
        # Path: {base}/bioemu_disto/{cluster_id}/bioemu/seed_000/sequence.fasta
        base_dir = pickle_path.parent.parent  # Go up from colabfold to cluster dir
        fasta_path = base_dir / "bioemu" / "seed_000" / "sequence.fasta"
        
        if not fasta_path.exists():
            msg = f"sequence.fasta not found: {fasta_path}"
            print(f"  Warning: {msg}")
            errors.append({"type": "MissingFasta", "pickle": str(pickle_path), "expected_fasta": str(fasta_path)})
            continue
        
        # Read sequence length from fasta
        try:
            with open(fasta_path, "r") as f:
                lines = f.readlines()
            # Skip header lines (starting with >)
            seq_lines = [line.strip() for line in lines if not line.startswith(">")]
            sequence = "".join(seq_lines)
            seq_length = len(sequence)
        except Exception as e:
            msg = f"Failed to read fasta {fasta_path}: {e}"
            print(f"  Warning: {msg}")
            errors.append({"type": "FastaReadError", "fasta": str(fasta_path), "message": str(e)})
            continue
        
        # Determine method_type (need to check which set this cluster belongs to)
        # This is tricky - we need to check distogram_analysis_data or infer from path
        # For now, try all three types
        method_type = None
        yaml_tag = None
        
        # Try to load distogram_analysis_data to find the correct method_type
        # This is a bit of a hack, but necessary since bioemu doesn't have method_type in path
        # We'll try each method_type until we find one that works
        for trial_type in ["apo-monomers", "ligand-induced", "protein-induced"]:
            target_dir = output_dir / trial_type / cluster_id
            if target_dir.parent.parent.exists():
                # Found a valid parent, assume this is correct
                method_type = trial_type
                # For bioemu, yaml_tag is typically cluster_id + some suffix
                # We need to infer this from the analysis data
                # For now, just use cluster_id as base
                yaml_tag = cluster_id
                break
        
        if not method_type:
            # Default to apo-monomers if can't determine
            method_type = "apo-monomers"
            yaml_tag = cluster_id
        
        # Create target filename: sample_{model}.npz
        target_name = "seed_1_distogram.npz"
        
        # Create target directory
        target_dir = output_dir / method_type / cluster_id
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_file = target_dir / target_name
        
        if target_file.exists() or target_file.is_symlink():
            skipped += 1
        else:
            # Create absolute symlink
            try:
                target_file.symlink_to(pickle_path)
                created += 1
            except Exception as e:
                print(f"  Warning: Failed to create symlink {target_file} -> {pickle_path}: {e}")
                errors.append({"type": "SymlinkError", "target": str(target_file), "source": str(pickle_path), "message": str(e)})
        
        # Create chain_seq_mapping.json once per target directory
        dir_key = str(target_dir)
        mapping_file = target_dir / "chain_seq_mapping.json"
        
        distogram_dict = np.load(target_file, allow_pickle=True)
        distogram_size = distogram_dict["distogram"]["logits"].shape[0]
        assert distogram_size == seq_length, f"Distogram size {distogram_size} does not match sequence length {seq_length} for {target_file}"
        if dir_key not in processed_dirs and (force or not mapping_file.exists()):
            processed_dirs.add(dir_key)
            
            # BioEmu only has one chain (A) with the full sequence
            chain_mapping = {
                "chains": {
                    "A": {
                        "start": 0,
                        "end": seq_length - 1,
                        "length": seq_length,
                        "mol_type": "polymer",
                    }
                },
                "total_length": seq_length,
                "distogram_size": distogram_size,
                "size_match": distogram_size == seq_length
            }
            
            try:
                with open(mapping_file, "w") as f:
                    json.dump(chain_mapping, f, indent=2)
                mappings_created += 1
            except Exception as e:
                print(f"  Warning: Failed to write chain mapping {mapping_file}: {e}")
                errors.append({"type": "MappingWriteError", "mapping_file": str(mapping_file), "message": str(e)})
    
    print(f"bioemu: Created {created} symlinks, skipped {skipped} existing, {mappings_created} chain mappings")
    
    if errors:
        # Save errors for bioemu
        err_path = output_dir / "mapping_errors_bioemu.json"
        try:
            with open(err_path, "w") as ef:
                json.dump(errors, ef, indent=2)
            print(f"  Saved {len(errors)} errors to {err_path}")
        except Exception as e:
            print(f"  Warning: Failed to save errors to {err_path}: {e}")
    
    return created, errors



# Answer-map -> on-disk name translation for ``generate_distogram_tasks``.
# Answer-map uses dashed methods (``boltz-1``); on-disk tree uses dash-less.
_DISTOGRAM_METHODS: Tuple[str, ...] = ("af3", "boltz-1", "boltz-2", "bioemu")
_DASH_TO_DISK_METHOD: Dict[str, str] = {"boltz-1": "boltz1", "boltz-2": "boltz2"}
# bioemu-only set alias: pipeline ``intrinsic`` -> on-disk ``apo-monomers``.
_BIOEMU_SET_ALIASES: Dict[str, str] = {"intrinsic": "apo-monomers"}


def _disk_method(method: str) -> str:
    """Translate answer-map method key -> on-disk directory name."""
    return _DASH_TO_DISK_METHOD.get(method, method)


def _disk_set(method: str, set_name: str) -> str:
    """Translate answer-map set key -> on-disk directory name (bioemu only)."""
    if method == "bioemu":
        return _BIOEMU_SET_ALIASES.get(set_name, set_name)
    return set_name


def generate_distogram_tasks(
    distogram_json: Path,
    output_base: Path,
    output_json: Path,
    method_filter: Optional[str] = None,
) -> int:
    """
    Generate distogram comparison tasks from distogram_analysis_data_final.json.

    This creates tasks for comparing distograms between different conformations:
    - Apo predictions: compare with all apo and holo references
    - Holo predictions: compare with all apo and corresponding holo (itself)

    The distogram paths are taken from the symlinked output directory structure:
    {output_base}/{method}/{set}/{cluster}/{yaml}/seed_{seed}_*.npz

    Args:
        distogram_json: Path to distogram_analysis_data_final.json
        output_base: Base directory containing symlinked distograms (e.g., 0101/distogram)
        output_json: Output path for tasks JSON
        method_filter: Optional method filter (e.g., 'bioemu', 'boltz1', 'boltz2', 'af3')

    Returns:
        Number of tasks generated
    """
    print(f"\n{'='*60}")
    print("Generating distogram comparison tasks")
    print(f"{'='*60}")
    print(f"Input JSON: {distogram_json}")
    print(f"Distogram base: {output_base}")
    print(f"Output JSON: {output_json}")

    if not distogram_json.exists():
        raise FileNotFoundError(f"Distogram JSON not found: {distogram_json}")

    # Load distogram analysis data
    with open(distogram_json, "r") as f:
        analysis_data = json.load(f)

    tasks = []

    for method_type, clusters in analysis_data.items():
        print(f"\nProcessing {method_type}...")

        for cluster_id, cluster_data in clusters.items():
            print(f"  Cluster {cluster_id}...")

            # Get reference information
            apo_references = cluster_data.get("apo_references", {})
            holo_references = cluster_data.get("holo_references", {})

            # Process apo predictions
            if "apo_predictions" in cluster_data:
                for method, method_info in cluster_data["apo_predictions"].items():
                    if method not in _DISTOGRAM_METHODS:
                        continue
                    if method_filter and method != method_filter:
                        continue
                    yaml_tag = method_info.get("yaml_tag", "")
                    if not yaml_tag and method != "bioemu":
                        continue

                    # Answer-map keys -> on-disk path segments.
                    disk_m = _disk_method(method)
                    disk_s = _disk_set(method, method_type)

                    # Pattern: {output_base}/{method}/{set}/{cluster}/{yaml}/seed_*.npz
                    distogram_pattern = (
                        output_base / disk_m / disk_s / cluster_id / yaml_tag / "seed_*.npz"
                    )
                    if method == "bioemu":
                        distogram_pattern = (
                            output_base / disk_m / disk_s / cluster_id / "seed_*.npz"
                        )
                    distogram_files = sorted(glob.glob(str(distogram_pattern)))
                    chain_mapping_pattern = (
                        output_base / disk_m / disk_s / cluster_id / yaml_tag / "chain_seq_mapping.json"
                    )
                    if method == "bioemu":
                        chain_mapping_pattern = (
                            output_base / disk_m / disk_s / cluster_id / "chain_seq_mapping.json"
                        )
                    method_type_ori = method_type
                    if not distogram_files and method_type_ori == "apo-monomers":
                        distogram_pattern = output_base / disk_m / "protein-induced" / cluster_id / yaml_tag / "seed_*.npz"
                        distogram_files = sorted(glob.glob(str(distogram_pattern)))
                        chain_mapping_pattern = output_base / disk_m / "protein-induced" / cluster_id / yaml_tag / "chain_seq_mapping.json"
                    if not distogram_files and method_type_ori == "apo-monomers":
                        distogram_pattern = output_base / disk_m / "ligand-induced" / cluster_id / yaml_tag / "seed_*.npz"
                        distogram_files = sorted(glob.glob(str(distogram_pattern)))
                        chain_mapping_pattern = output_base / disk_m / "ligand-induced" / cluster_id / yaml_tag / "chain_seq_mapping.json"
                    if not distogram_files:
                        print(f"    No distogram files found for {output_base}/{disk_m}/{disk_s}/{cluster_id}/{yaml_tag}")
                        continue

                    chain_mapping_files = list(glob.glob(str(chain_mapping_pattern)))
                    assert len(chain_mapping_files) == 1, f"Multiple chain_seq_mapping.json files found for {chain_mapping_pattern}"
                    chain_mapping_files = chain_mapping_files[0]
                    # For apo: compare with ALL references (apo + holo)
                    all_references = {**apo_references, **holo_references}

                    # Local helper: find conf label from cluster_data
                    def find_conf_label_local(yaml_tag_local: str) -> str:
                        base_tag = yaml_tag_local
                        if base_tag.endswith("_m"):
                            base_tag = base_tag[:-2]
                        elif base_tag.endswith("_x"):
                            base_tag = base_tag[:-2]

                        for entry in cluster_data.get("apo", []):
                            if "_conf_" in entry:
                                entry_base = entry.rsplit("_conf_", 1)[0]
                                if entry_base == base_tag:
                                    return "conf_" + entry.rsplit("_conf_", 1)[1]

                        for entry in cluster_data.get("holo", []):
                            if "_conf_" in entry:
                                entry_base = entry.rsplit("_conf_", 1)[0]
                                if entry_base == base_tag:
                                    return "conf_" + entry.rsplit("_conf_", 1)[1]

                        if yaml_tag_local.endswith("_m"):
                            return "conf_m"
                        elif yaml_tag_local.endswith("_x"):
                            return "conf_x"
                        return "unknown"

                    # Dash-less on-disk label for downstream consumers.
                    prediction_method = _disk_method(method)

                    for ref_yaml_tag, ref_info in all_references.items():
                        # Reference state
                        ref_state = "apo" if ref_yaml_tag in apo_references else "holo"

                        # Determine conformations
                        target_conf = find_conf_label_local(yaml_tag)
                        ref_conf = find_conf_label_local(ref_yaml_tag)

                        # Try to find a representative mobile_cif from model pattern (if present)
                        mobile_cif_pattern = method_info.get("pattern", "")
                        mobile_cif_path = ""
                        if mobile_cif_pattern:
                            try:
                                matches = glob.glob(mobile_cif_pattern)
                                if matches:
                                    mobile_cif_path = str(Path(matches[0]).resolve())
                            except Exception:
                                mobile_cif_path = ""

                        # mobile chain (prediction target_chain) and ref_chain
                        mobile_chain = method_info.get("target_chain", "")
                        ref_chain = ref_info.get("target_chain", "")

                        # Output path (where comparison results will be written)
                        output_comp_dir = output_base / "comparisons" / prediction_method / method_type / cluster_id / yaml_tag
                        output_comp_file = output_comp_dir / f"{yaml_tag}_to_{ref_yaml_tag}.json"

                        task = {
                            "prediction_yaml_tag": yaml_tag,
                            "reference_yaml_tag": ref_yaml_tag,
                            "ref_cif": ref_info.get("reference_cif_path", ""),
                            "reference_cb_json": ref_info.get("reference_cb_json", ""),
                            "mobile_cif": mobile_cif_path,
                            "mobile_cif_pattern": mobile_cif_pattern,
                            "prediction_distograms": distogram_files,
                            "chain_mapping_file": chain_mapping_files,
                            "output_cif": str(output_comp_file.resolve()),
                            "ref_chain": ref_chain,
                            "mobile_chain": mobile_chain,
                            "target_conformation": target_conf,
                            "target_state": "apo",
                            "reference_conformation": ref_conf,
                            "reference_state": ref_state,
                            "prediction_method": prediction_method,
                            "method": method,
                            "cluster_id": cluster_id,
                            "method_type": method_type,
                        }
                        tasks.append(task)

            # Process holo predictions
            if "holo_predictions" in cluster_data:
                for conformation, conformation_data in cluster_data["holo_predictions"].items():
                    for method, method_info in conformation_data.items():
                        if method not in _DISTOGRAM_METHODS:
                            continue
                        if method_filter and method != method_filter:
                            continue
                        yaml_tag = method_info.get("yaml_tag", "")
                        if not yaml_tag:
                            continue

                        # Answer-map keys -> on-disk segments (see apo branch).
                        disk_m = _disk_method(method)
                        disk_s = _disk_set(method, method_type)

                        # Find all distogram files for this yaml_tag
                        distogram_pattern = (
                            output_base / disk_m / disk_s / cluster_id / yaml_tag / "seed_*.npz"
                        )


                        distogram_files = list(glob.glob(str(distogram_pattern)))

                        if not distogram_files:
                            print(f"    No distogram files found for {disk_m}/{disk_s}/{cluster_id}/{yaml_tag}")
                            continue
                        distogram_chain_mapping_pattern = (
                                                        output_base / disk_m / disk_s / cluster_id / yaml_tag / "chain_seq_mapping.json"
                        )
                        chain_mapping_files = list(glob.glob(str(distogram_chain_mapping_pattern)))
                        assert len(chain_mapping_files) == 1, f"Multiple chain_seq_mapping.json files found for {distogram_chain_mapping_pattern}"
                        chain_mapping_files = chain_mapping_files[0]
                        # For holo: compare with all apo + corresponding holo (itself)
                        target_references = {**apo_references}
                        if yaml_tag in holo_references:
                            target_references[yaml_tag] = holo_references[yaml_tag]

                        for ref_yaml_tag, ref_info in target_references.items():
                            # Reference state
                            ref_state = "apo" if ref_yaml_tag in apo_references else "holo"

                            # Determine conformations using same helper logic as apo
                            def find_conf_label_local(yaml_tag_local: str) -> str:
                                base_tag = yaml_tag_local
                                if base_tag.endswith("_m"):
                                    base_tag = base_tag[:-2]
                                elif base_tag.endswith("_x"):
                                    base_tag = base_tag[:-2]

                                for entry in cluster_data.get("apo", []):
                                    if "_conf_" in entry:
                                        entry_base = entry.rsplit("_conf_", 1)[0]
                                        if entry_base == base_tag:
                                            return "conf_" + entry.rsplit("_conf_", 1)[1]

                                for entry in cluster_data.get("holo", []):
                                    if "_conf_" in entry:
                                        entry_base = entry.rsplit("_conf_", 1)[0]
                                        if entry_base == base_tag:
                                            return "conf_" + entry.rsplit("_conf_", 1)[1]

                                if yaml_tag_local.endswith("_m"):
                                    return "conf_m"
                                elif yaml_tag_local.endswith("_x"):
                                    return "conf_x"
                                return "unknown"

                            target_conf = find_conf_label_local(yaml_tag)
                            ref_conf = find_conf_label_local(ref_yaml_tag)

                            # Compute mobile_cif, prediction_method and output paths
                            mobile_cif_pattern = method_info.get("pattern", "")
                            mobile_cif_path = ""
                            if mobile_cif_pattern:
                                try:
                                    matches = glob.glob(mobile_cif_pattern)
                                    if matches:
                                        mobile_cif_path = str(Path(matches[0]).resolve())
                                except Exception:
                                    mobile_cif_path = ""

                            prediction_method = _disk_method(method)

                            mobile_chain = method_info.get("target_chain", "")
                            ref_chain = ref_info.get("target_chain", "")

                            output_comp_dir = output_base / "comparisons" / prediction_method / method_type / cluster_id / yaml_tag
                            output_comp_file = output_comp_dir / f"{yaml_tag}_to_{ref_yaml_tag}.json"

                            # Create task for this comparison (same fields as apo, except target_state and holo_conformation)
                            task = {
                                "prediction_yaml_tag": yaml_tag,
                                "reference_yaml_tag": ref_yaml_tag,
                                "ref_cif": ref_info.get("reference_cif_path", ""),
                                "reference_cb_json": ref_info.get("reference_cb_json", ""),
                                "mobile_cif": mobile_cif_path,
                                "mobile_cif_pattern": mobile_cif_pattern,
                                "prediction_distograms": distogram_files,
                                "chain_mapping_file": chain_mapping_files,
                                "output_cif": str(output_comp_file.resolve()),
                                "ref_chain": ref_chain,
                                "mobile_chain": mobile_chain,
                                "target_conformation": target_conf,
                                "target_state": "holo",
                                "reference_conformation": ref_conf,
                                "reference_state": ref_state,
                                "prediction_method": prediction_method,
                                "method": method,
                                "cluster_id": cluster_id,
                                "method_type": method_type,
                                "holo_conformation": conformation,
                            }
                            tasks.append(task)

    # Group tasks by (method, method_type, cluster, prediction_yaml_tag) to avoid redundancy
    # Same prediction can have multiple references
    # Different methods can have same yaml_tag, so we need method in the key
    grouped_tasks = {}
    for task in tasks:
        # Create unique key for each prediction (method + location + yaml_tag)
        group_key = (
            task["method"],
            task["method_type"],
            task["cluster_id"],
            task["prediction_yaml_tag"]
        )
        
        if group_key not in grouped_tasks:
            # Create new entry with prediction-level info
            grouped_tasks[group_key] = {
                "prediction_yaml_tag": task["prediction_yaml_tag"],
                "prediction_distograms": task["prediction_distograms"],
                "chain_mapping_file": task["chain_mapping_file"],
                "mobile_cif": task["mobile_cif"],
                "mobile_cif_pattern": task["mobile_cif_pattern"],
                "mobile_chain": task["mobile_chain"],
                "target_conformation": task["target_conformation"],
                "target_state": task["target_state"],
                "prediction_method": task["prediction_method"],
                "method": task["method"],
                "cluster_id": task["cluster_id"],
                "method_type": task["method_type"],
                "references": [],
                "reference_keys": set()  # Track reference keys to avoid duplicates
            }
            # Add holo_conformation if present
            if "holo_conformation" in task:
                grouped_tasks[group_key]["holo_conformation"] = task["holo_conformation"]
        
        # Add reference info (avoid duplicates)
        ref_key = task["reference_yaml_tag"]
        if ref_key not in grouped_tasks[group_key]["reference_keys"]:
            ref_entry = {
                "reference_yaml_tag": task["reference_yaml_tag"],
                "ref_cif": task["ref_cif"],
                "reference_cb_json": task["reference_cb_json"],
                "ref_chain": task["ref_chain"],
                "reference_conformation": task["reference_conformation"],
                "reference_state": task["reference_state"],
                "output_cif": task["output_cif"],
            }
            grouped_tasks[group_key]["references"].append(ref_entry)
            grouped_tasks[group_key]["reference_keys"].add(ref_key)
    
    # Convert to list (remove reference_keys helper set)
    final_tasks = []
    for task_data in grouped_tasks.values():
        task_data.pop("reference_keys", None)  # Remove helper set
        final_tasks.append(task_data)

    # Save tasks to JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(final_tasks, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated {len(final_tasks)} grouped prediction tasks (from {len(tasks)} comparisons)")
    print(f"Saved to: {output_json}")
    print(f"{'='*60}")

    return len(final_tasks)


def main():
    parser = argparse.ArgumentParser(description="Collect distogram files via symlinks")
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        choices=["boltz1", "boltz2", "af3", "bioemu"],
        help="Method to collect distograms for",
    )
    parser.add_argument("--all", action="store_true", help="Collect all methods")
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Distogram analysis JSON to find distogram patterns (e.g., distogram_analysis_data_with_cb_paths.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for symlinked distograms (default: eval.dirs.distogram)",
    )
    parser.add_argument(
        "--af3-chain-mapping-root",
        type=str,
        default=None,
        help="Path to the consolidated AF3 chain-mapping JSON "
        "(flat dict keyed by '{method_type}/{cluster_id}/{yaml_tag}', value "
        "carries a 'mapping' field of the form {cif_id: target_id}). "
        "For AF3/--all: pass this or set eval.external.af3_chain_mapping_root.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force overwrite existing chain_seq_mapping.json",
    )
    parser.add_argument(
        "--generate-tasks",
        action="store_true",
        help="Generate distogram comparison tasks JSON (skip distogram collection)",
    )
    parser.add_argument(
        "--tasks-output",
        type=str,
        default=None,
        help="Output path for distogram tasks JSON (default: {output_dir}/distogram_tasks.json)",
    )
    args = parser.parse_args()

    output_base = E.distogram_collect_output_dir(args.output_dir)
    af3_root = E.distogram_af3_chain_mapping_root(args.af3_chain_mapping_root)

    # Handle task generation
    if args.generate_tasks:
        json_path = Path(args.json)
        if not json_path.exists():
            raise FileNotFoundError(f"Distogram JSON not found: {json_path}")

        tasks_output = Path(args.tasks_output) if args.tasks_output else output_base / "distogram_tasks.json"
        generate_distogram_tasks(json_path, output_base, tasks_output, method_filter=args.method)
        return

    if (args.all or args.method == "af3") and af3_root is None:
        parser.error(
            "--af3-chain-mapping-root is required for AF3 collection (--method af3 or --all)"
        )

    # Original distogram collection logic
    output_base.mkdir(parents=True, exist_ok=True)

    total = 0

    # Load JSON patterns (required)
    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"Distogram JSON not found: {json_path}")
    patterns = load_patterns_from_distogram_json(json_path)
    if not patterns:
        print(f"Warning: No distogram patterns found in {json_path}")
    else:
        print(f"Loaded {len(patterns)} distogram patterns from {json_path}")

    all_errors = []

    if args.all or args.method == "boltz1":
        created_b1, errors_b1 = collect_boltz_distograms(output_base, args.force, patterns=patterns, boltz1_or_2="boltz1")
        total += created_b1
        all_errors.extend(errors_b1)

    if args.all or args.method == "boltz2":
        created_b2, errors_b2 = collect_boltz_distograms(output_base, args.force, patterns=patterns, boltz1_or_2="boltz2")
        total += created_b2
        all_errors.extend(errors_b2)

    if args.all or args.method == "af3":
        created_a, errors_a = collect_af3_distograms(
            output_base,
            args.force,
            patterns=patterns,
            af3_chain_mapping_root=af3_root,
        )
        total += created_a
        all_errors.extend(errors_a)

    if args.all or args.method == "bioemu":
        created_a, errors_a = collect_bioemu_distograms(output_base, args.force, patterns=patterns)
        total += created_a
        all_errors.extend(errors_a)
    if all_errors:
        err_all_path = output_base / "mapping_errors_all.json"
        try:
            with open(err_all_path, "w") as ef:
                json.dump(all_errors, ef, indent=2)
            print(f"Saved combined {len(all_errors)} mapping errors to {err_all_path}")
        except Exception as e:
            print(f"Warning: Failed to save combined errors to {err_all_path}: {e}")
        # Exit non-zero to signal failure
        import sys

        print("Errors encountered during mapping extraction. Exiting with status 1.")
        sys.exit(1)

    if not args.all and not args.method:
        parser.print_help()
        return

    print(f"\nTotal symlinks created: {total}")
    print(f"Output directory: {output_base}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract CB coordinates for distogram evaluation.

Primary mode in this repo is ``--answer-map``: read the
``seq_cluster_to_answer_map.json`` from ``curation.make_pairs``, write per-reference
``*_cb.json`` files, and emit an augmented JSON that adds ``reference_cb_json``
paths so downstream distogram tasks can load coordinates.

Notes
-----
The legacy mode that accepted ``alignment_tasks_*.json`` and wrote
``references_distogram.json`` has been removed. No downstream module reads
``references_distogram.json`` in this repository; distogram steps consume
per-reference ``*_cb.json`` via ``reference_cb_json``.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Any
import argparse
import json
import gzip
import gemmi

from utils._config import eval_cfg as E

# Standard amino acids (20 canonical)
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def to_standard_aa(one_letter: str) -> str:
    """Convert to standard amino acid, return 'X' for non-standard."""
    return one_letter if one_letter in STANDARD_AA else "X"


def read_cif_doc(cif_path: Path) -> gemmi.cif.Document:
    if cif_path.suffix == ".gz":
        with gzip.open(cif_path, "rt", encoding="utf-8") as f:
            content = f.read()
        return gemmi.cif.read_string(content)
    else:
        return gemmi.cif.read_file(str(cif_path))


def entity_poly_to_dict(block):
    """Read sequences from _entity_poly_seq (residue by residue)."""
    out = {}
    seq_loop = block.find("_entity_poly_seq.", ["entity_id", "mon_id"])
    if seq_loop.width() == 0:
        return out

    seqs_by_entity = {}
    for row in seq_loop:
        entity_id = str(row[0])
        mon_id = str(row[1])
        one_letter = gemmi.find_tabulated_residue(mon_id).one_letter_code
        one_letter = to_standard_aa(one_letter)
        if entity_id not in seqs_by_entity:
            seqs_by_entity[entity_id] = []
        seqs_by_entity[entity_id].append(one_letter)

    for eid, seq_list in seqs_by_entity.items():
        out[eid] = {"seq_can": "".join(seq_list)}

    return out


def get_entity_types(block) -> Dict[str, str]:
    """
    Get entity types from _entity table.
    Returns: {entity_id: entity_type} where type is 'polymer', 'non-polymer', 'water', etc.
    """
    entity_types = {}
    entity_loop = block.find("_entity.", ["id", "type"])
    if entity_loop.width() > 0:
        for row in entity_loop:
            entity_id = str(row[0])
            entity_type = str(row[1]).lower()
            entity_types[entity_id] = entity_type
    return entity_types


def extract_all_chains_for_distogram(
    cif_path: Path,
    error_list: list = None,
    use_auth_chain: bool = True,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Extract coordinates for ALL chains in a CIF file for distogram calculation.

    For polymer chains: CB atoms (CA for GLY)
    For non-polymer (ligand, ion, etc.): all atoms

    Returns dict of {chain_id: chain_info} where chain_info contains:
      - chain: chain ID
      - entity_type: 'polymer', 'non-polymer', 'water', etc.
      - sequence: observed single letter amino acid sequence (for polymer only)
      - full_sequence: full polymer sequence from entity (for polymer only)
      - seq_id_to_coord: {index: [x, y, z]} mapping (0-based)
      - atom_names: list of atom names (for non-polymer only)
      - comp_id: ligand 3-letter code (for non-polymer only)
      - num_atoms: number of atoms/residues
      - full_length: full sequence length (polymer) or num atoms (non-polymer)
    """

    def log_error(error_type: str, message: str):
        if error_list is not None:
            error_list.append(
                {
                    "cif_path": str(cif_path),
                    "error_type": error_type,
                    "message": message,
                }
            )

    try:
        doc = read_cif_doc(cif_path)
        block = doc.sole_block()
    except Exception as e:
        log_error("ReadError", str(e))
        return None

    # Get entity types
    entity_types = get_entity_types(block)

    # Get full sequence from _entity_poly_seq
    seqs_by_entity = entity_poly_to_dict(block)

    # Find label_asym_id -> entity_id mapping
    asym_to_entity = {}
    asym_loop = block.find("_struct_asym.", ["id", "entity_id"])
    if asym_loop.width() > 0:
        for row in asym_loop:
            asym_to_entity[str(row[0])] = str(row[1])
    # Read _atom_site
    atom_loop = block.find(
        "_atom_site.",
        [
            "label_atom_id",  # 0: atom name (CA, CB, etc.)
            "label_asym_id",  # 1: label chain id
            "label_seq_id",  # 2: sequence id (1-based)
            "label_comp_id",  # 3: residue name (3-letter)
            "Cartn_x",  # 4: x
            "Cartn_y",  # 5: y
            "Cartn_z",  # 6: z
            "auth_asym_id",  # 7: author chain id
        ],
    )

    if atom_loop.width() == 0:
        log_error("NoAtomSite", "Could not find _atom_site data")
        return None

    # Collect atoms by chain
    # For polymer: {chain_id: {label_seq_id: {'CA': coord, 'CB': coord, 'res_name': ...}}}
    # For non-polymer: {label_asym_id: [(atom_name, coord, comp_id), ...]}
    polymer_atoms = {}  # {chain_id: {label_seq_id: {'CA': coord, 'CB': coord, 'res_name': str, 'label_asym': str}}}
    nonpolymer_atoms = {}  # {label_asym_id: [(atom_name, coord, comp_id), ...]}

    # Determine entity type for each label_asym_id first
    # (for non-polymer, we always use label_asym_id as the chain identifier)

    for row in atom_loop:
        atom_name = str(row[0])
        label_asym_id = str(row[1])
        auth_asym_id = str(row[7])

        # Get entity type for this label_asym_id
        entity_id = asym_to_entity.get(label_asym_id)
        if entity_id is None:
            continue

        entity_type = entity_types.get(entity_id, "unknown")
        is_polymer = entity_type == "polymer"

        x = float(str(row[4]))
        y = float(str(row[5]))
        z = float(str(row[6]))
        coord = [x, y, z]

        if is_polymer:
            # For polymer, use auth_asym_id if use_auth_chain, else label_asym_id
            chain_id = auth_asym_id if use_auth_chain else label_asym_id

            # Only collect CA and CB for polymer
            if atom_name not in ("CA", "CB"):
                continue

            seq_id_str = str(row[2])
            if seq_id_str == "." or seq_id_str == "?":
                continue

            label_seq_id = int(seq_id_str) - 1  # 0-based
            res_name = str(row[3])

            if chain_id not in polymer_atoms:
                polymer_atoms[chain_id] = {}
            if label_seq_id not in polymer_atoms[chain_id]:
                polymer_atoms[chain_id][label_seq_id] = {
                    "res_name": res_name,
                    "label_asym": label_asym_id,
                    "entity_id": entity_id,
                }

            if atom_name == "CA" and "CA" not in polymer_atoms[chain_id][label_seq_id]:
                polymer_atoms[chain_id][label_seq_id]["CA"] = coord
            elif (
                atom_name == "CB" and "CB" not in polymer_atoms[chain_id][label_seq_id]
            ):
                polymer_atoms[chain_id][label_seq_id]["CB"] = coord
        else:
            # For non-polymer, use np_ prefix with auth_asym_id to avoid conflicts
            # (auth_asym_id might conflict with polymer chains)
            chain_id = f"np_{auth_asym_id}"

            # Collect all atoms for non-polymer
            comp_id = str(row[3])
            if chain_id not in nonpolymer_atoms:
                nonpolymer_atoms[chain_id] = {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "atoms": [],
                }
            nonpolymer_atoms[chain_id]["atoms"].append((atom_name, coord, comp_id))

    # Build result for all chains
    result = {}

    # Process polymer chains
    for chain_id, residues in polymer_atoms.items():
        seq_id_to_coord = {}
        sequence = []

        for label_seq_id in sorted(residues.keys()):
            res_data = residues[label_seq_id]

            # Use CB if available, else CA (for GLY or missing CB)
            if "CB" in res_data:
                coord = res_data["CB"]
            elif "CA" in res_data:
                coord = res_data["CA"]
            else:
                continue

            seq_id_to_coord[label_seq_id] = coord

            res_name = res_data["res_name"]
            single_letter = gemmi.find_tabulated_residue(res_name).one_letter_code
            single_letter = to_standard_aa(single_letter)
            sequence.append(single_letter)

        if not seq_id_to_coord:
            continue

        # Get full sequence from entity (use entity_id from first residue)
        full_sequence = ""
        first_res = next(iter(residues.values()))
        entity_id = first_res.get("entity_id")
        if entity_id and entity_id in seqs_by_entity:
            full_sequence = seqs_by_entity[entity_id].get("seq_can", "")

        result[chain_id] = {
            "chain": chain_id,
            "entity_type": "polymer",
            "sequence": "".join(sequence),
            "full_sequence": full_sequence,
            "seq_id_to_coord": seq_id_to_coord,
            "num_atoms": len(seq_id_to_coord),
            "full_length": len(full_sequence)
            if full_sequence
            else len(seq_id_to_coord),
        }

    # Process non-polymer chains
    for chain_id, chain_data in nonpolymer_atoms.items():
        atoms = chain_data["atoms"]
        if not atoms:
            continue

        seq_id_to_coord = {}
        atom_names = []
        comp_id = atoms[0][2]  # Use first atom's comp_id

        for i, (atom_name, coord, _) in enumerate(atoms):
            seq_id_to_coord[i] = coord
            atom_names.append(atom_name)

        entity_type = chain_data.get("entity_type", "non-polymer")

        result[chain_id] = {
            "chain": chain_id,
            "entity_type": entity_type,
            "comp_id": comp_id,
            "atom_names": atom_names,
            "seq_id_to_coord": seq_id_to_coord,
            "num_atoms": len(seq_id_to_coord),
            "full_length": len(seq_id_to_coord),
        }

    return result if result else None


def extract_coords_for_distogram(
    cif_path: Path,
    chain_id: str,
    error_list: list = None,
    use_auth_chain: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Extract coordinates for distogram calculation.

    For polymer chains: CB atoms (CA for GLY)
    For non-polymer (ligand, ion, etc.): all atoms

    Returns dict with:
      - chain: chain ID
      - entity_type: 'polymer' or 'non-polymer'
      - sequence: observed single letter amino acid sequence (for polymer only)
      - full_sequence: full polymer sequence from entity (for polymer only)
      - seq_id_to_coord: {index: [x, y, z]} mapping (0-based)
                         For polymer: index is label_seq_id
                         For non-polymer: index is atom sequential number
      - atom_names: list of atom names (for non-polymer only)
      - num_atoms: number of atoms/residues
      - full_length: full sequence length (polymer) or num atoms (non-polymer)
    """

    def log_error(error_type: str, message: str):
        if error_list is not None:
            error_list.append(
                {
                    "cif_path": str(cif_path),
                    "chain_id": chain_id,
                    "error_type": error_type,
                    "message": message,
                }
            )

    try:
        doc = read_cif_doc(cif_path)
        block = doc.sole_block()
    except Exception as e:
        log_error("ReadError", str(e))
        return None

    # Get entity types
    entity_types = get_entity_types(block)
    print(entity_types)
    # Get full sequence from _entity_poly_seq
    seqs_by_entity = entity_poly_to_dict(block)

    # Find label_asym_id -> entity_id mapping
    asym_to_entity = {}
    asym_loop = block.find("_struct_asym.", ["id", "entity_id"])
    if asym_loop.width() > 0:
        for row in asym_loop:
            asym_to_entity[str(row[0])] = str(row[1])

    # Also build auth_asym_id -> label_asym_id mapping from atom_site
    auth_to_label = {}

    # Read _atom_site
    atom_loop = block.find(
        "_atom_site.",
        [
            "label_atom_id",  # 0: atom name (CA, CB, etc.)
            "label_asym_id",  # 1: label chain id
            "label_seq_id",  # 2: sequence id (1-based)
            "label_comp_id",  # 3: residue name (3-letter)
            "Cartn_x",  # 4: x
            "Cartn_y",  # 5: y
            "Cartn_z",  # 6: z
            "auth_asym_id",  # 7: author chain id
        ],
    )

    if atom_loop.width() == 0:
        log_error("NoAtomSite", "Could not find _atom_site data")
        return None

    # First, determine entity type for this chain
    target_entity_id = None
    target_label_asym = None

    for row in atom_loop:
        label_asym_id = str(row[1])
        auth_asym_id = str(row[7])

        asym_id = auth_asym_id if use_auth_chain else label_asym_id

        if asym_id == chain_id:
            target_label_asym = label_asym_id
            auth_to_label[auth_asym_id] = label_asym_id
            if label_asym_id in asym_to_entity:
                target_entity_id = asym_to_entity[label_asym_id]
            break

    if target_entity_id is None:
        log_error("NoEntity", f"Could not find entity for chain={chain_id}")
        return None

    entity_type = entity_types.get(target_entity_id, "unknown")
    is_polymer = entity_type == "polymer"

    if is_polymer:
        # Extract CB atoms (CA for GLY) for polymer
        return _extract_polymer_coords(
            atom_loop,
            chain_id,
            use_auth_chain,
            asym_to_entity,
            seqs_by_entity,
            error_list,
        )
    else:
        # Extract all atoms for non-polymer
        return _extract_nonpolymer_coords(
            atom_loop, chain_id, use_auth_chain, entity_type, error_list
        )


def _extract_polymer_coords(
    atom_loop,
    chain_id: str,
    use_auth_chain: bool,
    asym_to_entity: dict,
    seqs_by_entity: dict,
    error_list: list = None,
) -> Optional[Dict[str, Any]]:
    """Extract CB/CA coords for polymer chains."""

    ca_atoms = {}
    cb_atoms = {}
    entity_id = None

    for row in atom_loop:
        atom_name = str(row[0])
        label_asym_id = str(row[1])
        auth_asym_id = str(row[7])

        asym_id = auth_asym_id if use_auth_chain else label_asym_id

        if asym_id != chain_id:
            continue

        if atom_name not in ("CA", "CB"):
            continue

        seq_id_str = str(row[2])
        if seq_id_str == "." or seq_id_str == "?":
            continue

        label_seq_id = int(seq_id_str) - 1  # 0-based
        res_name = str(row[3])
        x = float(str(row[4]))
        y = float(str(row[5]))
        z = float(str(row[6]))
        coord = [x, y, z]

        if atom_name == "CA":
            if label_seq_id not in ca_atoms:
                ca_atoms[label_seq_id] = (coord, res_name, label_asym_id)
        else:  # CB
            if label_seq_id not in cb_atoms:
                cb_atoms[label_seq_id] = (coord, res_name, label_asym_id)

        # Get entity info once
        if entity_id is None and label_asym_id in asym_to_entity:
            entity_id = asym_to_entity[label_asym_id]

    # Build output: use CB, fall back to CA for GLY
    seq_id_to_coord = {}
    sequence = []
    full_sequence = ""

    all_seq_ids = sorted(set(ca_atoms.keys()) | set(cb_atoms.keys()))

    for label_seq_id in all_seq_ids:
        if label_seq_id in cb_atoms:
            coord, res_name, label_asym = cb_atoms[label_seq_id]
        elif label_seq_id in ca_atoms:
            coord, res_name, label_asym = ca_atoms[label_seq_id]
        else:
            continue

        seq_id_to_coord[label_seq_id] = coord
        single_letter = gemmi.find_tabulated_residue(res_name).one_letter_code
        single_letter = to_standard_aa(single_letter)
        sequence.append(single_letter)

    if entity_id and entity_id in seqs_by_entity:
        full_sequence = seqs_by_entity[entity_id].get("seq_can", "")

    if not seq_id_to_coord:
        return None

    return {
        "chain": chain_id,
        "entity_type": "polymer",
        "sequence": "".join(sequence),
        "full_sequence": full_sequence,
        "seq_id_to_coord": seq_id_to_coord,
        "num_atoms": len(seq_id_to_coord),
        "full_length": len(full_sequence) if full_sequence else len(seq_id_to_coord),
    }


def _extract_nonpolymer_coords(
    atom_loop,
    chain_id: str,
    use_auth_chain: bool,
    entity_type: str,
    error_list: list = None,
) -> Optional[Dict[str, Any]]:
    """Extract all atom coords for non-polymer (ligand, ion, etc.)."""

    coords = []
    atom_names = []
    comp_id = None

    for row in atom_loop:
        atom_name = str(row[0])
        label_asym_id = str(row[1])
        auth_asym_id = str(row[7])

        asym_id = auth_asym_id if use_auth_chain else label_asym_id

        if asym_id != chain_id:
            continue

        x = float(str(row[4]))
        y = float(str(row[5]))
        z = float(str(row[6]))
        coord = [x, y, z]

        coords.append(coord)
        atom_names.append(atom_name)

        if comp_id is None:
            comp_id = str(row[3])  # ligand name

    if not coords:
        return None

    # Build 0-based sequential index
    seq_id_to_coord = {i: coord for i, coord in enumerate(coords)}

    return {
        "chain": chain_id,
        "entity_type": entity_type,
        "comp_id": comp_id,  # ligand 3-letter code
        "atom_names": atom_names,
        "seq_id_to_coord": seq_id_to_coord,
        "num_atoms": len(coords),
        "full_length": len(coords),
    }


def extract_cb_info(
    cif_path: Path,
    chain_id: str,
    error_list: list = None,
    use_auth_chain: bool = True,  # Reference CIFs typically use auth_asym_id
) -> Optional[Dict[str, Any]]:
    """
    Extract CB coordinates from CIF file.
    For GLY residues, use CA instead of CB.

    Returns dict with:
      - chain: chain ID
      - sequence: observed single letter amino acid sequence
      - full_sequence: full polymer sequence from entity
      - seq_id_to_coord: {label_seq_id: [x, y, z]} mapping (0-based)
      - num_residues: number of residues with CB/CA
      - full_length: full sequence length
    """

    def log_error(error_type: str, message: str):
        if error_list is not None:
            error_list.append(
                {
                    "cif_path": str(cif_path),
                    "chain_id": chain_id,
                    "error_type": error_type,
                    "message": message,
                }
            )

    try:
        doc = read_cif_doc(cif_path)
        block = doc.sole_block()
    except Exception as e:
        log_error("ReadError", str(e))
        return None

    # Get full sequence from _entity_poly_seq
    seqs_by_entity = entity_poly_to_dict(block)

    # Find label_asym_id -> entity_id mapping
    asym_to_entity = {}
    asym_loop = block.find("_struct_asym.", ["id", "entity_id"])
    if asym_loop.width() > 0:
        for row in asym_loop:
            asym_to_entity[str(row[0])] = str(row[1])

    # Read _atom_site
    atom_loop = block.find(
        "_atom_site.",
        [
            "label_atom_id",  # 0: atom name (CA, CB)
            "label_asym_id",  # 1: label chain id
            "label_seq_id",  # 2: sequence id (1-based)
            "label_comp_id",  # 3: residue name (3-letter)
            "Cartn_x",  # 4: x
            "Cartn_y",  # 5: y
            "Cartn_z",  # 6: z
            "auth_asym_id",  # 7: author chain id
        ],
    )

    if atom_loop.width() == 0:
        log_error("NoAtomSite", "Could not find _atom_site data")
        return None

    # First pass: collect all CA and CB atoms
    ca_atoms = {}  # {label_seq_id: (coord, res_name, label_asym_id)}
    cb_atoms = {}  # {label_seq_id: (coord, res_name, label_asym_id)}

    for row in atom_loop:
        atom_name = str(row[0])
        label_asym_id = str(row[1])
        auth_asym_id = str(row[7])

        asym_id = auth_asym_id if use_auth_chain else label_asym_id

        if asym_id != chain_id:
            continue

        if atom_name not in ("CA", "CB"):
            continue

        seq_id_str = str(row[2])
        if seq_id_str == "." or seq_id_str == "?":
            continue

        label_seq_id = int(seq_id_str) - 1  # 0-based

        res_name = str(row[3])
        x = float(str(row[4]))
        y = float(str(row[5]))
        z = float(str(row[6]))
        coord = [x, y, z]

        if atom_name == "CA":
            if label_seq_id not in ca_atoms:
                ca_atoms[label_seq_id] = (coord, res_name, label_asym_id)
        else:  # CB
            if label_seq_id not in cb_atoms:
                cb_atoms[label_seq_id] = (coord, res_name, label_asym_id)

    # Build output: use CB, fall back to CA for GLY
    seq_id_to_coord = {}
    sequence = []
    full_sequence = ""
    entity_id = None

    # Get all residue positions
    all_seq_ids = sorted(set(ca_atoms.keys()) | set(cb_atoms.keys()))

    for label_seq_id in all_seq_ids:
        if label_seq_id in cb_atoms:
            coord, res_name, label_asym = cb_atoms[label_seq_id]
        elif label_seq_id in ca_atoms:
            # Use CA for GLY or if CB is missing
            coord, res_name, label_asym = ca_atoms[label_seq_id]
        else:
            continue

        seq_id_to_coord[label_seq_id] = coord
        single_letter = gemmi.find_tabulated_residue(res_name).one_letter_code
        single_letter = to_standard_aa(single_letter)
        sequence.append(single_letter)

        # Get entity info once
        if entity_id is None and label_asym in asym_to_entity:
            entity_id = asym_to_entity[label_asym]
            if entity_id in seqs_by_entity:
                full_sequence = seqs_by_entity[entity_id]["seq_can"] or ""

    if not seq_id_to_coord:
        log_error("NoCBAtoms", f"No CB atoms found for chain={chain_id}")
        return None

    return {
        "chain": chain_id,
        "sequence": "".join(sequence),
        "full_sequence": full_sequence,
        "seq_id_to_coord": seq_id_to_coord,
        "num_residues": len(seq_id_to_coord),
        "full_length": len(full_sequence) if full_sequence else len(seq_id_to_coord),
    }


def process_distogram_file(
    distogram_json: Path,
    ref_output_base: Path,
    skip_existing: bool = True,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
):
    """Process a distogram_analysis_data JSON and extract all chains from referenced CIFs.

    Saves one JSON per reference CIF at: {ref_output_base}/{TWO_LETTER}/{asm_file}.json
    Each file contains: {"cif_path": <path>, "all_chains": {...}}
    """

    with open(distogram_json, "r") as f:
        data = json.load(f)

    # Collect unique reference CIFs from apo_references and holo_references
    ref_cifs = set()

    for dataset, items in data.items():
        for pdb_id, content in items.items():
            # apo_references and holo_references may be absent
            for section in ("apo_references", "holo_references"):
                refs = content.get(section, {})
                for key, info in refs.items():
                    ref_path = info.get("reference_cif_path")
                    if ref_path:
                        ref_cifs.add(ref_path)

    ref_list = sorted(ref_cifs)

    # Apply start/end slicing
    if end_idx is None:
        end_idx = len(ref_list)
    subset = ref_list[start_idx:end_idx]

    print(
        f"Processing {len(subset)} reference CIFs from {distogram_json} [{start_idx}:{end_idx}]"
    )

    errors = []
    total_processed = 0

    # Mapping from original CIF path to generated CB JSON path
    ref_map = {}

    def get_ref_output_path_local(ref_cif: str, base: Path) -> Path:
        ref_path = Path(ref_cif)
        parts = ref_path.parts
        for i, part in enumerate(parts):
            if part == "cif-asms" and i + 1 < len(parts):
                two_letter = parts[i + 1]
                filename = ref_path.stem + "_cb.json"
                return base / two_letter / filename
        filename = ref_path.stem + "_cb.json"
        return base / filename

    for i, ref_cif in enumerate(subset):
        ref_path = Path(ref_cif)

        if not ref_path.exists():
            errors.append({"cif_path": ref_cif, "error": "file not found"})
            continue

        out_path = get_ref_output_path_local(ref_cif, ref_output_base)
        
        # Smart check for existing reference CB JSON
        if skip_existing and out_path.exists():
            try:
                with open(out_path, 'r') as f:
                    existing_data = json.load(f)
                
                # Extract current chains from CIF
                current_all_chains = extract_all_chains_for_distogram(
                    ref_path, error_list=errors, use_auth_chain=True
                )
                
                if current_all_chains:
                    existing_chains = set(existing_data.get("all_chains", {}).keys())
                    expected_chains = set(current_all_chains.keys())
                    missing_chains = expected_chains - existing_chains
                    
                    if not missing_chains:
                        # All chains already exist, skip processing
                        ref_map[str(ref_cif)] = str(out_path)
                        continue
                    else:
                        print(f"  Found missing chains in {out_path.name}: {missing_chains}")
                        print(f"    Existing: {len(existing_chains)}, Expected: {len(expected_chains)}")
                        # Continue to reprocess with all chains
                else:
                    # Failed to extract, skip
                    ref_map[str(ref_cif)] = str(out_path)
                    continue
                    
            except (json.JSONDecodeError, KeyError, Exception) as e:
                print(f"  Error reading existing CB JSON {out_path}: {e}")
                print(f"  Will regenerate the file")

        # Extract all chains for distogram
        all_chains = extract_all_chains_for_distogram(
            ref_path, error_list=errors, use_auth_chain=True
        )
        if not all_chains:
            continue

        # Save file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"cif_path": ref_cif, "all_chains": all_chains}, f, indent=2)
        # Record mapping from CIF path to generated CB JSON path
        ref_map[str(ref_cif)] = str(out_path)
        total_processed += 1

    # Write mapping file CSV: ref CIF -> generated CB JSON path
    if "ref_map" in locals() and ref_map:
        map_out = ref_output_base / "distogram_ref_cb_map.json"
        try:
            with open(map_out, "w") as mf:
                json.dump(ref_map, mf, indent=2)
            print(f"Saved reference CB mapping to {map_out}")
        except Exception as e:
            errors.append({"error": "write_map_failed", "message": str(e)})

        # Also create an augmented version of the input distogram JSON that includes the
        # 'reference_cb_json' field for each apo/holo reference where available.
        try:
            import copy

            augmented = copy.deepcopy(data)
            for dataset, items in augmented.items():
                for pdb_id, content in items.items():
                    for section in ("apo_references", "holo_references"):
                        refs = content.get(section, {})
                        for key, info in refs.items():
                            ref_path = info.get("reference_cif_path")
                            if ref_path and ref_path in ref_map:
                                info["reference_cb_json"] = ref_map[ref_path]

            augmented_out = (
                distogram_json.parent / f"{distogram_json.stem}_with_cb_paths.json"
            )
            with open(augmented_out, "w") as af:
                json.dump(augmented, af, indent=2)
            print(f"Saved augmented distogram JSON with cb paths to {augmented_out}")
        except Exception as e:
            errors.append({"error": "augment_write_failed", "message": str(e)})

    return total_processed, errors


def create_symlinks(source_method: str, base_dir: Path):
    """
    Create symlinks from source_method directories to other methods.

    This is a legacy helper that operates on per-directory
    ``references_distogram.json`` files (which are no longer produced by the
    main extraction flow in this repo). Keep only if you have those files from
    older runs and want to mirror them across methods.
    """
    other_methods = ["af3", "chai-1", "bioemu"]
    source_base = base_dir / source_method

    if not source_base.exists():
        raise ValueError(f"Source method directory not found: {source_base}")

    created = 0
    skipped = 0

    # Walk through source method directories
    for refs_cb in source_base.rglob("references_distogram.json"):
        # Get relative path from source method
        rel_path = refs_cb.relative_to(source_base)

        for other_method in other_methods:
            target_dir = base_dir / other_method / rel_path.parent
            target_file = target_dir / "references_distogram.json"

            # Create target directory if needed
            if not target_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)

            if target_file.exists() or target_file.is_symlink():
                print(target_file)
                skipped += 1
                continue

            # Create relative symlink
            # Calculate relative path from target to source
            source_file = base_dir / source_method / rel_path
            rel_source = Path("../" * len(rel_path.parts)) / source_method / rel_path

            target_file.symlink_to(rel_source)
            created += 1

    print(f"Created {created} symlinks, skipped {skipped} existing")


def main():
    parser = argparse.ArgumentParser(
        description="Extract CB coordinates from reference CIF files"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start index for parallel processing"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="End index for parallel processing"
    )
    parser.add_argument(
        "--symlink",
        type=str,
        help="Create symlinks from this method to others (e.g., 'boltz')",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for --symlink (default: eval.external.aligned_cif_dir)",
    )
    parser.add_argument(
        "--answer-map",
        type=str,
        help=(
            "Path to seq_cluster_to_answer_map.json (from curation.make_pairs); "
            "extracts Cb for every reference CIF and writes "
            "<stem>_with_cb_paths.json next to the input."
        ),
    )
    parser.add_argument(
        "--ref-output",
        type=str,
        default=None,
        help="Output root for *_cb.json (default: eval.dirs.ref_coords / external.ref_coords_dir)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing reference JSON files",
    )
    args = parser.parse_args()

    if args.symlink:
        base = E.distogram_aligned_cif_dir(args.base_dir)
        if base is None:
            parser.error(
                "--base-dir or eval.external.aligned_cif_dir is required when using --symlink"
            )
        create_symlinks(args.symlink, base)
    elif args.answer_map:
        # Process answer-map JSON and extract Cb for every reference CIF it lists.
        answer_map_path = Path(args.answer_map)
        if not answer_map_path.exists():
            raise FileNotFoundError(f"Answer-map JSON not found: {answer_map_path}")

        ref_output = E.distogram_ref_coords_dir(args.ref_output)
        total_refs, errors = process_distogram_file(
            answer_map_path,
            ref_output,
            skip_existing=not args.no_skip,
            start_idx=args.start,
            end_idx=args.end,
        )

        # Save errors if any
        if errors:
            error_path = (
                answer_map_path.parent
                / f"extract_cb_errors_answer_map_{args.start}_{args.end or 'end'}.json"
            )
            with open(error_path, "w") as f:
                json.dump(errors, f, indent=2)
            print(f"Saved {len(errors)} errors to {error_path}")

        print(f"\nSummary:")
        print(f"  Total reference CIFs processed: {total_refs}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

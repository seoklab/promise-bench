"""Extract CA coordinates from mmCIF or PDB (foundation ``extract_ca_from_cif`` logic).

Used by ``struct_align_batch`` so ``seq_id_to_coord`` keys match step6 / a3m mapping.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gemmi
import numpy as np

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

_PDB_SUFFIXES = {".pdb", ".ent"}


def to_standard_aa(one_letter: str) -> str:
    return one_letter if one_letter in STANDARD_AA else "X"


def _real_suffix(path: Path) -> str:
    if path.suffix == ".gz" and len(path.suffixes) >= 2:
        return path.suffixes[-2].lower()
    return path.suffix.lower()


def is_pdb_path(path: Path) -> bool:
    return _real_suffix(path) in _PDB_SUFFIXES


def read_cif_doc(cif_path: Path) -> gemmi.cif.Document:
    if cif_path.suffix == ".gz":
        with gzip.open(cif_path, "rt", encoding="utf-8") as f:
            content = f.read()
        return gemmi.cif.read_string(content)
    return gemmi.cif.read_file(cif_path.as_posix())


def entity_poly_to_dict(block: gemmi.cif.Block) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    seq_loop = block.find("_entity_poly_seq.", ["entity_id", "mon_id"])
    if seq_loop.width() > 0:
        seqs_by_entity: Dict[str, List[str]] = {}
        for row in seq_loop:
            entity_id = str(row[0])
            mon_id = str(row[1])
            one_letter = gemmi.find_tabulated_residue(mon_id).one_letter_code
            one_letter = to_standard_aa(one_letter)
            seqs_by_entity.setdefault(entity_id, []).append(one_letter)
        for eid, seq_list in seqs_by_entity.items():
            s = "".join(seq_list)
            out[eid] = {"seq_can": s, "seq": s}
    return out


def read_cif_structure(cif_path: Path) -> gemmi.Structure:
    """Read a CIF or PDB file into a ``gemmi.Structure``.

    The name is kept for backward compatibility, but PDB / .ent (optionally .gz)
    inputs are also supported transparently via gemmi's format auto-detection.
    """
    if is_pdb_path(cif_path):
        return gemmi.read_structure(cif_path.as_posix())
    if cif_path.suffix == ".gz":
        with gzip.open(cif_path, "rt", encoding="utf-8") as f:
            content = f.read()
        doc = gemmi.cif.read_string(content)
        return gemmi.make_structure_from_block(doc[0])
    return gemmi.read_structure(cif_path.as_posix())


def extract_ca_idx_map(
    cif_path: Path, chain_id: str
) -> Tuple[Dict[int, np.ndarray], np.ndarray, Dict[int, int], str]:
    structure = read_cif_structure(cif_path)
    target_chain = None
    for model in structure:
        for chain in model:
            if chain.name == chain_id:
                target_chain = chain
                break
        if target_chain:
            break
    if target_chain is None:
        raise ValueError(f"Chain {chain_id} not found in {cif_path}")

    coords_list: List[np.ndarray] = []
    sequence: List[str] = []
    for residue in target_chain:
        ca_atom = residue.find_atom("CA", "*")
        if not ca_atom:
            continue
        pos = ca_atom.pos
        coords_list.append(np.array([pos.x, pos.y, pos.z], dtype=np.float32))
        aa_name = residue.name
        sequence.append(gemmi.find_tabulated_residue(aa_name).one_letter_code)
    if not coords_list:
        raise ValueError(f"No CA atoms found for chain={chain_id} in {cif_path}")

    idx_to_ca = {i: coords_list[i] for i in range(len(coords_list))}
    coords = np.stack(coords_list, axis=0)
    res_to_coord_map = {i: i for i in range(len(coords_list))}
    return idx_to_ca, coords, res_to_coord_map, "".join(sequence)


def _extract_ca_info_pdb(
    pdb_path: Path,
    chain_id: str,
    log_error,
) -> Optional[Dict[str, Any]]:
    """CA extraction for PDB inputs.

    Keys ``seq_id_to_coord`` by 0-based residue index in the requested chain so
    downstream a3m mapping (sequential model indices) lines up the same way as
    the CIF path does for a clean monomer.
    """
    try:
        structure = read_cif_structure(pdb_path)
    except Exception as e:
        log_error("ReadError", str(e))
        return None

    target_chain = None
    for model in structure:
        for chain in model:
            if chain.name == chain_id:
                target_chain = chain
                break
        if target_chain is not None:
            break
    if target_chain is None:
        log_error("ChainNotFound", f"Chain {chain_id} not found in {pdb_path}")
        return None

    seq_id_to_coord: Dict[int, List[float]] = {}
    sequence: List[str] = []
    idx = 0
    for residue in target_chain:
        ca_atom = residue.find_atom("CA", "*")
        if not ca_atom:
            continue
        pos = ca_atom.pos
        seq_id_to_coord[idx] = [float(pos.x), float(pos.y), float(pos.z)]
        one_letter = gemmi.find_tabulated_residue(residue.name).one_letter_code
        sequence.append(to_standard_aa(one_letter))
        idx += 1

    if not seq_id_to_coord:
        log_error("NoCAAtoms", f"No CA atoms found for chain={chain_id}")
        return None

    seq_str = "".join(sequence)
    return {
        "chain": chain_id,
        "sequence": seq_str,
        "full_sequence": seq_str,
        "seq_id_to_coord": seq_id_to_coord,
        "num_residues": len(seq_id_to_coord),
        "full_length": len(seq_str),
    }


def extract_ca_info(
    cif_path: Path,
    chain_id: str,
    error_list: Optional[list] = None,
    use_auth_chain: bool = False,
) -> Optional[Dict[str, Any]]:
    def log_error(error_type: str, message: str) -> None:
        print(f"  {error_type}: {message}")
        if error_list is not None:
            error_list.append(
                {
                    "cif_path": cif_path,
                    "chain_id": chain_id,
                    "error_type": error_type,
                    "message": message,
                }
            )

    if is_pdb_path(cif_path):
        return _extract_ca_info_pdb(cif_path, chain_id, log_error)

    try:
        doc = read_cif_doc(cif_path)
        block = doc.sole_block()
    except Exception as e:
        log_error("ReadError", str(e))
        return None

    seqs_by_entity = entity_poly_to_dict(block)
    asym_to_entity: Dict[str, str] = {}
    asym_loop = block.find("_struct_asym.", ["id", "entity_id"])
    if asym_loop.width() > 0:
        for row in asym_loop:
            asym_to_entity[str(row[0])] = str(row[1])

    atom_loop = block.find(
        "_atom_site.",
        [
            "label_atom_id",
            "label_asym_id",
            "label_seq_id",
            "label_comp_id",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
            "auth_asym_id",
        ],
    )
    if atom_loop.width() == 0:
        log_error("NoAtomSite", "Could not find _atom_site data")
        return None

    coords_list: List[Tuple[int, List[float], str]] = []
    full_sequence = ""
    entity_id: Optional[str] = None
    seen_seq_ids: set = set()

    for row in atom_loop:
        atom_name = str(row[0])
        label_asym_id = str(row[1])
        auth_asym_id = str(row[7])
        asym_id = auth_asym_id if use_auth_chain else label_asym_id
        if asym_id != chain_id or atom_name != "CA":
            continue
        seq_id_str = str(row[2])
        if seq_id_str in (".", "?"):
            continue
        label_seq_id = int(seq_id_str) - 1
        if label_seq_id in seen_seq_ids:
            continue
        seen_seq_ids.add(label_seq_id)
        res_name = str(row[3])
        single_letter = gemmi.find_tabulated_residue(res_name).one_letter_code
        single_letter = to_standard_aa(single_letter)
        x = float(str(row[4]))
        y = float(str(row[5]))
        z = float(str(row[6]))
        coords_list.append((label_seq_id, [x, y, z], single_letter))
        if entity_id is None and label_asym_id in asym_to_entity:
            entity_id = asym_to_entity[label_asym_id]
            if entity_id in seqs_by_entity:
                full_sequence = seqs_by_entity[entity_id]["seq_can"] or ""

    if not coords_list:
        log_error("NoCAAtoms", f"No CA atoms found for chain={chain_id}")
        return None

    coords_list.sort(key=lambda x: x[0])
    seq_id_to_coord: Dict[int, List[float]] = {}
    sequence: List[str] = []
    for label_seq_id, coord, single_letter in coords_list:
        seq_id_to_coord[label_seq_id] = coord
        sequence.append(single_letter)

    return {
        "chain": chain_id,
        "sequence": "".join(sequence),
        "full_sequence": full_sequence,
        "seq_id_to_coord": seq_id_to_coord,
        "num_residues": len(seq_id_to_coord),
        "full_length": len(full_sequence),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cif_path", type=Path)
    p.add_argument("--chain", "-c", required=True)
    args = p.parse_args()
    info = extract_ca_info(args.cif_path, args.chain)
    if info is None:
        raise SystemExit(1)
    print(info["num_residues"], "CA", info["sequence"][:60])


if __name__ == "__main__":
    main()

import io
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser

metals = ["MG", "ZN", "MN", "CA", "FE", "NI", "CO", "CU", "K", "NA"]


@dataclass
class Structure:
    atom_positions: np.ndarray  # [n_atoms, 3]
    atom_names: np.ndarray  # [n_atoms,]
    atom_elements: np.ndarray  # [n_atoms,]
    atom_residues: np.ndarray  # [n_atoms,]
    residue_idxs: np.ndarray  # [n_atoms,]
    residue_inserts: np.ndarray  # [n_atoms,]
    chain_ids: np.ndarray  # [n_atoms,]
    is_ligand: np.ndarray  # [n_atoms,]
    metal_positions: np.ndarray = None  # [n_metals, 3]
    metal_types: np.ndarray = None  # [n_metals,]


@dataclass
class StructureWithGrid:
    atom_positions: np.ndarray
    atom_names: np.ndarray
    atom_elements: np.ndarray
    atom_residues: np.ndarray
    residue_idxs: np.ndarray
    residue_inserts: np.ndarray
    chain_ids: np.ndarray
    is_ligand: np.ndarray
    grid_positions: np.ndarray  # [n_grids, 3]
    metal_positions: np.ndarray = None  # [n_metals, 3]
    metal_types: np.ndarray = None  # [n_metals,]


def _build_structure_from_model(model) -> Structure:
    """Build a Structure from a Biopython Model object."""
    data = defaultdict(list)

    for chain in model:
        for res in chain:
            hetflag, resseq, icode = res.id

            if hetflag == " ":
                for atom in res:
                    if atom.element not in ["H", "D"]:
                        data["atom_positions"].append(atom.coord)
                        data["atom_elements"].append(atom.element)
                        data["atom_residues"].append(res.get_resname())
                        data["atom_names"].append(atom.name)
                        data["is_ligand"].append(0)
                        data["residue_idxs"].append(resseq)
                        data["residue_inserts"].append(icode)
                        data["chain_ids"].append(chain.id)

            elif "H_" in hetflag:
                if res.get_resname() not in ["HOH", "DOD"]:
                    for atom in res.get_atoms():
                        # Check if this is a metal atom
                        if atom.element in metals and atom.element in res.get_resname():
                            data["metal_positions"].append(atom.coord)
                            data["metal_types"].append(atom.element)
                        else:
                            if atom.element not in ["H", "D"]:
                                data["atom_positions"].append(atom.coord)
                                data["atom_elements"].append(atom.element)
                                data["atom_residues"].append(res.get_resname())
                                data["atom_names"].append(atom.name)
                                data["is_ligand"].append(1)
                                data["residue_idxs"].append(resseq)
                                data["residue_inserts"].append(icode)
                                data["chain_ids"].append(chain.id)

    np_data = {k: np.array(v) for k, v in data.items()}

    if "metal_positions" not in np_data:
        np_data["metal_positions"] = np.zeros((0, 3), dtype=float)
        np_data["metal_types"] = np.array([], dtype=object)

    return Structure(**np_data)


def read_pdb(pdb_path) -> Structure:
    """Read a PDB file and return a Structure."""
    with open(pdb_path, "r") as f:
        pdb_str = f.read()
    pdb_fh = io.StringIO(pdb_str)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    model = list(structure.get_models())[0]

    return _build_structure_from_model(model)


def read_mmcif(cif_path) -> Structure:
    """Read an mmCIF file and return a Structure."""
    with open(cif_path, "r") as f:
        cif_str = f.read()
    cif_fh = io.StringIO(cif_str)

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("none", cif_fh)
    model = list(structure.get_models())[0]

    return _build_structure_from_model(model)

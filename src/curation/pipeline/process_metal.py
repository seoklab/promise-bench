import io
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import click
import numpy as np
import pandas as pd
import tqdm
from Bio.PDB.MMCIFParser import MMCIFParser
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

from ._data_root import DataRootCommand

_METAL_ELEMENTS = ["MG", "ZN", "MN", "CA", "FE", "NI", "CO", "CU", "K", "NA"]


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


def _build_structure_from_model(model) -> Structure:
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
                        if (
                            atom.element in _METAL_ELEMENTS
                            and atom.element in res.get_resname()
                        ):
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


def _read_mmcif(cif_path: str) -> Structure:
    with open(cif_path, "r") as f:
        cif_str = f.read()
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("none", io.StringIO(cif_str))
    model = list(structure.get_models())[0]
    return _build_structure_from_model(model)


def process_cif(
    pdb_file: str,
    coord_cutoff: float = 3.0,
    contact_cutoff: float = 7.0,
    coi: str = None,
) -> list:
    """Return metals within *coord_cutoff* of coordinating atoms.

    If *coi* (chain of interest) is given, only metals that have at least
    one coordinating atom belonging to that chain are returned.
    """
    structure = _read_mmcif(pdb_file)
    parts = Path(pdb_file).stem.split("_")
    pdb_id = parts[1] if len(parts) > 1 else parts[0]
    asm_id = parts[2] if len(parts) > 2 else ""

    if structure.metal_positions is None or len(structure.metal_positions) == 0:
        return []

    metal_positions = np.array(structure.metal_positions)
    metal_types = np.array(structure.metal_types)
    atom_positions = structure.atom_positions
    atom_residues = structure.atom_residues
    chain_ids = structure.chain_ids
    atom_elements = np.array(structure.atom_elements)

    chain_idx_map = [i for i, ch in enumerate(chain_ids) if str(ch) == str(coi)]
    if not chain_idx_map:
        return []
    atom_positions_coi = atom_positions[chain_idx_map]
    tree = cKDTree(atom_positions_coi)
    neigh_indices = tree.query_ball_point(metal_positions, coord_cutoff)
    min_dists_coi, _ = tree.query(metal_positions, k=1)

    results = []
    for metal_idx, metal_pos in enumerate(metal_positions):
        metal_type = metal_types[metal_idx]
        neighbor_idx_list = neigh_indices[metal_idx]
        candidate_idxs = [chain_idx_map[i] for i in neighbor_idx_list]
        n_o_s_neigh_idx = [
            i
            for i in candidate_idxs
            if atom_elements[i].upper() in ["N", "O", "S", "SE"]
        ]
        binding_residues = [atom_residues[i] for i in n_o_s_neigh_idx]
        binding_chains_raw = [chain_ids[i] for i in n_o_s_neigh_idx]
        unique_chains = list(set(binding_chains_raw))

        try:
            min_dist = float(min_dists_coi[metal_idx])
        except Exception:
            min_dist = None

        if min_dist is None or (
            contact_cutoff is not None and min_dist >= contact_cutoff
        ):
            continue

        results.append(
            [
                pdb_id,
                asm_id,
                metal_type,
                [round(c, 3) for c in metal_pos.tolist()],
                binding_residues,
                unique_chains,
                round(min_dist, 3) if min_dist is not None else None,
            ]
        )
    return results


def _compute_processed_rows(df: pd.DataFrame, mmcif_dir: Path) -> List[Dict]:
    """Read CIFs, remove low-coordination metals (coord <= 2) from ligand lists."""
    processed_rows: List[Dict] = []
    _cif_cache: Dict[str, list] = {}  # cache per (cif_path, coi)

    for _, row in df.iterrows():
        pdb = str(row.get("pdb", "")).lower()
        asm_id = str(row.get("assembly_id", ""))
        cif_path = (
            mmcif_dir / pdb[1:3].upper() / pdb.upper() / f"asm_{pdb}_{asm_id}.cif"
        )
        coi = str(row.get("chain_auth_asm", "") or "")

        cache_key = f"{cif_path}|{coi}"
        if cache_key in _cif_cache:
            metals = _cif_cache[cache_key]
        else:
            try:
                metals = process_cif(str(cif_path), coi=coi)
            except Exception:
                metals = []
            _cif_cache[cache_key] = metals

        ligands = (
            []
            if pd.isna(row.get("ligands"))
            else [x.strip() for x in str(row.get("ligands")).split(";") if x.strip()]
        )
        contact_ligands = (
            []
            if pd.isna(row.get("contact_ligands"))
            else [
                x.strip()
                for x in str(row.get("contact_ligands")).split(";")
                if x.strip()
            ]
        )

        # For each low-coordination metal (coord <= 2), remove one occurrence
        # from contact_ligands and ligands.
        for m in metals:
            coord_count = len(m[4])  # binding_residues
            if coord_count <= 2:
                mt = str(m[2]).upper()
                idx = next(
                    (i for i, v in enumerate(contact_ligands) if str(v).upper() == mt),
                    None,
                )
                if idx is not None:
                    contact_ligands.pop(idx)
                    idx2 = next(
                        (i for i, v in enumerate(ligands) if str(v).upper() == mt), None
                    )
                    if idx2 is not None:
                        ligands.pop(idx2)

        processed_rows.append(
            {
                **row.to_dict(),
                "pdb": pdb,
                "assembly_id": asm_id,
                "ligands": ligands,
                "contact_ligands": contact_ligands,
            }
        )

    return processed_rows


def process_cluster(
    csv_path: Path, in_dir: Path, out_dir: Path, mmcif_dir: Path
) -> Dict:
    """Read one cluster CSV, filter low-coord metals from ligand lists, write updated CSV."""
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    try:
        out_rel = csv_path.relative_to(in_dir)
    except Exception:
        out_rel = Path(csv_path.name)
    out_csv = out_dir / out_rel
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        df.to_csv(out_csv, index=False)
        return {"csv_path": str(csv_path), "n_rows": 0, "n_metals_removed": 0}

    processed_rows = _compute_processed_rows(df, mmcif_dir)

    # Build output dataframe with updated ligand columns
    out_rows = []
    n_metals_removed = 0
    for pr in processed_rows:
        row = {
            k: v
            for k, v in pr.items()
            if k not in ("chains", "metals_info", "ligand_list", "contact_ligands_str")
        }
        # overwrite ligand columns with filtered values
        row["ligands"] = ";".join(pr["ligands"]) if pr["ligands"] else ""
        row["contact_ligands"] = (
            ";".join(pr["contact_ligands"]) if pr["contact_ligands"] else ""
        )
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    return {"csv_path": str(csv_path), "n_rows": len(out_df)}


@click.command(cls=DataRootCommand, context_settings=dict(show_default=True))
@click.option(
    "--in-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    default="data/asms-subset",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    default="data/asms-metal",
)
@click.option(
    "--mmcif-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/cif-asms"),
    help="Root containing assembly CIFs.",
)
@click.option("--workers", type=int, default=32, help="Parallel workers.")
@click.option(
    "--glob",
    "glob_pat",
    type=str,
    default="**/*_asm_subset_filtered.csv",
    help="Glob pattern for input CSVs.",
)
def main(in_dir: Path, out_dir: Path, mmcif_dir: Path, workers: int, glob_pat: str):
    files = sorted(in_dir.glob(glob_pat))
    if not files:
        click.echo(f"[!] No inputs matching {in_dir}/{glob_pat}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        delayed(process_cluster)(p, in_dir, out_dir, mmcif_dir=mmcif_dir) for p in files
    ]

    results = Parallel(n_jobs=workers, prefer="processes")(
        tqdm.tqdm(tasks, desc="Metal filtering", unit="file")
    )

    n_ok = sum(1 for r in results if r is not None)
    total_rows = sum(r.get("n_rows", 0) for r in results if r is not None)
    click.echo(
        f"\n[done] {n_ok}/{len(files)} files processed, {total_rows} total rows written to {out_dir}"
    )


if __name__ == "__main__":
    main()

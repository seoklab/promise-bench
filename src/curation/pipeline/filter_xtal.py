from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import click
import gemmi
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .constants import _AA3, _NT, LIGAND_EXCLUDE


def _get_assembly(
    pdb: str, asm_id: str, assembled_cif_root: Optional[Path]
) -> gemmi.Structure:
    if assembled_cif_root is not None:
        cand = (
            assembled_cif_root
            / pdb[1:3].upper()
            / str(pdb).upper()
            / f"asm_{str(pdb).lower()}_{asm_id}.cif"
        )
        if cand.exists():
            st = gemmi.read_structure(str(cand))
            if len(st) > 0:
                return st
    else:
        print("[info] no assembled_cif_root given")
        return None


def _coords_and_ligs_from_model(model: gemmi.Model, include_h: bool = False):
    chain_coords: Dict[str, np.ndarray] = {}
    lig_instances: List[Dict[str, object]] = []

    def is_polymer_res(res: gemmi.Residue) -> bool:
        et = getattr(res, "entity_type", None)
        hf = getattr(res, "het_flag", "\0")
        comp_id = str(getattr(res, "name", "")).upper()
        if et == gemmi.EntityType.Polymer:
            return True

        if isinstance(hf, str) and (hf == "\0" or hf == " "):
            return comp_id in _AA3 or comp_id in _NT
        return False

    for sc in model:
        chain_name = sc.name
        xyz_poly: List[List[float]] = []
        for res in sc:
            comp_id = str(res.name).upper()
            if is_polymer_res(res):
                for at in res:
                    if (not include_h) and at.element.is_hydrogen:
                        continue
                    xyz_poly.append([at.pos.x, at.pos.y, at.pos.z])
            else:
                lig_xyz = []
                for at in res:
                    if (not include_h) and at.element.is_hydrogen:
                        continue
                    lig_xyz.append([at.pos.x, at.pos.y, at.pos.z])
                if lig_xyz:
                    lig_instances.append(
                        {
                            "comp_id": comp_id,
                            "auth_asym": chain_name,
                            "auth_seq_id": ".",
                            "ins_code": ".",
                            "coords": np.asarray(lig_xyz, dtype=float),
                        }
                    )
        if xyz_poly:
            chain_coords[chain_name] = np.asarray(xyz_poly, dtype=float)

    return chain_coords, lig_instances


def ligand_mediators_for_pair(
    a_xyz: np.ndarray,
    b_xyz: np.ndarray,
    lig_instances: List[Dict[str, object]],
    cutoff: float,
) -> Set[str]:
    mediators: Set[str] = set()
    if a_xyz.size == 0 or b_xyz.size == 0 or not lig_instances:
        return mediators
    c2 = cutoff * cutoff
    blk = 4096

    def near_chain(L: np.ndarray, P: np.ndarray) -> bool:
        for s in range(0, P.shape[0], blk):
            PP = P[s : s + blk]
            d2 = np.sum((PP[:, None, :] - L[None, :, :]) ** 2, axis=2)
            if np.any(d2 <= c2):
                return True
        return False

    for inst in lig_instances:
        cid = str(inst["comp_id"]).upper()
        if cid in LIGAND_EXCLUDE:
            continue
        L = inst["coords"]
        if near_chain(L, a_xyz) and near_chain(L, b_xyz):
            mediators.add(cid)
    return mediators


def get_contact_ligands(
    a_xyz: np.ndarray, lig_instances: List[Dict[str, object]], cutoff: float
) -> Set[str]:
    ligands: List[str] = list()
    if a_xyz.size == 0 or not lig_instances:
        return ligands
    c2 = cutoff * cutoff
    blk = 4096

    def near_chain(L: np.ndarray, P: np.ndarray) -> bool:
        for s in range(0, P.shape[0], blk):
            PP = P[s : s + blk]
            d2 = np.sum((PP[:, None, :] - L[None, :, :]) ** 2, axis=2)
            if np.any(d2 <= c2):
                return True
        return False

    for inst in lig_instances:
        cid = str(inst["comp_id"]).upper()
        if cid in LIGAND_EXCLUDE:
            continue
        L = inst["coords"]
        if near_chain(L, a_xyz):
            ligands.append(cid)
    return ligands


def get_contact_chains(
    coi_id: str, clone_coords: Dict[str, np.ndarray], cutoff: float
) -> List[str]:
    coi_xyz = clone_coords.get(coi_id)
    if coi_xyz is None or coi_xyz.size == 0:
        return []

    c2 = cutoff * cutoff
    blk = 4096
    hits: List[str] = []

    def near(P: np.ndarray, Q: np.ndarray) -> bool:
        for s in range(0, P.shape[0], blk):
            PP = P[s : s + blk]
            d2 = np.sum((PP[:, None, :] - Q[None, :, :]) ** 2, axis=2)
            if np.any(d2 <= c2):
                return True
        return False

    for cid, coords in clone_coords.items():
        if cid == coi_id:
            continue
        if coords is None or coords.size == 0:
            continue
        if near(coi_xyz, coords):
            hits.append(cid)
    return hits


def determine_subassemblies_clone_graph(
    coi_auth_asm: str,
    pairs: List[dict],
    present_clones: Set[str],
    clone_coords: Dict[str, np.ndarray],
    lig_instances_all: List[Dict[str, object]],
    chain_cutoff: float,
    ligand_cutoff: float,
    min_bio_prob: float,
    em_any: bool,
) -> Tuple[
    List[Set[str]],
    List[Tuple[str, str]],
    List[Dict[str, str]],
    List[Dict[str, str]],
    List[Dict[str, object]],
    Set[str],
    Dict[str, Set[str]],
]:
    def _norm_pair(u: str, v: str) -> Tuple[str, str]:
        return (u, v) if u < v else (v, u)

    pr_index: Dict[Tuple[str, str], dict] = {}
    for pr in pairs:
        a = str(pr["a_clone"])
        b = str(pr["b_clone"])
        if a == b:
            continue
        pr_index[_norm_pair(a, b)] = pr

    bio_edges: List[Tuple[str, str]] = []
    xtal_rows: List[Dict[str, str]] = []
    ligand_rows: List[Dict[str, str]] = []
    prodigy_rows: List[Dict[str, object]] = []

    for a, b in combinations(sorted(present_clones), 2):
        key = _norm_pair(a, b)
        pr = pr_index.get(key, None)

        if pr is None:
            continue

        p_bio = float(pr.get("p_bio", 0.0))
        if p_bio < min_bio_prob:
            lbl, reason = "XTAL", "XTAL"
        else:
            lbl, reason = "BIO", "BIO"

        # EM override
        if em_any:
            lbl, p_bio, reason = "BIO", 1.0, "em_forced_bio"
        else:
            mediators = ligand_mediators_for_pair(
                clone_coords.get(a, np.empty((0, 3))),
                clone_coords.get(b, np.empty((0, 3))),
                lig_instances_all,
                ligand_cutoff,
            )
            if mediators:
                lbl, p_bio, reason = "LIG", 1.0, "ligand_mediated"
                ligand_rows.append(
                    {
                        "pdb": str(pr.get("pdb", "")),
                        "assembly_id": str(pr.get("assembly_id", "")),
                        "chain_a": a,
                        "chain_b": b,
                        "mediators": ";".join(sorted(mediators)),
                    }
                )

        if lbl in ("BIO", "LIG"):
            bio_edges.append(key)
        else:
            xtal_rows.append(
                {
                    "pdb": str(pr.get("pdb", "")),
                    "assembly_id": str(pr.get("assembly_id", "")),
                    "chain_a": a,
                    "chain_b": b,
                    "reason": reason,
                    "p_bio": f"{p_bio:.6g}",
                }
            )

        prodigy_rows.append(
            {
                "pdb": str(pr.get("pdb", "")),
                "assembly_id": str(pr.get("assembly_id", "")),
                "chain_a": a,
                "chain_b": b,
                "p_bio": p_bio,
                "final_label": "BIO"
                if lbl == "BIO"
                else ("LIG" if lbl == "LIG" else "XTAL"),
                "reason": reason,
            }
        )

    g_clone: Dict[str, Set[str]] = {c: set() for c in present_clones}
    for u, v in bio_edges:
        g_clone[u].add(v)
        g_clone[v].add(u)

    def _cc(start: str) -> Set[str]:
        seen = {start}
        st = [start]
        while st:
            x = st.pop()
            for y in g_clone.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    st.append(y)
        return seen

    components: List[Set[str]] = []
    seen_all: Set[str] = set()

    for c in sorted(present_clones):
        if c in seen_all:
            continue
        comp = _cc(c)
        components.append(comp)
        seen_all |= comp

    contact_chains = list()
    contact_ligands = list()
    contact_ligands = get_contact_ligands(
        clone_coords.get(coi_auth_asm, np.empty((0, 3))),
        lig_instances_all,
        ligand_cutoff,
    )
    contact_chains = get_contact_chains(
        coi_id=coi_auth_asm, clone_coords=clone_coords, cutoff=chain_cutoff
    )

    return (
        components,
        bio_edges,
        xtal_rows,
        ligand_rows,
        prodigy_rows,
        present_clones,
        g_clone,
        contact_chains,
        contact_ligands,
    )


def process_group_with_cif(
    df: pd.DataFrame,
    group_index: Tuple[str, str, str],  # pdb, asm_id, coi_auth_asm
    assembled_cif_root: Optional[Path],
    min_bio_prob: float,
    chain_cutoff: float,
    ligand_cutoff: float,
    pair_index: Dict[Tuple[str, str], List[dict]],
):
    pdb, asm_id, coi_auth_asm = group_index
    g = df[
        (df["pdb"] == pdb)
        & (df["assembly_id"] == asm_id)
        & (df["chain_auth_asm"] == coi_auth_asm)
    ]
    if g.empty:
        return [], [], 0, [], (set(), set(), False)

    coi_auth_clones = set(g["chain_list_author"].iloc[0].split(";"))
    contact_chains = list()
    contact_ligands = list()

    try:
        st = _get_assembly(pdb, asm_id, assembled_cif_root)
        asm_model = st[0]
    except Exception as e:
        print(f"[warn] {pdb} asm {asm_id}: failed to load/build assembly: {e}")
        return [], [], 0, [], (set(), set(), False)

    clone_coords, lig_instances_all = _coords_and_ligs_from_model(
        asm_model, include_h=False
    )

    pairs = pair_index.get((pdb, asm_id), [])
    """
    if not pairs:
        print(f"[warn] {pdb} asm {asm_id}: no pairs found")
        return [], [], 0, [], (set(), set(), False)
    """

    em_any = (
        g.get("experimental_method") is not None
        and g["experimental_method"]
        .astype(str)
        .str.contains("ELECTRON", case=False, na=False)
        .any()
    )

    present_clones = set(clone_coords.keys())

    try:
        (
            components,
            bio_edges,
            xtal_rows,
            ligand_rows,
            prodigy_rows,
            clone_nodes,
            g_clone,
            contact_chains,
            contact_ligands,
        ) = determine_subassemblies_clone_graph(
            coi_auth_asm=coi_auth_asm,
            pairs=pairs,
            present_clones=present_clones,
            clone_coords=clone_coords,
            lig_instances_all=lig_instances_all,
            chain_cutoff=chain_cutoff,
            ligand_cutoff=ligand_cutoff,
            min_bio_prob=min_bio_prob,
            em_any=em_any,
        )
    except Exception as e:
        print(f"[warn] {pdb} asm {asm_id}: failed to determine subassemblies: {e}")
        return [], [], 0, [], ({}, {}, False)

    total_pairs_count = len(pairs)
    coi_component = set()
    is_diff = False
    contact_ligands_update = contact_ligands
    contact_chains_update = contact_chains

    for component in components:
        if coi_auth_asm in component:
            coi_component = component
            if coi_component != coi_auth_clones:
                is_diff = True
            else:
                is_diff = False
            break

    comp_diff = pdb, asm_id, coi_auth_asm, coi_auth_clones, coi_component, is_diff

    if len(coi_auth_clones) < len(coi_component):
        print(
            f"[error] {pdb} asm {asm_id} {coi_auth_asm}: coi_auth_clones < coi_component"
        )
    if len(coi_component) == 0:
        print(f"[error] {pdb} asm {asm_id} {coi_auth_asm}: coi_component is empty")
    return (
        ligand_rows,
        xtal_rows,
        total_pairs_count,
        prodigy_rows,
        comp_diff,
        contact_chains_update,
        contact_ligands_update,
    )


def process_file_worker(
    csv_path: Path,
    asm_raw_dir: Path,
    out_dir: Path,
    pair_index: Dict[Tuple[str, str], List[dict]],
    assembled_cif_root: Optional[Path],
    min_bio_prob: float,
    chain_cutoff: float,
    ligand_cutoff: float,
) -> Tuple[
    List[Dict[str, str]],  # ligand_edges_all
    List[Dict[str, str]],  # xtal_edges_all
    int,  # total_pairs_file
    List[Dict[str, object]],  # prodigy_rows_file
    List[Tuple[Set[str], Set[str], bool]],  # comp_diff_all
]:
    df = pd.read_csv(csv_path, dtype=str)
    rel = csv_path.relative_to(asm_raw_dir)
    out_csv = (
        out_dir / rel.parent / (csv_path.stem.replace("_asm_raw", "_asm_bio") + ".csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        df.to_csv(out_csv, index=False)
        return [], [], 0, [], []

    total_pairs_file = 0
    ligand_edges_all: List[Dict[str, str]] = []
    xtal_edges_all: List[Dict[str, str]] = []
    prodigy_rows_file: List[Dict[str, object]] = []
    comp_diff_all: List[Tuple[Set[str], Set[str], bool]] = []

    if "contact_chains" not in df.columns:
        df["contact_chains"] = ""
    if "contact_ligands" not in df.columns:
        df["contact_ligands"] = ""

    group_keys = list(
        df.groupby(["pdb", "assembly_id", "chain_auth_asm"], dropna=False).groups.keys()
    )

    for grp in group_keys:
        pdb_g, asm_g, chain_auth_asm_g = grp
        try:
            (
                ligand_rows,
                xtal_rows,
                n_pairs,
                prodigy_rows,
                comp_diff,
                contact_chains,
                contact_ligands,
            ) = process_group_with_cif(
                df=df,
                group_index=grp,
                assembled_cif_root=assembled_cif_root,
                min_bio_prob=min_bio_prob,
                chain_cutoff=chain_cutoff,
                ligand_cutoff=ligand_cutoff,
                pair_index=pair_index,
            )
        except Exception as e:
            print(f"[warn] group {grp} failed??: {e}")
            continue

        try:
            if comp_diff[5]:
                mask_drop = (
                    (df["pdb"] == str(comp_diff[0]))
                    & (df["assembly_id"] == str(comp_diff[1]))
                    & (df["chain_auth_asm"] == str(comp_diff[2]))
                )
                df = df[~mask_drop].copy()
        except Exception:
            pass

        if df.empty or df["conf_label"].nunique() <= 1:
            n_conf = int(df["conf_label"].nunique()) if not df.empty else 0
            print(
                f"Skipping CSV creation for {out_csv}, rows={len(df)}, conf_labels={n_conf}"
            )
            if out_csv.exists():
                out_csv.unlink()
            return [], [], 0, [], []

        ligand_edges_all.extend(ligand_rows or [])
        xtal_edges_all.extend(xtal_rows or [])
        prodigy_rows_file.extend(prodigy_rows or [])
        total_pairs_file += int(n_pairs or 0)
        comp_diff_all.append(comp_diff)

        m_group = (
            (df["pdb"] == str(pdb_g))
            & (df["assembly_id"] == str(asm_g))
            & (df["chain_auth_asm"] == str(chain_auth_asm_g))
        )
        if not m_group.any():
            continue

        df.loc[m_group, "contact_chains"] = ";".join(contact_chains)
        df.loc[m_group, "contact_ligands"] = ";".join(contact_ligands)

    df.to_csv(out_csv, index=False)

    return (
        ligand_edges_all,
        xtal_edges_all,
        total_pairs_file,
        prodigy_rows_file,
        comp_diff_all,
    )


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--asm-raw-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    default="data/asms-raw",
    help="Root directory containing *_asm_raw.csv files (mirrors cluster tree from step 1).",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    default="data/asms-bio",
    help="Where to write *_asm_bio.csv files (same tree).",
)
@click.option(
    "--pair-calls-csv",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    default="data/pair-calls.csv",
    help="Precomputed Prodigy pair calls (clone-level).",
)
@click.option(
    "--assembled-cif-root",
    type=click.Path(path_type=Path, exists=False, file_okay=False),
    default="data/cif-asms",
)
@click.option(
    "--min-bio-prob",
    type=float,
    default=0.5,
    show_default=True,
    help="Keep edge as BIO only if (label_raw==BIO and p_bio >= threshold) barring overrides.",
)
@click.option(
    "--chain-cutoff",
    type=float,
    default=5.0,
    show_default=True,
    help="Cutoff (Å) for ligand proximity and ligand-mediated overrides.",
)
@click.option(
    "--ligand-cutoff",
    type=float,
    default=7.0,
    show_default=True,
    help="Cutoff (Å) for ligand proximity and ligand-mediated overrides.",
)
@click.option("--workers", type=int, default=16, show_default=True)
def main(
    asm_raw_dir: Path,
    out_dir: Path,
    pair_calls_csv: Path,
    assembled_cif_root: Optional[Path],
    min_bio_prob: float,
    chain_cutoff: float,
    ligand_cutoff: float,
    workers: int,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    asm_raw_dir = Path(asm_raw_dir)
    assembled_cif_root = (
        Path(assembled_cif_root) if assembled_cif_root is not None else None
    )

    # Load pair calls
    pc_df = pd.read_csv(pair_calls_csv, dtype=str)
    required = {
        "pdb",
        "assembly_id",
        "chain_a_auth",
        "chain_b_auth",
        "p_bio",
        "label_raw",
    }
    missing = required - set(pc_df.columns)
    if missing:
        raise SystemExit(f"[!] pair_calls CSV missing columns: {sorted(missing)}")
    pc_df["p_bio"] = pd.to_numeric(pc_df["p_bio"], errors="coerce")

    pair_index: Dict[Tuple[str, str], List[dict]] = {}
    for _, r in pc_df.iterrows():
        key = (str(r["pdb"]).strip(), str(r["assembly_id"]).strip())
        pair_index.setdefault(key, []).append(
            {
                "pdb": key[0],
                "assembly_id": key[1],
                "a_clone": str(r["chain_a_auth"]).strip(),
                "b_clone": str(r["chain_b_auth"]).strip(),
                "p_bio": float(r["p_bio"]) if pd.notna(r["p_bio"]) else 0.0,
                "label_raw": str(r["label_raw"]).strip().upper(),
            }
        )

    # rint(f"pair_index: {pair_index}")

    asm_raw_files = sorted(asm_raw_dir.rglob("*_asm_raw.csv"))
    if not asm_raw_files:
        raise SystemExit(f"[!] No *_asm_raw.csv files under {asm_raw_dir}")

    # Process all files
    results = Parallel(n_jobs=workers)(
        delayed(process_file_worker)(
            p,
            asm_raw_dir,
            out_dir,
            pair_index,
            assembled_cif_root,
            min_bio_prob,
            chain_cutoff,
            ligand_cutoff,
        )
        for p in tqdm(asm_raw_files, desc="File", unit="file")
    )

    print(f"\n[done] processed {len(asm_raw_files)} files")


if __name__ == "__main__":
    main()

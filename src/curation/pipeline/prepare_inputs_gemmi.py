from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import click
import gemmi
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..utils.constants import (
    LIGAND_EXCLUDE,
    NUCLEOTIDE_3C,
)

CSV_HEADERS = [
    "pdb",
    "chain_author",
    "chain_auth_asm",
    "assembly_id",
    "conf_label",
    "protein_count",
    "ligand_count",
    "chain_list_label",
    "chain_list_author",
    "ligands",
    "resolution",
    "experimental_method",
    "desc",
]
MMCIF_DIR: Path = Path(".")
_LABEL_BASE_RX = re.compile(r"\d+$")


def _label_base(label_or_clone: str) -> str:
    return _LABEL_BASE_RX.sub("", str(label_or_clone))


def clusters_root_rglob_csv(root: Path):
    return root.rglob("*.csv")


def _parse_member_cell(cell):
    s = cell.strip()
    if "_" in s:
        pdb, ch = s.split("_", 1)
    elif ":" in s:
        pdb, ch = s.split(":", 1)
    else:
        if len(s) >= 5:
            pdb, ch = s[:4], s[4:]
        else:
            raise ValueError(f"Bad member format: {s}")
    return pdb.lower(), ch  # ch: auth_asym_id


def read_cluster_members(csv_path):
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames or "chain" not in rdr.fieldnames:
            raise SystemExit(f"[!] {csv_path}: header must contain 'chain', 'label'.")
        for r in rdr:
            cell = (r.get("chain") or "").strip()
            if not cell:
                continue
            lab = (r.get("label") or "").strip()
            pdb, auth = _parse_member_cell(cell)
            rows.append((pdb, auth, lab))
    return rows


def load_centers_from_file(path) -> dict:
    df = pd.read_csv(path)

    def std(chain: str) -> str:
        pdb, chain_id = _parse_member_cell(chain)
        return f"{pdb.lower()}_{chain_id}"

    out = {
        str(center).lower(): sorted(
            {
                std(c)
                for c in pd.concat([g["chain-x"], g["chain-y"]], ignore_index=True)
                .dropna()
                .map(lambda s: str(s).strip())
            }
        )
        for center, g in df.groupby("center", sort=False)
    }
    return out


@dataclass
class _Meta:
    resolution: float
    method: str
    keywords: str
    block: gemmi.cif.Block


def _load_metadata(cif_path: Path) -> _Meta:
    doc = gemmi.cif.read(str(cif_path))
    blk = doc.sole_block()

    def _get_res() -> float:
        try:
            v = blk.find_value("_reflns.d_resolution_high")
            if v and v not in ("?", "."):
                return float(v)
        except Exception:
            pass
        return float("inf")

    method = blk.find_value("_exptl.method") or ""
    if method:
        method = method.strip("'")

    keywords = blk.find_value("_struct_keywords.pdbx_keywords") or ""
    if keywords:
        keywords = keywords.strip("'")

    return _Meta(resolution=_get_res(), method=method, keywords=keywords, block=blk)


def _protein_label_ids(block: gemmi.cif.Block) -> Set[str]:
    lab_ids = block.find_values("_struct_asym.id")
    ent_ids = block.find_values("_struct_asym.entity_id")
    lab2ent: Dict[str, str] = {}
    if lab_ids and ent_ids and len(lab_ids) == len(ent_ids):
        for la, eid in zip(lab_ids, ent_ids):
            if la and eid and la not in (".", "?") and eid not in (".", "?"):
                lab2ent[la] = eid

    ep_eid = block.find_values("_entity_poly.entity_id")
    ep_typ = block.find_values("_entity_poly.type")
    ent2type: Dict[str, str] = {}
    if ep_eid and ep_typ and len(ep_eid) == len(ep_typ):
        for eid, ty in zip(ep_eid, ep_typ):
            if eid and ty:
                ent2type[eid] = ty

    prot_labels: Set[str] = set()
    for la, eid in lab2ent.items():
        ty = (ent2type.get(eid) or "").lower()
        if "polypeptide" in ty:
            prot_labels.add(la)

    return prot_labels


def _label_to_auth_map(block: gemmi.cif.Block) -> Dict[str, str]:
    protein_labels = _protein_label_ids(block)
    if not protein_labels:
        return {}

    labs = block.find_values("_atom_site.label_asym_id")
    auth = block.find_values("_atom_site.auth_asym_id")
    if not labs or not auth or len(labs) != len(auth):
        return {}

    seen_auths: Dict[str, Set[str]] = defaultdict(set)
    for la, au in zip(labs, auth):
        if not la or la in (".", "?"):
            continue
        if la not in protein_labels:
            continue
        if not au or au in (".", "?"):
            continue
        seen_auths[la].add(au)

    mapping: Dict[str, str] = {}
    for la, aus in seen_auths.items():
        if len(aus) == 0:
            continue
        if len(aus) > 1:
            raise ValueError(f"label '{la}' mapped to multiple auths: {sorted(aus)}")
        mapping[la] = next(iter(aus))

    return mapping


def _label_base_to_entity_poly_type(block: gemmi.cif.Block) -> Dict[str, str]:
    label_ids = block.find_values("_struct_asym.id")
    ent_ids = block.find_values("_struct_asym.entity_id")
    lab2ent: Dict[str, str] = {}
    if label_ids and ent_ids and len(label_ids) == len(ent_ids):
        for la, eid in zip(label_ids, ent_ids):
            if la and eid and la not in ("?", ".") and eid not in ("?", "."):
                lab2ent[la] = eid

    ep_entity_id = block.find_values("_entity_poly.entity_id")
    ep_type = block.find_values("_entity_poly.type")
    ent2type: Dict[str, str] = {}
    if ep_entity_id and ep_type and len(ep_entity_id) == len(ep_type):
        for eid, ty in zip(ep_entity_id, ep_type):
            if eid and ty:
                ent2type[eid] = ty

    labbase2type: Dict[str, str] = {}
    for la, eid in lab2ent.items():
        ty = ent2type.get(eid, "")
        labbase2type[_label_base(la)] = ty
    return labbase2type


def _assembly_rows(block: gemmi.cif.Block) -> List[Dict[str, str]]:
    tbl = block.find_mmcif_category("_pdbx_struct_assembly")
    if not tbl:
        return []
    full_tags = [str(t) for t in tbl.tags]
    short_keys = [t.split(".")[-1] for t in full_tags]

    out: List[Dict[str, str]] = []
    for row in tbl:
        rec: Dict[str, str] = {}
        for short in short_keys:
            rec[short] = row[short]
        out.append(rec)
    return out


def fetch_assemblies(
    pdb_code: str,
    cif_path: Path,
    save_dir: Path,
    resolution_cutoff: float,
    max_polymer_instances: int,
    max_lig_instances: int,
    exclude_na: bool = True,
):
    try:
        meta = _load_metadata(cif_path)
        if not (meta.resolution <= resolution_cutoff):
            return []

        st_asu = gemmi.read_structure(str(cif_path))
        st_asu.remove_hydrogens()

        labbase2poly = _label_base_to_entity_poly_type(meta.block)

        asm_rows = _assembly_rows(meta.block)
        results: List[dict] = []

        for row in asm_rows:
            asm_id = row.get("id", "")
            details = (row.get("details") or "").lower()
            if not asm_id:
                continue

            if "author" not in details:
                continue

            try:
                st_asm = st_asu.clone()
                st_asm.transform_to_assembly(
                    assembly_name=asm_id, how=gemmi.HowToNameCopiedChain.AddNumber
                )
            except Exception as e:
                print(
                    f"[warn] transform_to_assembly failed for {pdb_code} asm {asm_id}: {e}"
                )
                continue

            mdl = st_asm[0]
            clone_chain_names: List[str] = []
            for ch in mdl.subchains():
                clone_chain_names.append(ch.subchain_id())

            polymer_clones: List[str] = []
            for cid in clone_chain_names:
                base = _label_base(cid)
                poly_type = labbase2poly.get(base, "")
                if "polypeptide" in poly_type:
                    polymer_clones.append(cid)

            if exclude_na:
                has_na = False
                for base, poly_type in labbase2poly.items():
                    if poly_type and "nucleotide" in poly_type.lower():
                        if any(_label_base(c) == base for c in clone_chain_names):
                            has_na = True
                            break
                if has_na:
                    continue

            if len(polymer_clones) > max_polymer_instances:
                continue

            mdl = st_asm[0]

            auth_to_labels: Dict[str, set[str]] = defaultdict(set)

            for ch in mdl:
                auth_clone = ch.name
                local_labels: set[str] = set()
                for res in ch:
                    label_clone = res.subchain
                    if not label_clone:
                        continue

                    base = _label_base(label_clone)
                    poly_type = (labbase2poly.get(base) or "").lower()
                    if "polypeptide" not in poly_type:
                        continue
                    local_labels.add(label_clone)

                if local_labels:
                    auth_to_labels[auth_clone].update(local_labels)

            assembly_auth_to_mmcif: Dict[str, List[str]] = {
                au: sorted(list(labels)) for au, labels in auth_to_labels.items()
            }

            chain_list_author = sorted(assembly_auth_to_mmcif.keys())
            chain_list_label = sorted(
                {lb for lbs in assembly_auth_to_mmcif.values() for lb in lbs}
            )

            if len(chain_list_label) > max_polymer_instances:
                continue

            for au, clones in assembly_auth_to_mmcif.items():
                if len(clones) > 1:
                    print(
                        f"WARNING: Auth chain '{au}' maps to multiple clones: {clones} in assembly {asm_id}"
                    )

            ligand_instances: set[tuple] = set()

            for ch in mdl:
                clone_id = ch.name
                for res in ch:
                    if res.het_flag != "H":
                        continue
                    comp_id = (res.name or "").strip()
                    if not comp_id:
                        continue
                    if comp_id in LIGAND_EXCLUDE or comp_id in NUCLEOTIDE_3C:
                        continue

                    key = (comp_id, clone_id, res.seqid.num, res.seqid.icode)
                    ligand_instances.add(key)

            instances_by_comp: dict[str, set[tuple]] = defaultdict(set)
            for comp_id, clone_id, num, icode in ligand_instances:
                instances_by_comp[comp_id].add((clone_id, num, icode))

            ligand_list: list[str] = [
                comp_id
                for comp_id, insts in instances_by_comp.items()
                for _ in range(len(insts))
            ]
            total_lig_instances = sum(
                len(insts) for insts in instances_by_comp.values()
            )

            if total_lig_instances > max_lig_instances:
                continue

            results.append(
                {
                    "assembly": st_asm,
                    "assembly_id": str(asm_id),
                    "bio_assembly": row,
                    "chain_list_label": chain_list_label,
                    "chain_list_author": chain_list_author,
                    "ligand_list": ligand_list,
                    "polymer_count": len(polymer_clones),
                    "ligand_count": len(ligand_list),
                    "resolution": meta.resolution,
                    "method": meta.method,
                    "desc": meta.keywords,
                    "assembly_auth_to_mmcif": dict(assembly_auth_to_mmcif),
                }
            )

        return results

    except Exception as e:
        print(f"Error processing {pdb_code}: {e}", file=sys.stderr)
        return []


def check_auth_base(auth: str, auth_chain: str) -> bool:
    if not auth_chain.startswith(auth):
        return False

    suffix = auth_chain[len(auth) :]
    if suffix == "":
        return True
    if re.fullmatch(r"[1-9][0-9]?", suffix):
        return True
    return False


def process_cluster_file(
    in_csv,
    clusters_root,
    out_root,
    center_chain_map,
    save_assemblies_dir,
    max_polymer_instances,
    max_lig_instances,
    max_resolution,
    exclude_na,
):
    members = read_cluster_members(in_csv)
    rel = in_csv.relative_to(clusters_root)
    out_csv = out_root / rel.parent / (in_csv.stem + "_asm_raw.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not members:
        return str(out_csv), 0

    per_pdb_assemblies = {}
    out_rows = []

    for pdb, auth, conf_label in members:
        chain = pdb.lower() + "_" + auth
        if chain not in center_chain_map.get(in_csv.stem.lower(), []):
            print(f"Skipping {chain} not in centers_file for {in_csv.stem.lower()}")
            continue
        cif_path = MMCIF_DIR / f"{pdb.lower()}.cif"
        if not cif_path.exists():
            print(f"[!] Missing CIF: {cif_path} (skip {pdb})")
            per_pdb_assemblies[pdb] = []
            continue

        save_dir = (
            save_assemblies_dir
            if save_assemblies_dir is None
            else Path(save_assemblies_dir)
        )
        assemblies_filtered = fetch_assemblies(
            pdb,
            cif_path,
            save_dir,
            resolution_cutoff=max_resolution,
            max_polymer_instances=max_polymer_instances,
            max_lig_instances=max_lig_instances,
            exclude_na=exclude_na,
        )
        per_pdb_assemblies[pdb] = assemblies_filtered  # asms per pdb

    for pdb, auth, conf_label in members:
        chain = pdb.lower() + "_" + auth
        if chain not in center_chain_map.get(in_csv.stem.lower(), []):
            continue
        assemblies_filtered = per_pdb_assemblies[pdb]
        matched_assemblies = []

        for asm_data in assemblies_filtered:  # pdb
            assembly = asm_data["assembly"]  # st_asm
            assembly_id = asm_data["assembly_id"]
            assembly_auth_to_mmcif = asm_data["assembly_auth_to_mmcif"]
            assembly_chain_list_auth = asm_data["chain_list_author"]

            for auth_chain in assembly_chain_list_auth:
                if check_auth_base(auth, auth_chain):  # auth in auth_chain:
                    auth_asm = auth_chain
                    matched_assemblies.append(str(assembly_id))
                    break
            else:
                continue  # not found, skip to next assembly

            # drop homomer duplicates: only if everything is same except auth_asm
            duplicate = False

            def list_to_str(lst):
                if not lst:
                    return ""
                return ";".join(str(x) for x in lst)

            for asm in out_rows:
                if (
                    asm[3] == str(asm_data["assembly_id"])
                    and asm[4] == str(conf_label)
                    and asm[7] == list_to_str(asm_data["chain_list_label"])
                    and asm[8] == list_to_str(asm_data["chain_list_author"])
                    and asm[9] == list_to_str(asm_data["ligand_list"])
                    and asm[2] != str(auth_asm)
                ):
                    print(
                        f"[warn] {pdb} chain {auth_asm} (conf_label={conf_label}) homomer duplicate: {asm[2]} {auth_asm}"
                    )
                    duplicate = True
                    break
            if duplicate:
                continue

            # save assembly CIF file
            pdb_upper = pdb.upper()
            rep = pdb_upper[1:3]
            cif_file = save_dir / rep / pdb_upper / f"asm_{pdb}_{assembly_id}.cif"
            cif_file.parent.mkdir(parents=True, exist_ok=True)
            assembly.make_mmcif_block().write_file(str(cif_file))

            map_file = save_dir / rep / pdb_upper / f"asm_{pdb}_{assembly_id}_map.json"
            with map_file.open("w", encoding="utf-8") as f:
                json.dump(assembly_auth_to_mmcif, f, indent=2, ensure_ascii=False)

            out_rows.append(
                [
                    pdb,
                    auth,
                    auth_asm,
                    str(assembly_id),
                    str(conf_label),
                    int(asm_data["polymer_count"]),
                    int(asm_data["ligand_count"]),
                    list_to_str(asm_data["chain_list_label"]),
                    list_to_str(asm_data["chain_list_author"]),
                    list_to_str(asm_data["ligand_list"]),
                    float(asm_data["resolution"]),
                    str(asm_data["method"]),
                    str(asm_data["desc"]),
                ]
            )
        if len(matched_assemblies) > 1:
            print(
                f"[warn] {pdb} chain {auth} (conf_label={conf_label}) multiple matched assemblies: {', '.join(matched_assemblies)}"
            )

    if not out_rows or len({r[4] for r in out_rows}) <= 1:
        print(
            f"Skipping CSV creation for {out_csv}, {len(out_rows)} rows, {len({r[4] for r in out_rows})} conf_label"
        )
        if out_csv.exists():
            out_csv.unlink()
        return str(out_csv), 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADERS)
        w.writerows(out_rows)

    return str(out_csv), len(out_rows)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--mmcif-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Directory containing PDB mmCIF files (external input).",
)
@click.option(
    "--clusters-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    default="data/clusters",
    show_default=True,
    help="Root directory of cluster CSV files.",
)
@click.option(
    "--out-root",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    default="data/asms-raw",
)
@click.option("--save-assemblies", default=True, show_default=True)
@click.option(
    "--save-assemblies-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default="data/cif-asms",
)
@click.option("--max-polymer-instances", type=int, default=12)
@click.option("--max-lig-instances", type=int, default=12)
@click.option("--max-resolution", type=float, default=5.0)
@click.option("--exclude-na", default=True)
@click.option("--workers", type=int, default=1)
@click.option(
    "--centers-file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    default="data/filtered-pairs.csv",
    show_default=True,
    help="CSV with cluster centers / chain pairs.",
)
def main(
    mmcif_dir,
    clusters_root,
    out_root,
    save_assemblies,
    save_assemblies_dir,
    max_polymer_instances,
    max_lig_instances,
    max_resolution,
    exclude_na,
    workers,
    centers_file,
):
    global MMCIF_DIR
    MMCIF_DIR = Path(mmcif_dir)
    clusters_root = Path(clusters_root)
    out_root = Path(out_root)
    files = sorted(clusters_root_rglob_csv(clusters_root))

    save_dir = Path(save_assemblies_dir) if save_assemblies else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    if centers_file:
        center_chain_map = load_centers_from_file(centers_file)
        allowed = set(center_chain_map.keys())
        # print(allowed)
        if not allowed:
            click.echo("[!] No centers in centers_files.")
            raise SystemExit(1)
        before = len(files)
        files = [p for p in files if p.stem.lower() in allowed]
        after = len(files)
        miss = len(allowed - {p.stem.lower() for p in files})
        click.echo(
            f"[i] filter centers : {before} -> {after} files (Failed matchings: {miss})"
        )
        if not files:
            click.echo(f"[!] No matching csv with centers in {clusters_root}.")
            raise SystemExit(1)

    if not files:
        click.echo(f"[!] No CSV under {clusters_root}")
        raise SystemExit(1)

    results = Parallel(n_jobs=workers)(
        delayed(process_cluster_file)(
            p,
            clusters_root,
            out_root,
            center_chain_map,
            save_dir,
            max_polymer_instances,
            max_lig_instances,
            max_resolution,
            exclude_na,
        )
        for p in tqdm(files, desc="Clusters", unit="file")
    )
    for out_csv, n in results:
        click.echo(f"[fetch] {out_csv} ({n} rows)")


if __name__ == "__main__":
    main()

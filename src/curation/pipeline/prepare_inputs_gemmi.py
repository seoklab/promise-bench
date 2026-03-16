from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import click
import gemmi
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..utils.constants import (
    LIGAND_EXCLUDE,
    NUCLEOTIDE_3C,
)
from utils._config import pipeline_cfg as C
from utils._data_root import DataRootCommand
from ..utils._geometry import coords_and_ligs_from_model, get_contact_chains, get_contact_ligands
from ..utils._pdb_helpers import check_auth_base, label_base, parse_member_cell
from .types import AssemblyEntry, AssemblyResult, CifMeta


def _rglob_csv(root: Path):
    return root.rglob("*.csv")


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
            pdb, auth = parse_member_cell(cell)
            rows.append((pdb, auth, lab))
    return rows


def load_centers_from_file(path) -> dict:
    df = pd.read_csv(path)

    def std(chain: str) -> str:
        pdb, chain_id = parse_member_cell(chain)
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


def _load_cif_meta(cif_path: Path) -> CifMeta:
    """Read an mmCIF file and return structured metadata."""
    doc = gemmi.cif.read(str(cif_path))
    blk = doc.sole_block()

    def _get_res() -> float:
        for tag in (
            "_reflns.d_resolution_high",
            "_em_3d_reconstruction.resolution",
            "_refine.ls_d_res_high",
        ):
            try:
                v = blk.find_value(tag)
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

    return CifMeta(resolution=_get_res(), method=method, keywords=keywords, block=blk)


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
        meta = _load_cif_meta(cif_path)
        if not (meta.resolution <= resolution_cutoff):
            return []

        st_asu = gemmi.read_structure(str(cif_path))
        st_asu.remove_hydrogens()

        labbase2poly = meta.label_base_to_entity_poly_type()

        asm_rows = meta.assembly_rows()
        results: list[AssemblyResult] = []

        for row in asm_rows:
            if not row.id or not row.is_author_defined:
                continue

            asm_id = row.id

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
            clone_chain_names: list[str] = []
            for ch in mdl.subchains():
                clone_chain_names.append(ch.subchain_id())

            polymer_clones: list[str] = []
            for cid in clone_chain_names:
                base = label_base(cid)
                poly_type = labbase2poly.get(base, "")
                if "polypeptide" in poly_type:
                    polymer_clones.append(cid)

            if exclude_na:
                has_na = False
                for base, poly_type in labbase2poly.items():
                    if poly_type and "nucleotide" in poly_type.lower():
                        if any(label_base(c) == base for c in clone_chain_names):
                            has_na = True
                            break
                if has_na:
                    continue

            if len(polymer_clones) > max_polymer_instances:
                continue

            mdl = st_asm[0]

            auth_to_labels: dict[str, set[str]] = defaultdict(set)

            for ch in mdl:
                auth_clone = ch.name
                local_labels: set[str] = set()
                for res in ch:
                    label_clone = res.subchain
                    if not label_clone:
                        continue

                    base = label_base(label_clone)
                    poly_type = (labbase2poly.get(base) or "").lower()
                    if "polypeptide" not in poly_type:
                        continue
                    local_labels.add(label_clone)

                if local_labels:
                    auth_to_labels[auth_clone].update(local_labels)

            assembly_auth_to_mmcif: dict[str, list[str]] = {
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

            chain_coords, lig_instances = coords_and_ligs_from_model(mdl)

            results.append(
                AssemblyResult(
                    assembly=st_asm,
                    assembly_id=str(asm_id),
                    bio_assembly=row,
                    chain_list_label=chain_list_label,
                    chain_list_author=chain_list_author,
                    ligand_list=ligand_list,
                    polymer_count=len(polymer_clones),
                    ligand_count=len(ligand_list),
                    resolution=meta.resolution,
                    method=meta.method,
                    desc=meta.keywords,
                    assembly_auth_to_mmcif=dict(assembly_auth_to_mmcif),
                    chain_coords=chain_coords,
                    lig_instances=lig_instances,
                )
            )

        return results

    except Exception as e:
        print(f"Error processing {pdb_code}: {e}", file=sys.stderr)
        return []


def process_cluster_file(
    in_csv,
    clusters_root,
    out_root,
    center_chain_map,
    save_assemblies_dir,
    mmcif_dir,
    max_polymer_instances,
    max_lig_instances,
    max_resolution,
    exclude_na,
    chain_cutoff,
    ligand_cutoff,
):
    members = read_cluster_members(in_csv)
    rel = in_csv.relative_to(clusters_root)
    out_csv = out_root / rel.parent / (in_csv.stem + "_asm_raw.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not members:
        return str(out_csv), 0

    per_pdb_assemblies = {}
    out_entries: list[AssemblyEntry] = []

    for pdb, auth, conf_label in members:
        chain = pdb.lower() + "_" + auth
        if chain not in center_chain_map.get(in_csv.stem.lower(), []):
            print(f"Skipping {chain} not in centers_file for {in_csv.stem.lower()}")
            continue
        cif_path = mmcif_dir / f"{pdb.lower()}.cif"
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

        for asm_result in assemblies_filtered:
            for auth_chain in asm_result.chain_list_author:
                if check_auth_base(auth, auth_chain):
                    auth_asm = auth_chain
                    matched_assemblies.append(asm_result.assembly_id)
                    break
            else:
                continue  # not found, skip to next assembly

            # build typed entry with contacts computed here
            contact_ch = get_contact_chains(
                auth_asm, asm_result.chain_coords, chain_cutoff
            )
            contact_lig = get_contact_ligands(
                asm_result.chain_coords.get(auth_asm, np.empty((0, 3))),
                asm_result.lig_instances,
                ligand_cutoff,
            )
            entry = asm_result.to_entry(
                pdb=pdb,
                chain_author=auth,
                chain_auth_asm=auth_asm,
                conf_label=str(conf_label),
                contact_chains=contact_ch,
                contact_ligands=contact_lig,
            )

            # drop homomer duplicates
            if any(entry.is_homomer_duplicate(e) for e in out_entries):
                print(
                    f"[warn] {pdb} chain {auth_asm} (conf_label={conf_label}) homomer duplicate"
                )
                continue

            # save assembly CIF file (skip if already exists)
            pdb_upper = pdb.upper()
            rep = pdb_upper[1:3]
            cif_file = save_dir / rep / pdb_upper / f"asm_{pdb}_{asm_result.assembly_id}.cif"
            map_file = save_dir / rep / pdb_upper / f"asm_{pdb}_{asm_result.assembly_id}_map.json"

            if not (cif_file.exists() and map_file.exists()):
                cif_file.parent.mkdir(parents=True, exist_ok=True)
                asm_result.assembly.make_mmcif_block().write_file(str(cif_file))
                with map_file.open("w", encoding="utf-8") as f:
                    json.dump(asm_result.assembly_auth_to_mmcif, f, indent=2, ensure_ascii=False)

            out_entries.append(entry)

        if len(matched_assemblies) > 1:
            print(
                f"[warn] {pdb} chain {auth} (conf_label={conf_label}) multiple matched assemblies: {', '.join(matched_assemblies)}"
            )

    if not out_entries or len({e.conf_label for e in out_entries}) <= 1:
        print(
            f"Skipping CSV creation for {out_csv}, {len(out_entries)} rows, "
            f"{len({e.conf_label for e in out_entries})} conf_label"
        )
        if out_csv.exists():
            out_csv.unlink()
        return str(out_csv), 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(AssemblyEntry.csv_headers())
        w.writerows(e.to_csv_row() for e in out_entries)

    return str(out_csv), len(out_entries)


@click.command(
    cls=DataRootCommand, context_settings=dict(help_option_names=["-h", "--help"])
)
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
    default=str(C.dir("clusters")),
    show_default=True,
    help="Root directory of cluster CSV files.",
)
@click.option(
    "--out-root",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    default=str(C.dir("asms_raw")),
)
@click.option("--save-assemblies", default=True, show_default=True)
@click.option(
    "--save-assemblies-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=str(C.dir("cif_asms")),
)
@click.option("--max-polymer-instances", type=int, default=12)
@click.option("--max-lig-instances", type=int, default=12)
@click.option("--max-resolution", type=float, default=5.0)
@click.option("--exclude-na", default=True)
@click.option("--chain-cutoff", type=float, default=5.0, show_default=True,
              help="Cutoff (Å) for contact-chain proximity.")
@click.option("--ligand-cutoff", type=float, default=7.0, show_default=True,
              help="Cutoff (Å) for contact-ligand proximity.")
@click.option("--workers", type=int, default=8)
@click.option(
    "--centers-file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    default=str(C.file("filtered_pairs")),
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
    chain_cutoff,
    ligand_cutoff,
    workers,
    centers_file,
):
    mmcif_dir = Path(mmcif_dir)
    clusters_root = Path(clusters_root)
    out_root = Path(out_root)
    files = sorted(_rglob_csv(clusters_root))

    save_dir = Path(save_assemblies_dir) if save_assemblies else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    if centers_file:
        center_chain_map = load_centers_from_file(centers_file)
        allowed = set(center_chain_map.keys())
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
            mmcif_dir,
            max_polymer_instances,
            max_lig_instances,
            max_resolution,
            exclude_na,
            chain_cutoff,
            ligand_cutoff,
        )
        for p in tqdm(files, desc="Clusters", unit="file")
    )
    skipped = sum(1 for _, n in results if n == -1)
    for out_csv, n in results:
        if n == -1:
            continue
        click.echo(f"[fetch] {out_csv} ({n} rows)")
    click.echo(f"[done] {len(results)} clusters, {skipped} skipped (already exist)")


if __name__ == "__main__":
    main()

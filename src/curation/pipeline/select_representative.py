from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import click
import pandas as pd
from Bio.PDB import MMCIFParser, NeighborSearch
from joblib import Parallel, delayed

from ..utils._config import pipeline_cfg as C
from ..utils._data_root import DataRootCommand
from ..utils.constants import AA_3TO1
from .types import DatasetPair

CONTACT_DISTANCE = 5.0


def parse_a3m(path: Path) -> Dict[str, str]:
    """Return {header: aligned_sequence} from an a3m file.

    Lowercase letters (insertions) are stripped so that every sequence has the
    same number of alignment columns.
    """
    seqs: Dict[str, str] = {}
    hdr: Optional[str] = None
    buf: list[str] = []
    with path.open() as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if hdr is not None:
                    seqs[hdr] = re.sub(r"[a-z]", "", "".join(buf))
                hdr = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
        if hdr is not None:
            seqs[hdr] = re.sub(r"[a-z]", "", "".join(buf))
    return seqs


def _find_header(a3m: Dict[str, str], pdb: str, chain_letter: str) -> Optional[str]:
    """Find best matching header for pdb_chain in MSA."""
    targets = [
        f"{pdb.lower()}_{chain_letter}",
        f"{pdb.lower()}_{chain_letter.upper()}",
        f"{pdb.lower()}_{chain_letter.lower()}",
        f"{pdb.upper()}_{chain_letter}",
    ]
    for h in a3m:
        for t in targets:
            if h == t or h.startswith(t + " ") or h.startswith(t + "\t"):
                return h
    return None


def _ungapped(seq: str) -> str:
    return seq.replace("-", "")


_cif_parser = MMCIFParser(QUIET=True)


def _get_cif_path(cif_root: Path, pdb: str, asm: str) -> Path:
    prefix = pdb.upper()[1:3]
    return cif_root / prefix / pdb.upper() / f"asm_{pdb.lower()}_{asm}.cif"


def _extract_binding_site(
    cif_path: Path,
    target_chain: str,
    partner_type: str,  # "protein" | "ligand"
    partner_ids: List[str],  # chain ids  or  ligand comp_ids
) -> Tuple[List[int], Dict[int, str]]:
    """Return (sorted_resnums, {resnum: 1-letter AA}) of target chain residues
    within CONTACT_DISTANCE of partner atoms."""
    try:
        structure = _cif_parser.get_structure("s", str(cif_path))
    except Exception:
        return [], {}

    target_residues: list = []
    resnum_aa: Dict[int, str] = {}
    partner_atoms: list = []

    for model in structure:
        for chain in model:
            cid = chain.get_id()
            if cid == target_chain:
                for res in chain:
                    if res.get_id()[0] == " ":
                        target_residues.append(res)
                        rn = res.get_id()[1]
                        resnum_aa[rn] = AA_3TO1.get(res.get_resname(), "X")

            if partner_type == "protein":
                if cid in partner_ids:
                    for res in chain:
                        if res.get_id()[0] == " ":
                            for a in res.get_atoms():
                                if a.element != "H":
                                    partner_atoms.append(a)
            else:  # ligand
                for res in chain:
                    if (
                        res.get_id()[0].startswith("H_")
                        and res.get_resname() in partner_ids
                    ):
                        for a in res.get_atoms():
                            if a.element != "H":
                                partner_atoms.append(a)
        break  # first model only

    if not target_residues or not partner_atoms:
        return [], {}

    ns = NeighborSearch(partner_atoms)
    bs: Set[int] = set()
    for res in target_residues:
        for a in res:
            if a.element == "H":
                continue
            if ns.search(a.coord, CONTACT_DISTANCE):
                bs.add(res.get_id()[1])
                break

    bs_aa = {r: resnum_aa[r] for r in bs if r in resnum_aa}
    return sorted(bs), bs_aa


def _get_chain_seq(cif_path: Path, chain_id: str) -> Tuple[str, List[int]]:
    """Return (1-letter sequence string, sorted resnums list)."""
    try:
        structure = _cif_parser.get_structure("s", str(cif_path))
    except Exception:
        return "", []
    seq, resnums = [], []
    for model in structure:
        for ch in model:
            if ch.get_id() == chain_id:
                for res in ch:
                    if res.get_id()[0] == " ":
                        resnums.append(res.get_id()[1])
                        seq.append(AA_3TO1.get(res.get_resname(), "X"))
                break
        break
    return "".join(seq), resnums


def _map_resnums_to_msa_cols(
    struct_seq: str,
    struct_resnums: List[int],
    aligned_seq: str,
) -> Dict[int, int]:
    """Map structure residue numbers to 0-based MSA column indices.

    `aligned_seq` comes from the a3m (uppercase + '-').
    `struct_seq` is the 1-letter sequence extracted from the CIF.
    """
    ungapped = _ungapped(aligned_seq)

    # Build ungapped-position to MSA-column map
    ungap_to_col: Dict[int, int] = {}
    ui = 0
    for col, ch in enumerate(aligned_seq):
        if ch != "-":
            ungap_to_col[ui] = col
            ui += 1

    # If struct_seq length == ungapped length, direct 1:1
    if len(struct_seq) == len(ungapped):
        return {
            rn: ungap_to_col[i]
            for i, rn in enumerate(struct_resnums)
            if i in ungap_to_col
        }

    # Otherwise fall back to pairwise alignment
    try:
        from Bio.Align import PairwiseAligner

        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -10
        aligner.extend_gap_score = -0.5
        aln = aligner.align(struct_seq, ungapped)[0]
        a_s, a_t = str(aln[0]), str(aln[1])
    except Exception:
        return {}

    si, ti = 0, 0
    rn_to_ungap: Dict[int, int] = {}
    for k in range(len(a_s)):
        if a_s[k] != "-" and a_t[k] != "-":
            if si < len(struct_resnums):
                rn_to_ungap[struct_resnums[si]] = ti
            si += 1
            ti += 1
        elif a_s[k] != "-":
            si += 1
        else:
            ti += 1

    return {
        rn: ungap_to_col[up] for rn, up in rn_to_ungap.items() if up in ungap_to_col
    }


@dataclass
class EntryInfo:
    pdb: str
    asm: str
    chain_auth_asm: str  # e.g. "A1"
    chain_letter: str  # e.g. "A"
    header: str = ""  # MSA header
    aligned_seq: str = ""
    ungapped_seq: str = ""

    # binding site: MSA-column to 1-letter AA
    bs_col_aa: Dict[int, str] = field(default_factory=dict)

    has_binding_site: bool = False
    error: str = ""


def _process_cluster(
    cluster_stem: str,
    entry_dicts: List[dict],
    msa_root: Path,
    cif_root: Path,
) -> Optional[dict]:
    """Return dict with representative info + per-entry compatibility.

    ``entry_dicts`` is a list of dicts with keys:
        pdb, asm, chain (chain_auth_asm), contact_chains, contact_ligands
    collected from the pair CSVs.
    """

    prefix = cluster_stem[1:3]
    a3m_path = msa_root / prefix / f"{cluster_stem}.a3m"
    if not a3m_path.exists():
        return None

    a3m = parse_a3m(a3m_path)
    if not a3m:
        return None

    if not entry_dicts:
        return None

    entries: List[EntryInfo] = []

    for ed in entry_dicts:
        pdb = ed["pdb"]
        asm = ed["asm"]
        chain_asm = ed["chain"]
        chain_letter = re.match(r"([A-Za-z]+)", chain_asm)
        chain_letter = chain_letter.group(1) if chain_letter else chain_asm

        e = EntryInfo(
            pdb=pdb, asm=asm, chain_auth_asm=chain_asm, chain_letter=chain_letter
        )

        # Find in MSA
        hdr = _find_header(a3m, pdb, chain_letter)
        if hdr is None:
            e.error = "not_in_msa"
            entries.append(e)
            continue
        e.header = hdr
        e.aligned_seq = a3m[hdr]
        e.ungapped_seq = _ungapped(e.aligned_seq)

        # Determine binding partner
        contact_chains_raw = ed.get("contact_chains", "")
        contact_ligs_raw = ed.get("contact_ligands", "")

        has_protein_partner = bool(contact_chains_raw)
        has_ligand_partner = bool(contact_ligs_raw)

        if not has_protein_partner and not has_ligand_partner:
            e.has_binding_site = False
            entries.append(e)
            continue

        # CIF path
        cif_path = _get_cif_path(cif_root, pdb, asm)
        if not cif_path.exists():
            e.error = "cif_missing"
            entries.append(e)
            continue

        # Extract binding-site residues
        all_bs_resnums: Set[int] = set()
        all_bs_aa: Dict[int, str] = {}

        if has_protein_partner:
            partner_ids = [
                c.strip() for c in contact_chains_raw.split(";") if c.strip()
            ]
            rn, aa = _extract_binding_site(cif_path, chain_asm, "protein", partner_ids)
            all_bs_resnums.update(rn)
            all_bs_aa.update(aa)

        if has_ligand_partner:
            partner_ids = [c.strip() for c in contact_ligs_raw.split(";") if c.strip()]
            rn, aa = _extract_binding_site(cif_path, chain_asm, "ligand", partner_ids)
            all_bs_resnums.update(rn)
            all_bs_aa.update(aa)

        if not all_bs_resnums:
            e.error = "no_bs_residues"
            entries.append(e)
            continue

        # Drop entries with modified (non-standard) residues in binding site
        if any(aa == "X" for aa in all_bs_aa.values()):
            e.error = "modified_residue_in_bs"
            entries.append(e)
            continue

        # Map resnums to MSA columns
        struct_seq, struct_resnums = _get_chain_seq(cif_path, chain_asm)
        if not struct_seq:
            e.error = "no_struct_seq"
            entries.append(e)
            continue

        rn_to_col = _map_resnums_to_msa_cols(struct_seq, struct_resnums, e.aligned_seq)

        bs_col_aa: Dict[int, str] = {}
        for rn in sorted(all_bs_resnums):
            if rn in rn_to_col and rn in all_bs_aa:
                bs_col_aa[rn_to_col[rn]] = all_bs_aa[rn]

        e.bs_col_aa = bs_col_aa
        e.has_binding_site = len(bs_col_aa) > 0
        entries.append(e)

    valid = [e for e in entries if e.header and not e.error]
    if not valid:
        return None

    # For each candidate, compute support = number of other entries with
    # no binding-site mismatch.
    def _compatible(cand: EntryInfo, other: EntryInfo) -> bool:
        """Check if `other`'s binding-site residues match `cand`'s sequence."""
        if not other.has_binding_site:
            return True  # no binding site, always compatible
        cand_seq = cand.aligned_seq
        for col, aa in other.bs_col_aa.items():
            if col >= len(cand_seq):
                return False
            cand_aa = cand_seq[col]
            if cand_aa == "-":
                return False  # gap at binding-site position
            if cand_aa != aa:
                return False
        return True

    best_cand: Optional[EntryInfo] = None
    best_support = -1

    for cand in valid:
        support = sum(1 for o in valid if o is not cand and _compatible(cand, o))
        if support > best_support or (
            support == best_support
            and best_cand is not None
            and len(cand.ungapped_seq) > len(best_cand.ungapped_seq)
        ):
            best_support = support
            best_cand = cand

    if best_cand is None:
        return None

    compat_keys: Set[str] = set()  # "pdb_asm_chain"
    incompat_keys: Set[str] = set()
    compat_rows: List[dict] = []

    for e in entries:
        key = f"{e.pdb}_{e.asm}_{e.chain_letter}"
        if e.error:
            incompat_keys.add(key)
            compat_rows.append(
                {
                    "cluster": cluster_stem,
                    "pdb": e.pdb,
                    "asm": e.asm,
                    "chain": e.chain_letter,
                    "compatible": False,
                    "reason": e.error,
                }
            )
            continue
        ok = _compatible(best_cand, e)
        if ok:
            compat_keys.add(key)
        else:
            incompat_keys.add(key)
        compat_rows.append(
            {
                "cluster": cluster_stem,
                "pdb": e.pdb,
                "asm": e.asm,
                "chain": e.chain_letter,
                "compatible": ok,
                "reason": "" if ok else "bs_mismatch",
            }
        )

    return {
        "cluster": cluster_stem,
        "rep_header": best_cand.header,
        "rep_seq": best_cand.ungapped_seq,
        "rep_length": len(best_cand.ungapped_seq),
        "support": best_support,
        "total_entries": len(valid),
        "compat_keys": compat_keys,
        "incompat_keys": incompat_keys,
        "compat_rows": compat_rows,
    }


def _clean(val: str) -> str:
    v = str(val).strip()
    return "" if v in ("", "nan", "NaN", "None", "NONE") else v


def _collect_entries(dataset_dir: Path) -> Dict[str, List[dict]]:
    """Read all pair CSVs and return {cluster_stem: [entry_dict, ...]}.

    Each entry_dict has keys: pdb, asm, chain, contact_chains, contact_ligands.
    Entries are deduplicated per cluster by (pdb, asm, chain).
    """
    cluster_entries: Dict[str, Dict[str, dict]] = defaultdict(dict)

    for csv_path in sorted(dataset_dir.glob("*.csv")):
        try:
            pairs = DatasetPair.load_csv(csv_path)
        except Exception:
            continue
        if not pairs:
            continue

        for pair in pairs:
            cluster_stem = pair.cluster_stem
            if not cluster_stem:
                continue

            for side in (pair.a, pair.b):
                pdb = side.pdb.lower()
                asm = side.assembly_id
                chain = side.chain
                cc = side.contact_chains
                cl = side.contact_ligands
                if not pdb or not chain:
                    continue

                key = f"{pdb}_{asm}_{chain}"
                if key not in cluster_entries[cluster_stem]:
                    cluster_entries[cluster_stem][key] = {
                        "pdb": pdb,
                        "asm": asm,
                        "chain": chain,
                        "contact_chains": cc,
                        "contact_ligands": cl,
                    }
                else:
                    # merge contact info (union) in case different pairs
                    # contribute different partners
                    existing = cluster_entries[cluster_stem][key]
                    if cc and cc not in existing["contact_chains"]:
                        prev = existing["contact_chains"]
                        existing["contact_chains"] = f"{prev};{cc}" if prev else cc
                    if cl and cl not in existing["contact_ligands"]:
                        prev = existing["contact_ligands"]
                        existing["contact_ligands"] = f"{prev};{cl}" if prev else cl

    return {stem: list(entries.values()) for stem, entries in cluster_entries.items()}


@click.command(
    cls=DataRootCommand, context_settings={"help_option_names": ["-h", "--help"]}
)
@click.option(
    "--dataset-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=C.dir("combinations"),
    show_default=True,
    help="Directory with pair CSVs (curate_sets output).",
)
@click.option(
    "--msa-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    default=C.dir("msas"),
    show_default=True,
    help="Root of MSA a3m files.",
)
@click.option(
    "--cif-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=C.dir("cif_asms"),
    show_default=True,
    help="Root of assembly CIF files.",
)
@click.option(
    "--out-json",
    type=click.Path(path_type=Path),
    default=C.file("representative_sequences"),
    show_default=True,
    help="Output representative sequences JSON.",
)
@click.option(
    "--out-compat-csv",
    type=click.Path(path_type=Path),
    default=C.file("binding_site_compatibility"),
    show_default=True,
    help="Per-entry compatibility CSV.",
)
@click.option(
    "--out-dataset",
    type=click.Path(path_type=Path),
    default=C.dir("combinations_filtered"),
    show_default=True,
    help="Output directory for filtered pair CSVs.",
)
@click.option(
    "--workers", "-j", type=int, default=8, show_default=True, help="Parallel workers."
)
def main(
    dataset_dir: Path,
    msa_root: Path,
    cif_root: Path,
    out_json: Path,
    out_compat_csv: Path,
    out_dataset: Path,
    workers: int,
):
    """Select representative sequences based on binding-site compatibility."""

    # 1. Collect unique entries per cluster from pair CSVs
    click.echo(f"Scanning pair CSVs in {dataset_dir} ...")
    cluster_entries = _collect_entries(dataset_dir)
    n_entries = sum(len(v) for v in cluster_entries.values())
    click.echo(
        f"Found {n_entries} unique entries across {len(cluster_entries)} clusters"
    )

    # 2. Process each cluster
    click.echo(f"Processing with {workers} workers ...")
    results = Parallel(n_jobs=workers, backend="loky")(
        delayed(_process_cluster)(stem, edicts, msa_root, cif_root)
        for stem, edicts in cluster_entries.items()
    )

    # 3. Aggregate
    rep_json: Dict[str, dict] = {}
    all_compat_rows: List[dict] = []
    cluster_incompat: Dict[str, Set[str]] = {}

    ok_count = 0
    for r in results:
        if r is None:
            continue
        ok_count += 1
        cl = r["cluster"]
        rep_json[cl] = {
            "header": r["rep_header"],
            "seq": r["rep_seq"],
            "length": r["rep_length"],
            "support": r["support"],
            "total_entries": r["total_entries"],
        }
        all_compat_rows.extend(r["compat_rows"])
        if r["incompat_keys"]:
            cluster_incompat[cl] = r["incompat_keys"]

    click.echo(
        f"Representatives selected for {ok_count} / {len(cluster_entries)} clusters"
    )
    incompat_clusters = sum(1 for v in cluster_incompat.values() if v)
    incompat_entries = sum(len(v) for v in cluster_incompat.values())
    click.echo(
        f"Clusters with incompatible entries: {incompat_clusters}  "
        f"(total incompatible entries: {incompat_entries})"
    )

    # 4. Write representative_sequences.json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as fh:
        json.dump(rep_json, fh, indent=2, ensure_ascii=False)
    click.echo(f"Wrote {out_json}")

    # 5. Write per-entry compatibility CSV
    if all_compat_rows:
        pd.DataFrame(all_compat_rows).to_csv(out_compat_csv, index=False)
        click.echo(f"Wrote {out_compat_csv}")

    # 6. Filter pair CSVs — drop pairs involving incompatible entries
    out_dataset.mkdir(parents=True, exist_ok=True)

    for ds in sorted(dataset_dir.glob("*.csv")):
        try:
            pairs = DatasetPair.load_csv(ds)
        except Exception:
            continue
        if not pairs:
            DatasetPair.save_csv([], out_dataset / ds.name)
            continue

        n_before = len(pairs)
        kept: list[DatasetPair] = []

        for pair in pairs:
            stem = pair.cluster_stem
            if stem not in cluster_incompat:
                kept.append(pair)
                continue
            bad = cluster_incompat[stem]
            ka = f"{pair.a.pdb.lower()}_{pair.a.assembly_id}_{pair.a.chain_letter}"
            kb = f"{pair.b.pdb.lower()}_{pair.b.assembly_id}_{pair.b.chain_letter}"
            if ka not in bad and kb not in bad:
                kept.append(pair)

        DatasetPair.save_csv(kept, out_dataset / ds.name)
        click.echo(
            f"  {ds.name}: {n_before} -> {len(kept)} pairs  "
            f"(dropped {n_before - len(kept)})"
        )

    click.echo("Done.")


if __name__ == "__main__":
    main()

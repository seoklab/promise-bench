from __future__ import annotations

import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import click
import gemmi
import pandas as pd
import parasail
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..utils._config import pipeline_cfg as C
from ..utils._data_root import DataRootCommand
from ..utils.constants import _AA3, _NT

DBG_DEFAULTS = {
    "stage": "",
    "coi_lig_small": "",
    "coi_lig_large": "",
    "coi_lig_equal": "",
    "len_small": "",
    "len_large": "",
    "same_len": "",
    "u_small": "",
    "v_large": "",
    "fident": None,
    "pass_id": "",
    "lig_u": "",
    "lig_v": "",
    "pass_ligand": "",
    "matched_all": "",
}
EDGE_COLS = [
    "conf_label",
    "pdb_small",
    "asm_small",
    "pdb_large",
    "asm_large",
    "stage",
    "coi_lig_small",
    "coi_lig_large",
    "coi_lig_equal",
    "len_small",
    "len_large",
    "same_len",
    "u_small",
    "v_large",
    "fident",
    "pass_id",
    "lig_u",
    "lig_v",
    "pass_ligand",
    "matched_all",
]

_RX_LISTY = re.compile(r"^[\[(].*[\])]$")


def _norm_list_cell(cell: str) -> List[str]:
    if cell is None:
        return []
    s = str(cell).strip()
    if not s:
        return []
    if _RX_LISTY.match(s):
        try:
            v = json.loads(s.replace("'", '"'))
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
    parts = [
        t.strip() for t in (s.split(";") if ";" in s and "," not in s else s.split(","))
    ]
    return [p for p in parts if p]


def _to_int(x: str, default: int = 0) -> int:
    try:
        return int(float(str(x)))
    except Exception:
        return default


def _to_float(x: str, default: float = math.inf) -> float:
    try:
        return float(str(x))
    except Exception:
        return default


def _group_by_conf(rows: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for conf, sub in rows.groupby(rows["conf_label"].astype(str)):
        out[str(conf)] = sub.copy()
    return out


def _build_maps(
    sub: pd.DataFrame, coi_col: str, neigh_col: str, lig_col: str
) -> Tuple[
    Dict[Tuple[str, str], List[str]],  # neigh (pdb, asm, coi) -> [neighbors]
    Dict[Tuple[str, str], List[str]],  # lig  (pdb, asm, coi) -> ligs near that chain
]:
    neigh: Dict[Tuple[str, str], List[str]] = {}
    ligm: Dict[Tuple[str, str], List[str]] = {}
    for _, r in sub.iterrows():
        pdb = str(r.get("pdb", "")).strip()
        asm = str(r.get("assembly_id", "")).strip()
        coi = str(r.get(coi_col, "")).strip()
        neigh[(pdb, asm, coi)] = _norm_list_cell(str(r.get(neigh_col, "")))
        ligm[(pdb, asm, coi)] = _norm_list_cell(str(r.get(lig_col, "")))
    return neigh, ligm


# -----------------------------
# sequences (assembly/auth_asm)
# -----------------------------
def _aa1_from_residue(res: gemmi.Residue) -> Optional[str]:
    if res.het_flag == "H":
        return None
    rt = gemmi.find_tabulated_residue(str(res.name).strip().upper())

    def is_polymer_res(res: gemmi.Residue) -> bool:
        et = getattr(res, "entity_type", None)
        if et == gemmi.EntityType.Polymer:
            return True
        hf = getattr(res, "het_flag", "\0")
        if isinstance(hf, str) and (hf == "\0" or hf == " "):
            return True
        comp_id = str(getattr(res, "name", "")).upper()
        return comp_id in _AA3 or comp_id in _NT

    if is_polymer_res(res):
        return (rt.one_letter_code or "X") if rt else "X"
    return None


def _replace_nonstandard_with_X(raw: str) -> str:
    """Replace parenthesized non-standard residues with 'X' and strip whitespace."""
    if not raw:
        return ""
    s = re.sub(r"\([^)]*\)", "X", raw)
    s = s.replace("\n", "").replace(" ", "").strip()
    return s


def extract_auth_asm_sequences(cif_path: Path) -> Dict[str, str]:
    """Extract {auth_chain_id: one_letter_sequence} from _entity_poly in an mmCIF file."""
    if not cif_path.exists():
        return {}

    try:
        doc = gemmi.cif.read_file(str(cif_path))
        block = doc.sole_block()
    except Exception:
        return {}

    # Extract strand_id and sequence columns from _entity_poly table.
    # Columns prefixed with '?' are optional.
    table = block.find(
        "_entity_poly.",
        [
            "pdbx_strand_id",
            "?pdbx_seq_one_letter_code_can",
            "?pdbx_seq_one_letter_code",
        ],
    )
    if not table:
        return {}

    strand_to_seq: Dict[str, str] = {}

    for row in table:
        # Column 0: pdbx_strand_id (e.g. "A1", "C1" or "A,B")
        strand_field = row[0].strip()
        if not strand_field or strand_field in {".", "?"}:
            continue

        # Column 1: canonical one-letter sequence (preferred)
        # Column 2: non-canonical one-letter sequence (fallback)
        seq_val = ""
        if row.has(1):
            seq_val = row[1]
        elif row.has(2):
            seq_val = row[2]

        if not seq_val or seq_val in {".", "?"}:
            continue

        seq_clean = _replace_nonstandard_with_X(seq_val)
        if not seq_clean:
            continue

        for auth_id in strand_field.replace(",", " ").split():
            auth_id = auth_id.strip()
            if not auth_id:
                continue
            if len(seq_clean) > len(strand_to_seq.get(auth_id, "")):
                strand_to_seq[auth_id] = seq_clean

    return strand_to_seq


def _strip_trailing_digits(s: str) -> str:
    return re.sub(r"\d+$", "", s or "")


def _load_sequences_for_pdb(mmcif_dir: Path, pdb: str, asms: str) -> Dict[str, str]:
    pdb = pdb.lower()
    # try direct
    direct = mmcif_dir / pdb[1:3].upper() / pdb.upper() / f"asm_{pdb}_{asms}.cif"
    if direct.exists():
        return extract_auth_asm_sequences(direct)
    return {}


def _dbg_log(dbg: List[Dict[str, object]], ctx: Dict[str, str], stage: str, **extra):
    row = {**DBG_DEFAULTS, **ctx, "stage": stage}
    for k, v in extra.items():
        if isinstance(v, list):
            v = ";".join(map(str, v))
        row[k] = v
    dbg.append(row)


def _seq_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    res = parasail.nw_stats_striped_16(a, b, 10, 1, parasail.blosum62)
    length: Optional[int] = getattr(res, "len", None)
    if length is None:
        length = getattr(res, "length", 0)
    matches: Optional[int] = getattr(res, "matches", None)
    if matches is None:
        matches = getattr(res, "identity", 0)
    return (matches / length) if length and length > 0 else 0.0


def _bruteforce_perm(
    U: List[str], V: List[str], ident: Dict[Tuple[int, int], float]
) -> Tuple[bool, Dict[str, str]]:
    n = len(U)
    if n == 0:
        return True, {}
    if n != len(V):
        return False, {}

    best_sum = -1.0
    best_map: Dict[str, str] = {}
    for perm in itertools.permutations(range(n)):
        s = 0.0
        for i, j in enumerate(perm):
            s += ident.get((i, j), 0.0)
        if s > best_sum:
            best_sum = s
            best_map = {U[i]: V[j] for i, j in enumerate(perm)}

    # print(U, V)
    return (best_sum >= 0.0 and len(best_map) == n), best_map


def _superset_for_coi(
    neigh_small: List[str],
    neigh_large: List[str],
    lig_coi_small: List[str],
    lig_coi_large: List[str],
    chain_ligs_small: Dict[str, List[str]],
    chain_ligs_large: Dict[str, List[str]],
    seq_small: Dict[str, str],
    seq_large: Dict[str, str],
    id_thr: float,
    dbg: List[Dict[str, object]],
    ctx: Dict[str, str],
) -> bool:
    # 1) COI ligand equality
    coi_equal = Counter(lig_coi_small) == Counter(lig_coi_large)
    _dbg_log(
        dbg,
        ctx,
        "coi_lig",
        coi_lig_small=lig_coi_small,
        coi_lig_large=lig_coi_large,
        coi_lig_equal=coi_equal,
    )
    if not coi_equal:
        return False

    U = neigh_small
    V = neigh_large

    # 2) neighbor lengths
    _dbg_log(
        dbg, ctx, "len", len_small=len(U), len_large=len(V), same_len=(len(U) == len(V))
    )
    if len(U) == 0 and len(V) == 0:
        _dbg_log(dbg, ctx, "monomer")
        return True
    if len(U) != len(V):
        return False

    # 3) identities + ligand equality per candidate edge
    ident: Dict[Tuple[int, int], float] = {}
    for i, u in enumerate(U):
        su = seq_small.get(u, "") or seq_small.get(_strip_trailing_digits(u), "")
        lig_u = chain_ligs_small.get(u, [])
        for j, v in enumerate(V):
            sv = seq_large.get(v, "") or seq_large.get(_strip_trailing_digits(v), "")
            lig_v = chain_ligs_large.get(v, [])
            pid = _seq_identity(su, sv) if (su and sv) else 0.0
            # print(pid)
            lig_ok = Counter(lig_u) == Counter(lig_v)
            ident[(i, j)] = pid
            _dbg_log(
                dbg,
                ctx,
                "edge",
                u_small=u,
                v_large=v,
                fident=float(f"{pid:.4f}"),
                pass_id=(pid >= id_thr),
                lig_u=lig_u,
                lig_v=lig_v,
                pass_ligand=lig_ok,
            )

    # 4) matching
    perfect, mapping = _bruteforce_perm(U, V, ident)
    if perfect:
        for i, u in enumerate(U):
            v = mapping[u]
            j = V.index(v)
            if ident.get((i, j), 0.0) < id_thr:
                perfect = False
                break

    _dbg_log(dbg, ctx, "matching", matched_all=perfect)
    return perfect


def process_file(
    csv_path: Path,
    asm_bio_root: Path,
    out_root: Path,
    *,
    mmcif_dir: Path,
    id_thr: float,
    dump_edges: bool,
) -> Dict[str, object]:
    rel = csv_path.relative_to(asm_bio_root)
    dst_dir = out_root / rel.parent
    dst_dir.mkdir(parents=True, exist_ok=True)
    keep_csv = dst_dir / (
        csv_path.stem.replace("_asm_bio", "_asm_subset_filtered") + ".csv"
    )
    drop_csv = dst_dir / (
        csv_path.stem.replace("_asm_bio", "_asm_subset_dropped") + ".csv"
    )
    edge_tsv = dst_dir / (csv_path.stem.replace("_asm_bio", "_edges") + ".tsv")

    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
    except Exception as e:
        return {"file": str(csv_path), "error": f"read_failed: {e}"}

    if df.empty:
        df.to_csv(keep_csv, index=False)
        pd.DataFrame().to_csv(drop_csv, index=False)
        if dump_edges:
            pd.DataFrame().to_csv(edge_tsv, sep="\t", index=False)
        return {
            "file": str(csv_path),
            "kept_rows": 0,
            "dropped_rows": 0,
            "removed": 0,
            "assemblies": 0,
            "centers": 0,
            "out_csv": str(keep_csv),
            "drop_csv": str(drop_csv),
        }

    coi_col = "chain_auth_asm"
    neigh_col = "contact_chains"
    lig_col = "contact_ligands"

    for c in ["pdb", "assembly_id", "conf_label", "protein_count", "resolution"]:
        if c not in df.columns:
            return {"file": str(csv_path), "error": f"missing_column:{c}"}

    # Split by conf_label
    by_conf = _group_by_conf(df)

    seq_cache: Dict[Tuple[str, str], Dict[str, str]] = {}
    removed_set: Set[Tuple[str, str, str]] = set()  # (conf_label, pdb, asm)
    drop_logs: List[Dict[str, str]] = []
    dbg_rows: List[Dict[str, object]] = []

    centers = 0
    assemblies_total = 0

    for conf, sub in by_conf.items():
        centers += 1
        assemblies_total += len(
            set((pdb.lower(), asm) for pdb, asm in zip(sub["pdb"], sub["assembly_id"]))
        )
        neigh, ligm = _build_maps(
            sub, coi_col=coi_col, neigh_col=neigh_col, lig_col=lig_col
        )
        rows = [sub.iloc[i] for i in range(len(sub))]
        for r in rows:
            pdb, asm = str(r["pdb"]).lower(), str(r["assembly_id"])
            if (pdb, asm) not in seq_cache:
                seq_cache[(pdb, asm)] = _load_sequences_for_pdb(mmcif_dir, pdb, asm)

        m = len(rows)
        for i in range(m):
            ri = rows[i]
            pdb_i = str(ri["pdb"]).lower()
            asm_i = str(ri["assembly_id"])
            coi_i = str(ri[coi_col])
            if (conf, pdb_i, asm_i) in removed_set:
                continue
            for j in range(i + 1, m):
                rj = rows[j]
                pdb_j = str(rj["pdb"]).lower()
                asm_j = str(rj["assembly_id"])
                coi_j = str(rj[coi_col])
                if (conf, pdb_j, asm_j) in removed_set:
                    continue
                if (pdb_i == pdb_j) and (asm_i == asm_j):
                    continue

                # choose small/large by protein_count
                pc_i = _to_int(ri.get("protein_count", "0"))
                pc_j = _to_int(rj.get("protein_count", "0"))
                if pc_i < pc_j:
                    small, large = ri, rj
                    coi_small, coi_large = coi_i, coi_j
                elif pc_i > pc_j:
                    small, large = rj, ri
                    coi_small, coi_large = coi_j, coi_i
                else:
                    small, large = ri, rj
                    coi_small, coi_large = coi_i, coi_j

                ctx = {
                    "conf_label": conf,
                    "pdb_small": str(small["pdb"]).lower(),
                    "asm_small": str(small["assembly_id"]),
                    "pdb_large": str(large["pdb"]).lower(),
                    "asm_large": str(large["assembly_id"]),
                }

                lig_small = ligm.get(
                    (str(small["pdb"]).lower(), str(small["assembly_id"]), coi_small),
                    [],
                )
                lig_large = ligm.get(
                    (str(large["pdb"]).lower(), str(large["assembly_id"]), coi_large),
                    [],
                )
                neis_small = neigh.get(
                    (str(small["pdb"]).lower(), str(small["assembly_id"]), coi_small),
                    [],
                )
                neis_large = neigh.get(
                    (str(large["pdb"]).lower(), str(large["assembly_id"]), coi_large),
                    [],
                )

                chain_ligs_small = {
                    u: ligm.get(
                        (str(small["pdb"]).lower(), str(small["assembly_id"]), u), []
                    )
                    for u in neis_small
                }
                chain_ligs_large = {
                    v: ligm.get(
                        (str(large["pdb"]).lower(), str(large["assembly_id"]), v), []
                    )
                    for v in neis_large
                }

                seq_small = seq_cache.get((small["pdb"], small["assembly_id"]), {})
                seq_large = seq_cache.get((large["pdb"], large["assembly_id"]), {})

                equal = _superset_for_coi(
                    neis_small,
                    neis_large,
                    lig_small,
                    lig_large,
                    chain_ligs_small,
                    chain_ligs_large,
                    seq_small,
                    seq_large,
                    id_thr,
                    dbg_rows,
                    ctx,
                )
                if not equal:
                    continue

                # tie-breaks on equality -> decide drop
                drop_row, keep_row, reason = large, small, None
                if _to_int(small.get("protein_count", "0")) == _to_int(
                    large.get("protein_count", "0")
                ):
                    res_s = _to_float(small.get("resolution", "inf"))
                    res_l = _to_float(large.get("resolution", "inf"))
                    if res_s < res_l:
                        drop_row, keep_row, reason = large, small, "resolution"
                    elif res_s > res_l:
                        drop_row, keep_row, reason = small, large, "resolution"
                    else:
                        key_s = (str(small["pdb"]).lower(), str(small["assembly_id"]))
                        key_l = (str(large["pdb"]).lower(), str(large["assembly_id"]))
                        if key_s > key_l:
                            drop_row, keep_row = small, large
                        else:
                            drop_row, keep_row = large, small
                        reason = "identical"
                if reason is None:
                    reason = f"superset_of:{keep_row['assembly_id']}"

                removed_set.add(
                    (conf, str(drop_row["pdb"]).lower(), str(drop_row["assembly_id"]))
                )
                drop_logs.append(
                    {
                        "pdb": str(drop_row["pdb"]).lower(),
                        "conf_label": conf,
                        "removed_assembly_id": str(drop_row["assembly_id"]),
                        "reason": reason,
                        "kept_assembly_id": str(keep_row["assembly_id"]),
                        "evidence_json": json.dumps(
                            {
                                "protein_count_small": _to_int(
                                    keep_row.get("protein_count", "0")
                                ),
                                "protein_count_large": _to_int(
                                    drop_row.get("protein_count", "0")
                                ),
                                "resolution_small": _to_float(
                                    keep_row.get("resolution", "inf")
                                ),
                                "resolution_large": _to_float(
                                    drop_row.get("resolution", "inf")
                                ),
                            }
                        ),
                    }
                )

    # Write outputs
    kept_mask = []
    for _, r in df.iterrows():
        key = (str(r["conf_label"]), str(r["pdb"]).lower(), str(r["assembly_id"]))
        kept_mask.append(key not in removed_set)
    kept_df = df[kept_mask]

    reason_map = {
        (d["conf_label"], d["pdb"], d["removed_assembly_id"]): d for d in drop_logs
    }
    dropped_aug_rows: List[Dict[str, str]] = []
    for _, r in df.iterrows():
        key = (str(r["conf_label"]), str(r["pdb"]).lower(), str(r["assembly_id"]))
        if key in removed_set:
            meta = reason_map.get(key, {})
            row = {
                **{c: str(r[c]) for c in df.columns},
                "drop_reason": meta.get("reason", ""),
                "kept_assembly_id": meta.get("kept_assembly_id", ""),
                "evidence_json": meta.get("evidence_json", ""),
            }
            dropped_aug_rows.append(row)
    drop_df = pd.DataFrame(dropped_aug_rows)

    kept_df.to_csv(keep_csv, index=False)
    drop_df.to_csv(drop_csv, index=False)

    if dump_edges:
        try:
            df_dbg = pd.DataFrame(dbg_rows)
            if not df_dbg.empty:
                df_dbg = df_dbg.reindex(columns=EDGE_COLS)
            df_dbg.to_csv(edge_tsv, sep="\t", index=False)
        except Exception:
            pass

    return {
        "file": str(csv_path),
        "out_csv": str(keep_csv),
        "drop_csv": str(drop_csv),
        "centers": len(by_conf),
        "assemblies": assemblies_total,
        "removed": len(drop_logs),
        "kept_rows": int(kept_df.shape[0]),
        "dropped_rows": int(drop_df.shape[0]),
    }


@click.command(cls=DataRootCommand, context_settings=dict(show_default=True))
@click.option(
    "--asm-bio-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    default=str(C.dir("asms_bio")),
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    default=str(C.dir("asms_subset")),
)
@click.option(
    "--mmcif-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=C.dir("cif_asms"),
    help="Root containing assembly CIFs; tries <pdb>.cif or **/asm_<pdb>_*.cif, **/<pdb>/asm*.cif",
)
@click.option(
    "--id-threshold",
    "id_thr",
    type=float,
    default=0.95,
    help="Sequence identity threshold for neighbor matching.",
)
@click.option("--workers", type=int, default=8, help="Parallel workers (per file).")
@click.option(
    "--dump-edges",
    is_flag=True,
    default=True,
    help="Write per-pair diagnostics TSV files.",
)
def main(
    asm_bio_dir: Path,
    out_dir: Path,
    mmcif_dir: Path,
    id_thr: float,
    workers: int,
    dump_edges: bool,
):
    files = sorted(asm_bio_dir.glob("**/*_asm_bio.csv"))
    if not files:
        click.echo(f"[!] No inputs: {asm_bio_dir}/**/*_asm_bio.csv")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        delayed(process_file)(
            p,
            asm_bio_dir,
            out_dir,
            mmcif_dir=mmcif_dir,
            id_thr=id_thr,
            dump_edges=dump_edges,
        )
        for p in files
    ]
    results = Parallel(n_jobs=workers, prefer="processes")(
        tqdm(tasks, desc="Subset filtering", unit="file")
    )

    tot_files = 0
    agg_centers = 0
    agg_asms = 0
    agg_removed = 0
    for r in results:
        tot_files += 1
        if r.get("error"):
            click.echo(f"[warn] {r['file']}: {r['error']}")
            continue
        agg_centers += int(r.get("centers", 0))
        agg_asms += int(r.get("kept_rows", 0))
        agg_removed += int(r.get("dropped_rows", 0))
        click.echo(
            f"[+] {Path(r['file']).name}: kept={r['kept_rows']}, dropped={r['dropped_rows']}"
        )

    pct = (
        (100.0 * agg_removed / (agg_asms + agg_removed))
        if agg_asms + agg_removed
        else 0.0
    )
    click.echo("\n==== Global Summary ====")
    click.echo(f"Files processed   : {tot_files}")
    click.echo(f"Centers processed : {agg_centers}")
    click.echo(f"Assemblies total  : {agg_asms + agg_removed}")
    click.echo(f"Assemblies removed: {agg_removed} ({pct:.1f}%)")


if __name__ == "__main__":
    main()

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd

from ._data_root import DataRootCommand


def _s(x) -> str:
    return "" if (pd.isna(x) or x is None) else str(x)


def _lig_list(raw: str) -> Tuple[str, ...]:
    s = _s(raw).strip()
    if not s:
        return tuple()
    parts = [p.strip() for p in s.split(";") if p.strip()]
    parts.sort()
    return tuple(parts)


def lig_equal(a: str, b: str) -> bool:
    return _lig_list(a) == _lig_list(b)


def is_apo(raw: str) -> bool:
    return len(_lig_list(raw)) == 0


def _pair(rows_a: pd.DataFrame, rows_b: pd.DataFrame):
    for _, ra in rows_a.iterrows():
        for _, rb in rows_b.iterrows():
            yield ra, rb


def emit_min_row(cluster_csv: str, ra: pd.Series, rb: pd.Series) -> Dict[str, Any]:
    return {
        "cluster_csv": cluster_csv[0:9],
        "a_pdb": _s(ra.get("pdb")),
        "a_assembly_id": _s(ra.get("assembly_id")),
        "a_chain": _s(ra.get("chain_auth_asm")),
        "a_conf_label": ra.get("conf_label"),
        "a_chains": _s(ra.get("chain_list_author")),
        "a_contact_chains": _s(ra.get("contact_chains", "")),
        "a_ligand_list": _s(ra.get("ligands", "")),
        "a_contact_ligands": _s(ra.get("contact_ligands", "")),
        "b_pdb": _s(rb.get("pdb")),
        "b_assembly_id": _s(rb.get("assembly_id")),
        "b_chain": _s(rb.get("chain_auth_asm")),
        "b_conf_label": rb.get("conf_label"),
        "b_chains": _s(rb.get("chain_list_author")),
        "b_contact_chains": _s(rb.get("contact_chains", "")),
        "b_ligand_list": _s(rb.get("ligands", "")),
        "b_contact_ligands": _s(rb.get("contact_ligands", "")),
        "a_desc": _s(ra.get("desc", "")),
        "b_desc": _s(rb.get("desc", "")),
    }


def normalize_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    s = df.copy()
    s.columns = [c.strip().lstrip("\ufeff") for c in s.columns]
    if "pdb" in s:
        s["pdb"] = s["pdb"].astype(str).str.strip()
    if "chain_auth_asm" in s:
        s["chain_auth_asm"] = s["chain_auth_asm"].astype(str).str.strip()
    if "chain_list_author" in s:
        s["chain_list_author"] = s["chain_list_author"].astype(str).str.strip()
    if "protein_count" in s:
        s["protein_count"] = pd.to_numeric(s["protein_count"], errors="coerce")
    if "conf_label" in s:
        s["conf_label"] = pd.to_numeric(s["conf_label"], errors="coerce")
    if "contact_ligands" in s:
        s["contact_ligands"] = (
            s["contact_ligands"]
            .astype(str)
            .str.strip()
            .replace({"nan": "", "NaN": "", "None": "", "NONE": ""})
        )

    s["_lig_key"] = s["contact_ligands"].apply(_lig_list)
    return s


def safe_drop_duplicates(df: pd.DataFrame, subset_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in subset_cols if c in df.columns]
    if not cols:
        return df
    return df.drop_duplicates(subset=cols)


def _condensed_index(n: int, i: int, j: int) -> int:
    if i == j:
        raise ValueError("no index for diagonal")
    if i > j:
        i, j = j, i
    return n * i - i * (i + 1) // 2 + (j - i - 1)


_tm_cache: Dict[str, Any] = {}


def _get_tm_score(pair: Dict[str, Any], tm_root: Path) -> Optional[float]:
    """Look up pairwise TM-score from precomputed .npz file."""
    cluster_key = str(pair.get("cluster_csv", ""))
    if cluster_key not in _tm_cache:
        tm_path = tm_root / f"{cluster_key}.npz"
        if tm_path.exists():
            npz = np.load(tm_path)
            _tm_cache[cluster_key] = (npz["chains"], npz["tm_scores"])
        else:
            _tm_cache[cluster_key] = None

    cached = _tm_cache.get(cluster_key)
    if cached is None:
        return None

    chains, tm_scores = cached
    pdb_a = str(pair.get("a_pdb", "")).strip().lower()
    coi_a = re.sub(r"\d+$", "", str(pair.get("a_chain", "")).strip())
    chain_a = f"{pdb_a}_{coi_a.upper()}"

    pdb_b = str(pair.get("b_pdb", "")).strip().lower()
    coi_b = re.sub(r"\d+$", "", str(pair.get("b_chain", "")).strip())
    chain_b = f"{pdb_b}_{coi_b.upper()}"

    idx_a = np.where(chains == chain_a)[0]
    idx_b = np.where(chains == chain_b)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        return None

    try:
        pidx = _condensed_index(len(chains), int(idx_a[0]), int(idx_b[0]))
        return float(tm_scores[pidx])
    except (ValueError, IndexError):
        return None


# ---------- Pair Finders (FILE-LEVEL) ----------
def f_monomer_multimer(df: pd.DataFrame, cluster_csv: str):
    out = []
    for _, g in df.groupby("_lig_key"):
        mons = g.loc[g["protein_count"].eq(1)]
        multis = g.loc[g["protein_count"].gt(1)]
        if mons.empty or multis.empty:
            continue
        subset_cols = [
            "pdb",
            "chain_auth_asm",
            "conf_label",
            "assembly_id",
            "contact_ligands",
        ]
        mons = safe_drop_duplicates(mons, subset_cols)
        multis = safe_drop_duplicates(multis, subset_cols)
        for ra, rb in _pair(mons, multis):
            if ra["conf_label"] == rb["conf_label"]:
                continue
            out.append(emit_min_row(cluster_csv, ra, rb))
    return out


def f_ligand_change(df: pd.DataFrame, cluster_csv: str):
    """ligand list different, conf different, same protein_count"""
    out = []
    for pc, g in df.groupby("protein_count"):
        if len(g) < 2 or pd.isna(pc):
            continue
        subset_cols = [
            "pdb",
            "chain_auth_asm",
            "conf_label",
            "assembly_id",
            "contact_ligands",
        ]
        g = safe_drop_duplicates(g, subset_cols)
        keys = list(g["_lig_key"].unique())
        if len(keys) < 2:
            continue
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ga = g[g["_lig_key"].apply(lambda x: x == keys[i])]
                gb = g[g["_lig_key"].apply(lambda x: x == keys[j])]
                for ra, rb in _pair(ga, gb):
                    if ra["conf_label"] == rb["conf_label"]:
                        continue
                    out.append(emit_min_row(cluster_csv, ra, rb))
    return out


def f_apo_mono_to_holo(df: pd.DataFrame, cluster_csv: str):
    out = []
    apo_mono = df[
        (df["protein_count"].eq(1))
        & (df["contact_ligands"].apply(lambda s: is_apo(_s(s))))
    ]
    holo_mono = df[
        (df["protein_count"].eq(1))
        & (df["contact_ligands"].apply(lambda s: not is_apo(_s(s))))
    ]
    if apo_mono.empty or holo_mono.empty:
        return out
    subset_cols = [
        "pdb",
        "chain_auth_asm",
        "conf_label",
        "assembly_id",
        "contact_ligands",
    ]
    for pc, a in apo_mono.groupby("protein_count"):
        if pd.isna(pc):
            continue
        b = holo_mono[holo_mono["protein_count"].eq(pc)]
        if b.empty:
            continue
        a = safe_drop_duplicates(a, subset_cols)
        b = safe_drop_duplicates(b, subset_cols)
        for ra, rb in _pair(a, b):
            if ra["conf_label"] == rb["conf_label"]:
                continue
            out.append(emit_min_row(cluster_csv, ra, rb))
    return out


def f_apo_mono_to_any(df: pd.DataFrame, cluster_csv: str):
    """apo monomer (lig=0, pc=1) vs anything else (NOT apo mono), conf different"""
    out = []
    apo_mono = df[
        (df["protein_count"].eq(1))
        & (df["contact_ligands"].apply(lambda s: is_apo(_s(s))))
    ]
    others = df.drop(apo_mono.index)
    if apo_mono.empty or others.empty:
        return out
    subset_cols = [
        "pdb",
        "chain_auth_asm",
        "conf_label",
        "assembly_id",
        "contact_ligands",
    ]
    apo_mono = safe_drop_duplicates(apo_mono, subset_cols)
    others = safe_drop_duplicates(others, subset_cols)
    for ra, rb in _pair(apo_mono, others):
        if ra["conf_label"] == rb["conf_label"]:
            continue
        out.append(emit_min_row(cluster_csv, ra, rb))
    return out


def f_apo_mono_pairs(df: pd.DataFrame, cluster_csv: str):
    """apo monomer vs apo monomer, conf different"""
    out = []
    g = df[
        (df["protein_count"].eq(1))
        & (df["contact_ligands"].apply(lambda s: is_apo(_s(s))))
    ]
    if len(g) < 2:
        return out
    subset_cols = [
        "pdb",
        "chain_auth_asm",
        "conf_label",
        "assembly_id",
        "contact_ligands",
    ]
    g = safe_drop_duplicates(g, subset_cols)
    for i, ra in g.iterrows():
        for j, rb in g.iterrows():
            if j <= i:
                continue
            if ra["conf_label"] == rb["conf_label"]:
                continue
            out.append(emit_min_row(cluster_csv, ra, rb))
    return out


COMBOS = {
    "protein-induced": f_monomer_multimer,
    "intrinsic": f_apo_mono_pairs,
    "ligand-induced": f_apo_mono_to_holo,
}


def process_file(
    fpath: Path,
    combos: List[str],
    root: Path,
    tm_root: Optional[Path] = None,
    tm_cutoff: float = 0.8,
):
    df_raw = pd.read_csv(fpath)
    if df_raw.empty:
        return {c: [] for c in combos}
    need = {"pdb", "chain_auth_asm", "conf_label", "protein_count", "chain_list_author"}
    miss = [c for c in need if c not in df_raw.columns]
    if miss:
        raise ValueError(f"{fpath}: missing {miss}")

    df = normalize_df(df_raw)
    cluster_csv = fpath.relative_to(root).as_posix()

    # pairs
    buckets: Dict[str, List[Dict[str, Any]]] = {c: [] for c in combos}
    for name in combos:
        buckets[name] += COMBOS[name](df, cluster_csv)

    # TM-score filtering: drop pairs that are too similar (TM > cutoff)
    if tm_root is not None:
        for name in combos:
            filtered = []
            for pair in buckets[name]:
                score = _get_tm_score(pair, tm_root)
                pair["tm_score"] = score
                if score is not None and score > tm_cutoff:
                    continue  # too similar -> drop
                filtered.append(pair)
            buckets[name] = filtered

    return buckets


def write_per_type(outdir: Path, accum: Dict[str, List[Dict[str, Any]]]):
    outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in accum.items():
        out_path = outdir / f"{name}.csv"
        if not rows:
            out_path.write_text("")
        else:
            pd.DataFrame(rows).drop_duplicates().to_csv(out_path, index=False)


@click.command(
    cls=DataRootCommand, context_settings=dict(help_option_names=["-h", "--help"])
)
@click.option(
    "--filtered-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Root directory (e.g., asms-metal).",
    default="data/asms-metal",
)
@click.option(
    "--pattern",
    default="**/*_asm_subset_filtered.csv",
    show_default=True,
    help="Glob pattern (recursive).",
)
@click.option(
    "--outdir",
    type=click.Path(dir_okay=True, file_okay=False, writable=True),
    default="data/combinations",
    show_default=True,
    help="Directory to write per-type CSV outputs.",
)
@click.option(
    "--combos",
    multiple=True,
    type=click.Choice(list(COMBOS.keys())),
    help="Select specific combo types. If omitted, all are run.",
)
@click.option(
    "--tm-root",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    default="data/scores",
    show_default=True,
    help="Root dir with precomputed TM-score .npz files.",
)
@click.option(
    "--tm-cutoff",
    type=float,
    default=0.8,
    show_default=True,
    help="Drop pairs with TM-score above this threshold (too structurally similar).",
)
def cli(filtered_dir, pattern, outdir, combos, tm_root, tm_cutoff):
    combos = list(combos) if combos else list(COMBOS.keys())
    root = Path(filtered_dir)
    tm_root_p = Path(tm_root) if tm_root else None
    files = sorted(root.rglob(pattern))

    if tm_root_p:
        click.echo(f"[tm] Filtering pairs with TM > {tm_cutoff} using {tm_root_p}")

    global_pairs = {c: [] for c in combos}
    processed = 0

    for f in files:
        try:
            buckets = process_file(
                f, combos, root, tm_root=tm_root_p, tm_cutoff=tm_cutoff
            )
            processed += 1
            for c in combos:
                global_pairs[c].extend(buckets[c])
        except Exception as e:
            click.echo(f"[!] Skip {f}: {e}")

    outdir_p = Path(outdir)
    write_per_type(outdir_p, global_pairs)

    for c in combos:
        n = len(global_pairs[c])
        click.echo(f"[+] {c}: {n} pairs")
    click.echo(f"[=] processed {processed} centers")


if __name__ == "__main__":
    cli()

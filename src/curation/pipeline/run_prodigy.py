from __future__ import annotations

import os
import re
import shlex as _shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import click
import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from utils._config import pipeline_cfg as C
from utils._data_root import DataRootCommand

_PRODIGY_CMD: list[str] = _shlex.split(os.environ.get("PRODIGY_CMD", "prodigy_cryst"))

RX_NO_CONTACTS = re.compile(r"No contacts found", re.I)
RX_PRODIGY_TUPLE = re.compile(
    r"\(\s*'?\b(BIO|XTAL)\b'?\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)"
)
RX_PRODIGY_SIMPLE = re.compile(
    r"\b(BIO|XTAL)\b.*?([0-9]*\.?[0-9]+).*?([0-9]*\.?[0-9]+)"
)


def _find_raw_csvs(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*_asm_raw.csv") if p.is_file())


def _cif_path(cif_root: Path, pdb: str, asm: str) -> Path:
    return cif_root / pdb[1:3].upper() / pdb.upper() / f"asm_{pdb.lower()}_{asm}.cif"


def _parse_auth_list(s: str) -> List[str]:
    """Parse comma/semicolon-separated chain list."""
    if s is None:
        return []
    txt = str(s).strip().strip('"').strip("'").strip("[").strip("]")
    if not txt:
        return []
    parts: List[str] = []
    for token in txt.split(","):
        parts.extend(t.strip() for t in token.split(";"))
    seen, out = set(), []
    for x in parts:
        x = x.strip("'")
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _chain_coords_from_cif(
    cif_path: Path, include_h: bool = False
) -> Dict[str, np.ndarray]:
    """Read atom coordinates per chain from mmCIF file."""
    try:
        d = MMCIF2Dict(str(cif_path))
    except Exception:
        return {}

    xs = d.get("_atom_site.Cartn_x", [])
    ys = d.get("_atom_site.Cartn_y", [])
    zs = d.get("_atom_site.Cartn_z", [])
    n = min(len(xs), len(ys), len(zs))
    if n == 0:
        return {}

    auth_asym = d.get("_atom_site.auth_asym_id", [])
    if len(auth_asym) < n:
        return {}

    type_symbol = d.get("_atom_site.type_symbol", [])
    auth_atom_id = d.get("_atom_site.auth_atom_id", [])

    coords_by: Dict[str, List[List[float]]] = defaultdict(list)
    for i in range(n):
        elem = ""
        if len(type_symbol) > i and type_symbol[i]:
            elem = str(type_symbol[i]).upper()
        elif len(auth_atom_id) > i and auth_atom_id[i]:
            elem = str(auth_atom_id[i])[0].upper()
        if (not include_h) and elem == "H":
            continue

        chain = auth_asym[i]
        if not chain:
            continue

        try:
            x, y, z = float(xs[i]), float(ys[i]), float(zs[i])
        except Exception:
            continue
        coords_by[str(chain)].append([x, y, z])

    return {k: np.asarray(v, dtype=np.float32) for k, v in coords_by.items() if v}


def _contact_pairs(
    coords: Dict[str, np.ndarray], cutoff: float, allowed: Set[str]
) -> List[Tuple[str, str]]:
    """Find chain pairs within contact distance."""
    r2 = cutoff * cutoff
    names = sorted([c for c in coords.keys() if c in allowed])
    pairs: List[Tuple[str, str]] = []
    for i, ai in enumerate(names):
        A = coords[ai]
        if A.size == 0:
            continue
        for bj in names[i + 1 :]:
            B = coords[bj]
            if B.size == 0:
                continue
            small, big = (A, B) if A.shape[0] <= B.shape[0] else (B, A)
            for k in range(small.shape[0]):
                d2 = np.sum((big - small[k]) ** 2, axis=1)
                if float(d2.min(initial=np.inf)) <= r2:
                    pairs.append((ai, bj))
                    break
    return pairs


def _run_prodigy(
    struct_path: Path, chain_a: str, chain_b: str
) -> Tuple[str, float, float]:
    """Run prodigy_cryst and parse output."""
    cmd = [*_PRODIGY_CMD, "-q", str(struct_path), "--selection", chain_a, chain_b]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")

    # Check for no contacts (can appear in stdout/stderr or as exit code 1)
    if RX_NO_CONTACTS.search(out) or "No contacts found" in out:
        return "XTAL", 0.0, 1.0

    # Check for non-zero exit code after checking for "no contacts"
    if proc.returncode != 0:
        err_msg = f"prodigy_cryst exited with code {proc.returncode}"
        if proc.stderr:
            err_msg += f"\nSTDERR:\n{proc.stderr}"
        if proc.stdout:
            err_msg += f"\nSTDOUT:\n{proc.stdout}"
        raise RuntimeError(err_msg)

    for ln in out.splitlines()[::-1]:
        m = RX_PRODIGY_TUPLE.search(ln) or RX_PRODIGY_SIMPLE.search(ln)
        if m:
            return m.group(1).upper(), float(m.group(2)), float(m.group(3))

    raise RuntimeError(
        f"[prodigy] failed to parse output:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def _collect_from_raw(
    frames: List[pd.DataFrame],
) -> Tuple[
    Dict[Tuple[str, str], Set[str]],
    Dict[Tuple[str, str], List[dict]],
]:
    """Collect chain information from raw CSV files."""
    auth_clones: Dict[Tuple[str, str], Set[str]] = {}
    rows_by_asm: Dict[Tuple[str, str], List[dict]] = {}

    required_cols = {
        "pdb",
        "assembly_id",
        "chain_list_author",
        "conf_label",
        "chain_author",
        "chain_auth_asm",
    }

    for df in frames:
        missing = required_cols - set(df.columns)
        if missing:
            raise SystemExit(f"Missing columns {sorted(missing)} in raw CSV")

        df = df.copy()
        df["assembly_id"] = df["assembly_id"].astype(str)

        for _, row in df.iterrows():
            pdb = str(row["pdb"]).strip().lower()
            asm = str(row["assembly_id"]).strip()
            key = (pdb, asm)

            auths = set(_parse_auth_list(row["chain_list_author"]))
            if not auths:
                continue

            if key not in auth_clones:
                auth_clones[key] = set(auths)
            elif auths != auth_clones[key]:
                print(f"[warn] chain mismatch for {pdb} asm {asm}, using first seen")

            rows_by_asm.setdefault(key, []).append(
                {
                    "conf_label": str(row["conf_label"]).strip(),
                    "coi_author": str(row["chain_author"]).strip(),
                    "coi_author_clone": str(row["chain_auth_asm"]).strip(),
                }
            )

    return auth_clones, rows_by_asm


def _precompute_assembly(
    key: Tuple[str, str],
    cif_root: Path,
    allowed_auths: Set[str],
    contact_cutoff: float,
) -> Tuple[Tuple[str, str], Dict[Tuple[str, str], Tuple[str, float, float]], int]:
    """Compute prodigy scores for all contacting chain pairs in an assembly."""
    pdb, asm_id = key
    cif_path = _cif_path(cif_root, pdb, asm_id)
    if not cif_path.exists():
        print(f"[warn] CIF not found: {cif_path}")
        return key, {}, 0

    coords = _chain_coords_from_cif(cif_path)
    allowed = set(s for s in allowed_auths if s in coords)
    pairs = _contact_pairs(coords, contact_cutoff, allowed)

    out: Dict[Tuple[str, str], Tuple[str, float, float]] = {}
    fail = 0
    for a, b in pairs:
        try:
            out[(a, b)] = _run_prodigy(cif_path, a, b)
        except Exception as e:
            fail += 1
            print(f"[warn] prodigy failed {pdb} asm {asm_id} {a}-{b}: {e}")
    return key, out, fail


def _compute_stats(s: pd.Series) -> str:
    """Compute summary statistics for a series."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "n=0"
    return (
        f"n={s.size}, mean={s.mean():.4f}, std={s.std(ddof=1) if s.size > 1 else 0:.4f}, "
        f"min={s.min():.4f}, q25={s.quantile(0.25):.4f}, median={s.median():.4f}, "
        f"q75={s.quantile(0.75):.4f}, max={s.max():.4f}"
    )


@click.command(
    cls=DataRootCommand, context_settings=dict(help_option_names=["-h", "--help"])
)
@click.option(
    "--raw-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    default=str(C.dir("asms_raw")),
    show_default=True,
    help="Directory containing *_asm_raw.csv files.",
)
@click.option(
    "--cif-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    default=str(C.dir("cif_asms")),
    show_default=True,
    help="Directory containing assembled CIFs.",
)
@click.option(
    "--out-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    default=str(C.file("pair_calls")),
    show_default=True,
    help="Output CSV file path for pair calls (e.g., pair_calls.csv).",
)
@click.option(
    "--cuts",
    type=str,
    default="0.5",
    show_default=True,
    help="Comma-separated p_bio thresholds for XTAL classification.",
)
@click.option(
    "--contact-cutoff",
    type=float,
    default=10.0,
    show_default=True,
    help="Distance cutoff (Å) for contact detection.",
)
@click.option(
    "--workers",
    type=int,
    default=16,
    show_default=True,
    help="Number of parallel workers.",
)
def main(
    raw_dir: Path,
    cif_root: Path,
    out_csv: Path,
    cuts: str,
    contact_cutoff: float,
    workers: int,
):
    # Load raw CSVs
    raw_paths = _find_raw_csvs(raw_dir)
    if not raw_paths:
        raise SystemExit(f"No *_asm_raw.csv found under {raw_dir}")

    frames = []
    for p in raw_paths:
        try:
            frames.append(pd.read_csv(p, dtype={"pdb": str}))
        except Exception as e:
            print(f"[warn] failed to read {p}: {e}")
    if not frames:
        raise SystemExit("No readable raw CSVs")

    # Collect assembly info
    auths_dict, rows_by_asm = _collect_from_raw(frames)
    asm_keys = sorted(auths_dict.keys())
    if not asm_keys:
        raise SystemExit("No assemblies found")

    # Run prodigy on all assemblies
    jobs = [(key, cif_root, auths_dict[key], contact_cutoff) for key in asm_keys]
    results = Parallel(n_jobs=workers, prefer="processes")(
        delayed(_precompute_assembly)(*args)
        for args in tqdm(jobs, desc="Assemblies", unit="asm")
    )

    asm_pair_scores: Dict[
        Tuple[str, str], Dict[Tuple[str, str], Tuple[str, float, float]]
    ] = {}
    total_fail = 0
    for key, pair_map, fail in results:
        asm_pair_scores[key] = pair_map
        total_fail += fail

    # Generate output rows
    out_rows = []
    for key in asm_keys:
        pair_map = asm_pair_scores.get(key, {})
        if not pair_map:
            continue
        for row in rows_by_asm.get(key, []):
            for (a, b), (lbl, p_bio, p_xtal) in pair_map.items():
                out_rows.append(
                    {
                        "pdb": key[0],
                        "assembly_id": key[1],
                        "conf_label": row["conf_label"],
                        "coi_author": row["coi_author"],
                        "coi_author_clone": row["coi_author_clone"],
                        "chain_a_auth": a,
                        "chain_b_auth": b,
                        "p_bio": p_bio,
                        "label_raw": lbl,
                    }
                )

    calls_df = pd.DataFrame(out_rows)

    # Save pair_calls.csv
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    calls_df.to_csv(out_csv, index=False)
    print(f"\n[output] Wrote {out_csv} ({len(calls_df)} rows)")

    # Compute statistics
    if not calls_df.empty:
        pb = pd.to_numeric(calls_df["p_bio"], errors="coerce").dropna()
        if not pb.empty:
            stats = _compute_stats(pb)
            print("\n=== p_bio Statistics ===")
            print(stats)

    # Cutoff analysis
    cuts_f = [float(x.strip()) for x in cuts.split(",") if x.strip()]
    asm_total = len(asm_keys)

    if calls_df.empty:
        total_interfaces = 0
        print("\n=== Cutoff Analysis ===")
        print("No interfaces found")
    else:
        unique_pairs = calls_df.drop_duplicates(
            subset=["pdb", "assembly_id", "chain_a_auth", "chain_b_auth"]
        ).copy()
        total_interfaces = len(unique_pairs)

        print("\n=== Cutoff Analysis ===")
        for c in cuts_f:
            unique_pairs["is_xtal"] = pd.to_numeric(
                unique_pairs["p_bio"], errors="coerce"
            ).lt(c)
            xtal_count = int(unique_pairs["is_xtal"].sum())
            asm_with_xtal = int(
                unique_pairs.groupby(["pdb", "assembly_id"])["is_xtal"].any().sum()
            )

            xtal_ratio = xtal_count / total_interfaces if total_interfaces else 0.0
            asm_ratio = asm_with_xtal / asm_total if asm_total else 0.0

            print(f"Cutoff p_bio < {c:.2f}:")
            print(
                f"  XTAL interfaces: {xtal_count}/{total_interfaces} ({xtal_ratio * 100:.1f}%)"
            )
            print(
                f"  Assemblies with XTAL: {asm_with_xtal}/{asm_total} ({asm_ratio * 100:.1f}%)"
            )

    # Print summary
    print("\n=== Summary ===")
    nonempty = sum(1 for k in asm_keys if asm_pair_scores.get(k))
    print(f"Assemblies: {len(asm_keys)} (with contacts: {nonempty})")
    print(f"Interfaces: {total_interfaces}")
    print(f"Prodigy failures: {total_fail}")


if __name__ == "__main__":
    main()

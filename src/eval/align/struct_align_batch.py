#!/usr/bin/env python3
"""
Batch superpose predictions onto reference CIFs using Match-Maker (no ChimeraX).

**Correspondence** matches ``foundation_model/.../step6_calc_rmsd_tmscore.py``:
``representative_sequences`` JSON (cluster → model a3m header), cluster ``.a3m`` under
``--msa-dir``, ``get_alignment_mapping_with_filter`` on aligned model vs reference rows,
and optional ``valid_pair`` column filtering via the two valid-pair entity rows in the MSA
(not model-vs-ref columns).

Writes JSON with **4×4 transform**, **rmsd_ca** / **tm_score_ca** matching step6
(``calc_rmsd`` / ``calc_tm_score`` on the same CA pairs after superposition).
TM uses ``len(ref_coords)`` = number of aligned pairs. **rmsd_ca_inliers** is
``MmResult.aligned_rmsd`` from the Match-Maker inlier set.

Aligned mobile mmCIF is written to each task's ``output_cif`` by default; use
``--no-write-cif`` to skip.

Usage::

  python -m eval.align.struct_align_batch --json alignment_tasks.json --results-json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gemmi
import numpy as np

from eval.align.extract_ca_from_cif import extract_ca_info, read_cif_structure
from nuri.tools import chimera as mm_tools
from utils._config import pipeline_cfg as C

_PATH_KEYS = ("ref_cif", "mobile_cif", "output_cif")


def _path_json_default(obj: object) -> str:
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Not JSON serializable: {type(obj)!r}")


def _parse_a3m(a3m_path: Path) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    current_header: Optional[str] = None
    current_seq: List[str] = []

    with open(a3m_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    sequences[current_header] = "".join(current_seq)
                current_header = line[1:].split()[0]
                current_seq = []
            else:
                filtered = "".join(c for c in line if not c.islower())
                current_seq.append(filtered)

        if current_header is not None:
            sequences[current_header] = "".join(current_seq)

    return sequences


def _get_common_alignment_positions(seq1: str, seq2: str) -> set[int]:
    common_cols: set[int] = set()
    for i in range(min(len(seq1), len(seq2))):
        if seq1[i] != "-" and seq2[i] != "-":
            common_cols.add(i)
    return common_cols


def _get_alignment_mapping_with_filter(
    model_seq: str,
    ref_seq: str,
    allowed_cols: Optional[set[int]],
) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    model_idx = 0
    ref_idx = 0

    for col_idx in range(len(model_seq)):
        mc = model_seq[col_idx]
        rc = ref_seq[col_idx]

        if mc != "-" and rc != "-":
            if allowed_cols is None or col_idx in allowed_cols:
                mapping[model_idx] = ref_idx

        if mc != "-":
            model_idx += 1
        if rc != "-":
            ref_idx += 1

    return mapping


def _yaml_to_alignment_id(yaml_tag: str) -> Optional[str]:
    parts = yaml_tag.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[2][:-1]}"
    return None


def _get_ref_id_from_cif(ref_cif: Path, ref_chain: str) -> Optional[str]:
    parts = ref_cif.stem.split("_")
    if len(parts) >= 2:
        return f"{parts[1]}_{ref_chain}"
    return None


def _resolve_ref_aligned_seq(
    alignments: Dict[str, str], ref_cif: Path, ref_chain: str
) -> Optional[str]:
    ref_id = _get_ref_id_from_cif(ref_cif, ref_chain)
    if ref_id and ref_id in alignments:
        return alignments[ref_id]

    stem_parts = ref_cif.stem.split("_")
    possible_ids = [
        ref_id,
        ref_id.lower() if ref_id else None,
        ref_id.upper() if ref_id else None,
        _yaml_to_alignment_id(ref_cif.stem),
        ref_cif.stem,
        f"{stem_parts[1]}_{ref_chain}" if len(stem_parts) > 1 else None,
    ]
    if ref_id and len(ref_id) > 1:
        possible_ids.append(ref_id[:-1])

    for pid in possible_ids:
        if pid and pid in alignments:
            return alignments[pid]
    return None


def _a3m_path_for_cluster(msa_dir: Path, cluster_id: str) -> Path:
    cid = str(cluster_id).upper()
    sub = str(cluster_id)[1:3].upper()
    return msa_dir / sub / f"{cid}.a3m"


def _calc_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Same as ``step6_calc_rmsd_tmscore.calc_rmsd`` (structures already superposed)."""
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def _calc_tm_score(
    coords1: np.ndarray,
    coords2: np.ndarray,
    seq_length: int,
) -> float:
    """Same as ``step6_calc_rmsd_tmscore.calc_tm_score`` (coords1=ref, coords2=model aligned).

    step6 passes ``seq_length=len(ref_coords)`` (aligned pair count), not ``full_length``.
    """
    if seq_length < 1 or len(coords1) < 1:
        return 0.0
    if seq_length < 22:
        d0 = 0.5
    else:
        d0 = 1.24 * (seq_length - 15) ** (1 / 3) - 1.8
    distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))
    tm_sum = np.sum(1.0 / (1.0 + (distances / d0) ** 2))
    return float(tm_sum / seq_length)


def _apply_transform_xyz(coords: np.ndarray, transform: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    hom = np.concatenate([coords, np.ones((n, 1), dtype=np.float64)], axis=1)
    return (transform @ hom.T).T[:, :3]


def _write_aligned_mobile_cif(mobile_path: Path, transform: np.ndarray, out_path: Path) -> None:
    st = read_cif_structure(mobile_path)
    st.setup_entities()
    rot = transform[:3, :3].astype(np.float64)
    trans = transform[:3, 3].astype(np.float64)

    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    p = atom.pos
                    v = np.array([p.x, p.y, p.z], dtype=np.float64)
                    vn = rot @ v + trans
                    atom.pos = gemmi.Position(float(vn[0]), float(vn[1]), float(vn[2]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    st.make_mmcif_document().write_file(out_path.as_posix())


def _default_results_json_for_tasks(tasks_json: Path) -> Path:
    return tasks_json.parent / f"{tasks_json.stem}_align_results.json"


def _rep_seq_header(rep_seqs: Dict[str, Any], cluster_id: str) -> Optional[str]:
    entry = rep_seqs.get(cluster_id)
    if isinstance(entry, dict):
        header = entry.get("header")
        return str(header) if header else None
    if isinstance(entry, str):
        return entry
    return None


def _build_correspondence_msa(
    task: Dict[str, Any],
    mob_s2c: Dict[int, Any],
    ref_s2c: Dict[int, Any],
    rep_seqs: Dict[str, Any],
    alignments: Dict[str, str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str], Optional[int]]:
    """
    Returns (mobile_coords, ref_coords, error, n_valid_pair_common_cols).

    ``mob_s2c`` / ``ref_s2c`` are ``seq_id_to_coord`` from ``extract_ca_info``
    (0-based ``label_seq_id`` keys, same as foundation step6).
    """
    cluster_id = task.get("cluster_id")
    if not cluster_id:
        return None, None, "missing_cluster_id", None

    model_header = _rep_seq_header(rep_seqs, str(cluster_id))
    if not model_header or model_header not in alignments:
        return None, None, f"model_header_not_in_a3m:{model_header}", None

    model_aligned_seq = alignments[model_header]
    ref_cif = Path(task["ref_cif"])
    ref_chain = str(task.get("ref_chain", "A"))
    ref_aligned_seq = _resolve_ref_aligned_seq(alignments, ref_cif, ref_chain)
    if ref_aligned_seq is None:
        return None, None, "ref_row_not_in_a3m", None

    valid_pair = task.get("valid_pair") or []
    allowed_cols: Optional[set[int]] = None
    n_common: Optional[int] = None

    if isinstance(valid_pair, list) and len(valid_pair) == 2:
        vp0_id = _yaml_to_alignment_id(str(valid_pair[0]))
        vp1_id = _yaml_to_alignment_id(str(valid_pair[1]))
        vp0_seq = alignments.get(vp0_id) if vp0_id else None
        vp1_seq = alignments.get(vp1_id) if vp1_id else None
        if vp0_seq and vp1_seq:
            allowed_cols = _get_common_alignment_positions(vp0_seq, vp1_seq)
            n_common = len(allowed_cols)

    mapping = _get_alignment_mapping_with_filter(
        model_aligned_seq, ref_aligned_seq, allowed_cols
    )

    mob_rows: List[List[float]] = []
    ref_rows: List[List[float]] = []
    for model_idx, ref_idx in mapping.items():
        mi, ri = int(model_idx), int(ref_idx)
        if mi in mob_s2c and ri in ref_s2c:
            mob_rows.append(list(mob_s2c[mi]))
            ref_rows.append(list(ref_s2c[ri]))

    if len(mob_rows) < 3:
        return None, None, f"few_mapped_pairs:{len(mob_rows)}", n_common

    return (
        np.asarray(mob_rows, dtype=np.float64),
        np.asarray(ref_rows, dtype=np.float64),
        None,
        n_common,
    )


def process_one_task(
    task: Dict[str, Any],
    *,
    rep_seqs: Dict[str, Any],
    alignments: Dict[str, str],
    cutoff: float,
    global_ratio: float,
    viol_ratio: float,
    write_cif: bool,
) -> Dict[str, Any]:
    for key in _PATH_KEYS:
        if key in task and task[key] is not None:
            task[key] = Path(task[key])
    ref_path = task["ref_cif"]
    mob_path = task["mobile_cif"]
    ref_chain = str(task.get("ref_chain", "A"))
    mob_chain = str(task.get("mobile_chain", "A"))

    row: Dict[str, Any] = dict(task)
    row["ref_chain"] = ref_chain
    row["mobile_chain"] = mob_chain

    if not ref_path.exists() or not mob_path.exists():
        return {**row, "ok": False, "error": "missing_cif"}

    ref_info = extract_ca_info(ref_path, ref_chain, use_auth_chain=True)
    mob_info = extract_ca_info(mob_path, mob_chain, use_auth_chain=False)
    if ref_info is None:
        return {**row, "ok": False, "error": "extract_ca_ref"}
    if mob_info is None:
        return {**row, "ok": False, "error": "extract_ca_mob"}

    ref_s2c: Dict[int, Any] = ref_info["seq_id_to_coord"]
    mob_s2c: Dict[int, Any] = mob_info["seq_id_to_coord"]
    if len(ref_s2c) < 3 or len(mob_s2c) < 3:
        return {**row, "ok": False, "error": "short_chain"}

    mob_xyz, ref_xyz, err, n_vp_common = _build_correspondence_msa(
        task, mob_s2c, ref_s2c, rep_seqs, alignments
    )
    if err or mob_xyz is None or ref_xyz is None:
        return {**row, "ok": False, "error": err or "no_correspondence"}

    try:
        mm = mm_tools.match_maker(mob_xyz, ref_xyz, cutoff, global_ratio, viol_ratio)
    except Exception as e:
        return {**row, "ok": False, "error": f"match_maker:{e}"}

    transform = np.asarray(mm.transform, dtype=np.float64)
    n_inliers = int(len(mm.selected))

    mob_super = _apply_transform_xyz(mob_xyz, transform)
    n_aligned = int(ref_xyz.shape[0])
    rmsd_ca = _calc_rmsd(ref_xyz, mob_super)
    tm_score_ca = _calc_tm_score(ref_xyz, mob_super, n_aligned)

    row.update(
        {
            "ok": True,
            "correspondence": "msa_a3m",
            "valid_pair_common_residues": n_vp_common,
            "transform_4x4": transform.tolist(),
            "n_pairs": int(mob_xyz.shape[0]),
            "n_inliers": n_inliers,
            "rmsd_ca": rmsd_ca,
            "tm_score_ca": tm_score_ca,
            "rmsd_ca_inliers": float(mm.aligned_rmsd),
        }
    )

    if write_cif:
        out_path = task.get("output_cif")
        if out_path:
            try:
                out_cif = Path(out_path)
                _write_aligned_mobile_cif(mob_path, transform, out_cif)
                row["aligned_cif_written"] = out_cif.resolve()
            except Exception as e:
                row["aligned_cif_written"] = None
                row["write_cif_error"] = str(e)
        else:
            row["aligned_cif_written"] = None
            row["write_cif_note"] = "skipped_no_output_cif_in_task"

    return row


def run_batch(
    json_path: Path,
    results_path: Path,
    *,
    rep_seq_json: Path,
    msa_dir: Path,
    cutoff: float,
    global_ratio: float,
    viol_ratio: float,
    write_cif: bool,
    start: int,
    end: Optional[int],
) -> None:
    with open(rep_seq_json) as f:
        rep_seqs = json.load(f)

    with open(json_path) as f:
        tasks: List[Dict[str, Any]] = json.load(f)
    if end is None:
        end = len(tasks)
    chunk = tasks[start:end]

    a3m_cache: Dict[str, Dict[str, str]] = {}

    def alignments_for(cluster_id: str) -> Optional[Dict[str, str]]:
        if cluster_id in a3m_cache:
            return a3m_cache[cluster_id]
        a3m_path = _a3m_path_for_cluster(msa_dir, cluster_id)
        if not a3m_path.exists():
            return None
        a3m_cache[cluster_id] = _parse_a3m(a3m_path)
        return a3m_cache[cluster_id]

    results: List[Dict[str, Any]] = []
    for i, raw in enumerate(chunk):
        global_idx = start + i
        task: Dict[str, Any] = dict(raw)
        for key in _PATH_KEYS:
            if key in task and task[key] is not None:
                task[key] = Path(task[key])

        cluster_id = task.get("cluster_id")
        if not cluster_id:
            results.append({**task, "task_idx": global_idx, "ok": False, "error": "missing_cluster_id"})
            continue

        aln = alignments_for(str(cluster_id))
        if aln is None:
            results.append({**task, "task_idx": global_idx, "ok": False, "error": "missing_a3m"})
            continue

        result = process_one_task(
            task,
            rep_seqs=rep_seqs,
            alignments=aln,
            cutoff=cutoff,
            global_ratio=global_ratio,
            viol_ratio=viol_ratio,
            write_cif=write_cif,
        )
        result["task_idx"] = global_idx
        results.append(result)

        if (i + 1) % 50 == 0 or i == 0:
            ok = sum(1 for r in results if r.get("ok"))
            print(f"  [{i + 1}/{len(chunk)}] ok={ok} last_err={result.get('error')}")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_path_json_default)
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"Wrote {len(results)} rows ({n_ok} ok) -> {results_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", "-j", type=Path, required=True, help="alignment_tasks.json")
    p.add_argument(
        "--results-json",
        "-o",
        type=Path,
        default=None,
        help="Output path (default: <json_stem>_align_results.json)",
    )
    p.add_argument(
        "--rep-seq-json",
        type=Path,
        default=C.file("rep_seq"),
        help="cluster_id → header JSON (default: pipeline.rep_seq)",
    )
    p.add_argument(
        "--msa-dir",
        type=Path,
        default=C.dir("msas"),
        help="MSA root with XX/*.a3m (default: pipeline.dirs.msas)",
    )
    p.add_argument("--cutoff", type=float, default=2.0)
    p.add_argument("--global-ratio", type=float, default=0.1)
    p.add_argument("--viol-ratio", type=float, default=0.5)
    p.add_argument(
        "--write-cif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write aligned mobile mmCIF to each task's output_cif",
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    args = p.parse_args()

    tasks_path = args.json.resolve()
    out_path = (
        args.results_json.resolve()
        if args.results_json is not None
        else _default_results_json_for_tasks(tasks_path)
    )

    run_batch(
        tasks_path,
        out_path,
        rep_seq_json=args.rep_seq_json.resolve(),
        msa_dir=args.msa_dir.resolve(),
        cutoff=args.cutoff,
        global_ratio=args.global_ratio,
        viol_ratio=args.viol_ratio,
        write_cif=args.write_cif,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()

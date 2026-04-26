#!/usr/bin/env python3
"""
Batch superpose predictions onto reference CIFs using Match-Maker (no ChimeraX).

This module is the **public** alignment batch runner for ProMiSE-bench. It is backed by
the NuriKit Match-Maker implementation, but the CLI / filenames are intentionally kept
tool-agnostic so downstream pipelines do not encode the library name.

**Correspondence** matches ``foundation_model/.../step6_calc_rmsd_tmscore.py``:
``representative_sequences`` JSON (cluster → model a3m header), cluster ``.a3m`` under
``--msa-dir``, ``get_alignment_mapping_with_filter`` on aligned model vs reference rows,
and optional ``valid_pair`` column filtering via ``get_common_alignment_positions``.

Writes JSON rows with **4×4 transform**, **rmsd_ca** / **tm_score_ca** matching the
foundation_model outputs on the same CA pairs (MSA mapping + optional ``valid_pair``
columns), after a rigid transform.

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


def _get_common_alignment_positions(seq1: str, seq2: str) -> set:
    common_cols: set = set()
    for i in range(min(len(seq1), len(seq2))):
        if seq1[i] != "-" and seq2[i] != "-":
            common_cols.add(i)
    return common_cols


def _get_alignment_mapping_with_filter(
    model_seq: str,
    ref_seq: str,
    allowed_cols: Optional[set],
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


def _get_ref_id_from_cif(ref_cif: str, ref_chain: str) -> Optional[str]:
    path = Path(ref_cif)
    name = path.stem
    parts = name.split("_")
    if len(parts) >= 2:
        pdb_id = parts[1]
        return f"{pdb_id}_{ref_chain}"
    return None


def _resolve_ref_aligned_seq(
    alignments: Dict[str, str], ref_cif: str, ref_chain: str
) -> Optional[str]:
    ref_id = _get_ref_id_from_cif(ref_cif, ref_chain)
    if ref_id and ref_id in alignments:
        return alignments[ref_id]
    possible_ids = [
        ref_id,
        _yaml_to_alignment_id(Path(ref_cif).stem),
        Path(ref_cif).stem,
    ]
    for pid in possible_ids:
        if pid and pid in alignments:
            return alignments[pid]
    return None


def _calc_rmsd(mobile: np.ndarray, ref: np.ndarray) -> float:
    if mobile.shape != ref.shape or mobile.size == 0:
        return float("nan")
    diff = mobile - ref
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _calc_tm_score(mobile: np.ndarray, ref: np.ndarray) -> float:
    # TM-score normalized by number of aligned pairs (foundation_model step6 behavior)
    if mobile.shape != ref.shape or mobile.shape[0] == 0:
        return float("nan")
    l = mobile.shape[0]
    d0 = 1.24 * (l - 15) ** (1 / 3) - 1.8 if l > 15 else 0.5
    d = np.sqrt(np.sum((mobile - ref) ** 2, axis=1))
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))


def _apply_transform_xyz(xyz: np.ndarray, tf: np.ndarray) -> np.ndarray:
    ones = np.ones((xyz.shape[0], 1), dtype=xyz.dtype)
    homo = np.concatenate([xyz, ones], axis=1)
    out = homo @ tf.T
    return out[:, :3]


def _write_aligned_mobile_cif(mobile_cif: str, out_cif: str, tf: np.ndarray) -> None:
    st = read_cif_structure(mobile_cif)
    for model in st:
        for chain in model:
            for res in chain:
                for at in res:
                    pos = np.array([at.pos.x, at.pos.y, at.pos.z], dtype=float)[None, :]
                    new_pos = _apply_transform_xyz(pos, tf)[0]
                    at.pos = gemmi.Position(*map(float, new_pos))
    Path(out_cif).parent.mkdir(parents=True, exist_ok=True)
    st.make_mmcif_document().write_file(out_cif)


def _default_results_json_for_tasks(tasks_json: Path) -> Path:
    # Keep the default filename tool-agnostic.
    return tasks_json.parent / f"{tasks_json.stem}_align_results.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", required=True, help="alignment task JSON (list of dicts)")
    p.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Output JSON (list of per-task results). Default: <tasks>_align_results.json",
    )
    p.add_argument(
        "--msa-dir",
        type=str,
        default=str(C.dir("msas")),
        help="Directory containing per-cluster a3m (default: pipeline.msas)",
    )
    p.add_argument(
        "--rep-seq-json",
        type=str,
        default=str(C.file("rep_seq")),
        help="representative_sequences JSON (default: pipeline.rep_seq)",
    )
    p.add_argument(
        "--write-cif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write aligned mobile CIF to each task's output_cif",
    )
    args = p.parse_args()

    tasks_path = Path(args.json).resolve()
    out_path = (
        Path(args.results_json).resolve()
        if args.results_json
        else _default_results_json_for_tasks(tasks_path)
    )

    msa_dir = Path(args.msa_dir).resolve()
    rep_seq_json = Path(args.rep_seq_json).resolve()

    with open(rep_seq_json) as f:
        rep = json.load(f)

    with open(tasks_path) as f:
        tasks: List[Dict[str, Any]] = json.load(f)

    results: List[Dict[str, Any]] = []
    for task_idx, t in enumerate(tasks):
        row: Dict[str, Any] = dict(t)
        row["task_idx"] = task_idx

        try:
            mobile_cif = str(t["mobile_cif"])
            ref_cif = str(t["ref_cif"])
            mobile_chain = str(t["mobile_chain"])
            ref_chain = str(t["ref_chain"])

            # Load CA coords
            mobile_ca = extract_ca_info(mobile_cif, mobile_chain)
            ref_ca = extract_ca_info(ref_cif, ref_chain)

            # MSA mapping
            cluster_id = str(t["cluster_id"])
            a3m_path = msa_dir / cluster_id / f"{cluster_id}.a3m"
            if not a3m_path.exists():
                raise FileNotFoundError(f"Missing a3m: {a3m_path}")
            ali = _parse_a3m(a3m_path)

            yaml_tag = str(t.get("prediction_yaml_tag") or t.get("model_entity") or "")
            header = rep.get(cluster_id)
            if isinstance(header, dict):
                header = header.get("header")
            if not header:
                raise KeyError(f"Missing rep header for cluster_id={cluster_id}")
            model_seq = ali.get(str(header))
            ref_seq = _resolve_ref_aligned_seq(ali, ref_cif, ref_chain)
            if not model_seq or not ref_seq:
                raise KeyError("Missing aligned sequences for model/ref")

            allowed_cols = None
            if t.get("valid_pair"):
                allowed_cols = _get_common_alignment_positions(model_seq, ref_seq)

            mapping = _get_alignment_mapping_with_filter(model_seq, ref_seq, allowed_cols)
            if not mapping:
                raise ValueError("Empty MSA mapping")

            # Build paired coords
            mobile_xyz = []
            ref_xyz = []
            for m_i, r_i in mapping.items():
                if m_i in mobile_ca.seq_id_to_coord and r_i in ref_ca.seq_id_to_coord:
                    mobile_xyz.append(mobile_ca.seq_id_to_coord[m_i])
                    ref_xyz.append(ref_ca.seq_id_to_coord[r_i])

            mobile_xyz = np.asarray(mobile_xyz, dtype=float)
            ref_xyz = np.asarray(ref_xyz, dtype=float)
            if mobile_xyz.shape[0] == 0:
                raise ValueError("No paired CA coordinates after mapping")

            mm = mm_tools.match_maker(ref_xyz, mobile_xyz)
            tf = np.asarray(mm.transform, dtype=float)

            mobile_aligned = _apply_transform_xyz(mobile_xyz, tf)
            row["ok"] = True
            row["transform_4x4"] = tf.tolist()
            row["rmsd_ca"] = _calc_rmsd(mobile_aligned, ref_xyz)
            row["tm_score_ca"] = _calc_tm_score(mobile_aligned, ref_xyz)
            row["rmsd_ca_inliers"] = getattr(mm, "aligned_rmsd", None)

            if args.write_cif and t.get("output_cif"):
                _write_aligned_mobile_cif(mobile_cif, str(t["output_cif"]), tf)

        except Exception as e:
            row["ok"] = False
            row["error"] = str(e)

        results.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()


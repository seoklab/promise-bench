#!/usr/bin/env python3
"""
Reference–reference structural metrics from ``distogram_tasks.json``.

Moved from ``eval/align`` to ``eval/struct`` because it produces structure-level
metrics (reference↔reference RMSD/TM-score/lDDT) rather than alignment batches.

Adapted from ``foundation_model/.../step7_calc_reference_structural_metrics.py`` with:

- **NuriKit** ``match_maker`` instead of ChimeraX (no subprocess / no step5 extract).
- **TM-score** normalized by **aligned pair count** (same as
  ``step6_calc_rmsd_tmscore.py``: ``len(ref_coords)``), not ``full_length``.

**Outputs** (under ``--output_dir``, default ``./reference_metrics``):

- ``reference_alignment_tasks.json`` — one row per ref1/ref2 pair (paths + metadata).
- ``aligned_references/<cluster_id>/<prediction_yaml_tag>/`` — aligned ref2 mmCIF
  (ref2 superposed onto ref1) when alignment runs.
- Per-pair JSON:
  ``.../<ref1_tag>_<ref1_conf>_vs_<ref2_tag>_<ref2_conf>_metrics.json``.

Usage::

  python -m eval.struct.calc_reference_structural_metrics --tasks distogram_tasks.json
  python -m eval.struct.calc_reference_structural_metrics -t tasks.json -o out_ref_metrics
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from nuri.tools import chimera as mm_tools

from eval.align.extract_ca_from_cif import extract_ca_info
from eval.align.struct_align_batch import (
    _apply_transform_xyz,
    _calc_rmsd,
    _calc_tm_score,
    _write_aligned_mobile_cif,
)
from utils._config import pipeline_cfg as C


@dataclass
class ReferenceComparison:
    task_idx: int
    prediction_yaml_tag: str
    cluster_id: str
    method: str
    method_type: str
    ref1_tag: str
    ref1_cif: str
    ref1_chain: str
    ref1_conf: str
    ref1_state: str
    ref2_tag: str
    ref2_cif: str
    ref2_chain: str
    ref2_conf: str
    ref2_state: str
    comparison_key: str


@dataclass
class StructuralMetrics:
    comparison_key: str
    alignment_rmsd: Optional[float] = None
    tm_score: Optional[float] = None
    lddt_score: Optional[float] = None
    ca_rmsd: Optional[float] = None
    aligned_residues: Optional[int] = None
    ref1_length: Optional[int] = None
    ref2_length: Optional[int] = None
    error: Optional[str] = None


def extract_reference_pairs(distogram_tasks: List[Dict]) -> List[ReferenceComparison]:
    comparisons: List[ReferenceComparison] = []
    seen_comparisons: set[str] = set()

    for task_idx, task in enumerate(distogram_tasks):
        references = task.get("references", [])
        if len(references) < 2:
            continue

        for i, ref1 in enumerate(references):
            for j, ref2 in enumerate(references):
                if i >= j:
                    continue

                unique_key = (
                    f"{ref1['ref_cif']}:{ref1['ref_chain']}_vs_{ref2['ref_cif']}:"
                    f"{ref2['ref_chain']}"
                )
                if unique_key in seen_comparisons:
                    continue
                seen_comparisons.add(unique_key)

                comparison_key = (
                    f"{ref1['reference_yaml_tag']}_vs_{ref2['reference_yaml_tag']}"
                )
                comparisons.append(
                    ReferenceComparison(
                        task_idx=task_idx,
                        prediction_yaml_tag=task["prediction_yaml_tag"],
                        cluster_id=task["cluster_id"],
                        method=task["method"],
                        method_type=task["method_type"],
                        ref1_tag=ref1["reference_yaml_tag"],
                        ref1_cif=ref1["ref_cif"],
                        ref1_chain=ref1["ref_chain"],
                        ref1_conf=ref1["reference_conformation"],
                        ref1_state=ref1["reference_state"],
                        ref2_tag=ref2["reference_yaml_tag"],
                        ref2_cif=ref2["ref_cif"],
                        ref2_chain=ref2["ref_chain"],
                        ref2_conf=ref2["reference_conformation"],
                        ref2_state=ref2["reference_state"],
                        comparison_key=comparison_key,
                    )
                )

    return comparisons


def aligned_ref2_cif_path(comp: ReferenceComparison, output_dir: Path) -> Path:
    aligned_dir = (
        output_dir / "aligned_references" / comp.cluster_id / comp.prediction_yaml_tag
    )
    filename = (
        f"{comp.ref2_tag}_{comp.ref2_conf}_aligned_to_{comp.ref1_tag}_{comp.ref1_conf}.cif"
    )
    return aligned_dir / filename


def create_alignment_tasks(
    comparisons: List[ReferenceComparison],
    output_dir: Path,
    skip_existing_alignments: bool = False,
) -> Path:
    """Write ``reference_alignment_tasks.json`` (provenance + paths for aligned CIFs)."""
    tasks: List[Dict[str, Any]] = []
    seen_tasks: set[str] = set()
    skipped = 0

    for comp in comparisons:
        aligned_cif = aligned_ref2_cif_path(comp, output_dir)
        if skip_existing_alignments and (
            aligned_cif.exists() or (aligned_cif.parent / (aligned_cif.name + ".gz")).exists()
        ):
            skipped += 1
            continue

        task_key = f"{comp.ref1_cif}:{comp.ref1_chain}_{comp.ref2_cif}:{comp.ref2_chain}"
        if task_key in seen_tasks:
            continue
        seen_tasks.add(task_key)

        tasks.append(
            {
                "ref_cif": comp.ref1_cif,
                "mobile_cif": comp.ref2_cif,
                "output_cif": str(aligned_cif),
                "ref_chain": comp.ref1_chain,
                "mobile_chain": comp.ref2_chain,
                "comparison_key": comp.comparison_key,
                "cluster_id": comp.cluster_id,
                "prediction_yaml_tag": comp.prediction_yaml_tag,
                "ref1_yaml_tag": comp.ref1_tag,
                "ref2_yaml_tag": comp.ref2_tag,
            }
        )

    out_file = output_dir / "reference_alignment_tasks.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"  Alignment tasks: created={len(tasks)}, skipped_existing={skipped}")
    return out_file


def parse_a3m(a3m_path: Path) -> Dict[str, str]:
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
                current_seq.append("".join(c for c in line if not c.islower()))
        if current_header is not None:
            sequences[current_header] = "".join(current_seq)

    return sequences


def get_ref_id_from_cif(ref_cif: str, ref_chain: str) -> Optional[str]:
    path = Path(ref_cif)
    parts = path.stem.split("_")
    if len(parts) >= 2:
        pdb_id = parts[1]
        return f"{pdb_id}_{ref_chain}"
    return None


def find_sequence_in_msa(
    ref_tag: str, ref_cif: str, ref_chain: str, msa_sequences: Dict[str, str]
) -> Tuple[Optional[str], Optional[str]]:
    if ref_tag in msa_sequences:
        return ref_tag, msa_sequences[ref_tag]

    ref_id = get_ref_id_from_cif(ref_cif, ref_chain)
    possible_ids = [
        ref_tag,
        ref_tag.lower() if ref_tag else None,
        ref_tag.upper() if ref_tag else None,
        ref_id,
        ref_id.lower() if ref_id else None,
        ref_id.upper() if ref_id else None,
        f"{Path(ref_cif).stem.split('_')[1]}_{ref_chain}"
        if len(Path(ref_cif).stem.split("_")) > 1
        else None,
        ref_id[:-1] if ref_id and len(ref_id) > 1 else None,
    ]
    for pid in possible_ids:
        if pid and pid in msa_sequences:
            return pid, msa_sequences[pid]
    return None, None


def get_alignment_mapping(model_seq: str, ref_seq: str) -> Dict[int, int]:
    """model_idx (first sequence) -> ref_idx (second sequence) at non-gap columns."""
    mapping: Dict[int, int] = {}
    model_idx = 0
    ref_idx = 0
    for i in range(len(model_seq)):
        mc, rc = model_seq[i], ref_seq[i]
        if mc != "-" and rc != "-":
            mapping[model_idx] = ref_idx
        if mc != "-":
            model_idx += 1
        if rc != "-":
            ref_idx += 1
    return mapping


def calc_lddt(
    ref_coords: np.ndarray,
    model_coords: np.ndarray,
    cutoff: float = 15.0,
    thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> float:
    n_residues = len(ref_coords)
    if n_residues < 2:
        return 0.0

    ref_dist = cdist(ref_coords, ref_coords, metric="euclidean")
    model_dist = cdist(model_coords, model_coords, metric="euclidean")

    mask = np.ones((n_residues, n_residues), dtype=bool)
    np.fill_diagonal(mask, False)
    for i in range(n_residues - 1):
        mask[i, i + 1] = False
        mask[i + 1, i] = False

    within_cutoff = (ref_dist < cutoff) & mask
    if not np.any(within_cutoff):
        return 0.0

    dist_diff = np.abs(ref_dist - model_dist)
    total_preserved = 0
    total_pairs = int(np.sum(within_cutoff))
    for thresh in thresholds:
        preserved = np.sum((dist_diff < thresh) & within_cutoff)
        total_preserved += int(preserved)

    return float(total_preserved / (len(thresholds) * total_pairs))


def calculate_structural_metrics(
    comparisons: List[ReferenceComparison],
    msa_dir: Path,
    output_dir: Path,
    *,
    skip_alignment: bool,
    skip_existing_alignments: bool,
    cutoff: float,
    global_ratio: float,
    viol_ratio: float,
) -> List[StructuralMetrics]:
    results: List[StructuralMetrics] = []

    for comp in comparisons:
        print(f"Processing comparison: {comp.comparison_key}")
        aligned_ref2_path = aligned_ref2_cif_path(comp, output_dir)

        try:
            cluster_prefix = comp.cluster_id[1:3].upper()
            msa_file = msa_dir / cluster_prefix / f"{comp.cluster_id.upper()}.a3m"
            if not msa_file.exists():
                results.append(
                    StructuralMetrics(
                        comparison_key=comp.comparison_key,
                        error=f"MSA file not found: {msa_file}",
                    )
                )
                continue

            msa_sequences = parse_a3m(msa_file)
            _, ref1_aligned_seq = find_sequence_in_msa(
                comp.ref1_tag, comp.ref1_cif, comp.ref1_chain, msa_sequences
            )
            _, ref2_aligned_seq = find_sequence_in_msa(
                comp.ref2_tag, comp.ref2_cif, comp.ref2_chain, msa_sequences
            )

            if ref1_aligned_seq is None or ref2_aligned_seq is None:
                results.append(
                    StructuralMetrics(
                        comparison_key=comp.comparison_key,
                        error="Reference sequences not found in MSA",
                    )
                )
                continue

            mapping = get_alignment_mapping(ref1_aligned_seq, ref2_aligned_seq)

            ref1_ca = extract_ca_info(
                Path(comp.ref1_cif), comp.ref1_chain, use_auth_chain=True
            )
            if not ref1_ca:
                results.append(
                    StructuralMetrics(
                        comparison_key=comp.comparison_key,
                        error="extract_ca ref1 failed",
                    )
                )
                continue

            ref1_coords = {int(k): v for k, v in ref1_ca["seq_id_to_coord"].items()}

            if skip_alignment:
                gz = aligned_ref2_path.parent / (aligned_ref2_path.name + ".gz")
                if aligned_ref2_path.exists():
                    p2 = aligned_ref2_path
                elif gz.exists():
                    p2 = gz
                else:
                    results.append(
                        StructuralMetrics(
                            comparison_key=comp.comparison_key,
                            error=f"Aligned ref2 not found (--skip-alignment): {aligned_ref2_path}",
                        )
                    )
                    continue
                ref2_ca = extract_ca_info(p2, comp.ref2_chain, use_auth_chain=True)
            else:
                ref2_ca = extract_ca_info(
                    Path(comp.ref2_cif), comp.ref2_chain, use_auth_chain=True
                )

            if not ref2_ca:
                results.append(
                    StructuralMetrics(
                        comparison_key=comp.comparison_key,
                        error="extract_ca ref2 failed",
                    )
                )
                continue

            ref2_coords = {int(k): v for k, v in ref2_ca["seq_id_to_coord"].items()}

            templ_rows: List[List[float]] = []
            q_rows: List[List[float]] = []
            for idx1, idx2 in mapping.items():
                if idx1 in ref1_coords and idx2 in ref2_coords:
                    templ_rows.append(list(ref1_coords[idx1]))
                    q_rows.append(list(ref2_coords[idx2]))

            if len(templ_rows) < 3:
                results.append(
                    StructuralMetrics(
                        comparison_key=comp.comparison_key,
                        error=f"Too few aligned coordinates: {len(templ_rows)}",
                    )
                )
                continue

            templ = np.asarray(templ_rows, dtype=np.float64)
            q = np.asarray(q_rows, dtype=np.float64)

            if skip_alignment:
                ca_rmsd = _calc_rmsd(templ, q)
                n_aligned = int(templ.shape[0])
                tm_score = _calc_tm_score(templ, q, n_aligned)
                lddt_score = calc_lddt(templ, q)
                T = None
            else:
                mm = mm_tools.match_maker(q, templ, cutoff, global_ratio, viol_ratio)
                T = np.asarray(mm.transform, dtype=np.float64)
                q_super = _apply_transform_xyz(q, T)
                ca_rmsd = _calc_rmsd(templ, q_super)
                n_aligned = int(templ.shape[0])
                tm_score = _calc_tm_score(templ, q_super, n_aligned)
                lddt_score = calc_lddt(templ, q_super)

                should_write = True
                if skip_existing_alignments and (
                    aligned_ref2_path.exists()
                    or (aligned_ref2_path.parent / (aligned_ref2_path.name + ".gz")).exists()
                ):
                    should_write = False
                if should_write:
                    aligned_ref2_path.parent.mkdir(parents=True, exist_ok=True)
                    _write_aligned_mobile_cif(Path(comp.ref2_cif), T, aligned_ref2_path)

            ref1_len = ref1_ca.get("full_length", len(ref1_coords))
            ref2_len = ref2_ca.get("full_length", len(ref2_coords))

            result = StructuralMetrics(
                comparison_key=comp.comparison_key,
                tm_score=tm_score,
                lddt_score=lddt_score,
                ca_rmsd=ca_rmsd,
                aligned_residues=n_aligned,
                ref1_length=int(ref1_len) if ref1_len is not None else None,
                ref2_length=int(ref2_len) if ref2_len is not None else None,
            )
            results.append(result)

            result_dict: Dict[str, Any] = {
                "cluster_id": comp.cluster_id,
                "prediction_yaml_tag": comp.prediction_yaml_tag,
                "comparison_key": comp.comparison_key,
                "ref1_tag": comp.ref1_tag,
                "ref1_conf": comp.ref1_conf,
                "ref1_state": comp.ref1_state,
                "ref2_tag": comp.ref2_tag,
                "ref2_conf": comp.ref2_conf,
                "ref2_state": comp.ref2_state,
                "tm_score": result.tm_score,
                "tm_norm_length": n_aligned,
                "lddt_score": result.lddt_score,
                "ca_rmsd": result.ca_rmsd,
                "aligned_residues": result.aligned_residues,
                "ref1_length": result.ref1_length,
                "ref2_length": result.ref2_length,
                "ref1_cif": comp.ref1_cif,
                "ref2_cif": comp.ref2_cif,
                "aligned_ref2_path": str(aligned_ref2_path),
                "alignment_backend": "nurikit match_maker",
                "skip_alignment_mode": skip_alignment,
                "error": result.error,
            }
            if T is not None:
                result_dict["transform"] = T.tolist()

            individual_results_dir = (
                output_dir / "aligned_references" / comp.cluster_id / comp.prediction_yaml_tag
            )
            individual_results_dir.mkdir(parents=True, exist_ok=True)
            result_filename = (
                f"{comp.ref1_tag}_{comp.ref1_conf}_vs_{comp.ref2_tag}_{comp.ref2_conf}_metrics.json"
            )
            with open(individual_results_dir / result_filename, "w") as f:
                json.dump(result_dict, f, indent=2)
            print(f"  Saved result to: {individual_results_dir / result_filename}")

        except Exception as e:
            results.append(StructuralMetrics(comparison_key=comp.comparison_key, error=str(e)))
            print(f"Error processing {comp.comparison_key}: {e}")
            err_dir = (
                output_dir / "aligned_references" / comp.cluster_id / comp.prediction_yaml_tag
            )
            err_dir.mkdir(parents=True, exist_ok=True)
            result_filename = (
                f"{comp.ref1_tag}_{comp.ref1_conf}_vs_{comp.ref2_tag}_{comp.ref2_conf}_metrics.json"
            )
            with open(err_dir / result_filename, "w") as f:
                json.dump(
                    {
                        "cluster_id": comp.cluster_id,
                        "prediction_yaml_tag": comp.prediction_yaml_tag,
                        "comparison_key": comp.comparison_key,
                        "ref1_tag": comp.ref1_tag,
                        "ref2_tag": comp.ref2_tag,
                        "error": str(e),
                    },
                    f,
                    indent=2,
                )

    return results


def process_distogram_tasks(
    tasks_json: Path,
    output_dir: Path,
    msa_dir: Path,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    skip_alignment: bool = False,
    skip_existing_alignments: bool = False,
    cutoff: float = 2.0,
    global_ratio: float = 0.1,
    viol_ratio: float = 0.5,
) -> None:
    with open(tasks_json) as f:
        distogram_tasks: List[Dict] = json.load(f)

    print(f"Loaded {len(distogram_tasks)} distogram tasks from {tasks_json}")

    if end_idx is None:
        end_idx = len(distogram_tasks)
    distogram_tasks = distogram_tasks[start_idx:end_idx]
    print(f"Processing tasks [{start_idx}:{end_idx}] ({len(distogram_tasks)} tasks)")

    print("Extracting reference pairs...")
    comparisons = extract_reference_pairs(distogram_tasks)
    print(f"Found {len(comparisons)} unique reference comparisons after deduplication")

    if not comparisons:
        print("No reference comparisons found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if not skip_alignment:
        print("Creating alignment task manifest...")
        create_alignment_tasks(comparisons, output_dir, skip_existing_alignments)

    print("Calculating structural metrics...")
    results = calculate_structural_metrics(
        comparisons,
        msa_dir,
        output_dir,
        skip_alignment=skip_alignment,
        skip_existing_alignments=skip_existing_alignments,
        cutoff=cutoff,
        global_ratio=global_ratio,
        viol_ratio=viol_ratio,
    )

    successful = sum(1 for r in results if r.error is None)
    failed = sum(1 for r in results if r.error is not None)
    print("\nSummary:")
    print(f"  Total comparisons: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", "-t", type=Path, required=True, help="distogram_tasks.json")
    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("reference_metrics"),
        help="Root output directory (default: ./reference_metrics)",
    )
    p.add_argument(
        "--msa-dir",
        type=Path,
        default=C.dir("msas"),
        help="MSA root with XX/*.a3m (default: pipeline.dirs.msas)",
    )
    p.add_argument("--start", "-s", type=int, default=0)
    p.add_argument("--end", "-e", type=int, default=None)
    p.add_argument(
        "--skip-alignment",
        action="store_true",
        help="Skip alignment; read ref2 from existing aligned CIF (e.g. prior run)",
    )
    p.add_argument(
        "--skip-existing-alignments",
        action="store_true",
        help="Skip writing aligned CIF when output already exists; still computes metrics",
    )
    p.add_argument("--cutoff", type=float, default=2.0)
    p.add_argument("--global-ratio", type=float, default=0.1)
    p.add_argument("--viol-ratio", type=float, default=0.5)

    args = p.parse_args()
    if not args.tasks.exists():
        raise FileNotFoundError(args.tasks)

    process_distogram_tasks(
        tasks_json=args.tasks.resolve(),
        output_dir=args.output_dir.resolve(),
        msa_dir=args.msa_dir.resolve(),
        start_idx=args.start,
        end_idx=args.end,
        skip_alignment=args.skip_alignment,
        skip_existing_alignments=args.skip_existing_alignments,
        cutoff=args.cutoff,
        global_ratio=args.global_ratio,
        viol_ratio=args.viol_ratio,
    )


if __name__ == "__main__":
    main()


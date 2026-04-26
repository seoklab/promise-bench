#!/usr/bin/env python3
"""
Check progress of distogram calculation tasks.

The output file is 'distogram_loss_final.json' in the prediction_distograms directory.
E.g., distogram/af3/apo-monomers/8ABP_1/2wrz_2_B1_m/distogram_loss_final.json

Usage (``PYTHONPATH=src``)::

    python -m eval.distogram.check_distogram_results --tasks data_eval/distogram/distogram_tasks.json
    python -m eval.distogram.check_distogram_results --tasks …/distogram_tasks.json --verbose
"""

from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import argparse
import json

from utils._config import eval_cfg as E
from eval.distogram.path_utils import (
    flatten_valid_pair_edges,
    reference_distogram_diff_path,
)


def get_output_dir(task: dict) -> Path:
    """Get the output directory for distogram_loss_final.json from task."""
    prediction_distograms = task.get("prediction_distograms", [])
    if prediction_distograms:
        # Output is in the same directory as the prediction distograms
        return Path(prediction_distograms[0]).parent
    return None


def get_expected_dynamic_pairs(
    task: dict,
    ref_distogram_dir: Path,
    valid_pair_edges: set[tuple[str, str]],
) -> set:
    """
    Get expected dynamic region pairs from reference_distogram_diff.json.
    Returns set of pair_keys like "ref_A|ref_B" (sorted alphabetically).
    """
    # Extract method_type from prediction_distograms path (more reliable than task["method_type"])
    prediction_distograms = task.get("prediction_distograms", [])
    if prediction_distograms:
        # Path format: distogram/METHOD/METHOD_TYPE/CLUSTER/PREDICTION_TAG/...
        path_parts = Path(prediction_distograms[0]).parts
        if len(path_parts) >= 3:
            method_type_from_path = path_parts[2]  # e.g., "ligand-induced"
        else:
            method_type_from_path = task.get("method_type", "")
    else:
        method_type_from_path = task.get("method_type", "")

    cluster_id = task.get("cluster_id", "")
    prediction_yaml_tag = task.get("prediction_yaml_tag", "")

    tcopy = dict(task)
    if method_type_from_path:
        tcopy["method_type"] = method_type_from_path
    ref_distogram_diff_path = reference_distogram_diff_path(ref_distogram_dir, tcopy)

    if not ref_distogram_diff_path.exists():
        return set()

    try:
        with open(ref_distogram_diff_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "pairwise_comparisons" not in data:
            return set()

        # Get all reference_yaml_tags from task
        reference_yaml_tags = set(
            ref.get("reference_yaml_tag")
            for ref in task.get("references", [])
            if ref.get("reference_yaml_tag")
        )

        expected_pairs = set()
        for comparison in data["pairwise_comparisons"]:
            ref_A = comparison.get("reference_A", "")
            ref_B = comparison.get("reference_B", "")
            if ref_A and ref_B:
                # Sorted (ref_A, ref_B) matches ``calc_distogram_loss`` / ``calc_reference_distogram_diff`` pair keys
                if (ref_A, ref_B) not in valid_pair_edges:
                    continue
                
                # Apply filtering logic for non-apo-monomers
                if method_type_from_path != 'apo-monomers':
                    # For each reference in the task, check if pair should be included
                    should_include = False
                    for reference_yaml_tag in reference_yaml_tags:
                        if reference_yaml_tag != prediction_yaml_tag:
                            # Check if pair involves the reference or prediction
                            if '_x' in reference_yaml_tag:
                                if ref_A == reference_yaml_tag or ref_B == reference_yaml_tag:
                                    should_include = True
                            if '_x' in prediction_yaml_tag:
                                if ref_A == prediction_yaml_tag or ref_B == prediction_yaml_tag:
                                    should_include = True
                        else:
                            # If reference == prediction, include all pairs
                            should_include = True
                    
                    if not should_include:
                        continue
                
                pair_key = "|".join(sorted([ref_A, ref_B]))
                expected_pairs.add(pair_key)

        return expected_pairs
    except Exception:
        return set()


def check_distogram_results(
    tasks_json: Path,
    ref_distogram_dir: Path,
    valid_pair_edges: set[tuple[str, str]],
    verbose: bool = False,
):
    """Check if distogram_loss_final.json exists and has all expected references and dynamic pairs for all tasks."""

    with open(tasks_json, "r") as f:
        tasks = json.load(f)

    total_tasks = len(tasks)
    missing = []  # File doesn't exist
    empty = []  # File exists but empty
    incomplete_refs = []  # File exists but missing some references
    incomplete_pairs = []  # File exists but missing some dynamic pairs
    success = []

    for i, task in enumerate(tasks):
        output_dir = get_output_dir(task)
        if output_dir is None:
            missing.append((i, task, None))
            continue

        result_file = output_dir / "distogram_loss_real_final.json"

        if not result_file.exists():
            missing.append((i, task, output_dir))
        else:
            # Check if file has content and all expected references
            try:
                with open(result_file) as f:
                    data = json.load(f)
                if len(data) == 0:
                    empty.append((i, task, output_dir))
                else:
                    # Get expected reference tags from task
                    expected_ref_tags = set(
                        ref.get("reference_yaml_tag")
                        for ref in task.get("references", [])
                        if ref.get("reference_yaml_tag")
                    )

                    # Get existing reference tags from result
                    existing_ref_tags = set()
                    existing_dynamic_pairs = set()
                    for entry in data:
                        for ref in entry.get("references", []):
                            tag = ref.get("reference_yaml_tag")
                            if tag:
                                existing_ref_tags.add(tag)
                            # Collect existing dynamic pairs from mean_dynamic_losses
                            mean_dynamic_losses = ref.get("mean_dynamic_losses", {})
                            for pair_key in mean_dynamic_losses.keys():
                                existing_dynamic_pairs.add(pair_key)

                    missing_refs = expected_ref_tags - existing_ref_tags

                    # Check dynamic pairs independently
                    expected_dynamic_pairs = get_expected_dynamic_pairs(
                        task, ref_distogram_dir, valid_pair_edges
                    )
                    missing_pairs = expected_dynamic_pairs - existing_dynamic_pairs

                    # Classify based on what's missing
                    has_missing_refs = len(missing_refs) > 0
                    has_missing_pairs = len(missing_pairs) > 0

                    if has_missing_refs:
                        incomplete_refs.append(
                            (
                                i,
                                task,
                                output_dir,
                                len(existing_ref_tags),
                                len(expected_ref_tags),
                                missing_refs,
                            )
                        )

                    if has_missing_pairs:
                        incomplete_pairs.append(
                            (
                                i,
                                task,
                                output_dir,
                                len(existing_dynamic_pairs),
                                len(expected_dynamic_pairs),
                                missing_pairs,
                            )
                        )

                    # Only mark as success if both refs and pairs are complete
                    if not has_missing_refs and not has_missing_pairs:
                        success.append((i, task, output_dir, len(data)))
            except Exception as e:
                missing.append((i, task, output_dir))
                if verbose:
                    print(f"Error reading {result_file}: {e}")
                    import traceback

                    traceback.print_exc()

    # Print summary
    total_incomplete = len(incomplete_refs) + len(incomplete_pairs)
    print(f"\n{'=' * 60}")
    print(f"Distogram Calculation Progress: {tasks_json.name}")
    print(f"{'=' * 60}")
    print(f"Total tasks: {total_tasks}")
    print(
        f"  ✓ Complete:            {len(success)} ({len(success) / total_tasks * 100:.1f}%)"
    )
    print(
        f"  △ Incomplete refs:     {len(incomplete_refs)} ({len(incomplete_refs) / total_tasks * 100:.1f}%)"
    )
    print(
        f"  △ Incomplete pairs:    {len(incomplete_pairs)} ({len(incomplete_pairs) / total_tasks * 100:.1f}%)"
    )
    print(
        f"  ✗ Missing:             {len(missing)} ({len(missing) / total_tasks * 100:.1f}%)"
    )
    print(
        f"  ○ Empty:               {len(empty)} ({len(empty) / total_tasks * 100:.1f}%)"
    )
    print(f"{'=' * 60}")

    # Show missing tasks
    if verbose or len(missing) <= 20:
        if missing:
            print(f"\nMissing distogram_loss_final.json ({len(missing)}):")
            for i, task, output_dir in missing[:50]:
                method = task.get("method", "unknown")
                cluster = task.get("cluster_id", "unknown")
                yaml_tag = task.get("prediction_yaml_tag", "unknown")
                print(f"  [{i:>4}] {method}/{cluster}/{yaml_tag}")
            if len(missing) > 50:
                print(f"  ... and {len(missing) - 50} more")

    if verbose or len(empty) <= 20:
        if empty:
            print(f"\nEmpty distogram_loss_final.json ({len(empty)}):")
            for i, task, output_dir in empty[:50]:
                method = task.get("method", "unknown")
                cluster = task.get("cluster_id", "unknown")
                yaml_tag = task.get("prediction_yaml_tag", "unknown")
                print(f"  [{i:>4}] {method}/{cluster}/{yaml_tag}")
            if len(empty) > 50:
                print(f"  ... and {len(empty) - 50} more")

    if verbose or len(incomplete_refs) <= 20:
        if incomplete_refs:
            print(f"\nIncomplete references ({len(incomplete_refs)}):")
            for (
                i,
                task,
                output_dir,
                existing,
                expected,
                missing_refs,
            ) in incomplete_refs[:50]:
                method = task.get("method", "unknown")
                pair_type = task.get("method_type", "unknown")
                cluster = task.get("cluster_id", "unknown")
                yaml_tag = task.get("prediction_yaml_tag", "unknown")
                print(
                    f"  [{i:>4}] {method}/{pair_type}/{cluster}/{yaml_tag} - {existing}/{expected} refs"
                )
                if verbose:
                    print(f"         Missing refs: {missing_refs}")
            if len(incomplete_refs) > 50:
                print(f"  ... and {len(incomplete_refs) - 50} more")

    if verbose or len(incomplete_pairs) <= 20:
        if incomplete_pairs:
            print(f"\nIncomplete dynamic pairs ({len(incomplete_pairs)}):")
            for (
                i,
                task,
                output_dir,
                existing,
                expected,
                missing_pairs,
            ) in incomplete_pairs[:50]:
                method = task.get("method", "unknown")
                pair_type = task.get("method_type", "unknown")
                cluster = task.get("cluster_id", "unknown")
                yaml_tag = task.get("prediction_yaml_tag", "unknown")
                print(
                    f"  [{i:>4}] {method}/{pair_type}/{cluster}/{yaml_tag} - {existing}/{expected} pairs"
                )
                if verbose:
                    print(f"         Missing pairs: {missing_pairs}")
            if len(incomplete_pairs) > 50:
                print(f"  ... and {len(incomplete_pairs) - 50} more")

    # Save missing + incomplete task indices to file for resubmission
    if missing or empty or incomplete_refs or incomplete_pairs:
        # Collect all task indices (use set to remove duplicates)
        all_indices = set()
        all_indices.update(i for i, _, _ in missing)
        all_indices.update(i for i, _, _ in empty)
        all_indices.update(i for i, _, _, _, _, _ in incomplete_refs)
        all_indices.update(i for i, _, _, _, _, _ in incomplete_pairs)

        missing_indices = [str(i) for i in sorted(all_indices)]
        missing_file = tasks_json.parent / f"missing_distogram_tasks.txt"
        with open(missing_file, "w") as f:
            f.write("\n".join(missing_indices))
        print(f"\nIncomplete task indices saved to: {missing_file}")

        # Also save as JSON for easier resubmission (remove duplicates by index)
        missing_tasks = [tasks[i] for i in sorted(all_indices)]
        missing_json = tasks_json.parent / f"missing_distogram_tasks.json"
        with open(missing_json, "w") as f:
            json.dump(missing_tasks, f, indent=2)
        print(f"Incomplete tasks saved to: {missing_json}")

    return (
        len(success),
        len(missing),
        len(empty),
        len(incomplete_refs),
        len(incomplete_pairs),
    )


def analyze_by_method(tasks_json: Path):
    """Analyze completion by method and method_type."""

    with open(tasks_json, "r") as f:
        tasks = json.load(f)

    # Group by method and method_type
    stats = defaultdict(lambda: {"total": 0, "completed": 0, "missing": 0, "empty": 0})

    for task in tasks:
        method = task.get("method", "unknown")
        method_type = task.get("method_type", "unknown")
        key = f"{method}/{method_type}"

        stats[key]["total"] += 1

        output_dir = get_output_dir(task)
        if output_dir is None:
            stats[key]["missing"] += 1
            continue

        result_file = output_dir / "distogram_loss_final.json"
        if not result_file.exists():
            stats[key]["missing"] += 1
        else:
            try:
                with open(result_file) as f:
                    data = json.load(f)
                if len(data) == 0:
                    stats[key]["empty"] += 1
                else:
                    stats[key]["completed"] += 1
            except:
                stats[key]["missing"] += 1

    print(f"\n{'=' * 80}")
    print(f"Progress by Method/Type")
    print(f"{'=' * 80}")
    print(
        f"{'Method/Type':<35} {'Total':<8} {'Complete':<10} {'Missing':<9} {'Empty':<7} {'%Done':<6}"
    )
    print(f"{'-' * 80}")

    for key in sorted(stats.keys()):
        s = stats[key]
        pct_done = s["completed"] / s["total"] * 100 if s["total"] > 0 else 0
        print(
            f"{key:<35} {s['total']:<8} {s['completed']:<10} {s['missing']:<9} {s['empty']:<7} {pct_done:5.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Check progress of distogram calculation tasks"
    )
    parser.add_argument(
        "--tasks", "-t", type=str, required=True, help="Path to distogram_tasks.json"
    )
    parser.add_argument(
        "--ref-distogram-dir",
        type=str,
        default=None,
        help="reference_distogram_diff root (default: eval.dirs.ref_distogram / eval.external.ref_distogram_dir)",
    )
    parser.add_argument(
        "--valid-pairs",
        type=str,
        default=None,
        help="valid_pairs.json (default: eval.files.valid_pairs)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all missing/empty tasks",
    )
    parser.add_argument(
        "--by-method", action="store_true", help="Show breakdown by method and type"
    )

    args = parser.parse_args()

    tasks_json = Path(args.tasks)
    if not tasks_json.exists():
        print(f"Error: Tasks file not found: {tasks_json}")
        return 1

    ref_distogram_dir = E.distogram_ref_distogram_dir(args.ref_distogram_dir)
    valid_pairs_path = E.distogram_valid_pairs_path(args.valid_pairs)
    with open(valid_pairs_path, "r") as f:
        vp = json.load(f)
    valid_pair_edges = set(flatten_valid_pair_edges(vp))

    success, missing, empty, incomplete_refs, incomplete_pairs = (
        check_distogram_results(
            tasks_json, ref_distogram_dir, valid_pair_edges, verbose=args.verbose
        )
    )

    # Show method breakdown if requested
    if args.by_method:
        analyze_by_method(tasks_json)

    # Return appropriate exit code
    total_incomplete = missing + empty + incomplete_refs + incomplete_pairs
    if total_incomplete == 0:
        print(
            "\n✓ All distogram calculations completed successfully with all references and dynamic pairs!"
        )
        return 0
    else:
        print(f"\n⚠ {total_incomplete} tasks still need to be processed or fixed")
        return 1


if __name__ == "__main__":
    exit(main())

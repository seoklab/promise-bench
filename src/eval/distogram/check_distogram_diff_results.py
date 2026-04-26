#!/usr/bin/env python3
"""
Check completion status of ``eval.distogram.calc_reference_distogram_diff`` outputs.

Verifies aggregated and per-comparison files under the ref_distogram root
(same layout as ``calc_reference_distogram_diff --output-dir``).

Usage (``PYTHONPATH=src``)::

    python -m eval.distogram.check_distogram_diff_results -t data_eval/distogram/distogram_tasks.json
    python -m eval.distogram.check_distogram_diff_results -t …/distogram_tasks.json --output-dir …/ref_distogram
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set
import itertools

from utils._config import eval_cfg as E


def get_output_dir(output_dir: str | Path) -> Path:
    """Resolve the ref_distogram output root (same as ``calc_reference_distogram_diff --output-dir``)."""
    return Path(output_dir)


def group_tasks_by_output(tasks: List[dict]) -> Dict[tuple, List[dict]]:
    """Group tasks by (cluster_id, prediction_yaml_tag, method_type).

    Tasks with the same key are merged into one output run by ``calc_reference_distogram_diff``.
    """
    groups = {}

    for task in tasks:
        cluster_id = task.get("cluster_id", "")
        prediction_yaml_tag = task.get("prediction_yaml_tag", "")
        method_type = task.get("method_type", "")

        key = (cluster_id, prediction_yaml_tag, method_type)

        if key not in groups:
            groups[key] = []
        groups[key].append(task)

    return groups


def get_expected_individual_files(
    task_group: List[dict], output_base_dir: Path
) -> List[Path]:
    """
    Get list of expected individual output files for a group of tasks.

    Since tasks with same cluster_id + prediction_yaml_tag + method_type
    get merged, we need to consider union of all references.

    Args:
        task_group: List of tasks with same cluster_id + prediction_yaml_tag + method_type
        output_base_dir: Base output directory

    Returns:
        List of expected file paths
    """
    expected_files = []

    if not task_group:
        return expected_files

    # All tasks in group should have same cluster_id, prediction_yaml_tag, method_type
    first_task = task_group[0]
    cluster_id = first_task.get("cluster_id", "")
    prediction_yaml_tag = first_task.get("prediction_yaml_tag", "")
    method_type = first_task.get("method_type", "")

    # Individual results directory
    individual_results_dir = (
        output_base_dir / method_type / cluster_id / prediction_yaml_tag
    )

    # Collect all unique references from all tasks in the group
    all_ref_infos = []
    seen_refs = set()

    for task in task_group:
        ref_infos = task.get("reference_infos", [])
        for ref_info in ref_infos:
            ref_tag = ref_info.get("reference_yaml_tag", "")
            ref_conf = ref_info.get("reference_conformation", "")
            ref_key = (ref_tag, ref_conf)

            if ref_key not in seen_refs:
                all_ref_infos.append(ref_info)
                seen_refs.add(ref_key)

    # Get reference pairs that should be compared - REMOVE THIS LINE, all_ref_infos already set above
    # ref_infos = task.get("reference_infos", [])

    if len(all_ref_infos) < 2:
        return expected_files

    # Generate all pairwise combinations from union of references
    for ref_A, ref_B in itertools.combinations(all_ref_infos, 2):
        ref_A_tag = ref_A.get("reference_yaml_tag", "")
        ref_A_conf = ref_A.get("reference_conformation", "")
        ref_B_tag = ref_B.get("reference_yaml_tag", "")
        ref_B_conf = ref_B.get("reference_conformation", "")

        # Individual comparison result file
        result_filename = (
            f"{ref_A_tag}_{ref_A_conf}_vs_{ref_B_tag}_{ref_B_conf}_distogram_diff.json"
        )
        result_file = individual_results_dir / result_filename
        expected_files.append(result_file)

        # Visualization file (optional, not always generated)
        viz_filename = (
            f"{ref_A_tag}_{ref_A_conf}_vs_{ref_B_tag}_{ref_B_conf}_distance_diff.png"
        )
        viz_file = individual_results_dir / viz_filename
        # Note: We don't include viz files in required files as they might not always be generated

    return expected_files


def get_expected_aggregated_file(task_group: List[dict], output_base_dir: Path) -> Path:
    """
    Get the expected aggregated result file path for a group of tasks.

    Args:
        task_group: List of tasks with same cluster_id + prediction_yaml_tag + method_type
        output_base_dir: Base output directory

    Returns:
        Expected aggregated result file path
    """
    if not task_group:
        return Path()

    first_task = task_group[0]
    cluster_id = first_task.get("cluster_id", "")
    prediction_yaml_tag = first_task.get("prediction_yaml_tag", "")
    method_type = first_task.get("method_type", "")

    # Aggregated result file
    result_file_all = (
        output_base_dir
        / method_type
        / cluster_id
        / prediction_yaml_tag
        / "reference_distogram_diff.json"
    )

    return result_file_all


def check_task_group_completion(task_group: List[dict], output_base_dir: Path) -> Dict:
    """
    Check if all expected output files exist for a group of tasks.

    Tasks with same cluster_id + prediction_yaml_tag + method_type get merged,
    so we check them as a group.

    Args:
        task_group: List of tasks with same cluster_id + prediction_yaml_tag + method_type
        output_base_dir: Base output directory

    Returns:
        Dictionary with completion status information
    """
    if not task_group:
        return {}

    first_task = task_group[0]
    cluster_id = first_task.get("cluster_id", "")
    prediction_yaml_tag = first_task.get("prediction_yaml_tag", "")
    method_type = first_task.get("method_type", "")

    # Count total tasks and references in group
    total_tasks_in_group = len(task_group)
    total_refs_in_group = sum(
        len(task.get("reference_infos", [])) for task in task_group
    )
    # Get expected files
    expected_individual_files = get_expected_individual_files(
        task_group, output_base_dir
    )
    expected_aggregated_file = get_expected_aggregated_file(task_group, output_base_dir)

    # Check individual files
    missing_individual = []
    existing_individual = []

    for file_path in expected_individual_files:
        if file_path.exists():
            existing_individual.append(file_path)
        else:
            missing_individual.append(file_path)

    # Check aggregated file
    aggregated_exists = expected_aggregated_file.exists()

    # Determine overall completion status
    is_complete = aggregated_exists and len(missing_individual) == 0

    return {
        "cluster_id": cluster_id,
        "prediction_yaml_tag": prediction_yaml_tag,
        "method_type": method_type,
        "tasks_in_group": total_tasks_in_group,
        "total_references": total_refs_in_group,
        "is_complete": is_complete,
        "aggregated_file": {
            "path": expected_aggregated_file,
            "exists": aggregated_exists,
        },
        "individual_files": {
            "total": len(expected_individual_files),
            "existing": len(existing_individual),
            "missing": len(missing_individual),
            "missing_files": [str(f) for f in missing_individual],
        },
        "expected_comparisons": len(expected_individual_files),
    }


def analyze_results_by_method(results: List[Dict]) -> Dict:
    """Analyze results by prediction method."""
    by_method = {}

    for result in results:
        # Try to determine method from prediction_yaml_tag or path
        yaml_tag = result["prediction_yaml_tag"]

        # Extract method from yaml tag (common patterns)
        method = "unknown"
        if "af3" in yaml_tag.lower():
            method = "AF3"
        elif "boltz" in yaml_tag.lower():
            method = "Boltz"
        elif "chai" in yaml_tag.lower():
            method = "Chai-1"
        elif "bioemu" in yaml_tag.lower():
            method = "BioEmu"

        if method not in by_method:
            by_method[method] = {
                "total": 0,
                "complete": 0,
                "incomplete": 0,
            }

        by_method[method]["total"] += 1
        if result["is_complete"]:
            by_method[method]["complete"] += 1
        else:
            by_method[method]["incomplete"] += 1

    return by_method


def main():
    parser = argparse.ArgumentParser(
        description="Check completion of calc_reference_distogram_diff outputs (ref_distogram tree)"
    )
    parser.add_argument(
        "--tasks", "-t", type=str, required=True, help="Path to distogram_tasks.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="ref_distogram root (default: eval.dirs.ref_distogram / external.ref_distogram_dir)",
    )
    parser.add_argument(
        "--start", "-s", type=int, default=0, help="Start index for checking"
    )
    parser.add_argument(
        "--end", "-e", type=int, default=None, help="End index for checking"
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Show detailed missing files for incomplete tasks",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Show only summary statistics",
    )

    args = parser.parse_args()

    # Load tasks
    tasks_json = Path(args.tasks)
    if not tasks_json.exists():
        raise FileNotFoundError(f"Tasks JSON not found: {tasks_json}")

    with open(tasks_json, "r") as f:
        tasks = json.load(f)

    output_base_dir = get_output_dir(E.distogram_ref_distogram_dir(args.output_dir))
    print("Checking calc_reference_distogram_diff results:")
    print(f"  Tasks file: {tasks_json}")
    print(f"  Output directory: {output_base_dir}")
    print(f"  Total tasks in file: {len(tasks)}")

    # Group tasks by (cluster_id, prediction_yaml_tag, method_type)
    task_groups = group_tasks_by_output(tasks)
    print(f"  Grouped into {len(task_groups)} unique output locations")

    # Show grouping info
    merged_count = sum(1 for group in task_groups.values() if len(group) > 1)
    if merged_count > 0:
        print(f"    {merged_count} locations have multiple tasks (will be merged)")

    # Apply start/end indices to groups
    group_items = list(task_groups.items())
    if args.end is None:
        args.end = len(group_items)

    groups_to_check = group_items[args.start : args.end]
    print(
        f"  Checking groups [{args.start}:{args.end}] ({len(groups_to_check)} groups)"
    )

    # Check each task group
    results = []
    complete_count = 0
    incomplete_count = 0
    total_expected_comparisons = 0
    total_existing_comparisons = 0

    for i, (key, task_group) in enumerate(groups_to_check):
        cluster_id, prediction_yaml_tag, method_type = key
        result = check_task_group_completion(task_group, output_base_dir)
        results.append(result)

        total_expected_comparisons += result["expected_comparisons"]
        total_existing_comparisons += result["individual_files"]["existing"]

        if result["is_complete"]:
            complete_count += 1
        else:
            incomplete_count += 1

            if not args.summary_only:
                print(
                    f"\n❌ Incomplete: {result['cluster_id']} / {result['prediction_yaml_tag']} ({result['tasks_in_group']} tasks, {result['total_references']} refs)"
                )
                if not result["aggregated_file"]["exists"]:
                    print(
                        f"   Missing aggregated file: {result['aggregated_file']['path']}"
                    )
                if result["individual_files"]["missing"] > 0:
                    print(
                        f"   Missing {result['individual_files']['missing']}/{result['individual_files']['total']} individual files"
                    )
                    if args.show_missing:
                        for missing_file in result["individual_files"]["missing_files"]:
                            print(f"     - {missing_file}")

        if not args.summary_only and result["is_complete"]:
            print(
                f"✅ Complete: {result['cluster_id']} / {result['prediction_yaml_tag']} ({result['individual_files']['total']} comparisons)"
            )

    # Summary statistics
    print(f"\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Original tasks: {len(tasks)}")
    print(f"  Unique output locations: {len(group_items)}")
    print(f"  Locations checked: {len(groups_to_check)}")
    print(f"  Complete locations: {complete_count}")
    print(f"  Incomplete locations: {incomplete_count}")
    n_checked = len(groups_to_check)
    if n_checked > 0:
        print(f"  Completion rate: {complete_count / n_checked * 100:.1f}%")
    else:
        print("  Completion rate: N/A (no groups in selected range)")
    print(f"  Total expected comparisons: {total_expected_comparisons}")
    print(f"  Existing comparisons: {total_existing_comparisons}")
    print(
        f"  Comparison completion rate: {total_existing_comparisons / total_expected_comparisons * 100:.1f}%"
        if total_expected_comparisons > 0
        else "  No comparisons expected"
    )

    # Method-wise analysis
    method_stats = analyze_results_by_method(results)
    if len(method_stats) > 1:
        print(f"\nBY METHOD:")
        for method, stats in method_stats.items():
            completion_rate = (
                stats["complete"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )
            print(
                f"  {method}: {stats['complete']}/{stats['total']} complete ({completion_rate:.1f}%)"
            )

    # Show incomplete groups summary
    if incomplete_count > 0 and not args.summary_only:
        print(f"\nINCOMPLETE GROUPS:")
        for result in results:
            if not result["is_complete"]:
                missing_info = []
                if not result["aggregated_file"]["exists"]:
                    missing_info.append("aggregated")
                if result["individual_files"]["missing"] > 0:
                    missing_info.append(
                        f"{result['individual_files']['missing']} individual"
                    )

                print(
                    f"  {result['cluster_id']}/{result['prediction_yaml_tag']} ({result['tasks_in_group']} tasks, {result['total_references']} refs) - Missing: {', '.join(missing_info)}"
                )

    print(f"\nOutput directory: {output_base_dir}")
    if n_checked == 0:
        print("No task groups in range to verify (empty tasks or empty [--start,--end] slice).")
    elif incomplete_count == 0:
        print("🎉 All checked reference distogram diff outputs are complete!")
    else:
        print(f"⚠️  {incomplete_count} output locations need to be processed or rerun.")


if __name__ == "__main__":
    main()

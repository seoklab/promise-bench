#!/usr/bin/env python3
"""
Generate sbatch jobs for parallel reference distogram difference calculation.

Usage:
    python -m eval.distogram.generate_distogram_diff_jobs --tasks distogram/distogram_tasks.json --num_jobs 20
    python -m eval.distogram.generate_distogram_diff_jobs --tasks distogram/distogram_tasks.json --num_jobs 50 --submit
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import subprocess
import os

from utils._config import eval_cfg as E

_REPO_ROOT = Path(__file__).resolve().parents[3]


def generate_sbatch_script(
    tasks_json: Path,
    job_idx: int,
    start_idx: int,
    end_idx: int,
    output_dir: Path,
    rep_seq: str,
    msa_dir: str,
    threshold: float,
    output_base_dir: str,
) -> Path:
    """Generate a single sbatch script."""

    script_content = f"""#!/bin/bash
#SBATCH --job-name=ref_diff_{job_idx}
#SBATCH --output={output_dir}/logs/ref_diff_{job_idx}_%j.out
#SBATCH --error={output_dir}/logs/ref_diff_{job_idx}_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

cd "{_REPO_ROOT}"

uv run --project "{_REPO_ROOT}" \\
    python -m eval.distogram.calc_reference_distogram_diff \\
    --tasks {tasks_json} \\
    --rep-seq {rep_seq} \\
    --msa-dir {msa_dir} \\
    --threshold {threshold} \\
    --output-dir {output_base_dir} \\
    --start {start_idx} \\
    --end {end_idx}

echo "Job {job_idx} completed: tasks [{start_idx}:{end_idx}]"
"""

    script_path = output_dir / f"job_{job_idx:03d}.sh"
    with open(script_path, "w") as f:
        f.write(script_content)

    return script_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate sbatch jobs for parallel reference distogram difference calculation"
    )
    parser.add_argument(
        "--tasks", "-t", type=str, required=True, help="Path to distogram_tasks.json"
    )
    parser.add_argument(
        "--num_jobs",
        "-n",
        type=int,
        default=20,
        help="Number of parallel jobs (default: 20)",
    )
    parser.add_argument(
        "--rep-seq",
        type=str,
        default=None,
        help="representative_sequences JSON (default: pipeline.files.rep_seq)",
    )
    parser.add_argument(
        "--msa-dir",
        type=str,
        default=None,
        help="a3m root (default: pipeline.dirs.msas)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Distance threshold (Angstroms) to report as significant difference",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory for sbatch scripts (default: <tasks_dir>/sbatch_ref_distogram_diff)",
    )
    parser.add_argument(
        "--result-output-dir",
        type=str,
        default=None,
        help="reference_distogram_diff root (default: eval.dirs.ref_distogram / external)",
    )
    parser.add_argument(
        "--submit", action="store_true", help="Submit jobs immediately after generation"
    )
    args = parser.parse_args()

    tasks_json = Path(args.tasks).resolve()
    if not tasks_json.exists():
        raise FileNotFoundError(f"Tasks JSON not found: {tasks_json}")

    # Count total tasks
    with open(tasks_json, "r") as f:
        tasks = json.load(f)
    total_tasks = len(tasks)
    print(f"Total tasks to process: {total_tasks}")

    # Calculate chunk size
    chunk_size = (total_tasks + args.num_jobs - 1) // args.num_jobs
    print(f"Chunk size: ~{chunk_size} tasks per job")

    # Create output directory for sbatch scripts
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = tasks_json.parent / "sbatch_ref_distogram_diff"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    if args.result_output_dir:
        result_output_dir = args.result_output_dir
    else:
        result_output_dir = str(E.distogram_ref_distogram_dir(None))

    rep_seq = str(E.distogram_rep_seq_json(args.rep_seq).resolve())
    msa_dir = str(E.distogram_msa_dir(args.msa_dir).resolve())
    result_output_dir = str(Path(result_output_dir).resolve())

    script_paths = []
    for job_idx in range(args.num_jobs):
        start_idx = job_idx * chunk_size
        end_idx = min((job_idx + 1) * chunk_size, total_tasks)

        if start_idx >= total_tasks:
            break

        script_path = generate_sbatch_script(
            tasks_json=tasks_json,
            job_idx=job_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            output_dir=output_dir,
            rep_seq=rep_seq,
            msa_dir=msa_dir,
            threshold=args.threshold,
            output_base_dir=result_output_dir,
        )
        script_paths.append(script_path)
        print(f"Generated: {script_path.name} (tasks {start_idx}-{end_idx})")

    print(f"\nGenerated {len(script_paths)} sbatch scripts in {output_dir}")
    print(f"Results will be saved to: {result_output_dir}")

    # Create submit all script
    submit_all_path = output_dir / "submit_all.sh"
    with open(submit_all_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(
            f"# Submit all {len(script_paths)} reference distogram difference jobs\n\n"
        )
        for script_path in script_paths:
            f.write(f"sbatch {script_path}\n")
    os.chmod(submit_all_path, 0o755)
    print(f"Created: {submit_all_path}")

    # Submit if requested
    if args.submit:
        print("\nSubmitting jobs...")
        for script_path in script_paths:
            result = subprocess.run(
                ["sbatch", str(script_path)], capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  Submitted: {script_path.name} -> {result.stdout.strip()}")
            else:
                print(f"  Failed: {script_path.name} -> {result.stderr.strip()}")
    else:
        print(f"\nTo submit all jobs, run: bash {submit_all_path}")
        print(f"Or use --submit flag to submit automatically")


if __name__ == "__main__":
    main()

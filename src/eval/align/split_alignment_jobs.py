#!/usr/bin/env python3
"""Split alignment_tasks.json into shards; emit local run_all.sh and/or SLURM sbatch scripts.

Input JSON is produced by ``python -m eval.align.generate_alignment_tasks``.
Each shard runs ``python -m eval.align.struct_align_batch``.

``--emit local`` runs shards sequentially on this host; ``--emit sbatch`` emits SLURM
only; ``--emit both`` (default) generates both.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

from utils._config import eval_cfg as E


def split_and_generate_jobs(
    input_json: str,
    num_parts: int,
    output_dir: Path,
    emit: str,
    sbatch_partition: str,
    sbatch_time: str,
    sbatch_mem: str,
    sbatch_cpus: int,
    *,
    python_exe: str,
    log_dir: Path,
    write_cif: bool,
) -> None:
    input_json = str(Path(input_json).resolve())
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(input_json, "r") as f:
        tasks = json.load(f)

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No tasks in JSON; nothing to generate.")
        return

    tasks_per_part = math.ceil(total_tasks / num_parts)

    print(f"Total tasks: {total_tasks}")
    print(f"Parts: {num_parts}")
    print(f"Tasks per part: ~{tasks_per_part}")
    print(f"Emit: {emit}")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    sbatch_dir = output_dir / "sbatch"
    if emit in ("sbatch", "both"):
        sbatch_dir.mkdir(parents=True, exist_ok=True)

    json_files: List[Path] = []
    for i in range(num_parts):
        start_idx = i * tasks_per_part
        end_idx = min((i + 1) * tasks_per_part, total_tasks)

        if start_idx >= total_tasks:
            break

        part_tasks = tasks[start_idx:end_idx]
        part_num = str(i + 1).zfill(4)
        json_filename = f"alignment_tasks_part{part_num}.json"
        json_path = json_dir / json_filename

        with open(json_path, "w") as f:
            json.dump(part_tasks, f, indent=2)

        json_files.append(json_path.resolve())
        print(f"  Part {i + 1}: {len(part_tasks)} tasks -> {json_path}")

    results_dir = output_dir / "align_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    wc = " --no-write-cif" if not write_cif else ""

    def one_local_line(jp: Path, part_idx: int) -> str:
        part_num = str(part_idx + 1).zfill(4)
        rp = (results_dir / f"align_part{part_num}.json").resolve()
        return (
            f"{python_exe} -m eval.align.struct_align_batch "
            f"--json {jp} --results-json {rp}{wc}\n"
        )

    def one_sbatch_inner(jp: Path, part_idx: int) -> str:
        return one_local_line(jp, part_idx)

    if emit in ("local", "both"):
        run_script_path = output_dir / "run_all.sh"
        with open(run_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Run alignment shards sequentially (no scheduler).\n")
            f.write(f"# Total tasks: {total_tasks}, shards: {len(json_files)}\n\n")
            f.write("set -euo pipefail\n\n")
            for idx, jp in enumerate(json_files):
                f.write(one_local_line(jp, idx))
        run_script_path.chmod(0o755)
        print(f"\nLocal runner: {run_script_path}")

    if emit in ("sbatch", "both"):
        sbatch_list: List[Path] = []
        for i, json_path in enumerate(json_files):
            jp = str(json_path)
            part_num = str(i + 1).zfill(4)
            sbatch_path = sbatch_dir / f"align_part{part_num}.sh"

            inner = one_sbatch_inner(json_path, i)

            with open(sbatch_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(f"#SBATCH --job-name=align_{part_num}\n")
                f.write(f"#SBATCH --output={log_dir}/align_{part_num}_%j.out\n")
                f.write(f"#SBATCH --error={log_dir}/align_{part_num}_%j.err\n")
                f.write(f"#SBATCH --time={sbatch_time}\n")
                f.write(f"#SBATCH --mem={sbatch_mem}\n")
                f.write(f"#SBATCH --cpus-per-task={sbatch_cpus}\n")
                f.write(f"#SBATCH --partition={sbatch_partition}\n\n")
                f.write(inner)

            sbatch_path.chmod(0o755)
            sbatch_list.append(sbatch_path.resolve())

        submit_all_path = sbatch_dir / "submit_all.sh"
        with open(submit_all_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Submit {len(sbatch_list)} SLURM jobs\n\n")
            for sp in sbatch_list:
                f.write(f"sbatch {sp}\n")

        submit_all_path.chmod(0o755)
        print(f"SLURM scripts: {sbatch_dir} ({len(sbatch_list)} jobs)")
        print(f"Submit all:    {submit_all_path}")


def main() -> None:
    # Keep defaults independent of an ``eval.dirs.align`` config key.
    # Put alignment artefacts under eval.dirs.output/align/ by convention.
    align_root = E.dir("output") / "align"
    default_input = align_root / "alignment_tasks.json"
    default_out = align_root / "job_batches"
    default_log = align_root / "logs"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(default_input),
        help="alignment_tasks.json from eval.align.generate_alignment_tasks",
    )
    p.add_argument(
        "--parts",
        "-p",
        type=int,
        default=10,
        help="Number of shards",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=default_out,
        help="Directory for json/, sbatch/, run_all.sh",
    )
    p.add_argument(
        "--emit",
        choices=("both", "local", "sbatch"),
        default="both",
        help="local = only run_all.sh; sbatch = only SLURM scripts; both = both",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter (must have promise-data + nurikit installed)",
    )
    p.add_argument(
        "--write-cif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit aligned mmCIF writes (default: on; use --no-write-cif in generated commands)",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=default_log,
        help="SLURM stdout/stderr directory",
    )
    p.add_argument("--sbatch-partition", default="normal.q")
    p.add_argument("--sbatch-time", default="24:00:00")
    p.add_argument("--sbatch-mem", default="16G")
    p.add_argument("--sbatch-cpus", type=int, default=1)

    args = p.parse_args()

    split_and_generate_jobs(
        input_json=args.input,
        num_parts=args.parts,
        output_dir=args.output_dir,
        emit=args.emit,
        sbatch_partition=args.sbatch_partition,
        sbatch_time=args.sbatch_time,
        sbatch_mem=args.sbatch_mem,
        sbatch_cpus=args.sbatch_cpus,
        python_exe=args.python,
        log_dir=args.log_dir,
        write_cif=args.write_cif,
    )


if __name__ == "__main__":
    main()

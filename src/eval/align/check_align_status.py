#!/usr/bin/env python3
"""
Sanity-check alignment runs by comparing ``alignment_tasks.json`` against the
sharded ``align_results/align_part*.json`` produced by SLURM (or local) workers.

Prints a per-prediction-method table:

    method       tasks   results        ok    ok%      fail   missing
    ----------------------------------------------------------------
    af3         209000    209000    209000  100.00%      0         0
    ...

Where ``missing = tasks - results`` (shard didn't complete, OOM, etc.) and
``fail = results - ok`` (alignment itself returned ``ok=False``).

Usage
-----

    python -m eval.align.check_align_status \\
        --tasks       data_eval/output/align/alignment_tasks.json \\
        --results-dir data_eval/output/align/job_batches_v2/align_results

Both flags fall back to the same defaults as ``eval.align.split_alignment_jobs``.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

from utils._config import eval_cfg as E


def collect_counts(tasks_path: Path, results_dir: Path) -> Dict[str, Dict[str, int]]:
    """Return ``{method: {tasks,results,ok,fail,missing}}`` for the given run."""
    with open(tasks_path) as f:
        tasks: List[Dict] = json.load(f)
    t_per_method = Counter(r["prediction_method"] for r in tasks)

    ok = Counter()
    fail = Counter()
    result_files = sorted(glob.glob(str(results_dir / "align_part*.json")))
    for rp in result_files:
        with open(rp) as f:
            rows = json.load(f)
        for row in rows:
            method = row["prediction_method"]
            if row.get("ok"):
                ok[method] += 1
            else:
                fail[method] += 1

    methods: Set[str] = set(t_per_method) | set(ok) | set(fail)
    out: Dict[str, Dict[str, int]] = {}
    for m in sorted(methods):
        nt = t_per_method.get(m, 0)
        nok = ok.get(m, 0)
        nf = fail.get(m, 0)
        nr = nok + nf
        out[m] = {
            "tasks": nt,
            "results": nr,
            "ok": nok,
            "fail": nf,
            "missing": nt - nr,
        }
    return out


def format_table(counts: Dict[str, Dict[str, int]]) -> str:
    """Render the per-method table + TOTAL row as a plain-text block."""
    header = f'{"method":10}{"tasks":>10}{"results":>10}{"ok":>10}{"ok%":>8}{"fail":>8}{"missing":>10}'
    sep = "-" * len(header)
    lines = [header, sep]
    tot = {k: 0 for k in ("tasks", "results", "ok", "fail", "missing")}
    for m, c in counts.items():
        pct = c["ok"] / c["tasks"] * 100 if c["tasks"] else 0.0
        lines.append(
            f"{m:10}{c['tasks']:>10}{c['results']:>10}{c['ok']:>10}{pct:>7.2f}%"
            f"{c['fail']:>8}{c['missing']:>10}"
        )
        for k in tot:
            tot[k] += c[k]
    pct = tot["ok"] / tot["tasks"] * 100 if tot["tasks"] else 0.0
    lines.append(sep)
    lines.append(
        f'{"TOTAL":10}{tot["tasks"]:>10}{tot["results"]:>10}{tot["ok"]:>10}'
        f"{pct:>7.2f}%{tot['fail']:>8}{tot['missing']:>10}"
    )
    return "\n".join(lines)


def main() -> None:
    align_root = E.dir("output") / "align"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tasks",
        type=Path,
        default=align_root / "alignment_tasks.json",
        help="alignment_tasks.json (from eval.align.generate_alignment_tasks).",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=align_root / "job_batches" / "align_results",
        help="Directory containing align_part*.json result shards.",
    )
    args = p.parse_args()

    if not args.tasks.exists():
        p.error(f"--tasks not found: {args.tasks}")
    if not args.results_dir.is_dir():
        p.error(f"--results-dir not a directory: {args.results_dir}")

    counts = collect_counts(args.tasks, args.results_dir)
    print(f"tasks:        {args.tasks}")
    print(f"results dir:  {args.results_dir}")
    print()
    print(format_table(counts))


if __name__ == "__main__":
    main()

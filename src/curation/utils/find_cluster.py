#!/usr/bin/env python3
"""Find which sequence cluster a given PDB_chain belongs to."""

import argparse
import subprocess
from pathlib import Path

SEQS_DIR = Path(__file__).resolve().parents[3] / "data" / "seqs"


def find_cluster(query: str, seqs_dir: Path = SEQS_DIR) -> list[str]:
    pdb, chain = query.split("_", 1)
    query_fmt = f"{pdb.lower()}_{chain}"
    result = subprocess.run(
        ["grep", "-rl", f"^>{query_fmt}$", str(seqs_dir)],
        capture_output=True,
        text=True,
    )
    return [Path(p).stem for p in result.stdout.strip().splitlines() if p]


def main():
    parser = argparse.ArgumentParser(description="Find sequence cluster for PDB_chain")
    parser.add_argument("queries", nargs="+", help="PDB_chain ids, e.g. 12E8_L 7RTB_R")
    parser.add_argument("--seqs", type=Path, default=SEQS_DIR)
    args = parser.parse_args()

    for q in args.queries:
        clusters = find_cluster(q, args.seqs)
        if clusters:
            print(f"{q}\t{', '.join(clusters)}")
        else:
            print(f"{q}\t(not found)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Check TMScoreResult npz files."""

from pathlib import Path

import click
import numpy as np


@click.command()
@click.argument("npz_file", type=click.Path(exists=True, path_type=Path))
def main(npz_file: Path):
    """Check contents of a TMScoreResult npz file."""
    data = np.load(npz_file, allow_pickle=True)

    print(f"File: {npz_file}")
    print(f"Keys: {list(data.keys())}")
    print()

    chains = data.get("chains")
    tm_scores = data.get("tm_scores")
    rmsds = data.get("rmsds")

    if chains is not None:
        print(f"Chains ({len(chains)}): {list(chains)}")
        print()

    # Print all pairwise scores
    if chains is not None and tm_scores is not None:
        print(f"Pairwise scores (total: {len(tm_scores)}):")
        print(f"{'Chain 1':<15} {'Chain 2':<15} {'TM-score':<10} {'RMSD':<10}")
        print("-" * 55)

        k = 0
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                tms = tm_scores[k]
                rmsd_val = rmsds[k] if rmsds is not None else float("nan")
                print(f"{chains[i]:<15} {chains[j]:<15} {tms:<10.4f} {rmsd_val:<10.4f}")
                k += 1


if __name__ == "__main__":
    main()

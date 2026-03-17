#!/usr/bin/env python3
"""
ESM-MSA-1b Contact Prediction for ProMiSE-bench.

Predict residue-residue contacts from MSA using ESM-MSA-1b model.

Usage:
    # Predict contacts for example data
    python -m src.eval.esm_run --examples-dir examples/msa-server

    # Predict for specific a3m files
    python -m src.eval.esm_run --input examples/msa-server/intrinsic/7OYW_1/7OYW_1.a3m

    # Multiple seeds for robustness
    python -m src.eval.esm_run --examples-dir examples/msa-server --multi-seed
"""

import sys
import os
import string
from pathlib import Path
from typing import List, Optional, Tuple

import click
import numpy as np
import torch
from Bio import SeqIO
from scipy.spatial.distance import cdist

import esm

from utils._config import eval_cfg as E


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_OUTPUT_DIR = str(E.dir('esm_contacts'))
DEFAULT_SAMPLE_SIZE = 1024
DEFAULT_NUM_SEQS = 128
DEFAULT_SEED = 42
DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Translation table to remove insertions
_deletekeys = dict.fromkeys(string.ascii_lowercase)
_deletekeys["."] = None
_deletekeys["*"] = None
_translation = str.maketrans(_deletekeys)


# ============================================================================
# MSA Processing
# ============================================================================

def remove_insertions(sequence: str) -> str:
    """Remove insertion characters from a3m sequence."""
    return sequence.translate(_translation)


def read_a3m(filepath: str) -> List[Tuple[str, str]]:
    """Read a3m MSA file and return aligned sequences."""
    sequences = []
    for record in SeqIO.parse(filepath, "fasta"):
        seq = remove_insertions(str(record.seq))
        sequences.append((record.description, seq))
    return sequences


def random_sample(
    msa: List[Tuple[str, str]], 
    sample_size: int, 
    seed: int = DEFAULT_SEED
) -> List[Tuple[str, str]]:
    """Randomly sample sequences from MSA. Always keeps query (first)."""
    if len(msa) <= sample_size:
        return msa
    
    np.random.seed(seed)
    query = msa[0]
    rest = msa[1:]
    indices = np.random.choice(len(rest), size=sample_size - 1, replace=False)
    indices = sorted(indices)
    return [query] + [rest[i] for i in indices]


def greedy_select(
    msa: List[Tuple[str, str]], 
    num_seqs: int, 
    mode: str = "max"
) -> List[Tuple[str, str]]:
    """Select diverse sequences using greedy max-Hamming algorithm."""
    if len(msa) <= num_seqs:
        return msa
    
    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)
    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    
    indices = sorted(indices)
    return [msa[idx] for idx in indices]


# ============================================================================
# ESM-MSA-1b Predictor
# ============================================================================

class ESMMSAPredictor:
    """ESM-MSA-1b model wrapper for contact prediction."""
    
    def __init__(self, device: str = DEFAULT_DEVICE):
        self.device = device
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        
    def load_model(self) -> "ESMMSAPredictor":
        """Load ESM-MSA-1b model (lazy loading)."""
        if self.model is None:
            print(f"Loading ESM-MSA-1b model on {self.device}...")
            self.model, self.alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
            self.model = self.model.eval().to(self.device)
            self.batch_converter = self.alphabet.get_batch_converter()
        return self
    
    def predict(
        self, 
        msa: List[Tuple[str, str]], 
        num_seqs: int = DEFAULT_NUM_SEQS
    ) -> np.ndarray:
        """Predict contacts from MSA. Returns LxL contact probability matrix."""
        self.load_model()
        
        if len(msa) > num_seqs:
            msa = greedy_select(msa, num_seqs=num_seqs)
        
        _, _, batch_tokens = self.batch_converter([msa])
        batch_tokens = batch_tokens.to(self.device)
        
        with torch.no_grad():
            contacts = self.model.predict_contacts(batch_tokens)[0].cpu().numpy()
        
        return contacts


# ============================================================================
# Processing Functions
# ============================================================================

def process_a3m_file(
    a3m_path: str,
    output_dir: str,
    predictor: ESMMSAPredictor,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    num_seqs: int = DEFAULT_NUM_SEQS,
    seed: int = DEFAULT_SEED,
) -> Optional[np.ndarray]:
    """Process a3m file and save contact predictions."""
    a3m_path = Path(a3m_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = a3m_path.stem
    output_npy = output_dir / f"{base_name}_seed{seed}_n{sample_size}_contacts.npy"
    
    print(f"\n{'='*60}")
    print(f"Processing: {a3m_path.name}")
    
    # Read MSA
    msa_full = read_a3m(str(a3m_path))
    if len(msa_full) == 0:
        print(f"  WARNING: No sequences")
        return None
    
    seq_lengths = set(len(seq) for _, seq in msa_full)
    if len(seq_lengths) > 1:
        print(f"  ERROR: Unaligned sequences")
        return None
    
    seq_length = seq_lengths.pop()
    print(f"  MSA: {len(msa_full)} seqs, L={seq_length}")
    
    # Sample and predict
    msa_sampled = random_sample(msa_full, sample_size=sample_size, seed=seed)
    print(f"  Sampled: {len(msa_sampled)} seqs (seed={seed})")
    
    contacts = predictor.predict(msa_sampled, num_seqs=num_seqs)
    print(f"  Contacts: {contacts.shape}, range=[{contacts.min():.3f}, {contacts.max():.3f}]")
    
    # Save
    np.save(output_npy, contacts)
    print(f"  Saved: {output_npy.name}")
    
    return contacts


def discover_a3m_files(examples_dir: str) -> List[Path]:
    """Find all a3m files in examples directory."""
    return sorted(Path(examples_dir).rglob("*.a3m"))


# ============================================================================
# CLI
# ============================================================================

@click.command()
@click.option('--examples-dir', type=click.Path(exists=True), default=None,
              help='examples/msa-server directory')
@click.option('--input', '-i', 'input_files', type=click.Path(exists=True),
              multiple=True, help='Input a3m file(s)')
@click.option('--output-dir', '-o', type=click.Path(),
              default=DEFAULT_OUTPUT_DIR,
              show_default=True, help='Output directory')
@click.option('--sample-size', '-s', type=int, default=DEFAULT_SAMPLE_SIZE,
              show_default=True)
@click.option('--num-seqs', '-n', type=int, default=DEFAULT_NUM_SEQS,
              show_default=True)
@click.option('--seed', type=int, default=DEFAULT_SEED,
              show_default=True)
@click.option('--multi-seed', is_flag=True,
              help=f'Use seeds: {DEFAULT_SEEDS}')
@click.option('--seeds', type=int, multiple=True,
              help='Custom seeds')
@click.option('--skip-existing', is_flag=True)
@click.option('--device', '-d', type=click.Choice(['cuda', 'cpu']),
              default=DEFAULT_DEVICE, show_default=True)
def main(examples_dir, input_files, output_dir, sample_size, num_seqs,
         seed, multi_seed, seeds, skip_existing, device):
    """ESM-MSA-1b Contact Prediction for ProMiSE-bench."""
    if not examples_dir and not input_files:
        raise click.UsageError('One of --examples-dir or --input is required.')

    # Collect input files
    if examples_dir:
        a3m_files = discover_a3m_files(examples_dir)
    else:
        a3m_files = [Path(f) for f in input_files if Path(f).exists()]

    if not a3m_files:
        raise click.ClickException('No input files found!')

    # Seeds
    if seeds:
        seed_list = list(seeds)
    elif multi_seed:
        seed_list = DEFAULT_SEEDS
    else:
        seed_list = [seed]

    print(f"\n{'#'*60}")
    print(f"ESM-MSA-1b Contact Prediction")
    print(f"{'#'*60}")
    print(f"Files: {len(a3m_files)}, Seeds: {seed_list}, Device: {device}")
    print(f"Output: {output_dir}")

    predictor = ESMMSAPredictor(device=device)

    success, skip_cnt, fail = 0, 0, 0

    for a3m_path in a3m_files:
        out_dir = Path(output_dir)

        for s in seed_list:
            if skip_existing:
                output_npy = out_dir / f"{a3m_path.stem}_seed{s}_n{sample_size}_contacts.npy"
                if output_npy.exists():
                    skip_cnt += 1
                    continue

            try:
                result = process_a3m_file(
                    str(a3m_path), str(out_dir), predictor,
                    sample_size, num_seqs, s,
                )
                success += 1 if result is not None else 0
                fail += 0 if result is not None else 1
            except Exception as e:
                print(f"ERROR: {e}")
                fail += 1

    print(f"\n{'#'*60}")
    print(f"Done! Success: {success}, Skip: {skip_cnt}, Fail: {fail}")

    raise SystemExit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()

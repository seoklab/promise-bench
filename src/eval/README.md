# Evaluation Module

Quantify prediction biases on ProMiSE-bench conformational pairs.

---

## Quick Start

```bash
conda activate promise

# 1. Prepare structures (renumber to MSA positions)
python -m src.eval.cif_to_renumbered_pdb --targets-dir examples/targets

# 2. Predict contacts from MSA
python -m src.eval.esm_run --examples-dir examples/msa-server

# 3. Compute MSA bias
python -m src.eval.msa_bias -o data/eval/msa_bias.csv
```

---

## Pipeline Overview

| # | Step | Script | Output |
|---|------|--------|--------|
| 1 | Renumber structures | `cif_to_renumbered_pdb.py` | `data/eval/renumbered_pdbs/` |
| 2 | ESM contact prediction | `esm_run.py` | `data/eval/msas/*.npy` |
| 3 | MSA bias calculation | `msa_bias.py` | `data/eval/msa_bias.csv` |

---

## MSA Bias

Measures whether MSA coevolution signals favor one conformation over another.

### Step 1: Renumber Structures

Convert CIF assemblies to PDB with residue numbers aligned to the representative MSA position.

```bash
python -m src.eval.cif_to_renumbered_pdb \
    --targets-dir examples/targets \
    --msa-dir data/msas \
    --rep-seq data/rep_seq.json \
    -o data/eval/renumbered_pdbs
```

### Step 2: ESM Contact Prediction

Predict residue contacts using ESM-MSA-1b.

```bash
python -m src.eval.esm_run \
    --examples-dir examples/msa-server \
    -o data/eval/msas \
    --multi-seed  # 10 seeds for robustness
```

| Option | Description |
|--------|-------------|
| `--sample-size N` | Random sample from MSA (default: 1024) |
| `--num-seqs N` | Sequences for ESM (default: 128) |
| `--device cuda/cpu` | Inference device (default: auto) |

### Step 3: MSA Bias Calculation

Compute MSA preference scores for each conformational pair.

```bash
python -m src.eval.msa_bias \
    --valid-pairs data/dataset/valid_pairs.json \
    --pdb-dir data/eval/renumbered_pdbs \
    --esm-dir data/eval/msas \
    -o data/eval/msa_bias.csv
```

**Output columns:**

| Column | Description |
|--------|-------------|
| `conf1_name` | Apo conformer tag |
| `conf2_name` | Holo conformer tag |
| `msa_pref` | MSA preference (−1 = apo, +1 = holo) |
| `common_count` | Contacts in both conformations |
| `conf1_unique_count` | Contacts only in conf1 |
| `conf2_unique_count` | Contacts only in conf2 |

---

## Project Structure

```
src/eval/
├── cif_to_renumbered_pdb.py   Renumber CIF → PDB (MSA-aligned)
├── esm_run.py                 ESM-MSA-1b contact prediction
├── msa_bias.py                MSA coevolution bias calculation
└── README.md
```

---

## 
- **Distogram Bias** 
- **Structure Bias** 
- **Training Bias**

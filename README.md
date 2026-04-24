# ProMiSE-bench

ProMiSE: **Pro**tein **M**ult**i**-**S**tate **E**valuation Benchmark in Biological Contexts

A curated benchmark dataset of protein conformational changes derived from experimentally determined structures in the Protein Data Bank (PDB).

## Overview

ProMiSE-bench provides high-quality protein conformational change pairs for:
- Assessing protein structure prediction models (e.g., AlphaFold3, Boltz-1,2, Chai-1)
- Evaluating multi-state conformation sampling capabilities of prediction models with novel metrics



### Key Features

- **🧬 Biology-Aware Pairs**: High-resolution pairs capturing binder-induced conformational changes
- **🔍 Stringent QC Pipeline**: Removal of crystal artifacts and redundant assemblies to ensure physiological relavance
- **📊 Advanced Evaluation**: Multi-state success metrics and rigorous leakage analysis beyond traditional RMSD

### Quick Install

```bash
git clone https://github.com/ProMiSE-bench/ProMiSE-bench.git
cd ProMiSE-bench
bash install.sh
```

This script requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (used to install the editable `promise-data` package into the conda env). It creates two conda environments:
- `promise`: Main curation pipeline (Python 3.9+)
- `prodigy-cryst`: Crystal contact classifier (Python 3.8, used internally)

To install only the Python package and dependencies into a local virtualenv (for example when reproducing eval scripts), run `uv sync` from the repository root, or `uv pip install -e .` into an existing environment. The full curation pipeline still expects the conda-provided tools (FAMSA, Foldseek, MMseqs2, and so on) from `environment.yaml`.


## Usage

### Dataset

The ProMiSE dataset is available in [data/dataset](data/dataset), where each CSV file is assigned to one of three categories: intrinsic dynamics, ligand-induced, or protein-induced.

### Running the Full Pipeline

```bash
conda activate promise
cd ProMiSE-bench

promise_data run \
    --spec data/clusters.json \
    --mmcif-store /path/to/pdb_mmcif/mmcif_files
```
`data/clusters.json` is provided in the repo. However, mmcif files should be manually downloaded with `src/curation/utils/download_mmcif.py`. Refer to [src/curation/README.md](src/curation/README.md) for details.

### Curation Pipeline Overview

The curation pipeline consists of 11 steps:

1. **Create key files**: Align conformers with FAMSA, etc.
2. **TM-Score Computation**: Calculate structural similarity
3. **Clustering**: Sub-cluster by TM-score
4. **Input Preparation**: Parse mmCIF assemblies
5. **Crystal Contact Detection**: Classify interfaces with PRODIGY-cryst
6. **Crystal Filtering**: Remove crystallographic artifacts
7. **Subset Filtering**: Filter by sequence identity
8. **Metal Processing**: Remove low-coordination metal ions
9. **Set Curation**: Extract conformational pairs
10. **Representative Selection**: Filter by binding compatibility
11. **Sequence Clustering**: Remove redundancy (MMseqs2 @ 40%)

See [src/curation/README.md](src/curation/README.md) for step-by-step details.

## Output Structure

### Key Pipeline Outputs (available for download)
```
data/
├── seqs/                      # FASTA files for each sequence cluster
├── msas/                      # Multiple sequence alignments 
├── clusters/                  # Conformational clustering results
├── scores/                    # TM-score matrices
├── filtered-pairs.csv         # Conformational pairs passing filters
└── representative_sequences_total.json  # Representative sequences for inference

```

### Intermediate Files (generated during pipeline run)
```
data/
├── asms-raw/                  # Parsed assembly information
├── asms-bio/                  # Crystal-filtered assemblies
├── asms-subset/               # Sequence identity filtered
├── asms-metal/                # Metal coordination filtered
├── combinations/              # Conformational pair combinations
├── combinations-filtered/     # Representative pairs
├── seqcluster_work/           # MMseqs2 result for sequence clustering
├── pair-calls.csv             # Crystal artifact probability from PRODIGY-cryst
└── binding_site_compatibility.csv  # Binding site comparison for representative selection

```

### Final Output

```
data/
├── representative_sequences.json  # Representative sequences for inference
└── dataset-pipeline/          # Curated dataset from the pipeline
```

**Download Key Pipeline Outputs**: (https://drive.google.com/drive/folders/1BALc--RHPy8QVZaFNtI3LLXFfGWL_z4V?usp=drive_link)

Pre-computed pipeline outputs are available via the Google Drive link above. These files allow you to start from Step 4 and skip the computationally expensive Steps 1–3 (Create key files, TM-score computation, and conformation clustering). Since some outputs are required for evaluation, we strongly recommend downloading them.

After downloading, extract `data.tar.gz` in the `data/` directory:

```bash
tar -xzvf data.tar.gz
```

Then start the pipeline from Step 4 using the `--start-from` option (see [src/curation/README.md](src/curation/README.md) for more details):

```bash
promise_data run \
    --spec data/clusters.json \
    --mmcif-store /path/to/pdb_mmcif/mmcif_files \
    --start-from prepare_inputs
```

## Project Structure

```
promise-bench/
├── README.md                               
├── pyproject.toml                          
├── install.sh                              # Installation script
├── environment.yaml                        # Main conda environment
├── environment-prodigy.yaml                # Prodigy-cryst environment
├── data/                                   
│   ├── clusters.json                       # Input cluster specification (provided)
│   ├── preference_score.json               # Preference metrics of final curated dataset
│   └── dataset/                            # Final curated dataset
└── src/
    └── curation/                           
        ├── README.md                       
        ├── run.py                          
        ├── pipeline/                       # Pipeline orchestration & step modules
        └── utils/                          # Utility functions
    
```

Pre-computed outputs downloaded (`data.tar.gz`) should be unzipped under `data/` directory.

## Evaluation

### Preference Scores
Pre-computed preference scores (`data/preference_scores.json`) for each prediction model across all conformational pairs. This file aggregates multiple evaluation metrics to assess how well each model captures the holo (target) conformation.

**Structure:**
```
{model} → {category} → {cluster_id} → {pair_id} → {scores}
```

- **Models**: `alphafold3`, `boltz1`, `boltz2`, `chai`, `bioemu`
- **Categories**: `intrinsic-dynamics`, `ligand-induced`, `protein-induced`
- **Cluster ID**: Sequence cluster identifier (e.g., `8ABP_1`)
- **Pair ID**: Conformational pair identifier in format `{pdb1}_{asm1}_{chain1}-{pdb2}_{asm2}_{chain2}`
  - Example: `2wrz_2_B1-2wrz_1_A1` represents conformer 1 (PDB: 2wrz, assembly: 2, chain: B1) vs conformer 2 (PDB: 2wrz, assembly: 1, chain: A1) 

**Score Fields:**

| Field | Description |
|-------|-------------|
| `msa_holo` | MSA-based preference score toward holo conformation (sum of per-residue preferences) |
| `rmsd_conf1_conf2` | RMSD (Å) between the two reference conformations |
| `struct_holo` | Structure-based (ConfBench) preference score toward holo conformation |
| `disto_holo` | Distogram-based preference score toward holo conformation |
| `dyndisto_holo` | Dynamic distogram-based preference score toward holo conformation |
| `bias_entry1_hits` | Number of PDB training set hits for entry 1 |
| `bias_entry2_hits` | Number of PDB training set hits for entry 2 |
| `train_holo` | Training data bias toward holo conformation (ratio difference of training hits) |
| `after_training_cutoff` | Whether the pair entries are deposited after the model's training cutoff date |

Positive values of `msa_holo`, `struct_holo`, `disto_holo`, and `dyndisto_holo` indicate a preference toward the holo conformation, while negative values indicate a preference toward the apo conformation. `train_holo` quantifies preference of the training data.


## Contributing

Contributions are welcome! Please open an issue or pull request.

## Contact

For questions or issues, please:
- Open a [GitHub issue](https://github.com/ProMiSE-bench/ProMiSE-bench/issues)


# ProMiSE-bench

**Pro**tein **M**ult**i**-**S**tate **E**valuation Benchmark in Biological Contexts

A curated benchmark dataset of protein conformational changes derived from experimentally determined structures in the Protein Data Bank (PDB).

## Overview

ProMiSE-bench provides high-quality protein conformational change pairs for:
- Assessing protein structure prediction models (e.g., AlphaFold3, Boltz-1,2, Chai-1)
- Evaluating conformational sampling capabilities of prediction models with novel metrics


### Key Features

- **🧬 Biology-Aware Pairs**: High-resolution pairs capturing binder-induced conformational changes
- **🔍 Stringent QC Pipeline**: Removal of crystal artifacts and redundant assemblies to ensure physiological relavance
- **📊 Advanced Evaluation**: Multi-state success metrics and rigorous leakage analysis beyond traditional RMSD

### Quick Install

```bash
git clone https://github.com/ProMiSE-bench/ProMiSE-bench.git
cd promise-bench
bash install.sh
```

This creates two conda environments:
- `promise`: Main curation pipeline (Python 3.9+)
- `prodigy-cryst`: Crystal contact classifier (Python 3.8, used internally)


## Usage

### Running the Full Pipeline

```bash
conda activate promise
cd promise-bench

promise_data run \
    --spec data/clusters.json \
    --mmcif-store /path/to/pdb_mmcif/mmcif_files
```
`clusters.json` is provided in the repo. However, mmcif files should be manually downloaded. Refer to src/curation/README.md for details.
### Pipeline Overview

The curation pipeline consists of 11 steps:

1. **MSA Creation**: Align conformers with FAMSA
2. **TM-Score Computation**: Calculate structural similarity
3. **Clustering**: Sub-cluster by TM-score
4. **Input Preparation**: Parse mmCIF assemblies
5. **Crystal Contact Detection**: Classify interfaces with PRODIGY-CRYST
6. **Crystal Filtering**: Remove crystallographic artifacts
7. **Subset Filtering**: Filter by sequence identity
8. **Metal Processing**: Remove low-coordination metal ions
9. **Set Curation**: Extract conformational pairs
10. **Representative Selection**: Filter by binding compatibility
11. **Sequence Clustering**: Remove redundancy (MMseqs2 @ 40%)

See [src/curation/README.md](src/curation/README.md) for step-by-step details.

## Output Structure

```
data/
├── clusters.json              # Input cluster specification
├── msas/                      # Multiple sequence alignments
├── scores/                    # TM-score matrices
├── filtered-pairs.csv         # Conformational pairs passing filters
├── asms-raw/                  # Parsed assembly information
├── asms-bio/                  # Crystal-filtered assemblies
├── asms-subset/               # Sequence identity filtered
├── asms-metal/                # Metal coordination filtered
├── combinations/              # Conformational pair combinations
├── combinations-filtered/     # Representative pairs
└── dataset-pipeline/          # Final redundancy-filtered dataset
```

## Examples

See [examples/](examples/) for:
- Sample input cluster specifications
- Expected output formats
- Analysis scripts

## Project Structure

```
promise-bench/
├── README.md                  # This file
├── pyproject.toml             # Package metadata
├── install.sh                 # Installation script
├── environment.yaml           # Main conda environment
├── environment-prodigy.yaml   # Prodigy-cryst environment
├── data/                      # Working directory (generated)
│   └── clusters.json          # Input cluster specification
├── examples/                  # Example data and scripts
└── src/
    └── curation/              # Data curation pipeline
        ├── README.md          # Detailed pipeline documentation
        ├── pipeline.py        # Pipeline orchestration
        ├── run.py             # CLI entry point (promise_data)
        └── *.py               # Individual pipeline steps
```

## Citation

If you use ProMiSE-bench in your research, please cite:

```
[Citation to be added upon publication]
```

## License

[License to be determined]

## Contributing

Contributions are welcome! Please open an issue or pull request.

## Contact

For questions or issues, please:
- Open a [GitHub issue](https://github.com/ProMiSE-bench/ProMiSE-bench/issues)
- Contact: [your-email@example.com]


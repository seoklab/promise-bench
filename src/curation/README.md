# ProMiSE-Bench Curation Pipeline

Data curation pipeline for building the **ProMiSE** (Protein Multi-State Evaluation) benchmark — a curated set of conformational-change pairs
from the PDB.

---

## Prerequisites

- `install.sh`


## Quick Start

```bash
conda activate promise

promise_data run \
    --spec data/clusters.json \
    --mmcif-store /path/to/pdb_mmcif/mmcif_files
```

| Option | Description |
|--------|-------------|
| `--spec` | Cluster specification JSON (GroupSet format) |
| `--mmcif-store` | Directory of PDB mmCIF files (`*.cif`) |
| `--keep-intermediates` | Keep intermediate directories (asms-raw, asms-bio, etc.) under `data/`. By default, they are written to a temporary directory and deleted when the pipeline finishes. |

All outputs are written under `data/`. Final curated dataset: `data/dataset-pipeline/`.

### Downloading mmCIF Files

```bash
# Specific PDB IDs
python -m curation.download_mmcif --data-dir /path --pdb-list ids.txt

# Full PDB mirror (~600 GB, requires stable connection)
python -m curation.download_mmcif --data-dir /path
```

**Warning**: Full PDB mirror download is ~600 GB and may take several hours to days depending on network speed. Ensure sufficient disk space and stable internet connection before proceeding.

---

## Pipeline Overview

```
promise_data steps

 1. create_msa             Build MSAs (FAMSA) and extract Cα coords
 2. pairwise_tm            Pairwise TM-scores within each cluster
 3. cluster_by_tmscore     Sub-cluster by TM-score, extract pairs
 4. prepare_inputs         Parse assembly info from mmCIF
 5. run_prodigy            Classify crystal contacts (PRODIGY-cryst)
 6. filter_xtal            Remove crystal-contact assemblies
 7. subsets                Filter by sequence identity
 8. process_metal          Remove low-coordination metal ions
 9. curate_sets            Extract conformational-change pairs
10. select_representative  Select representatives by binding-site compatibility
11. filter_seq_clusters    Remove redundant clusters (MMseqs2, 40% identity)
```

| # | Step | Key Outputs |
|---|------|-------------|
| 1 | `create_msa` | `data/msas/`, `data/coords/` |
| 2 | `pairwise_tm` | `data/scores/` |
| 3 | `cluster_by_tmscore` | `data/clusters/`, `data/filtered-pairs.csv` |
| 4 | `prepare_inputs` | `data/asms-raw/`, `data/cif-asms/` |
| 5 | `run_prodigy` | `data/pair-calls.csv` |
| 6 | `filter_xtal` | `data/asms-bio/` |
| 7 | `subsets` | `data/asms-subset/` |
| 8 | `process_metal` | `data/asms-metal/` |
| 9 | `curate_sets` | `data/combinations/` |
| 10 | `select_representative` | `data/combinations-filtered/` |
| 11 | `filter_seq_clusters` | `data/dataset-pipeline/` |

### Partial Execution

```bash
# Resume from a specific step
promise_data run --spec spec.json --mmcif-store /path --start-from curate_sets

# Run only steps 1-3
promise_data run --spec spec.json --mmcif-store /path --stop-after cluster_by_tmscore

# Run a range
promise_data run --spec spec.json --mmcif-store /path \
    --start-from prepare_inputs --stop-after process_metal
```

### Custom Output Directory

```bash
promise_data run --spec spec.json --mmcif-store /path -C /work/output
```

### Running Individual Steps

```bash
python -m curation.create_msa --help
python -m curation.curate_sets --help
```

---

## Project Structure

```
src/curation/
│
├── __main__.py                     
├── run.py                          
│
├── utils/                          
│   ├── constants.py                
│   ├── typedefs.py                 
│   ├── pdb_utils.py                
│   └── download_mmcif.py           # Download mmCIF files from RCSB
│
└── pipeline/                       
    ├── __init__.py                 
    ├── create_msa.py               1.  Build MSAs, extract Cα coordinates
    ├── pairwise_tm_multiprocessing.py  2.  Pairwise TM-score computation
    ├── cluster_by_tmscore.py       3.  Agglomerative clustering by TM-score
    ├── prepare_inputs_gemmi.py     4.  Assembly extraction from mmCIF
    ├── run_prodigy.py              5.  Crystal contact classification
    ├── filter_xtal.py              6.  Crystal-contact assembly filtering
    ├── subsets.py                  7.  Sequence-identity based filtering
    ├── process_metal.py            8.  Low-coordination metal filtering
    ├── curate_sets.py              9.  Conformational-change pair extraction
    ├── select_representative.py   10.  Representative selection by binding-site
    └── filter_seq_clusters.py     11.  MMseqs2 redundancy removal
```

Note that the final dataset presented involves a manual curation step. Therefore, simply running the full pipeline will not reproduce the exact dataset used in this work. After Step 9 (curate_sets.py), we manually removed a small number of structure pairs that exhibited conformational changes but lacked binders or were associated with crystal artifacts.

We provide the file `representative_sequences_total.json`, which contains the representative sequences selected for the released dataset. This file may differ from the one generated automatically by the pipeline.


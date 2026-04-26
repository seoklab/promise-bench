# Evaluation pipeline (`src/eval/`)

This directory contains the **evaluation pipeline** (alignment, structure-based scoring, and distogram-based scoring).

`curation.make_pairs` writes **`seq_cluster_to_answer_map.json`** (MSA paths, reference CIFs, `apo_references` / `holo_references`, distogram patterns, chains). For reference Cβ extraction, pass **that file’s path** to `extract_reference_cb --distogram`. The flag label is older than this filename; it still means “the cluster map JSON,” not a separate preprocessing output.

Below: **what runs where**, **inputs → outputs**, and **recommended order**. Paths depend on `config/config.yaml` and CLI flags; see each module’s `--help`.

---

## Recommended run order (end-to-end)

This section is the **sequential** “do this, then that” ordering for a fresh run.
Some branches can run in parallel on a cluster, but the dependencies below must hold.

### Step 1 — Curation (required)

1. Generate the base pair definitions and enriched map:

   - `python -m curation.make_pairs`

   Outputs (you will use these later):
   - `valid_pairs.json`
   - `seq_cluster_to_answer_map.json` (enriched map; this is also the `--distogram` input below)

### Step 2 — Distogram task preparation (required for both §3 and §4)

2. (Optional but common) Materialize per-reference Cβ coordinate JSONs from the enriched map (Step 1 output):

   - `python -m eval.distogram.extract_reference_cb --distogram <seq_cluster_to_answer_map.json>`

   Output:
   - augmented JSON (`seq_cluster_to_answer_map_with_cb_paths.json`, or `<stem>_with_cb_paths.json` if you used another filename) that adds `reference_cb_json`

3. Build the distogram symlink tree and generate `distogram_tasks.json`:

   - `python -m eval.distogram.collect_distograms --json <seq_cluster_to_answer_map_with_cb_paths.json> ...`

   Outputs:
   - a `distogram/...` symlink tree under your chosen output dir
   - `distogram_tasks.json` (when task-generation flags are enabled; see `--help`)

**Note:** `distogram_tasks.json` is a key dependency for **Struct reference↔reference metrics** (§3.1).

### Step 3 — Alignment (required for structure-based ConfBench; optional otherwise)

4. Author alignment tasks:

   - `python -m eval.align.generate_alignment_tasks ...`

   Output:
   - `alignment_tasks.json`

5. Run alignment in shards (recommended) or directly:

   - `python -m eval.align.split_alignment_jobs ...` → generates `run_all.sh` / `sbatch` scripts
   - each shard runs `python -m eval.align.struct_align_batch --json alignment_tasks_partXXXX.json --results-json align_partXXXX.json`

   Outputs:
   - `align_part*.json` (prediction↔reference metrics)

### Step 4 — Struct scoring (required for `merge_all`)

6. Compute reference↔reference structural metrics (requires `distogram_tasks.json` from Step 2):

   - `python -m eval.struct.calc_reference_structural_metrics --tasks <distogram_tasks.json> --output-dir <reference_metrics>`

7. Compute structure-based ConfBench scores (requires alignment results from Step 3):

   - `python -m eval.struct.calc_confbench_score_valid_pairs --align-results <job_batches>/align_results --ref-metrics-dir <reference_metrics>`

   Output:
   - `confbench_scores_valid_pairs.json` (plus summary CSV / validation report)

### Step 5 — Distogram scoring (required for `merge_all`)

8. Run distogram loss and downstream steps (often sharded with `--start/--end`):

   - `python -m eval.distogram.calc_distogram_loss --tasks <distogram_tasks.json> ...`
   - `python -m eval.distogram.calc_reference_distogram_diff --tasks <distogram_tasks.json> ...`
   - `python -m eval.distogram.calc_distogram_confbench ...`

   Output:
   - `confbench_scores_distogram.json` (for `eval.merge_all --confbench-distogram-json`)

### Step 6 — Merge (optional; paper analysis)

9. Merge all metrics into final JSON/CSV:

   - `python -m eval.merge_all --valid-pairs-json ... --confbench-json ... --confbench-distogram-json ... --msa-pref-csv ...`

---

## 1. Curation (`src/curation/`)

Run **in this order**.

### 1.1 `make_pairs`

| | |
|---|-----|
| **Command** | `python -m curation.make_pairs` (see `--help` for `--csv-dir`, `--outdir`, `--examples-dir`, …) |
| **Typical inputs** | Cluster / dataset tables under your data root, example prediction layouts under `examples_dir`, and other inputs described in `make_pairs`’s docstring and Click options. |
| **Main outputs** | **`seq_cluster_to_answer_map.json`** and **`valid_pairs.json`**. The map is the file passed to **`extract_reference_cb --distogram`** and **`collect_distograms --json`** (after optional `_with_cb_paths` augmentation). |

---

## 2. Alignment (`src/eval/align/`)

Run **after** §1.1 has produced `valid_pairs.json` and the enriched map JSON.

### 2.1 `generate_alignment_tasks` (task authoring)

| | |
|---|-----|
| **Command** | `python -m eval.align.generate_alignment_tasks` (see `--help`) |
| **Typical inputs** | `valid_pairs.json` + `seq_cluster_to_answer_map.json` from `curation.make_pairs`. |
| **Main outputs** | `alignment_tasks.json` (task list for structure alignment). |

This step is **required** for the alignment pipeline, because `split_alignment_jobs` consumes `alignment_tasks.json`.

### 2.2 `struct_align_batch` (execution)

| | |
|---|-----|
| **How it is usually run** | Indirectly via `python -m eval.align.split_alignment_jobs` → generated `run_all.sh` / sbatch scripts. |
| **Direct command** | `python -m eval.align.struct_align_batch --json <alignment_tasks_partXXXX.json> --results-json <align_partXXXX.json>` |
| **Inputs** | Sharded `alignment_tasks_part*.json` (each row contains `ref_cif`, `mobile_cif`, `ref_chain`, `mobile_chain`, plus metadata like `cluster_id`, `pair_type`, `valid_pair`). |
| **Outputs** | `align_part*.json` with per-task results including `rmsd_ca` and `tm_score_ca` (prediction↔reference). Optional aligned CIF writing depends on flags / the generated scripts. |

This output is later consumed by **structure-based ConfBench** (`eval.struct.calc_confbench_score_valid_pairs`).

This branch is **independent** of the distogram scripts below: it does **not** need `extract_reference_cb` to finish first, as long as structures referenced in tasks exist.

**Alignment modules (index)**:
- `eval/align/generate_alignment_tasks.py`: write `alignment_tasks.json`
- `eval/align/split_alignment_jobs.py`: shard tasks + emit `run_all.sh` / sbatch
- `eval/align/struct_align_batch.py`: run alignment and write `align_part*.json` (prediction↔reference RMSD/TM-score)

---

## 3. Struct (`src/eval/struct/`)

### 3.1 Reference↔reference structural metrics (optional)

To produce reference↔reference metrics (e.g. CA RMSD) used by downstream structure-based
ConfBench scoring, run:

- `python -m eval.struct.calc_reference_structural_metrics --tasks <distogram_tasks.json> --output-dir <reference_metrics>`

This step requires `distogram_tasks.json` (authored by `eval.distogram.collect_distograms`).

This writes per-pair `*_metrics.json` under `<reference_metrics>/aligned_references/...`.

### 3.2 Structure-based ConfBench (optional)

If you want **RMSD-based ConfBench scores** (used by `eval/merge_all.py` as `--confbench-json`),
run:

- `python -m eval.struct.calc_confbench_score_valid_pairs --align-results <job_batches>/align_results --ref-metrics-dir <reference_metrics>`

This consumes:
- per-task alignment results (`align_part*.json`) for prediction↔reference RMSDs
- reference↔reference metrics emitted by `eval.struct.calc_reference_structural_metrics`

---

## 4. Distogram (`src/eval/distogram/`)

These modules live under **`eval/distogram`**, not under `curation`.

### 4.1 Relationship to §1–2 (parallelism)

- **After `make_pairs` (§1.1)** you can start **§4** without waiting for alignment (§2), as long as the map and on-disk prediction/reference paths are consistent.
- **Timeline:** alignment (§2) and distogram prep/compute (§4) are often run **in parallel** on a cluster: e.g. Slurm alignment jobs while you run `extract_reference_cb` → `collect_distograms` → loss shards.
- **Logical order inside distogram:** follow the steps below; many clusters also run **`calc_distogram_loss`** (and later steps) **in parallel** over task index ranges (`--start` / `--end`) or many jobs.

### 4.2 `extract_reference_cb`

`extract_reference_cb` is the step that materializes **per-reference CB coordinate JSON**
files used by downstream distogram tasks.

#### Mode: `--distogram` (enriched cluster map)

| | |
|---|-----|
| **Command** | `python -m eval.distogram.extract_reference_cb --distogram path/to/seq_cluster_to_answer_map.json` |
| **Input** | Same schema as **`seq_cluster_to_answer_map.json`** from `make_pairs` (`apo_references` / `holo_references` with `reference_cif_path`). Legacy filenames like `distogram_analysis_data_final.json` refer to the same shape. |
| **Output** | Writes per-reference `*_cb.json` under the configured `ref_coords` root, emits `distogram_ref_cb_map.json`, and writes an **augmented** distogram JSON that adds `reference_cb_json` paths. |
This augmented JSON (the `*_with_cb_paths.json` next to the input) is what you should pass to `collect_distograms --json ...` so that `distogram_tasks.json` includes `reference_cb_json`.

### 4.3 `collect_distograms`

| | |
|---|-----|
| **Command** | `python -m eval.distogram.collect_distograms` (`--help`) |
| **Typical inputs** | **`seq_cluster_to_answer_map.json`** or its **`_with_cb_paths`** variant from `extract_reference_cb` (see `--json` in `--help`); method filters (`--method` / `--all`); optional AF3 chain-mapping root. |
| **Main outputs** | **`distogram/…`** symlink tree under your chosen output dir; optionally **`distogram_tasks.json`** (via task-generation flags). |

### 4.4 `calc_distogram_loss` → `calc_reference_distogram_diff` → `calc_distogram_confbench`

| Step | Command (pattern) | Typical inputs | Typical outputs |
|:----:|---------------------|----------------|-----------------|
| Loss | `python -m eval.distogram.calc_distogram_loss --tasks …` | Final **`distogram_tasks.json`**, prediction npz paths in tasks, MSAs, rep-seq JSON, ref-diff roots | Per-prediction dir: e.g. **`distogram_loss_real_final.json`** (see module for exact filenames) |
| Ref–ref diff | `python -m eval.distogram.calc_reference_distogram_diff --tasks …` | Same task JSON + MSA/rep-seq + ref layout | Reference–reference distogram diff tree under your ref_distogram root |
| ConfBench-style | `python -m eval.distogram.calc_distogram_confbench …` | `valid_pairs`, `distogram_tasks`, ref_distogram paths | Aggregated scores JSON (see `--help`) |

**Ref–ref diff** and **confbench** both assume the **loss** stage has populated the paths/state they read; they are “downstream” of loss in the same task tree, not alternatives to each other.

---

## Optional: Slurm job generators

The following job generators are **optional** (useful when you want many small jobs / sbatch scripts).

- **Alignment**: `python -m eval.align.split_alignment_jobs`
  - **Use when**: `alignment_tasks.json` is large and you want `run_all.sh` / many `sbatch` shards.
  - **Skip when**: you can run `python -m eval.align.struct_align_batch` directly on a single JSON (or you shard manually).

- **Distogram**: `python -m eval.distogram.generate_distogram_calc_jobs` and `python -m eval.distogram.generate_distogram_diff_jobs`
  - **Use when**: task count is large and you want many pre-generated `sbatch` scripts (instead of hand-writing array jobs).
  - **Skip when**: you can run `calc_distogram_loss` / `calc_reference_distogram_diff` directly (with `--start`/`--end` or your own scheduler wrappers).

---

## Optional: progress checks

**`check_distogram_results`** and **`check_distogram_diff_results`** are for **verification when a long run may be incomplete** (failed nodes, partial arrays, interrupted sessions).

- They compare **`distogram_tasks.json`** (and `valid_pairs` where applicable) to files on disk and summarize **missing or empty** artefacts.
- Run them before resubmitting slices so you do not blindly rerun finished work.

---

## How to run (quick reference)

Use `PYTHONPATH=src` (or an editable install) and module invocations, e.g.:

```bash
python -m eval.distogram.collect_distograms --help
```

If CLI paths are omitted, defaults come from `config/config.yaml` (`eval` section) and `EvalConfig.distogram_*` in `utils._config.eval_cfg`.

---

## Module index

| Module | Role |
|--------|------|
| `eval/distogram/extract_reference_cb` | Reference Cβ extraction from **`seq_cluster_to_answer_map.json`** via `--distogram` (adds `reference_cb_json` for downstream tasks) |
| `eval/distogram/collect_distograms` | Symlink distograms into a tree; emit `distogram_tasks.json` |
| `eval/distogram/calc_distogram_loss` | Prediction vs reference distogram loss |
| `eval/distogram/calc_reference_distogram_diff` | Pairwise reference distogram differences |
| `eval/distogram/calc_distogram_confbench` | ConfBench-like scores from distogram distances |
| `eval/distogram/check_distogram_results` | **Optional** — audit loss-stage outputs vs tasks |
| `eval/distogram/check_distogram_diff_results` | **Optional** — audit ref-diff outputs vs tasks |
| `eval/distogram/generate_distogram_calc_jobs` | **Optional** — split `calc_distogram_loss` into many Slurm jobs |
| `eval/distogram/generate_distogram_diff_jobs` | **Optional** — split `calc_reference_distogram_diff` into many Slurm jobs |

Shared path helpers: `eval/distogram/path_utils.py`.

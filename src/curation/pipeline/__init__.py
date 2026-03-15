"""Curation pipeline definition.

Each step is a pair (label, callable) where the callable accepts
``(spec, mmcif_store, workdir)`` and returns the Click args list to be
invoked on that step's ``main`` / ``cli`` click command.

Intermediate outputs
--------------------
By default the runner writes intermediate artefacts (``asms-raw``,
``asms-bio``, …) under a temporary directory and deletes them once the
pipeline finishes.  Pass ``keep_intermediates=True`` to write them to
the normal ``data/`` tree instead.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import click

# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

# Directories inside ``data/`` that are intermediate artefacts.
# They will be written to tmpdir and cleaned up unless the user
# explicitly requests keeping them.
INTERMEDIATE_DIRS: set[str] = {
    "asms-raw",
    "asms-bio",
    "asms-subset",
    "asms-metal",
    "combinations",
    "combinations-filtered",
    "seqcluster_work",
}

# Final output directory name (replaces "combinations-seqfiltered")
FINAL_OUTPUT_DIR = "dataset-pipeline"


class Step:
    """One step of the pipeline."""

    __slots__ = ("label", "module_path", "entry", "args_fn")

    def __init__(
        self,
        label: str,
        module_path: str,
        entry: str = "main",
        args_fn=None,
    ):
        self.label = label
        self.module_path = module_path  # e.g. "curation.create_msa"
        self.entry = entry  # click command name inside module
        self.args_fn = args_fn  # (spec, mmcif_store, workdir, data, tmp) -> list[str]


def _resolve_prodigy() -> Optional[str]:
    """Build a prodigy_cryst invocation command.

    If a ``prodigy-cryst`` conda env exists, returns
    ``conda run --no-banner -n prodigy-cryst prodigy_cryst``
    so the binary runs inside that env.  Otherwise falls back
    to a bare ``prodigy_cryst`` (must be on PATH).
    """
    import subprocess as sp

    try:
        sp.run(
            ["conda", "run", "-n", "prodigy-cryst", "prodigy_cryst", "--help"],
            capture_output=True,
            check=True,
        )
        return "conda run -n prodigy-cryst prodigy_cryst"
    except Exception:
        if shutil.which("prodigy_cryst"):
            return "prodigy_cryst"
        return None


def _d(data: Path, tmp: Path, name: str) -> str:
    """Resolve an intermediate directory.

    If *tmp* differs from *data*, intermediates go under *tmp*;
    otherwise they land in the normal ``data/`` tree.
    """
    if name in INTERMEDIATE_DIRS and tmp != data:
        p = tmp / name
    else:
        p = data / name
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _f(data: Path, tmp: Path, name: str) -> str:
    """Resolve an intermediate *file* (e.g. ``pair-calls.csv``)."""
    if tmp != data:
        p = tmp / name
    else:
        p = data / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


STEPS: list[Step] = [
    # --- Phase 1: extract pairs ---
    Step(
        "create_msa",
        "curation.pipeline.create_msa",
        args_fn=lambda s, m, w, d, t: ["--mmcif-dir", str(m), str(s)],
    ),
    Step(
        "pairwise_tm",
        "curation.pipeline.pairwise_tm_multiprocessing",
    ),
    Step(
        "cluster_by_tmscore",
        "curation.pipeline.cluster_by_tmscore",
    ),
    # --- Phase 2: curation ---
    Step(
        "prepare_inputs",
        "curation.pipeline.prepare_inputs_gemmi",
        args_fn=lambda s, m, w, d, t: [
            "--mmcif-dir",
            str(m),
            "--out-root",
            _d(d, t, "asms-raw"),
            "--save-assemblies-dir",
            str(d / "cif-asms"),  # always keep
        ],
    ),
    Step(
        "run_prodigy",
        "curation.pipeline.run_prodigy",
        args_fn=lambda s, m, w, d, t: [
            "--raw-dir",
            _d(d, t, "asms-raw"),
            "--out-csv",
            _f(d, t, "pair-calls.csv"),
        ],
    ),
    Step(
        "filter_xtal",
        "curation.pipeline.filter_xtal",
        args_fn=lambda s, m, w, d, t: [
            "--asm-raw-dir",
            _d(d, t, "asms-raw"),
            "--out-dir",
            _d(d, t, "asms-bio"),
            "--pair-calls-csv",
            _f(d, t, "pair-calls.csv"),
        ],
    ),
    Step(
        "subsets",
        "curation.pipeline.subsets",
        args_fn=lambda s, m, w, d, t: [
            "--asm-bio-dir",
            _d(d, t, "asms-bio"),
            "--out-dir",
            _d(d, t, "asms-subset"),
        ],
    ),
    Step(
        "process_metal",
        "curation.pipeline.process_metal",
        args_fn=lambda s, m, w, d, t: [
            "--in-dir",
            _d(d, t, "asms-subset"),
            "--out-dir",
            _d(d, t, "asms-metal"),
        ],
    ),
    Step(
        "curate_sets",
        "curation.pipeline.curate_sets",
        entry="cli",
        args_fn=lambda s, m, w, d, t: [
            "--filtered-dir",
            _d(d, t, "asms-metal"),
            "--outdir",
            _d(d, t, "combinations"),
            "--filtered-pairs",
            _f(d, t, "filtered-pairs.csv"),
        ],
    ),
    Step(
        "select_representative",
        "curation.pipeline.select_representative",
        args_fn=lambda s, m, w, d, t: [
            "--dataset-dir",
            _d(d, t, "combinations"),
            "--out-dataset",
            _d(d, t, "combinations-filtered"),
        ],
    ),
    Step(
        "filter_seq_clusters",
        "curation.pipeline.filter_seq_clusters",
        args_fn=lambda s, m, w, d, t: [
            "--dataset-dir",
            _d(d, t, "combinations"),
            "--out-dir",
            str(d / FINAL_OUTPUT_DIR),
            "--work-dir",
            _d(d, t, "seqcluster_work"),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _import_click_cmd(step: Step) -> click.BaseCommand:
    """Dynamically import a step's Click command object."""
    import importlib

    mod = importlib.import_module(step.module_path)
    cmd = getattr(mod, step.entry)
    if not isinstance(cmd, click.BaseCommand):
        raise TypeError(f"{step.module_path}.{step.entry} is not a Click command")
    return cmd


def run_pipeline(
    spec: Path,
    mmcif_store: Path,
    workdir: Path,
    *,
    start_from: Optional[str] = None,
    stop_after: Optional[str] = None,
    keep_intermediates: bool = False,
    data_root: Optional[Path] = None,
):
    """Execute the full pipeline (or a slice of it).

    Parameters
    ----------
    spec : Path
        GroupSet JSON file.
    mmcif_store : Path
        Directory containing PDB mmCIF files.
    workdir : Path
        Working directory where ``data/`` is written.
    start_from : str, optional
        Resume from this step label (inclusive).
    stop_after : str, optional
        Stop after this step label (inclusive).
    keep_intermediates : bool
        If *False* (default), intermediate artefacts (``asms-raw``,
        ``asms-bio``, ``asms-subset``, ``asms-metal``, ``combinations``,
        ``combinations-filtered``, ``seqcluster_work``) are written to
        a temporary directory and deleted after the run.  Set to *True*
        to keep them under ``data/``.
    """
    labels = [s.label for s in STEPS]

    start_idx = 0
    if start_from:
        if start_from not in labels:
            raise click.UsageError(
                f"Unknown step '{start_from}'. Choose from: {labels}"
            )
        start_idx = labels.index(start_from)

    stop_idx = len(STEPS) - 1
    if stop_after:
        if stop_after not in labels:
            raise click.UsageError(
                f"Unknown step '{stop_after}'. Choose from: {labels}"
            )
        stop_idx = labels.index(stop_after)

    selected = STEPS[start_idx : stop_idx + 1]

    # Resolve PRODIGY_CMD once if the prodigy step is in scope
    if any(s.label == "run_prodigy" for s in selected):
        prodigy = _resolve_prodigy()
        if prodigy:
            os.environ["PRODIGY_CMD"] = prodigy
            click.echo(f"Using PRODIGY_CMD: {prodigy}")
        else:
            click.echo(
                "Warning: prodigy_cryst not found. "
                "Install the prodigy-cryst conda env or put prodigy_cryst on PATH.\n"
                "  See: install.sh",
                err=True,
            )

    # Switch to workdir so relative data/ paths resolve correctly
    orig_cwd = Path.cwd()
    os.chdir(workdir)

    data_dir = data_root if data_root is not None else Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Intermediate artefacts location
    tmp_ctx = None
    if keep_intermediates:
        tmp_dir = data_dir
    else:
        tmp_ctx = tempfile.TemporaryDirectory(
            prefix="promise_intermediate_", dir=workdir
        )
        tmp_dir = Path(tmp_ctx.name)

    click.echo("")
    click.echo("=" * 48)
    click.echo("promise-bench curation pipeline")
    click.echo(f"  spec:               {spec}")
    click.echo(f"  mmcif-store:        {mmcif_store}")
    click.echo(f"  workdir:            {workdir}")
    click.echo(f"  logs:               {workdir / log_dir}")
    click.echo(f"  steps:              {start_idx + 1}..{stop_idx + 1} / {len(STEPS)}")
    click.echo(f"  keep-intermediates: {keep_intermediates}")
    if not keep_intermediates:
        click.echo(f"  tmpdir:             {tmp_dir}")
    click.echo(f"  final output:       data/{FINAL_OUTPUT_DIR}")
    click.echo("=" * 48)

    try:
        for i, step in enumerate(selected, 1):
            step_num = labels.index(step.label) + 1
            log_path = log_dir / f"{step_num:02d}_{step.label}.log"
            click.echo(f"\n[{step_num}/{len(STEPS)}] {step.label} ", nl=False)

            t0 = time.time()
            cmd = _import_click_cmd(step)

            args: list[str] = []
            if data_root is not None:
                args.extend(["-C", str(data_dir)])
            if step.args_fn is not None:
                args.extend(step.args_fn(spec, mmcif_store, workdir, data_dir, tmp_dir))

            # Redirect stdout/stderr to log file
            with open(log_path, "w") as log_f:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = log_f
                sys.stderr = log_f
                try:
                    cmd.main(list(args), standalone_mode=False)
                except Exception:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                    elapsed = time.time() - t0
                    click.echo(f"FAILED ({elapsed:.1f}s)")
                    click.echo(f"  see {log_path}", err=True)
                    raise
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr

            elapsed = time.time() - t0
            click.echo(f"OK ({elapsed:.1f}s) -> {log_path}")
    finally:
        # Clean up intermediates
        if tmp_ctx is not None:
            total_size = (
                sum(f.stat().st_size for f in tmp_dir.rglob("*") if f.is_file())
                if tmp_dir.exists()
                else 0
            )
            click.echo(
                f"\nCleaning up intermediates ({total_size / (1024**2):.1f} MB) ..."
            )
            tmp_ctx.cleanup()
            click.echo("  done.")

    os.chdir(orig_cwd)

    click.echo("")
    click.echo("=" * 48)
    click.echo("Curation pipeline complete!")
    click.echo(f"  Final output: {workdir / data_dir / FINAL_OUTPUT_DIR}")
    click.echo("=" * 48)

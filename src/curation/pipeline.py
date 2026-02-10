"""Curation pipeline definition.

Each step is a pair (label, callable) where the callable accepts
``(spec, mmcif_store, workdir)`` and returns the Click args list to be
invoked on that step's ``main`` / ``cli`` click command.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import click

# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------


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
        self.args_fn = args_fn  # (spec, mmcif_store, workdir) -> list[str]


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


STEPS: list[Step] = [
    # --- Phase 1: extract pairs ---
    Step(
        "create_msa",
        "curation.create_msa",
        args_fn=lambda s, m, w: ["--mmcif-dir", str(m), str(s)],
    ),
    Step(
        "pairwise_tm",
        "curation.pairwise_tm_multiprocessing",
    ),
    Step(
        "cluster_by_tmscore",
        "curation.cluster_by_tmscore",
    ),
    # --- Phase 2: curation ---
    Step(
        "prepare_inputs",
        "curation.prepare_inputs_gemmi",
        args_fn=lambda s, m, w: ["--mmcif-dir", str(m)],
    ),
    Step(
        "run_prodigy",
        "curation.run_prodigy",
    ),
    Step(
        "filter_xtal",
        "curation.filter_xtal",
    ),
    Step(
        "subsets",
        "curation.subsets",
    ),
    Step(
        "process_metal",
        "curation.process_metal",
    ),
    Step(
        "curate_sets",
        "curation.curate_sets",
        entry="cli",
    ),
    Step(
        "select_representative",
        "curation.select_representative",
    ),
    Step(
        "filter_seq_clusters",
        "curation.filter_seq_clusters",
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
    total = len(selected)

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

    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    click.echo("")
    click.echo("=" * 48)
    click.echo("promise-bench curation pipeline")
    click.echo(f"  spec:        {spec}")
    click.echo(f"  mmcif-store: {mmcif_store}")
    click.echo(f"  workdir:     {workdir}")
    click.echo(f"  logs:        {workdir / log_dir}")
    click.echo(f"  steps:       {start_idx + 1}..{stop_idx + 1} / {len(STEPS)}")
    click.echo("=" * 48)

    for i, step in enumerate(selected, 1):
        step_num = labels.index(step.label) + 1
        log_path = log_dir / f"{step_num:02d}_{step.label}.log"
        click.echo(f"\n[{step_num}/{len(STEPS)}] {step.label} ", nl=False)

        t0 = time.time()
        cmd = _import_click_cmd(step)

        args: list[str] = []
        if step.args_fn is not None:
            args = step.args_fn(spec, mmcif_store, workdir)

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

    os.chdir(orig_cwd)

    click.echo("")
    click.echo("=" * 48)
    click.echo("Curation pipeline complete!")
    click.echo("=" * 48)

"""CLI entry point for promise-bench data curation.

Install the package (``pip install -e .``) then run::

    promise_data run --spec spec.json --mmcif-store /path/to/mmcif_files

Or invoke via ``python -m curation``::

    python -m curation run --spec spec.json --mmcif-store /path/to/mmcif_files
"""

from __future__ import annotations

from pathlib import Path

import click

from .pipeline import STEPS, run_pipeline

STEP_LABELS = [s.label for s in STEPS]


@click.group()
@click.version_option(package_name="promise-data")
def promise_data():
    """promise-bench data curation toolkit."""


@promise_data.command()
@click.option(
    "--spec",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="GroupSet JSON spec file.",
)
@click.option(
    "--mmcif-store",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing PDB mmCIF files.",
)
@click.option(
    "--workdir",
    "-C",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Working directory (data/ is written here).",
)
@click.option(
    "--start-from",
    type=click.Choice(STEP_LABELS, case_sensitive=False),
    default=None,
    help="Resume from this step (inclusive).",
)
@click.option(
    "--stop-after",
    type=click.Choice(STEP_LABELS, case_sensitive=False),
    default=None,
    help="Stop after this step (inclusive).",
)
def run(
    spec: Path,
    mmcif_store: Path,
    workdir: Path,
    start_from: str | None,
    stop_after: str | None,
):
    """Run the full curation pipeline (or a slice of it)."""
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    run_pipeline(
        spec.resolve(),
        mmcif_store.resolve(),
        workdir,
        start_from=start_from,
        stop_after=stop_after,
    )


@promise_data.command("steps")
def list_steps():
    """List all pipeline steps."""
    for i, step in enumerate(STEPS, 1):
        click.echo(f"  {i:2d}. {step.label:<28s}  ({step.module_path})")

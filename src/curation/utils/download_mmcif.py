#!/usr/bin/env python3
"""Download PDB mmCIF files from RCSB rsync server.

Downloads mmCIF files and extracts them to a flat directory structure.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from subprocess import PIPE, Popen
from time import sleep
from typing import Optional

import click
from tqdm.auto import tqdm


def download_mmcif_batch(
    data_dir: Path,
    two_char_code: str,
    retries: int = 5,
    keep_compressed: bool = False,
) -> int:
    """Download mmCIF files for a specific two-character code batch.

    Parameters
    ----------
    data_dir : Path
        Target directory where .cif files will be stored (flat structure)
    two_char_code : str
        Two-character code (e.g., '1a', '2b') representing the batch
    retries : int
        Number of retry attempts on failure
    keep_compressed : bool
        Keep .gz files after extraction (default: False)

    Returns
    -------
    int
        Number of files successfully downloaded and extracted
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Temporary directory for compressed files
    tmp_dir = data_dir / ".tmp" / two_char_code
    tmp_dir.mkdir(parents=True, exist_ok=True)

    SERVER = "rsync.rcsb.org::ftp_data"

    # Download compressed mmCIF files
    command = [
        "rsync",
        "-rlpt",
        "-v",
        "-z",
        "--delete",
        f"{SERVER}/structures/divided/mmCIF/{two_char_code}/",
        str(tmp_dir) + "/",
    ]

    try:
        proc = Popen(command, stderr=PIPE, stdout=PIPE)
        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            click.echo(f"[ERROR] rsync failed for {two_char_code}", err=True)
            for ln in stderr.decode().splitlines():
                click.echo(f"  {ln.strip()}", err=True)

            if retries > 0:
                retries -= 1
                sleep_time = 2 ** (6 - retries)
                click.echo(
                    f"[RETRY] {retries} attempts remaining, sleeping {sleep_time}s"
                )
                sleep(sleep_time)
                return download_mmcif_batch(
                    data_dir, two_char_code, retries, keep_compressed
                )
            return 0
    except Exception as e:
        click.echo(f"[ERROR] Exception during rsync: {e}", err=True)
        return 0

    # Extract .cif.gz files to flat directory
    gz_files = list(tmp_dir.glob("*.cif.gz"))
    extracted = 0

    for gz_path in gz_files:
        pdb_id = gz_path.stem.replace(".cif", "")  # Remove .cif from .cif.gz
        out_path = data_dir / f"{pdb_id}.cif"

        try:
            with gzip.open(gz_path, "rb") as f_in:
                with open(out_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            extracted += 1
        except Exception as e:
            click.echo(f"[WARN] Failed to extract {gz_path.name}: {e}", err=True)

        # Remove compressed file if not keeping
        if not keep_compressed:
            gz_path.unlink(missing_ok=True)

    # Clean up temp directory
    if not keep_compressed:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return extracted


def get_two_char_codes_from_pdbs(pdb_ids: list[str]) -> list[str]:
    """Extract unique two-character codes from PDB IDs.

    Parameters
    ----------
    pdb_ids : list[str]
        List of PDB IDs (e.g., ['1abc', '2def'])

    Returns
    -------
    list[str]
        Sorted unique two-character codes (e.g., ['ab', 'de'])
    """
    codes = set()
    for pdb_id in pdb_ids:
        if len(pdb_id) >= 4:
            codes.add(pdb_id[1:3].lower())
    return sorted(codes)


def get_all_two_char_codes() -> list[str]:
    """Get all available two-character codes from RCSB rsync server.

    Returns
    -------
    list[str]
        List of all two-character codes available
    """
    from subprocess import check_output

    cmd = [
        "rsync",
        "--list-only",
        "rsync.rcsb.org::ftp_data/structures/divided/mmCIF/",
    ]

    try:
        output = check_output(cmd).decode("utf-8").split("\n")
        codes = [
            line.split()[-1]
            for line in output
            if line.startswith("d") and not line.endswith(".")
        ]
        return sorted(codes)
    except Exception as e:
        click.echo(f"[ERROR] Failed to list two-char codes: {e}", err=True)
        return []


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory for mmCIF files (flat structure: pdbid.cif)",
)
@click.option(
    "--pdb-list",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Text file with PDB IDs (one per line). If not provided, downloads all.",
)
@click.option(
    "--two-char-codes",
    type=str,
    default=None,
    help="Comma-separated two-char codes (e.g., '1a,2b,3c'). Overrides --pdb-list.",
)
@click.option(
    "--keep-compressed",
    is_flag=True,
    help="Keep .cif.gz files in .tmp/ directory after extraction.",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    help="Number of parallel downloads (sequential download per batch).",
)
def main(
    data_dir: Path,
    pdb_list: Optional[Path],
    two_char_codes: Optional[str],
    keep_compressed: bool,
    workers: int,
):
    """Download PDB mmCIF files from RCSB rsync server.

    Examples:

        # Download all mmCIF files
        python download_mmcif.py --data-dir /path/to/mmcif_files

        # Download specific PDB IDs from file
        python download_mmcif.py --data-dir ./mmcif --pdb-list pdb_ids.txt

        # Download specific two-char code batches
        python download_mmcif.py --data-dir ./mmcif --two-char-codes 1a,2b,3c
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Determine which batches to download
    if two_char_codes:
        codes = [c.strip() for c in two_char_codes.split(",")]
        click.echo(f"Downloading {len(codes)} two-char code batches: {codes}")
    elif pdb_list:
        pdb_ids = pdb_list.read_text().strip().split("\n")
        pdb_ids = [p.strip().lower() for p in pdb_ids if p.strip()]
        codes = get_two_char_codes_from_pdbs(pdb_ids)
        click.echo(f"Extracted {len(codes)} two-char codes from {len(pdb_ids)} PDB IDs")
    else:
        click.echo(
            "No --pdb-list or --two-char-codes specified, downloading ALL mmCIF files"
        )
        codes = get_all_two_char_codes()
        click.echo(f"Found {len(codes)} two-char code batches on RCSB server")

        if not click.confirm(
            "This will download the entire PDB (~hundreds of GB). Continue?"
        ):
            return

    # Download batches
    total_extracted = 0

    for code in tqdm(codes, desc="Batches", unit="batch"):
        extracted = download_mmcif_batch(
            data_dir=data_dir,
            two_char_code=code,
            keep_compressed=keep_compressed,
        )
        total_extracted += extracted

        if extracted > 0:
            tqdm.write(f"  [{code}] Extracted {extracted} files")

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Total files extracted: {total_extracted}")
    click.echo(f"Output directory: {data_dir}")
    click.echo(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""Central pipeline path configuration.

Loads ``config/pipeline.yaml`` once and exposes helpers for resolving
directory and file paths relative to the data root (``data/``).

Typical usage in pipeline modules
---------------------------------
::

    from curation.utils._config import pipeline_cfg as C

    @click.option("--seqs", ..., default=C.dir("seqs"))      # Path("data/seqs")
    @click.option("--out", ..., default=C.file("pair_calls")) # Path("data/pair-calls.csv")
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Locate the YAML file
# ---------------------------------------------------------------------------

# config/ lives at the repo root, next to src/
# _config.py is at src/curation/utils/_config.py -> 3 parents up to repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO_ROOT / "config" / "pipeline.yaml"


@functools.lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    """Read and cache the pipeline YAML."""
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)["pipeline"]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

_DATA_PREFIX = "data"


def _data_root() -> str:
    """Return the configured data root (default ``'data'``)."""
    return _load().get("data_root", _DATA_PREFIX)


class PipelineConfig:
    """Thin accessor around the YAML config dict."""

    @staticmethod
    def data_root() -> str:
        """Configured data root directory name (e.g. ``'data'`` or ``'data-95'``)."""
        return _data_root()

    # -- raw access --

    @staticmethod
    def raw() -> dict[str, Any]:
        """Return the full ``pipeline:`` mapping."""
        return _load()

    # -- directory helpers --

    @staticmethod
    def dirname(key: str) -> str:
        """Bare directory name, e.g. ``"asms-raw"``."""
        return _load()["dirs"][key]

    @staticmethod
    def dir(key: str) -> Path:
        """``Path("<data_root>/<dirname>")``, suitable as a Click default."""
        return Path(_data_root()) / _load()["dirs"][key]

    # -- file helpers --

    @staticmethod
    def filename(key: str) -> str:
        """Bare file name, e.g. ``"pair-calls.csv"``."""
        return _load()["files"][key]

    @staticmethod
    def file(key: str) -> Path | None:
        """``Path("<data_root>/<filename>")`` or *None* if the value is null.

        If the configured value is an absolute path, it is returned as-is.
        """
        val = _load()["files"][key]
        if val is None:
            return None
        p = Path(val)
        if p.is_absolute():
            return p
        return Path(_data_root()) / val

    # -- special --

    @staticmethod
    def final_output() -> str:
        """The final output directory name (bare)."""
        return _load()["final_output"]

    @staticmethod
    def final_output_dir() -> Path:
        """``Path("<data_root>/<final_output>")``."""
        return Path(_data_root()) / _load()["final_output"]

    @staticmethod
    def intermediate_keys() -> list[str]:
        """Config keys of directories that are intermediate artefacts."""
        return _load()["intermediate"]

    @staticmethod
    def intermediate_dirs() -> set[str]:
        """Set of bare directory names that are intermediate artefacts."""
        cfg = _load()
        return {cfg["dirs"][k] for k in cfg["intermediate"]}


# Singleton instance — import as ``from curation.utils._config import pipeline_cfg as C``
pipeline_cfg = PipelineConfig()

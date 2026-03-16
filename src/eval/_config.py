"""Central evaluation path configuration.

Loads ``config/eval.yaml`` once and exposes helpers for resolving
directory and file paths relative to the eval data root (``data_eval/``).

Usage
-----
::

    from eval._config import eval_cfg as E

    E.dir("training_bias")  # -> Path("data_eval/train/training_bias")
    E.file("valid_pairs")   # -> Path("data_eval/valid_pairs.json")
    E.external("meta_data_dir")  # -> Path or None
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Locate the YAML file
# ---------------------------------------------------------------------------

# _config.py is at src/eval/_config.py -> 2 parents up to repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config" / "eval.yaml"

_DATA_PREFIX = "data_eval"


@functools.lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    """Read and cache the eval YAML."""
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)["eval"]


def _data_root() -> str:
    """Return the configured data root (default ``'data_eval'``)."""
    return _load().get("data_root", _DATA_PREFIX)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


class EvalConfig:
    """Thin accessor around the YAML config dict."""

    @staticmethod
    def data_root() -> str:
        """Configured data root directory name."""
        return _data_root()

    @staticmethod
    def raw() -> dict[str, Any]:
        """Return the full ``eval:`` mapping."""
        return _load()

    # -- directory helpers --

    @staticmethod
    def dirname(key: str) -> str:
        """Bare directory path segment, e.g. ``"train/training_bias"``."""
        return _load()["dirs"][key]

    @staticmethod
    def dir(key: str) -> Path:
        """``Path("<data_root>/<dirname>")``, suitable as a default."""
        return Path(_data_root()) / _load()["dirs"][key]

    # -- file helpers --

    @staticmethod
    def filename(key: str) -> str:
        """Bare file path segment, e.g. ``"valid_pairs.json"``."""
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

    # -- external (machine-specific) paths --

    @staticmethod
    def external(key: str) -> Path | None:
        """Return an external path or *None* if unset/null."""
        val = _load().get("external", {}).get(key)
        if val is None:
            return None
        return Path(val)


# Singleton instance
eval_cfg = EvalConfig()

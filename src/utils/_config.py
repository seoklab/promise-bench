"""Central path configuration for both curation pipeline and evaluation.

Loads ``config/config.yaml`` once and exposes section-specific accessors
for resolving directory and file paths relative to each section's data root.

Usage
-----
::

    from utils._config import pipeline_cfg as C
    from utils._config import eval_cfg as E

    C.dir("seqs")            # -> Path("data/seqs")
    C.file("filtered_pairs") # -> Path("data/filtered-pairs.csv")
    E.dir("training_bias")   # -> Path("data_eval/train/training_bias")
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

# config/ lives at the repo root, next to src/
# _config.py is at src/utils/_config.py -> 2 parents up to repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"


@functools.lru_cache(maxsize=1)
def _load_all() -> dict[str, Any]:
    """Read and cache the full config YAML."""
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


# =====================================================================
# Generic section accessor
# =====================================================================


class SectionConfig:
    """Base accessor for a single top-level YAML section."""

    def __init__(self, section: str, default_data_root: str) -> None:
        self._section = section
        self._default_data_root = default_data_root

    def _cfg(self) -> dict[str, Any]:
        return _load_all()[self._section]

    # -- raw access --

    def raw(self) -> dict[str, Any]:
        """Return the full section mapping."""
        return self._cfg()

    # -- data root --

    def data_root(self) -> str:
        """Configured data root directory name."""
        return self._cfg().get("data_root", self._default_data_root)

    # -- directory helpers --

    def dirname(self, key: str) -> str:
        """Bare directory path segment, e.g. ``"asms-raw"``."""
        return self._cfg()["dirs"][key]

    def dir(self, key: str) -> Path:
        """``Path("<data_root>/<dirname>")``, suitable as a Click default."""
        return Path(self.data_root()) / self._cfg()["dirs"][key]

    # -- file helpers --

    def filename(self, key: str) -> str:
        """Bare file path segment, e.g. ``"pair-calls.csv"``."""
        return self._cfg()["files"][key]

    def file(self, key: str) -> Path | None:
        """``Path("<data_root>/<filename>")`` or *None* if the value is null.

        If the configured value is an absolute path, it is returned as-is.
        """
        val = self._cfg()["files"][key]
        if val is None:
            return None
        p = Path(val)
        if p.is_absolute():
            return p
        return Path(self.data_root()) / val

    # -- external (machine-specific) paths --

    def external(self, key: str) -> Path | None:
        """Return an external path or *None* if unset/null."""
        val = self._cfg().get("external", {}).get(key)
        if val is None:
            return None
        return Path(val)


# =====================================================================
# Pipeline-specific extensions
# =====================================================================


class PipelineConfig(SectionConfig):
    """Pipeline accessor with extra helpers for final output / intermediates."""

    def __init__(self) -> None:
        super().__init__("pipeline", "data")

    def final_output(self) -> str:
        """The final output directory name (bare)."""
        return self._cfg()["final_output"]

    def final_output_dir(self) -> Path:
        """``Path("<data_root>/<final_output>")``."""
        return Path(self.data_root()) / self._cfg()["final_output"]

    def intermediate_keys(self) -> list[str]:
        """Config keys of directories that are intermediate artefacts."""
        return self._cfg()["intermediate"]

    def intermediate_dirs(self) -> set[str]:
        """Set of bare directory names that are intermediate artefacts."""
        cfg = self._cfg()
        return {cfg["dirs"][k] for k in cfg["intermediate"]}


# =====================================================================
# Eval accessor
# =====================================================================


class EvalConfig(SectionConfig):
    """Eval accessor — no extra methods beyond the base."""

    def __init__(self) -> None:
        super().__init__("eval", "data_eval")


# =====================================================================
# Singleton instances
# =====================================================================

# Curation pipeline: ``from utils._config import pipeline_cfg as C``
pipeline_cfg = PipelineConfig()

# Evaluation: ``from utils._config import eval_cfg as E``
eval_cfg = EvalConfig()

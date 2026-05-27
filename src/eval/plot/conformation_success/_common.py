"""Shared constants for conformation-success analysis.

Kept in sync with the legacy step8 series in ``dynamic_set/final/`` so that
figures and numerical thresholds remain comparable.

Naming conventions
------------------
``promise-bench`` align rows use the dashed model labels (``boltz-1``,
``boltz-2``, ``chai-1``); the rest of this package uses the dash-less
``METHODS`` keys (``boltz1``, ``boltz2``, ``chai``). ``normalize_method``
translates between the two.
"""

from __future__ import annotations

from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Numeric thresholds
# ---------------------------------------------------------------------------

# A sample "covers" a conformation iff TM(conf) >= TM_THRESHOLD AND
# TM(conf) is strictly greater than the TM of every other conformation
# for the same sample.
TM_THRESHOLD: float = 0.8


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

METHODS: Tuple[str, ...] = ("af3", "boltz1", "boltz2", "chai", "bioemu")

# Methods with per-sample confidence scores (Bioemu has none in this pipeline).
METHODS_WITH_CONFIDENCE: Tuple[str, ...] = ("af3", "boltz1", "boltz2", "chai")

METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "af3": "AF3",
    "boltz1": "Boltz-1",
    "boltz2": "Boltz-2",
    "chai": "Chai-1",
    "bioemu": "Bioemu",
}

# align_part rows use dashed labels; translate to METHODS keys.
_DASHED_TO_DASHLESS: Dict[str, str] = {
    "boltz-1": "boltz1",
    "boltz-2": "boltz2",
    "chai-1": "chai",
}


def normalize_method(method: str) -> str:
    """Map dashed align-row labels to the dash-less METHODS keys.

    ``"boltz-1"`` -> ``"boltz1"``; pass-through for ``"af3"`` / ``"bioemu"``
    / already-normalised labels.
    """
    if method is None:
        return method  # type: ignore[return-value]
    return _DASHED_TO_DASHLESS.get(method, method)


# ---------------------------------------------------------------------------
# Set types
# ---------------------------------------------------------------------------

# Canonical set names used internally. promise-bench align rows already use
# ``intrinsic`` (not the legacy ``apo-monomers``).
SET_TYPES: Tuple[str, ...] = ("intrinsic", "ligand-induced", "protein-induced")

# Aliases accepted from legacy inputs.
_SET_ALIASES: Dict[str, str] = {
    "apo-monomers": "intrinsic",
    "apo": "intrinsic",
}


def normalize_set_type(set_type: str) -> str:
    """Map legacy ``apo-monomers`` to the canonical ``intrinsic``."""
    if set_type is None:
        return set_type  # type: ignore[return-value]
    return _SET_ALIASES.get(set_type, set_type)


# ---------------------------------------------------------------------------
# Colors (used by figures.py / figures_topk.py later)
# ---------------------------------------------------------------------------

SET_TYPE_COLORS: Dict[str, str] = {
    "intrinsic": "#8D9FCF",
    "ligand-induced": "#EEBEC3",
    "protein-induced": "#8CC6C0",
}

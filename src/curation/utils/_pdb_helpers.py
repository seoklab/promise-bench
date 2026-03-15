"""General PDB / mmCIF chain-ID helpers.

Small, pure-Python utilities that do not depend on gemmi or heavy
third-party libraries.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# label_base — strip trailing digits from clone IDs (e.g. "A2" → "A")
# ---------------------------------------------------------------------------

_LABEL_BASE_RX = re.compile(r"\d+$")


def label_base(label_or_clone: str) -> str:
    """Return the label *base* by removing a trailing numeric clone suffix."""
    return _LABEL_BASE_RX.sub("", str(label_or_clone))


# ---------------------------------------------------------------------------
# parse_member_cell — split "4hhb_A" / "4hhb:A" / "4hhbA" into (pdb, chain)
# ---------------------------------------------------------------------------


def parse_member_cell(cell: str) -> tuple[str, str]:
    """Parse a cluster member string into ``(pdb_code, auth_asym_id)``.

    Accepted formats:
    - ``4hhb_A``  (underscore separated)
    - ``4hhb:A``  (colon separated)
    - ``4hhbA``   (concatenated, 4-char PDB code + chain)
    """
    s = cell.strip()
    if "_" in s:
        pdb, ch = s.split("_", 1)
    elif ":" in s:
        pdb, ch = s.split(":", 1)
    else:
        if len(s) >= 5:
            pdb, ch = s[:4], s[4:]
        else:
            raise ValueError(f"Bad member format: {s}")
    return pdb.lower(), ch


# ---------------------------------------------------------------------------
# check_auth_base — e.g. auth="A", auth_chain="A" or "A2" → True
# ---------------------------------------------------------------------------


def check_auth_base(auth: str, auth_chain: str) -> bool:
    """Check whether *auth_chain* starts with *auth* (+ optional numeric suffix)."""
    if not auth_chain.startswith(auth):
        return False
    suffix = auth_chain[len(auth):]
    if suffix == "":
        return True
    if re.fullmatch(r"[1-9][0-9]?", suffix):
        return True
    return False

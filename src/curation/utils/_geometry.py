"""Geometry helpers for coordinate extraction and proximity queries.

Shared across pipeline scripts (prepare_inputs_gemmi, filter_xtal, …).
"""

from __future__ import annotations

from typing import Dict, List, Set

import gemmi
import numpy as np

from ..utils.constants import _AA3, _NT, LIGAND_EXCLUDE


# ---------------------------------------------------------------------------
# Coordinate extraction
# ---------------------------------------------------------------------------


def coords_and_ligs_from_model(
    model: gemmi.Model,
    include_h: bool = False,
) -> tuple[Dict[str, np.ndarray], List[dict]]:
    """Extract per-chain polymer coordinates and ligand instances.

    Returns
    -------
    chain_coords : dict[str, np.ndarray]
        Mapping *auth_asym_id* → (N, 3) heavy-atom coordinate array.
    lig_instances : list[dict]
        Each dict has keys ``comp_id``, ``auth_asym``, ``coords``.
    """
    chain_coords: Dict[str, np.ndarray] = {}
    lig_instances: List[dict] = []

    def _is_polymer(res: gemmi.Residue) -> bool:
        et = getattr(res, "entity_type", None)
        hf = getattr(res, "het_flag", "\0")
        comp_id = str(getattr(res, "name", "")).upper()
        if et == gemmi.EntityType.Polymer:
            return True
        if isinstance(hf, str) and hf in ("\0", " "):
            return comp_id in _AA3 or comp_id in _NT
        return False

    for sc in model:
        chain_name = sc.name
        xyz_poly: list[list[float]] = []
        for res in sc:
            comp_id = str(res.name).upper()
            if _is_polymer(res):
                for at in res:
                    if (not include_h) and at.element.is_hydrogen:
                        continue
                    xyz_poly.append([at.pos.x, at.pos.y, at.pos.z])
            else:
                lig_xyz: list[list[float]] = []
                for at in res:
                    if (not include_h) and at.element.is_hydrogen:
                        continue
                    lig_xyz.append([at.pos.x, at.pos.y, at.pos.z])
                if lig_xyz:
                    lig_instances.append(
                        {
                            "comp_id": comp_id,
                            "auth_asym": chain_name,
                            "auth_seq_id": ".",
                            "ins_code": ".",
                            "coords": np.asarray(lig_xyz, dtype=float),
                        }
                    )
        if xyz_poly:
            chain_coords[chain_name] = np.asarray(xyz_poly, dtype=float)

    return chain_coords, lig_instances


# ---------------------------------------------------------------------------
# Proximity helpers
# ---------------------------------------------------------------------------


def _near_block(
    P: np.ndarray,
    Q: np.ndarray,
    c2: float,
    blk: int = 4096,
) -> bool:
    """True if any atom in *P* is within ``sqrt(c2)`` of any atom in *Q*."""
    for s in range(0, P.shape[0], blk):
        PP = P[s : s + blk]
        d2 = np.sum((PP[:, None, :] - Q[None, :, :]) ** 2, axis=2)
        if np.any(d2 <= c2):
            return True
    return False


def get_contact_chains(
    coi_id: str,
    clone_coords: Dict[str, np.ndarray],
    cutoff: float,
) -> List[str]:
    """Return chain IDs within *cutoff* Å of chain *coi_id*."""
    coi_xyz = clone_coords.get(coi_id)
    if coi_xyz is None or coi_xyz.size == 0:
        return []
    c2 = cutoff * cutoff
    return [
        cid
        for cid, coords in clone_coords.items()
        if cid != coi_id
        and coords is not None
        and coords.size > 0
        and _near_block(coi_xyz, coords, c2)
    ]


def get_contact_ligands(
    a_xyz: np.ndarray,
    lig_instances: List[dict],
    cutoff: float,
) -> List[str]:
    """Return comp_ids of ligands within *cutoff* Å of *a_xyz*."""
    if a_xyz.size == 0 or not lig_instances:
        return []
    c2 = cutoff * cutoff
    hits: List[str] = []
    for inst in lig_instances:
        cid = str(inst["comp_id"]).upper()
        if cid in LIGAND_EXCLUDE:
            continue
        L = inst["coords"]
        if _near_block(a_xyz, L, c2):
            hits.append(cid)
    return hits


def ligand_mediators_for_pair(
    a_xyz: np.ndarray,
    b_xyz: np.ndarray,
    lig_instances: List[dict],
    cutoff: float,
) -> Set[str]:
    """Return comp_ids of ligands that bridge chains *a* and *b*."""
    mediators: Set[str] = set()
    if a_xyz.size == 0 or b_xyz.size == 0 or not lig_instances:
        return mediators
    c2 = cutoff * cutoff
    for inst in lig_instances:
        cid = str(inst["comp_id"]).upper()
        if cid in LIGAND_EXCLUDE:
            continue
        L = inst["coords"]
        if _near_block(L, a_xyz, c2) and _near_block(L, b_xyz, c2):
            mediators.add(cid)
    return mediators

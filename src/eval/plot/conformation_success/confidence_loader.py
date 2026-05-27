"""Per-prediction confidence / ranking score loader.

Ports ``get_confidence_score`` from ``dynamic_set/final/step8a-1_compute_pair_results.py``
with one promise-bench-specific extension: Boltz-2 confidences live under the
``_with_msa`` variant of the CIF stem.

Mapping
-------
- ``af3``    : ``<parent.parent>/ranking_scores.csv``, row matched by
              ``(seed, sample)`` parsed from ``parent.name``
              (``seed-<S>_sample-<I>``); returns ``ranking_score``.
- ``boltz1`` : ``<parent>/confidence_<stem>.json`` -> ``confidence_score``.
              (Legacy data path swap ``/galaxy4/`` -> ``/galaxy3/`` kept for
              compatibility with older runs.)
- ``boltz2`` : ``<parent>/confidence_<entity>_with_msa_model_<N>.json``
              (promise-bench naming) with fallback to plain
              ``confidence_<stem>.json`` -> ``confidence_score``.
- ``chai``   : ``<parent>/scores.model_idx_<N>.npz`` -> ``aggregate_score``.
- ``bioemu`` : no confidence available; returns ``None``.

CLI (single-method spot check):

    PYTHONPATH=src uv run --project . \\
      python -m eval.plot.conformation_success.confidence_loader \\
        --cluster 4KBF_1 --method boltz2 --max 5
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ._common import METHODS_WITH_CONFIDENCE, normalize_method


def _read_json_key(path: Path, key: str) -> Optional[float]:
    try:
        with open(path) as f:
            d = json.load(f)
        v = d.get(key)
        return float(v) if v is not None else None
    except Exception:
        return None


def _read_npz_key(path: Path, key: str) -> Optional[float]:
    try:
        a = np.load(path)
        v = a[key]
        return float(np.asarray(v).flat[0])
    except Exception:
        return None


# ranking_scores.csv is shared across all (seed, sample) entries from the
# same AF3 prediction directory; cache the parsed contents instead of
# rescanning the file for every sample.
@functools.lru_cache(maxsize=4096)
def _load_ranking_csv(csv_path_str: str) -> Optional[Dict[Tuple[int, int], float]]:
    """Return ``{(seed, sample): ranking_score}`` from ``ranking_scores.csv``.

    Returns ``None`` on missing file / parse failure; an empty dict is a
    valid file with no rows. Cached by absolute path string.
    """
    p = Path(csv_path_str)
    if not p.exists():
        return None
    try:
        out: Dict[Tuple[int, int], float] = {}
        with open(p) as f:
            for row in csv.DictReader(f):
                out[(int(row["seed"]), int(row["sample"]))] = float(row["ranking_score"])
        return out
    except Exception:
        return None


def _read_ranking_csv(csv_path: Path, seed: int, sample: int) -> Optional[float]:
    table = _load_ranking_csv(str(csv_path))
    if table is None:
        return None
    return table.get((seed, sample))


def clear_caches() -> None:
    """Drop cached ranking_scores.csv tables. Call between unrelated runs."""
    _load_ranking_csv.cache_clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_confidence_score(method: str, mobile_cif: Optional[str]) -> Optional[float]:
    """Return the confidence/ranking score for one prediction sample.

    ``method`` should already be dash-less (``"boltz1"``, ``"boltz2"``,
    ``"chai"`` etc.) or any value accepted by ``normalize_method``.
    Returns ``None`` if the score cannot be resolved (bioemu, missing
    files, parse errors).
    """
    if mobile_cif is None:
        return None
    m = normalize_method(method)
    if m not in METHODS_WITH_CONFIDENCE:
        return None

    p = Path(mobile_cif)

    if m == "af3":
        sample_dir = p.parent.name  # e.g. "seed-8_sample-0"
        csv_path = p.parent.parent / "ranking_scores.csv"
        mm = re.match(r"seed-(\d+)_sample-(\d+)", sample_dir)
        if not mm:
            return None
        return _read_ranking_csv(csv_path, int(mm.group(1)), int(mm.group(2)))

    if m == "boltz1":
        stem = p.stem  # e.g. 1cke_1_A1_m_model_5
        cand = p.parent / f"confidence_{stem}.json"
        if not cand.exists():
            cand = Path(str(cand).replace("/galaxy4/", "/galaxy3/"))
        return _read_json_key(cand, "confidence_score")

    if m == "boltz2":
        stem = p.stem
        cand = p.parent / f"confidence_{stem}.json"
        if cand.exists():
            return _read_json_key(cand, "confidence_score")
        # promise-bench Boltz-2 dirs use varying confidence prefixes:
        #   confidence_<entity>_with_msa_model_<N>.json
        #   confidence_<entity>_rerun_with_msa_model_<N>.json
        # Match by entity + model index via glob.
        if "_model_" in stem:
            ent, _, midx = stem.rpartition("_model_")
            for hit in p.parent.glob(f"confidence_{ent}_*_model_{midx}.json"):
                return _read_json_key(hit, "confidence_score")
        return None

    if m == "chai":
        name = p.name  # e.g. pred.model_idx_0.cif
        npz_name = name.replace("pred.", "scores.").replace(".cif", ".npz")
        return _read_npz_key(p.parent / npz_name, "aggregate_score")

    return None


# ---------------------------------------------------------------------------
# Annotation helper -- attach confidence to existing samples in a results dict
# ---------------------------------------------------------------------------


def attach_confidence_to_results(results: dict, *, methods_to_skip: tuple = ("bioemu",)) -> dict:
    """Mutate a ``pair_level_results.json``-shaped dict by inserting
    ``confidence_score`` onto every sample entry.

    Operates in-place and also returns ``results`` for convenience.
    """
    # intrinsic
    for cmap in results.get("intrinsic", {}).values():
        for method, res in cmap.items():
            if method in methods_to_skip:
                continue
            for s in res.get("samples", []) or []:
                s["confidence_score"] = get_confidence_score(method, s.get("mobile_cif"))

    # induced
    for st in ("ligand-induced", "protein-induced"):
        for cmap in results.get(st, {}).values():
            for method, pairs in cmap.items():
                if method in methods_to_skip:
                    continue
                for pair in pairs or []:
                    for s in pair.get("apo_samples", []) or []:
                        s["confidence_score"] = get_confidence_score(method, s.get("mobile_cif"))
                    for s in pair.get("holo_samples", []) or []:
                        s["confidence_score"] = get_confidence_score(method, s.get("mobile_cif"))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Probe confidence-score availability for one (cluster, method)."
    )
    parser.add_argument(
        "--align-dir",
        type=str,
        default=None,
        help="Dir of align_part*.json (default: eval.dirs.align_results).",
    )
    parser.add_argument("--cluster", "-c", type=str, required=True)
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        required=True,
        help="dash-less method key (af3 / boltz1 / boltz2 / chai / bioemu).",
    )
    parser.add_argument(
        "--set",
        type=str,
        default=None,
        help="Optional pair_type filter (intrinsic / ligand-induced / protein-induced).",
    )
    parser.add_argument("--max", type=int, default=10, help="Print at most N samples.")
    args = parser.parse_args()

    from utils._config import eval_cfg as E

    from .align_loader import load_align_rows

    align_dir = E.align_results_dir(args.align_dir)
    method = normalize_method(args.method)
    rows = load_align_rows(
        align_dir,
        cluster_filter=[args.cluster],
        method_filter=[method],
        set_filter=[args.set] if args.set else None,
    )
    seen: set = set()
    print(f"cluster={args.cluster}  method={method}  -> {len(rows)} rows")
    n = 0
    for r in rows:
        mc = r.get("mobile_cif")
        if not mc or mc in seen:
            continue
        seen.add(mc)
        score = get_confidence_score(method, mc)
        print(f"  {Path(mc).name:<55}  -> {score}")
        n += 1
        if n >= args.max:
            break


if __name__ == "__main__":
    _cli()

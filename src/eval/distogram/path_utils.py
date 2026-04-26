"""
Shared path helpers for reference distogram / valid-pairs layout.

**Where artefacts live (convention in this repo)**

- ``ref_coords/`` — ``extract_reference_cb --distogram``: per–reference-CIF ``*_cb.json``
  plus ``distogram_ref_cb_map.json`` and optional ``*_with_cb_paths.json`` next to the
  input JSON.
- ``<output-dir>/`` (``collect_distograms --output-dir``) — symlink tree
  ``{boltz1,boltz2,af3,bioemu}/{method_type}/{cluster}/{yaml}/…`` and
  ``comparisons/…``; ``distogram_tasks.json`` defaults here.
- ``<tasks_parent>/ref_distogram/`` — ``calc_reference_distogram_diff``: per-task
  ``reference_distogram_diff.json`` under ``{method_type}/{cluster}/{prediction_yaml}/``.
- Next to each prediction distogram dir — ``calc_distogram_loss`` writes
  ``distogram_loss_real_final.json``.
- ConfBench scores — ``calc_distogram_confbench --output`` or default
  ``<distogram_tasks_parent>/distogram_confbench_scores.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


def flatten_valid_pair_edges(valid_pairs: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Build undirected edge list (both directions) from valid_pairs.json structure:
    { pair_type: { cluster_id: [ [tag_a, tag_b], ... ] } }
    """
    edges: List[Tuple[str, str]] = []
    for _ptype, clusters in valid_pairs.items():
        if not isinstance(clusters, dict):
            continue
        for _cid, pairs_list in clusters.items():
            if not isinstance(pairs_list, list):
                continue
            for pairs in pairs_list:
                if isinstance(pairs, dict) and "valid_pair" in pairs:
                    vp = pairs["valid_pair"]
                    if len(vp) == 2:
                        a, b = str(vp[0]), str(vp[1])
                        edges.append((a, b))
                        edges.append((b, a))
                elif (
                    isinstance(pairs, (list, tuple))
                    and len(pairs) == 2
                    and isinstance(pairs[0], str)
                    and isinstance(pairs[1], str)
                ):
                    a, b = pairs[0], pairs[1]
                    edges.append((a, b))
                    edges.append((b, a))
    return edges


def reference_distogram_diff_path(ref_distogram_dir: Path, task: Dict[str, Any]) -> Path:
    """
    Resolve ``reference_distogram_diff.json`` for a distogram task.

    Matches ``calc_distogram_loss`` layout: under
    ``{ref_distogram_dir}/{method_type}/{cluster_id}/``, pick the newest
    ``*/reference_distogram_diff.json`` if any exist; otherwise fall back to
    ``.../{prediction_yaml_tag}/reference_distogram_diff.json``.
    """
    method_type = task.get("method_type", "") or ""
    cluster_id = task.get("cluster_id", "") or ""
    prediction_yaml_tag = task.get("prediction_yaml_tag", "") or ""

    base = ref_distogram_dir / method_type / cluster_id
    matches = list(base.glob("*/reference_distogram_diff.json"))
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime)
    return base / prediction_yaml_tag / "reference_distogram_diff.json"

#!/usr/bin/env python3
"""
make_pairs.py - Generate seq_cluster_to_answer_map.json and valid_pairs.json

Usage:
    python -m curation.make_pairs
    python -m curation.make_pairs --csv-dir data/dataset --outdir data/dataset --examples-dir examples
"""

import json
import csv
import string
import re
from glob import glob
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Set, Tuple, Optional

import click

from utils._config import pipeline_cfg


# ============================================================================
# Configuration
# ============================================================================
SET_NAMES = ["intrinsic", "protein-induced", "ligand-induced"]

MODEL_PATTERNS = {
    "af3": "{examples_dir}/af3/{set_name}/{cluster}/{tag}/seed_*/sample_*/model.cif",
    "bioemu": "{examples_dir}/bioemu/{set_name}/{cluster}/pdbs/sample_*.pdb",
    "boltz-1": "{examples_dir}/boltz-1/{set_name}/{cluster}/{tag}/seed_*/{tag}_model_*.cif",
    "boltz-2": "{examples_dir}/boltz-2/{set_name}/{cluster}/{tag}/seed_*/{tag}_model_*.cif",
    "chai-1": "{examples_dir}/chai-1/{set_name}/{cluster}/{tag}/seed_*/pred.model_idx_*.cif",
}


# ============================================================================
# Load Valid Centers from clusters.json
# ============================================================================
def load_valid_centers(clusters_json: Path) -> Set[str]:
    if not clusters_json.exists():
        click.echo(f"  [WARN] clusters.json not found: {clusters_json}")
        return set()
    with open(clusters_json) as f:
        clusters = json.load(f)
    return {c["center"] for c in clusters}


# ============================================================================
# Helper Functions
# ============================================================================
def make_entry_id(pdb: str, asm: str, chain: str, conf: str) -> str:
    return f"{pdb.lower()}_{asm}_{chain}_conf_{conf}"


def make_tag(pdb: str, asm: str, chain: str, suffix: str = "m") -> str:
    return f"{pdb.lower()}_{asm}_{chain}_{suffix}"


def get_conf(entry_id: str) -> Optional[int]:
    match = re.search(r'_conf_(\d+)$', entry_id)
    return int(match.group(1)) if match else None


def tag_to_key(tag: str) -> Tuple[str, str, str]:
    base = tag[:-2] if tag.endswith(('_m', '_x')) else tag
    parts = base.split('_')
    return (parts[0].lower(), parts[1], parts[2]) if len(parts) >= 3 else (None, None, None)


# ============================================================================
# Prediction Paths
# ============================================================================
def get_predictions(examples_dir: Path, set_name: str, cluster: str, tag: str) -> Dict[str, dict]:
    if not examples_dir or not examples_dir.exists():
        return {}
    
    predictions = {}
    for model, pattern_template in MODEL_PATTERNS.items():
        pattern = pattern_template.format(
            examples_dir=examples_dir, set_name=set_name, cluster=cluster, tag=tag
        )
        files = glob(pattern)
        if files:
            predictions[model] = {"pattern": pattern, "count": len(files)}
    return predictions


def add_predictions_to_data(data: Dict[str, Dict], examples_dir: Optional[Path]) -> Dict[str, Dict]:
    for set_name, clusters in data.items():
        for cluster_name, info in clusters.items():
            apo_preds = {}
            if info["apo_tags"]:
                apo_preds = get_predictions(examples_dir, set_name, cluster_name, info["apo_tags"][0])
            info["apo_predictions"] = apo_preds
            
            holo_preds = {}
            if info["holo_tags"]:
                holo_preds = get_predictions(examples_dir, set_name, cluster_name, info["holo_tags"][0])
            info["holo_predictions"] = holo_preds
    return data


# ============================================================================
# CSV Parsing (supports both old and new formats)
# ============================================================================
def load_csv(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        click.echo(f"  [WARN] CSV not found: {csv_path}")
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def get_cluster_name(row: dict) -> str:
    """Get cluster name from row (supports both formats)."""
    cluster = row.get('cluster_csv') or row.get('cluster', '')
    # Handle 'AB/8ABP_1' -> '8ABP_1'
    if '/' in cluster:
        return cluster.split('/')[1]
    return cluster


def load_csv_pairs(csv_path: Path) -> Set[Tuple]:
    """Load valid pairs from CSV."""
    pairs = set()
    for row in load_csv(csv_path):
        # Support both formats
        a_pdb = (row.get('a_pdb') or row.get('pdb_a', '')).lower()
        a_asm = row.get('a_assembly_id') or row.get('asm_a', '')
        a_chain = row.get('a_chain') or row.get('chain_a', '')
        
        b_pdb = (row.get('b_pdb') or row.get('pdb_b', '')).lower()
        b_asm = row.get('b_assembly_id') or row.get('asm_b', '')
        b_chain = row.get('b_chain') or row.get('chain_b', '')
        
        a = (a_pdb, a_asm, a_chain)
        b = (b_pdb, b_asm, b_chain)
        pairs.add((a, b))
        pairs.add((b, a))
    return pairs


# ============================================================================
# Process Sets
# ============================================================================
def process_intrinsic(rows: List[dict]) -> Dict[str, dict]:
    """Process intrinsic CSV (supports both formats)."""
    result = {}
    
    clusters = set(get_cluster_name(row) for row in rows)
    for cluster_name in clusters:
        cluster_rows = [r for r in rows if get_cluster_name(r) == cluster_name]
        
        entries_by_conf = defaultdict(set)
        for row in cluster_rows:
            # Support both formats
            for old_prefix, new_suffix in [('a', '_a'), ('b', '_b')]:
                pdb = (row.get(f'{old_prefix}_pdb') or row.get(f'pdb{new_suffix}', '')).lower()
                asm = row.get(f'{old_prefix}_assembly_id') or row.get(f'asm{new_suffix}', '')
                chain = row.get(f'{old_prefix}_chain') or row.get(f'chain{new_suffix}', '')
                conf = row.get(f'{old_prefix}_conf_label') or row.get(f'conf_label{new_suffix}', '')
                if pdb and asm and chain:
                    entries_by_conf[conf].add((pdb, asm, chain))
        
        apo_list, apo_tags = [], []
        for conf in sorted(entries_by_conf.keys()):
            pdb, asm, chain = sorted(entries_by_conf[conf])[0]
            apo_list.append(make_entry_id(pdb, asm, chain, conf))
            apo_tags.append(make_tag(pdb, asm, chain, "m"))
        
        result[cluster_name] = {
            "apo": apo_list,
            "holo": [],
            "apo_tags": apo_tags,
            "holo_tags": [],
        }
    
    return result


def process_induced_set(rows: List[dict]) -> Dict[str, dict]:
    """Process induced CSV (supports both formats)."""
    result = {}
    
    clusters = set(get_cluster_name(row) for row in rows)
    for cluster_name in clusters:
        cluster_rows = [r for r in rows if get_cluster_name(r) == cluster_name]
        
        apo_by_conf = defaultdict(set)
        holo_set = set()
        
        for row in cluster_rows:
            # a is apo (support both formats)
            a_pdb = (row.get('a_pdb') or row.get('pdb_a', '')).lower()
            a_asm = row.get('a_assembly_id') or row.get('asm_a', '')
            a_chain = row.get('a_chain') or row.get('chain_a', '')
            a_conf = row.get('a_conf_label') or row.get('conf_label_a', '')
            if a_pdb and a_asm and a_chain:
                apo_by_conf[a_conf].add((a_pdb, a_asm, a_chain))
            
            # b is holo
            b_pdb = (row.get('b_pdb') or row.get('pdb_b', '')).lower()
            b_asm = row.get('b_assembly_id') or row.get('asm_b', '')
            b_chain = row.get('b_chain') or row.get('chain_b', '')
            b_conf = row.get('b_conf_label') or row.get('conf_label_b', '')
            if b_pdb and b_asm and b_chain:
                holo_set.add((b_pdb, b_asm, b_chain, b_conf))
        
        apo_list, apo_tags = [], []
        for conf in sorted(apo_by_conf.keys()):
            pdb, asm, chain = sorted(apo_by_conf[conf])[0]
            apo_list.append(make_entry_id(pdb, asm, chain, conf))
            apo_tags.append(make_tag(pdb, asm, chain, "m"))
        
        holo_list, holo_tags = [], []
        for pdb, asm, chain, conf in sorted(holo_set):
            holo_list.append(make_entry_id(pdb, asm, chain, conf))
            holo_tags.append(make_tag(pdb, asm, chain, "x"))
        
        result[cluster_name] = {
            "apo": apo_list,
            "holo": holo_list,
            "apo_tags": apo_tags,
            "holo_tags": holo_tags,
        }
    
    return result


# ============================================================================
# Filtering
# ============================================================================
def filter_data(data: Dict[str, Dict], valid_centers: Set[str]) -> Dict[str, Dict]:
    filtered = {}
    
    for set_name, clusters in data.items():
        filtered[set_name] = {}
        is_induced = set_name in ['protein-induced', 'ligand-induced']
        
        for cluster_name, info in clusters.items():
            if valid_centers and cluster_name not in valid_centers:
                continue
            
            if is_induced:
                apo_confs = {get_conf(e) for e in info['apo']}
                holo_confs = {get_conf(e) for e in info['holo']}
                if apo_confs & holo_confs:
                    continue
                if len(info['apo']) < 1 or len(info['holo']) < 1:
                    continue
            else:
                if len(info['apo']) < 1:
                    continue
            
            filtered[set_name][cluster_name] = info
    
    return filtered


# ============================================================================
# Valid Pairs Generation
# ============================================================================
def generate_valid_pairs(data: Dict[str, Dict], csv_pairs: Dict[str, Set]) -> Dict[str, Dict]:
    result = {}
    
    for set_name, clusters in data.items():
        result[set_name] = {}
        pairs_set = csv_pairs.get(set_name, set())
        
        for cluster_name, info in clusters.items():
            valid = []
            
            if set_name == "intrinsic":
                for t1, t2 in combinations(info['apo_tags'], 2):
                    k1, k2 = tag_to_key(t1), tag_to_key(t2)
                    if (k1, k2) in pairs_set or (k2, k1) in pairs_set:
                        valid.append([t1, t2])
            else:
                for apo_tag in info['apo_tags']:
                    for holo_tag in info['holo_tags']:
                        k1, k2 = tag_to_key(apo_tag), tag_to_key(holo_tag)
                        if (k1, k2) in pairs_set or (k2, k1) in pairs_set:
                            valid.append([apo_tag, holo_tag])
            
            if valid:
                result[set_name][cluster_name] = valid
    
    return result



# ============================================================================
# Distogram enrichment (config: pipeline.distogram_enrich in config/config.yaml)
# ============================================================================
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _enrich_cfg() -> Dict:
    return pipeline_cfg.raw().get("distogram_enrich") or {}


def _resolve_path(val: object) -> Path | None:
    if not val or not isinstance(val, str):
        return None
    p = Path(val)
    return p if p.is_absolute() else _REPO_ROOT / p


def _map_cfg_path(s: str | None) -> str:
    p = _resolve_path(s)
    return str(p) if p is not None else ""


def _method_to_disto_key(method: str) -> str:
    return {
        "af3": "af3",
        "boltz-2": "boltz2",
        "boltz2": "boltz2",
        "boltz-1": "boltz1",
        "boltz1": "boltz1",
        "bioemu": "bioemu",
    }.get(method, method)


class DistogramPatternError(Exception):
    """Raised when distogram pattern is missing or matches no files."""

    pass


def get_chain_match_from_fasta(fasta_file: Path) -> Dict[str, str]:
    """
    Get chain ID mapping from FASTA file for chai method.
    Maps original chain ID to alphabet chain ID (A, B, C, ...).
    """
    chain_match = {}
    if not fasta_file.exists():
        return chain_match

    with open(fasta_file) as f:
        chain_ids = list(string.ascii_uppercase)
        idx = -1
        for line in f.readlines():
            if line.startswith(">"):
                idx += 1
                if idx < len(chain_ids):
                    chain_id = line.strip().split("=")[-1]
                    chain_match[chain_id] = chain_ids[idx]
    return chain_match


def _fallback_chain_letter(id_str: str) -> str:
    s = (id_str or "").strip()
    if not s:
        return ""
    return s[0] if s[0].isalpha() else s


def get_target_chain_for_method(
    method: str,
    yaml_tag: str,
    path_segment: str,
    cluster_id: str,
    map_set_name: str,
) -> str:
    """
    Get target chain ID for different prediction methods.
    ``path_segment`` comes from the prediction glob (set folder in the path);
    ``map_set_name`` is the answer-map set key (``intrinsic``, etc.) for chain JSON lookup.
    """
    de = _enrich_cfg()
    segment = (path_segment or "").strip() or "intrinsic"
    mkey = {
        "boltz-2": "boltz2",
        "boltz-1": "boltz1",
        "chai-1": "chai",
    }.get(method, method)
    if mkey == "af3":
        interested_chain = extract_chain_from_yaml_tag(yaml_tag)
        p = _map_cfg_path(de.get("af3_chain_mappings"))
        if p:
            m = get_chain_mapping(cluster_id, yaml_tag, map_set_name, p)
            if m is not None and interested_chain in m:
                return m[interested_chain]
        return _fallback_chain_letter(interested_chain)

    if mkey == "boltz2":
        interested_chain = extract_chain_from_yaml_tag(yaml_tag)
        p = _map_cfg_path(de.get("boltz_chain_mappings"))
        if p:
            m = get_chain_mapping(cluster_id, yaml_tag, map_set_name, p)
            if m is not None and interested_chain in m:
                return m[interested_chain]
        return _fallback_chain_letter(interested_chain)

    if mkey == "boltz1":
        return extract_chain_from_yaml_tag(yaml_tag)

    if mkey == "chai":
        interested_chain_id = extract_chain_from_yaml_tag(yaml_tag)
        if segment == "ligand-induced":
            return "A"
        if segment == "intrinsic":
            return "A"
        root = _resolve_path(de.get("chai_fasta_root"))
        if root is not None:
            fasta_path = root / segment / cluster_id / f"{yaml_tag}.fa"
            if fasta_path.exists():
                chain_match = get_chain_match_from_fasta(fasta_path)
                if interested_chain_id in chain_match:
                    return chain_match[interested_chain_id]
        return _fallback_chain_letter(interested_chain_id)

    if mkey == "bioemu":
        return "A"

    return extract_chain_from_yaml_tag(yaml_tag)


def extract_chain_from_yaml_tag(yaml_tag: str) -> str:
    """
    Extract chain ID from yaml tag.
    e.g., '2wrz_2_B1_m' -> 'B1'
    """
    parts = yaml_tag.split("_")
    if len(parts) >= 3:
        return parts[2]  # B1
    return ""


def get_chain_mapping(
    cluster_id: str, yaml_tag: str, method_type: str, mapping_json_path: str
) -> Optional[Dict]:
    """
    Get modeled→target chain mapping from a JSON file (AF3 or Boltz layout).
    Key shape: ``"{set_name}/{cluster_id}/{yaml_tag}"`` (same as the map set key).
    """
    if not mapping_json_path:
        return None
    mapping_path = Path(mapping_json_path)
    if not mapping_path.exists():
        return None
    key = f"{(method_type or '').strip()}/{cluster_id}/{yaml_tag}"
    with open(mapping_path, "r") as f:
        all_mappings = json.load(f)
    if key not in all_mappings:
        return None
    modeled = all_mappings[key]["mapping"]
    return {v: k for k, v in modeled.items()}


def get_msa_path(cluster_id: str) -> str:
    """
    MSA file path: ``<data_root>/msas/<2-letter>/{cluster_id}.a3m`` (``pipeline.dirs.msas``).
    """
    try:
        two_letter = cluster_id.split("_")[0][1:3]
        msa_path = pipeline_cfg.dir("msas") / two_letter / f"{cluster_id}.a3m"
    except Exception:
        return ""
    if msa_path.exists():
        return str(msa_path)
    print(f"Warning: MSA file not found for {cluster_id}")
    return ""


def get_reference_cif_path(yaml_tag: str) -> str:
    """
    Get reference CIF file path for a yaml tag under ``pipeline.dirs.cif_asms``.
    """
    parts = yaml_tag.split("_")
    if len(parts) < 3:
        return ""

    pdb_id = parts[0].upper()
    asm_num = parts[1]
    first_two = pdb_id[1:3]
    rel = Path(first_two) / pdb_id / f"asm_{pdb_id.lower()}_{asm_num}.cif"

    try:
        cif_path = pipeline_cfg.dir("cif_asms") / rel
        if cif_path.exists():
            return str(cif_path)
    except Exception:
        pass

    print(f"Warning: Reference CIF file not found for yaml_tag {yaml_tag}")
    return ""


def extract_yaml_tag_from_pattern(pattern: str) -> str:
    """
    Extract yaml_tag from a prediction pattern.
    e.g., ``.../af3/.../intrinsic/8ABP_1/2wrz_2_B1_m/seed_...`` -> ``2wrz_2_B1_m``
    """

    # Look for pattern like /cluster_id/yaml_tag/
    match = re.search(r"/[^/]+/([^/]+)/seed_", pattern)
    if match:
        return match.group(1)
    return ""


def extract_method_type_from_pattern(pattern: str) -> str:
    """Extract set segment from a prediction path: intrinsic, ligand-induced, or protein-induced."""
    match = re.search(
        r"/(intrinsic|ligand-induced|protein-induced)/[^/]+/", pattern
    )
    if match:
        return match.group(1)
    return ""


def get_distogram_path_pattern(
    method: str, method_type: str, cluster_id: str, yaml_tag: str
) -> str:
    """
    Resolve a distogram glob that exists on disk. Template list comes from
    ``config.pipeline.distogram_enrich.distogram`` (keys: af3, boltz1, boltz2, bioemu).
    Placeholders: {method_type} (intrinsic|ligand-induced|protein-induced), {cluster_id}, {yaml_tag}.
    """
    key = _method_to_disto_key(method)
    if key not in ("af3", "boltz1", "boltz2", "bioemu"):
        return ""
    de = _enrich_cfg()
    disto = de.get("distogram") or {}
    raw = disto.get(key) or disto.get(method)
    if raw is None or raw == []:
        return ""
    templates = [raw] if isinstance(raw, str) else list(raw)
    mt = (method_type or "").strip() or "intrinsic"
    for tmpl in templates:
        s = str(tmpl).format(
            method_type=mt, cluster_id=cluster_id, yaml_tag=yaml_tag
        )
        if s and glob(s):
            return s
    return ""


def enhance_cluster_data(
    cluster_id: str,
    cluster_data: Dict,
    set_name: str,
    representative_sequences: Dict,
) -> Dict:
    """
    Enhance cluster data with additional information needed for distogram analysis.
    Maintains the original structure while adding analysis metadata.
    """
    # Create a deep copy to avoid modifying original
    enhanced_data = json.loads(json.dumps(cluster_data))
    de = _enrich_cfg()
    af3_chain_json = _map_cfg_path(de.get("af3_chain_mappings"))
    boltz_chain_json = _map_cfg_path(de.get("boltz_chain_mappings"))

    # Add MSA path at cluster level
    enhanced_data["msa_path"] = get_msa_path(cluster_id)

    # Add representative sequence if available
    if cluster_id in representative_sequences:
        enhanced_data["representative_sequence_id"] = representative_sequences[
            cluster_id
        ].get("header", "")

    # Enhance apo tags with reference information
    if "apo_tags" in enhanced_data:
        apo_refs = {}
        for tag in enhanced_data["apo_tags"]:
            apo_refs[tag] = {
                "reference_cif_path": get_reference_cif_path(tag),
                "target_chain": extract_chain_from_yaml_tag(tag),
            }
        enhanced_data["apo_references"] = apo_refs

    # Enhance holo tags with reference information
    if "holo_tags" in enhanced_data:
        holo_refs = {}
        for tag in enhanced_data["holo_tags"]:
            holo_refs[tag] = {
                "reference_cif_path": get_reference_cif_path(tag),
                "target_chain": extract_chain_from_yaml_tag(tag),
            }
        enhanced_data["holo_references"] = holo_refs

    # Enhance predictions with target chain and distogram path information
    # Handle apo_predictions (method -> info structure)
    if "apo_predictions" in enhanced_data:
        for method, method_info in enhanced_data["apo_predictions"].items():
            # Create enhanced method info by copying original
            enhanced_method_info = json.loads(json.dumps(method_info))

            # Add target chain based on method
            if method == "bioemu":
                # BioEmu always uses chain A
                enhanced_method_info["target_chain"] = "A"

                pattern = get_distogram_path_pattern(
                    method, set_name, cluster_id, yaml_tag=""
                )
                if not pattern:
                    raise DistogramPatternError(
                        f"Distogram pattern not defined for method '{method}' cluster '{cluster_id}'"
                    )
                enhanced_method_info["distogram_pattern"] = pattern
            else:
                # For other methods, try to extract yaml_tag from pattern
                pattern = method_info.get("pattern", "")
                yaml_tag = extract_yaml_tag_from_pattern(pattern)
                pred_set = extract_method_type_from_pattern(pattern) or set_name

                if yaml_tag:
                    enhanced_method_info["yaml_tag"] = yaml_tag
                    enhanced_method_info["reference_cif_path"] = get_reference_cif_path(
                        yaml_tag
                    )

                    target_chain = get_target_chain_for_method(
                        method, yaml_tag, pred_set, cluster_id, set_name
                    )
                    enhanced_method_info["target_chain"] = target_chain

                    # Add distogram pattern for methods that support it
                    if method in ("af3", "boltz-1", "boltz-2"):
                        pattern = get_distogram_path_pattern(
                            method, pred_set, cluster_id, yaml_tag
                        )

                        if not pattern:
                            raise DistogramPatternError(
                                f"Distogram pattern not defined for method '{method}' cluster '{cluster_id}' yaml_tag '{yaml_tag}'"
                            )

                        enhanced_method_info["distogram_pattern"] = pattern

                        # Add AF3 / Boltz chain mapping if available
                        if method == "af3" and af3_chain_json:
                            af3_mapping = get_chain_mapping(
                                cluster_id,
                                yaml_tag,
                                set_name,
                                af3_chain_json,
                            )
                            if af3_mapping:
                                enhanced_method_info["chain_mapping"] = af3_mapping

                        elif method == "boltz-2" and boltz_chain_json:
                            boltz_mapping = get_chain_mapping(
                                cluster_id,
                                yaml_tag,
                                set_name,
                                boltz_chain_json,
                            )
                            if boltz_mapping:
                                enhanced_method_info["chain_mapping"] = boltz_mapping
                else:
                    # If yaml_tag extraction fails, add a default target_chain
                    print(
                        f"Warning: Could not extract yaml_tag from pattern for {method} in apo_predictions: {pattern}"
                    )
                    enhanced_method_info["target_chain"] = "A"  # fallback

            enhanced_data["apo_predictions"][method] = enhanced_method_info

    # Handle holo_predictions (conformation -> method -> info structure)
    if "holo_predictions" in enhanced_data:
        for conformation, conformation_data in enhanced_data[
            "holo_predictions"
        ].items():
            enhanced_conformation_data = {}

            for method, method_info in conformation_data.items():
                # Create enhanced method info by copying original
                enhanced_method_info = json.loads(json.dumps(method_info))

                # Add target chain based on method
                if method == "bioemu":
                    # BioEmu always uses chain A
                    enhanced_method_info["target_chain"] = "A"
                else:
                    # For other methods, try to extract yaml_tag from pattern
                    pattern = method_info.get("pattern", "")
                    yaml_tag = extract_yaml_tag_from_pattern(pattern)
                    path_method_type = (
                        extract_method_type_from_pattern(pattern) or set_name
                    )

                    if yaml_tag:
                        enhanced_method_info["yaml_tag"] = yaml_tag
                        enhanced_method_info["reference_cif_path"] = (
                            get_reference_cif_path(yaml_tag)
                        )

                        target_chain = get_target_chain_for_method(
                            method, yaml_tag, path_method_type, cluster_id, set_name
                        )
                        enhanced_method_info["target_chain"] = target_chain

                        # Add distogram pattern for methods that support it
                        if method in ("af3", "boltz-1", "boltz-2"):
                            pattern = get_distogram_path_pattern(
                                method, path_method_type, cluster_id, yaml_tag
                            )

                            if not pattern:
                                raise DistogramPatternError(
                                    f"Distogram pattern not defined for method '{method}' cluster '{cluster_id}' yaml_tag '{yaml_tag}'"
                                )

                            enhanced_method_info["distogram_pattern"] = pattern

                            if method == "af3" and af3_chain_json:
                                af3_mapping = get_chain_mapping(
                                    cluster_id,
                                    yaml_tag,
                                    set_name,
                                    af3_chain_json,
                                )
                                if af3_mapping:
                                    enhanced_method_info["chain_mapping"] = af3_mapping
                            elif method == "boltz-2" and boltz_chain_json:
                                boltz_mapping = get_chain_mapping(
                                    cluster_id,
                                    yaml_tag,
                                    set_name,
                                    boltz_chain_json,
                                )
                                if boltz_mapping:
                                    enhanced_method_info["chain_mapping"] = (
                                        boltz_mapping
                                    )
                    else:
                        # If yaml_tag extraction fails, add a default target_chain
                        print(
                            f"Warning: Could not extract yaml_tag from pattern for {method} in holo_predictions/{conformation}: {pattern}"
                        )
                        enhanced_method_info["target_chain"] = "A"  # fallback

                enhanced_conformation_data[method] = enhanced_method_info

            enhanced_data["holo_predictions"][conformation] = enhanced_conformation_data

    return enhanced_data


def enrich_seq_cluster_map(
    seq_cluster_data: Dict[str, Any],
    representative_sequences: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Enrich every cluster in a seq_cluster_to_answer_map (e.g. from make_pairs).
    The dynamics set is keyed ``intrinsic`` in the map and in
    ``distogram_enrich`` path templates.
    """
    enhanced_data: Dict[str, Any] = {}
    for set_name, clusters in seq_cluster_data.items():
        enhanced_data[set_name] = {}
        for cluster_id, cluster_data in clusters.items():
            enhanced_data[set_name][cluster_id] = enhance_cluster_data(
                cluster_id, cluster_data, set_name, representative_sequences
            )
    return enhanced_data

# ============================================================================
# Main CLI
# ============================================================================
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--csv-dir", type=click.Path(exists=True, file_okay=False), default="data/dataset", show_default=True)
@click.option("--clusters-json", type=click.Path(exists=True, dir_okay=False), default="data/clusters.json", show_default=True)
@click.option("--examples-dir", type=click.Path(file_okay=False), default="examples", show_default=True)
@click.option("--outdir", type=click.Path(file_okay=False), default="data/dataset", show_default=True)
def main(csv_dir, clusters_json, examples_dir, outdir):
    """Generate seq_cluster_to_answer_map.json and valid_pairs.json"""
    csv_dir = Path(csv_dir)
    clusters_json = Path(clusters_json)
    examples_dir = Path(examples_dir) if examples_dir else None
    out_dir = Path(outdir)
    
    click.echo("=" * 60)
    click.echo("make_pairs.py")
    click.echo("=" * 60)
    
    # 1. Load valid centers
    click.echo("\n[1] Loading valid centers from clusters.json...")
    valid_centers = load_valid_centers(clusters_json)
    click.echo(f"  Found {len(valid_centers)} valid centers")
    
    # 2. Load and process CSVs
    click.echo("\n[2] Loading CSVs...")
    data = {}
    
    rows = load_csv(csv_dir / "intrinsic.csv")
    data["intrinsic"] = process_intrinsic(rows)
    click.echo(f"  intrinsic: {len(data['intrinsic'])} clusters")
    
    for set_name in ["protein-induced", "ligand-induced"]:
        rows = load_csv(csv_dir / f"{set_name}.csv")
        data[set_name] = process_induced_set(rows)
        click.echo(f"  {set_name}: {len(data[set_name])} clusters")
    
    # 3. Filter
    click.echo("\n[3] Filtering (valid centers + conf overlap)...")
    data = filter_data(data, valid_centers)
    for set_name in SET_NAMES:
        click.echo(f"  {set_name}: {len(data.get(set_name, {}))} clusters")
    
    # 4. Add predictions
    click.echo("\n[4] Adding prediction paths...")
    if examples_dir and examples_dir.exists():
        click.echo(f"  Using examples from: {examples_dir}")
        data = add_predictions_to_data(data, examples_dir)
        model_counts = defaultdict(int)
        for clusters in data.values():
            for info in clusters.values():
                for model in info.get("apo_predictions", {}):
                    model_counts[model] += 1
        for model, count in sorted(model_counts.items()):
            click.echo(f"    {model}: {count} clusters with predictions")
    else:
        click.echo("  [SKIP] examples directory not found")
        data = add_predictions_to_data(data, None)

    rep_path = pipeline_cfg.file("rep_seq")
    if rep_path is None or not rep_path.exists():
        raise click.ClickException(
            "Enrichment requires pipeline.files.rep_seq in config to point to an existing JSON file."
        )
    click.echo("\n[4b] Enriching seq_cluster_to_answer_map (MSA, CIF, distogram, chains)...")
    with open(rep_path) as fh:
        rep_data = json.load(fh)
    data = enrich_seq_cluster_map(data, rep_data)
    click.echo("  Enrichment done.")

    # 5. Generate valid pairs
    click.echo("\n[5] Generating valid pairs...")
    csv_pairs = {
        "intrinsic": load_csv_pairs(csv_dir / "intrinsic.csv"),
        "protein-induced": load_csv_pairs(csv_dir / "protein-induced.csv"),
        "ligand-induced": load_csv_pairs(csv_dir / "ligand-induced.csv"),
    }
    
    valid_pairs = generate_valid_pairs(data, csv_pairs)
    for set_name in SET_NAMES:
        total = sum(len(v) for v in valid_pairs.get(set_name, {}).values())
        click.echo(f"  {set_name}: {len(valid_pairs.get(set_name, {}))} clusters, {total} pairs")
    
    # 6. Save outputs
    click.echo("\n[6] Saving...")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "seq_cluster_to_answer_map.json", 'w') as f:
        json.dump(data, f, indent=2)
    click.echo(f"  -> {out_dir / 'seq_cluster_to_answer_map.json'}")
    
    with open(out_dir / "valid_pairs.json", 'w') as f:
        json.dump(valid_pairs, f, indent=2)
    click.echo(f"  -> {out_dir / 'valid_pairs.json'}")
    
    click.echo("\nDone!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_pairs.py - Generate seq_cluster_to_answer_map.json and valid_pairs.json

Usage:
    python -m curation.make_pairs
    python -m curation.make_pairs --csv-dir data/dataset --outdir data/dataset --examples-dir examples
"""

import json
import csv
import re
from glob import glob
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple, Optional

import click


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


def get_csv_field(row: dict, old_name: str, new_name: str) -> str:
    """Get field value, supporting both old and new CSV formats."""
    return row.get(old_name) or row.get(new_name, "")


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

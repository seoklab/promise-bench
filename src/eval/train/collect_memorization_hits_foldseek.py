#!/usr/bin/env python3
"""
Collect memorization hits grouped by conformational clusters.

Usage:
    python collect_memorization_hits.py <hits_file> <threshold_type> <threshold_value> <model>
    
Examples:
    python collect_memorization_hits.py hits_foldseek.tsv tm 0.7 boltz_2
    python collect_memorization_hits.py hits_foldseek.tsv tm 0.8 af3
"""

import os
import sys
import json
from datetime import datetime
from collections import defaultdict

from utils._config import eval_cfg as E

# Paths (from config)
META_DATA_DIR = str(E.external("meta_data_dir") or "/home.galaxy4/seeun/DB/RCSB/meta_data")
CSV_DIR = str(E.external("combinations_dir") or "data/combinations-final")
HITS_DIR = str(E.external("foldseek_hits_dir") or "foldseek/out")
BASE_OUTPUT_DIR = str(E.dir("memorization_hits_foldseek"))

# Cutoff dates for models
CUTOFFS = {
    "af3": "2021-09-30",
    "chai_1": "2021-01-12",
    "boltz_2": "2023-06-01",
    "bioemu": "2019-08-28"
}

# CSV categories
CSV_CATEGORIES = [
    "apo-monomers",
    "ligand-induced",
    "protein-induced",
]


def parse_date(date_str):
    """Parse date string to datetime object"""
    return datetime.strptime(date_str, "%Y-%m-%d")


def get_pdb_date(pdb_id):
    """Get revision_date from meta_data JSON"""
    json_path = os.path.join(META_DATA_DIR, f"{pdb_id.lower()}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                return data.get("revision_date")
        except Exception as e:
            print(f"Warning: Failed to read {json_path}: {e}")
            return None
    return None


def load_conf_clusters(csv_file):
    """
    Load CSV and group by (cluster_csv, conf_label).
    Returns: dict[tuple[cluster, conf_label], list[asm_keys]]
    where asm_keys match the exact format in hits_foldseek.tsv:
    - Monomer: "asm_6dtq_2" (no chain)
    - Multimer: "asm_6dtq_2_B1" (with chain, where B1 is the COI)
    """
    conf_clusters = defaultdict(list)
    
    with open(csv_file, 'r') as f:
        header = next(f).strip().split(',')
        idx = {col: i for i, col in enumerate(header)}
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) <= max(idx.values()):
                continue
            
            cluster = parts[idx['cluster_csv']]
            
            # Process a side
            a_pdb = parts[idx['a_pdb']]
            a_asm = parts[idx['a_assembly_id']]
            a_chain = parts[idx['a_chain']]
            a_chains = parts[idx['a_chains']]
            a_conf = parts[idx['a_conf_label']]
            
            # Determine if multimer by checking if a_chains contains multiple chains
            # Note: Monomers can also have chain suffixes (e.g., asm_1nl5_1_A1)
            # We create both formats to match any query format in hits file
            if ';' in a_chains:
                # Multimer: use chain-specific format only
                a_asm_key = f"asm_{a_pdb}_{a_asm}_{a_chain}"
                conf_clusters[(cluster, a_conf)].append(a_asm_key)
            else:
                # Monomer: create both formats (with and without chain)
                a_asm_key_no_chain = f"asm_{a_pdb}_{a_asm}"
                a_asm_key_with_chain = f"asm_{a_pdb}_{a_asm}_{a_chain}"
                conf_clusters[(cluster, a_conf)].append(a_asm_key_no_chain)
                conf_clusters[(cluster, a_conf)].append(a_asm_key_with_chain)
            
            # Process b side
            b_pdb = parts[idx['b_pdb']]
            b_asm = parts[idx['b_assembly_id']]
            b_chain = parts[idx['b_chain']]
            b_chains = parts[idx['b_chains']]
            b_conf = parts[idx['b_conf_label']]
            
            # Determine if multimer by checking if b_chains contains multiple chains
            # Note: Monomers can also have chain suffixes (e.g., asm_1nl5_1_A1)
            # We create both formats to match any query format in hits file
            if ';' in b_chains:
                # Multimer: use chain-specific format only
                b_asm_key = f"asm_{b_pdb}_{b_asm}_{b_chain}"
                conf_clusters[(cluster, b_conf)].append(b_asm_key)
            else:
                # Monomer: create both formats (with and without chain)
                b_asm_key_no_chain = f"asm_{b_pdb}_{b_asm}"
                b_asm_key_with_chain = f"asm_{b_pdb}_{b_asm}_{b_chain}"
                conf_clusters[(cluster, b_conf)].append(b_asm_key_no_chain)
                conf_clusters[(cluster, b_conf)].append(b_asm_key_with_chain)
    
    # Remove duplicates
    for key in conf_clusters:
        conf_clusters[key] = list(set(conf_clusters[key]))
    
    return conf_clusters


def load_hits(hits_file, threshold_type, threshold_value, cutoff_date):
    """
    Load hits from TSV file and filter by threshold and cutoff date.
    Returns: dict[query_asm, list[hit_lines]]
    """
    hits_by_query = defaultdict(list)
    
    # Determine column index for threshold (0-indexed)
    # foldseek TSV: query, target, evalue, bits, alnlen, qcov, tcov, tmscore, fident
    #               0      1       2       3     4       5     6     7        8
    threshold_col = None
    if threshold_type == "tm":
        threshold_col = 7  # TM score column
    elif threshold_type == "fident":
        threshold_col = 8  # fident column
    else:
        print(f"Error: Unknown threshold type '{threshold_type}'")
        sys.exit(1)
    
    cutoff_dt = parse_date(cutoff_date)
    
    print(f"Loading hits from {hits_file}...")
    line_count = 0
    filtered_count = 0
    
    with open(hits_file, 'r') as f:
        for line in f:
            line_count += 1
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue
            
            query = parts[0]
            target = parts[1]
            
            # Check threshold
            try:
                score = float(parts[threshold_col])
                if score < threshold_value:
                    continue
            except (ValueError, IndexError):
                continue
            
            # Extract PDB ID from target (format: 1ba2_A or just 2gx6)
            target_pdb = target.split('_')[0][:4]
            
            # Check cutoff date
            pdb_date_str = get_pdb_date(target_pdb)
            if pdb_date_str:
                try:
                    pdb_date = parse_date(pdb_date_str)
                    if pdb_date > cutoff_dt:
                        continue
                except:
                    continue
            else:
                # Skip if we can't determine the date
                continue
            
            hits_by_query[query].append(line.strip())
            filtered_count += 1
    
    print(f"  Total lines: {line_count}")
    print(f"  Filtered hits: {filtered_count}")
    print(f"  Unique queries: {len(hits_by_query)}")
    
    return hits_by_query


def sanitize_cluster_name(cluster):
    """Convert cluster name to path-safe format"""
    # "A2/8A27_1" -> "8A27_1"
    return cluster.split('/')[-1] if '/' in cluster else cluster


def collect_hits(category, conf_clusters, hits_by_query, output_base):
    """
    Collect hits for each conformational cluster and write to files.
    """
    category_dir = os.path.join(output_base, category)
    
    total_clusters = 0
    total_hits = 0
    
    for (cluster, conf_label), asm_list in sorted(conf_clusters.items()):
        # Create cluster directory
        cluster_safe = sanitize_cluster_name(cluster)
        cluster_dir = os.path.join(category_dir, cluster_safe)
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Collect all hits for this conf cluster
        all_hits = []
        for asm_key in asm_list:
            if asm_key in hits_by_query:
                all_hits.extend(hits_by_query[asm_key])
        
        # Remove duplicates
        all_hits = list(set(all_hits))
        
        # Write to file
        output_file = os.path.join(cluster_dir, f"conf_{conf_label}.tsv")
        with open(output_file, 'w') as f:
            for hit_line in all_hits:
                f.write(hit_line + '\n')
        
        if all_hits:
            total_hits += len(all_hits)
            total_clusters += 1
    
    print(f"  {category}: {total_clusters} clusters with hits, {total_hits} total hits")


def main():
    if len(sys.argv) != 5:
        print("Usage: python collect_memorization_hits.py <hits_file> <threshold_type> <threshold_value> <model>")
        print("Examples:")
        print("  python collect_memorization_hits.py hits_foldseek.tsv tm 0.7 boltz_2")
        print("  python collect_memorization_hits.py hits_foldseek.tsv tm 0.8 af3")
        sys.exit(1)
    
    hits_filename = sys.argv[1]
    threshold_type = sys.argv[2]
    threshold_value = float(sys.argv[3])
    model = sys.argv[4]
    
    # Validate model
    if model not in CUTOFFS:
        print(f"Error: Unknown model '{model}'. Available: {list(CUTOFFS.keys())}")
        sys.exit(1)
    
    cutoff_date = CUTOFFS[model]
    hits_file = os.path.join(HITS_DIR, hits_filename)
    
    if not os.path.exists(hits_file):
        print(f"Error: Hits file not found: {hits_file}")
        sys.exit(1)
    
    # Create output directory structure
    output_base = os.path.join(
        BASE_OUTPUT_DIR,
        f"hits_{hits_filename.replace('.tsv', '')}_{threshold_type}_{threshold_value}",
        model
    )
    
    print(f"\n{'='*70}")
    print(f"Collecting memorization hits")
    print(f"{'='*70}")
    print(f"Hits file: {hits_file}")
    print(f"Threshold: {threshold_type} >= {threshold_value}")
    print(f"Model: {model}")
    print(f"Cutoff date: {cutoff_date}")
    print(f"Output: {output_base}")
    print(f"{'='*70}\n")
    
    # Load all hits with filtering
    hits_by_query = load_hits(hits_file, threshold_type, threshold_value, cutoff_date)
    
    # Process each CSV category
    for category in CSV_CATEGORIES:
        csv_file = os.path.join(CSV_DIR, f"{category}.csv")
        
        if not os.path.exists(csv_file):
            print(f"Warning: CSV file not found: {csv_file}")
            continue
        
        print(f"\nProcessing {category}...")
        conf_clusters = load_conf_clusters(csv_file)
        print(f"  Found {len(conf_clusters)} conformational clusters")
        
        collect_hits(category, conf_clusters, hits_by_query, output_base)
    
    print(f"\n{'='*70}")
    print(f"Done! Results written to:")
    print(f"{output_base}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


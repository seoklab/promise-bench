#!/usr/bin/env python3
"""
Calculate smoothed training data bias scores for each valid pair using conf labels from combinations-final
Uses smoothed bias formula: (holo_hits - apo_hits) / (holo_hits + apo_hits + smoothing_constant)
"""

import csv
import json
import os
from pathlib import Path
from collections import defaultdict

from curation.utils._config import eval_cfg as E


# Smoothing constant
SMOOTHING_CONSTANT = 0.0


def load_valid_pairs(valid_pairs_json_path, category):
    """Load valid pairs from JSON file for a specific category"""
    with open(valid_pairs_json_path, 'r') as f:
        data = json.load(f)
    return data.get(category, {})


def load_combinations_csv(csv_path):
    """Load combinations CSV to map entry names to conf_labels"""
    # Build a mapping: (pdb, assembly_id, chain) -> conf_label
    entry_to_conf = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Entry format in valid_pairs: "pdb_assembly_chain_suffix"
            # CSV has: a_pdb, a_assembly_id, a_chain, a_conf_label
            a_key = f"{row['a_pdb']}_{row['a_assembly_id']}_{row['a_chain']}"
            b_key = f"{row['b_pdb']}_{row['b_assembly_id']}_{row['b_chain']}"
            
            entry_to_conf[a_key] = row['a_conf_label']
            entry_to_conf[b_key] = row['b_conf_label']
    
    return entry_to_conf


def count_hits_in_tsv(tsv_path):
    """
    Calculate average number of hits per query in a TSV file.
    Each line represents a hit for a query, so we:
    1. Group hits by query (first column)
    2. Calculate average hits per query
    """
    if not os.path.exists(tsv_path):
        return 0.0
    
    query_hit_counts = defaultdict(int)
    
    with open(tsv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # First column is the query
            query = line.split('\t')[0]
            query_hit_counts[query] += 1
    
    # Calculate average hits per query
    if not query_hit_counts:
        return 0.0
    
    total_queries = len(query_hit_counts)
    total_hits = sum(query_hit_counts.values())
    avg_hits = total_hits / total_queries
    
    return avg_hits


def get_conf_label_hits(cluster_dir, conf_label):
    """Get hit count for a specific conf_label in a cluster"""
    tsv_path = cluster_dir / f"conf_{conf_label}.tsv"
    return count_hits_in_tsv(tsv_path)


def parse_entry_name(entry_name):
    """Parse entry name like '6wxn_2_A1_m' into components"""
    parts = entry_name.rsplit('_', 1)  # Split off suffix
    if len(parts) == 2:
        core, suffix = parts
        # Core is like '6wxn_2_A1'
        core_parts = core.split('_')
        if len(core_parts) >= 3:
            pdb = core_parts[0]
            assembly = core_parts[1]
            chain = '_'.join(core_parts[2:])  # Handle chain like 'A1' or 'AB1'
            return pdb, assembly, chain, suffix
    return None, None, None, None


def calculate_smoothed_bias(hits1, hits2, smoothing_constant=SMOOTHING_CONSTANT):
    """
    Calculate smoothed bias score
    Formula: (hits2 - hits1) / (hits1 + hits2 + smoothing_constant)
    Range: approximately [-1, 1], but smoother near extremes
    Returns 0.0 if both hits are 0 (no bias when no data).
    """
    total_with_smoothing = hits1 + hits2 + smoothing_constant
    if total_with_smoothing == 0:
        return 0.0  # No bias when both are 0
    bias = (hits2 - hits1) / total_with_smoothing
    return bias


def calculate_bias_scores(valid_pairs_json, csv_path, memorization_base_dir, model_name, category):
    """Calculate smoothed bias scores for all valid pairs"""
    # Load valid pairs for category
    valid_pairs_dict = load_valid_pairs(valid_pairs_json, category)
    
    # Load conf_label mapping from CSV
    entry_to_conf = load_combinations_csv(csv_path)
    
    results = []
    memorization_dir = Path(memorization_base_dir) / model_name / category
    
    total_pairs = sum(len(pairs) for pairs in valid_pairs_dict.values())
    processed = 0
    
    for cluster_name, pair_list in valid_pairs_dict.items():
        cluster_dir = memorization_dir / cluster_name
        
        # Skip if cluster directory doesn't exist (no hits for this threshold)
        if not cluster_dir.exists():
            continue
        
        for pair in pair_list:
            entry1, entry2 = pair[0], pair[1]
            
            # Parse entry names
            pdb1, asm1, chain1, suffix1 = parse_entry_name(entry1)
            pdb2, asm2, chain2, suffix2 = parse_entry_name(entry2)
            
            if not pdb1 or not pdb2:
                print(f"Warning: Could not parse entries: {entry1}, {entry2}")
                continue
            
            # Get conf labels from combinations CSV
            key1 = f"{pdb1}_{asm1}_{chain1}"
            key2 = f"{pdb2}_{asm2}_{chain2}"
            
            conf1 = entry_to_conf.get(key1)
            conf2 = entry_to_conf.get(key2)
            
            if conf1 is None or conf2 is None:
                print(f"Warning: Conf label not found for {key1} or {key2}")
                continue
            
            # Count hits for each conf_label
            hits1 = get_conf_label_hits(cluster_dir, conf1)
            hits2 = get_conf_label_hits(cluster_dir, conf2)
            
            # Calculate smoothed bias (returns 0.0 if both hits are 0)
            smoothed_bias = calculate_smoothed_bias(hits1, hits2)
            
            # Calculate total and ratios (for reference)
            total_hits = hits1 + hits2
            
            if total_hits == 0:
                ratio1 = 0.0
                ratio2 = 0.0
                ratio_diff = 0.0
            else:
                ratio1 = hits1 / total_hits
                ratio2 = hits2 / total_hits
                ratio_diff = ratio2 - ratio1  # entry2 - entry1, same direction as smoothed_bias
            
            result = {
                'cluster_name': cluster_name,
                'entry1': entry1,
                'entry1_pdb': pdb1,
                'entry1_conf_label': conf1,
                'entry1_hits': hits1,
                'entry1_ratio': round(ratio1, 4),
                'entry2': entry2,
                'entry2_pdb': pdb2,
                'entry2_conf_label': conf2,
                'entry2_hits': hits2,
                'entry2_ratio': round(ratio2, 4),
                'total_hits': total_hits,
                'ratio_difference': round(ratio_diff, 4),
                'smoothed_bias': round(smoothed_bias, 4)
            }
            
            results.append(result)
            processed += 1
            
            if processed % 100 == 0:
                print(f"Processed {processed}/{total_pairs} pairs...")
    
    return results


def main():
    valid_pairs_json = str(E.file("valid_pairs"))
    combinations_dir = str(E.external("combinations_dir") or "data/combinations-final")
    memorization_base_dir = str(E.dir("memorization_hits_intersection"))
    
    # Threshold combinations to process
    threshold_combinations = [
        (0.9, 0.8)
    ]
    
    # Categories to process
    categories = [
        ('apo-monomers', 'apo-monomers.csv'),
        ('ligand-induced', 'ligand-induced.csv'),
        ('protein-induced', 'protein-induced.csv')
    ]
    
    # Process for each model
    models = ['af3', 'boltz_2', 'chai_1', 'bioemu']
    
    print(f"Using smoothing constant: {SMOOTHING_CONSTANT}")
    print(f"Bias formula: (hits2 - hits1) / (hits1 + hits2 + {SMOOTHING_CONSTANT})\n")
    
    # Process for each threshold combination
    for tm_threshold, fident_threshold in threshold_combinations:
        print(f"\n{'='*70}")
        print(f"Processing: TM≥{tm_threshold}, fident≥{fident_threshold}")
        print(f"{'='*70}")
        
        # Build memorization directory path for this threshold
        threshold_dir = f"hits_tm_{tm_threshold}_fident_{fident_threshold}"
        threshold_base_dir = os.path.join(memorization_base_dir, threshold_dir)
        
        if not os.path.exists(threshold_base_dir):
            print(f"Warning: Threshold directory not found: {threshold_base_dir}")
            continue
        
        # Process for each model
        for model in models:
            print(f"\nModel: {model}")
            
            model_results = {}
            
            for category, csv_filename in categories:
                print(f"  Category: {category}")
                csv_path = os.path.join(combinations_dir, csv_filename)
                
                if not os.path.exists(csv_path):
                    print(f"  Warning: CSV file not found: {csv_path}")
                    continue
                
                results = calculate_bias_scores(
                    valid_pairs_json, 
                    csv_path, 
                    threshold_base_dir, 
                    model, 
                    category
                )
                
                model_results[category] = results
                
                print(f"    Total pairs processed: {len(results)}")
                
                # Print some statistics
                if results:
                    # Calculate statistics (all biases are now valid floats)
                    avg_smoothed_bias = sum(r['smoothed_bias'] for r in results) / len(results)
                    max_smoothed_bias = max(r['smoothed_bias'] for r in results)
                    min_smoothed_bias = min(r['smoothed_bias'] for r in results)
                    
                    print(f"    Smoothed bias: avg={avg_smoothed_bias:.4f}, min={min_smoothed_bias:.4f}, max={max_smoothed_bias:.4f}")
                    
                    # Count pairs with no hits
                    no_hits_count = sum(1 for r in results if r['total_hits'] == 0)
                    print(f"    Pairs with no hits: {no_hits_count} ({no_hits_count/len(results)*100:.1f}%)")
            
            # Save combined results for this model and threshold
            output_dir = str(E.dir("training_bias"))
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"training_bias_per_pair_{model}.json")
            with open(output_path, 'w') as f:
                json.dump(model_results, f, indent=2)
            
            print(f"  ✓ Saved to: {output_path}")


if __name__ == "__main__":
    main()

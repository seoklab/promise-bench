#!/usr/bin/env python3
"""
Merge all valid pair data for paper analysis.

Inputs (all configurable via CLI options):
  --valid-pairs-json          Base reference of valid conformational pairs (valid_pairs.json)
  --msa-pref-csv              MSA preference scores per pair (per_pair_summary.csv)
  --confbench-json            RMSD-based ConfBench scores (confbench_scores_valid_pairs.json)
  --confbench-distogram-json  Distogram-based ConfBench scores (confbench_scores_distogram.json)
  --training-bias-dir         Directory of per-model training bias JSONs (tm_0.9_fident_0.8)
  --survived-clusters-json    Training cutoff survival flags (survived_clusters_threshold60.json)

Outputs (written to --output-dir):
  merged_valid_pairs_data.json
      {model: {set_name: {cluster_id: {pair_key: {metric: value, ...}}}}}
  merged_valid_pairs_data.csv
      Flat table with columns: model, set_name, cluster_id, conf1_name, conf2_name,
      msa_pref_sum, msa_pref_avg, same_sign_sum_avg, over_coverage_0.1,
      rmsd_conf1_conf2, confbench_mean (apo-monomers) /
      confbench_apo_pred & confbench_holo_pred (induced sets),
      distogram_confbench*, distogram_dynamic_confbench*,
      bias_entry1_hits, bias_entry2_hits, bias_ratio_diff,
      after_training_cutoff
"""

import os
import json
import pandas as pd
import numpy as np
import click
from collections import defaultdict


# Model mapping
MODELS = ['alphafold3', 'boltz1', 'boltz2', 'chai', 'bioemu']
MODEL_KEY_MAP = {
    'alphafold3': 'af3',
    'boltz1': 'boltz1',
    'boltz2': 'boltz2',
    'chai': 'chai',
    'bioemu': 'bioemu'
}

# Bias file mapping (boltz1 uses af3 bias)
BIAS_FILE_MAP = {
    'alphafold3': 'training_bias_per_pair_af3.json',
    'boltz1': 'training_bias_per_pair_af3.json',  # boltz1 = af3
    'boltz2': 'training_bias_per_pair_boltz_2.json',
    'chai': 'training_bias_per_pair_chai_1.json',
    'bioemu': 'training_bias_per_pair_bioemu.json'
}

# Survived clusters model key mapping
SURVIVED_MODEL_MAP = {
    'alphafold3': 'af3',
    'boltz1': 'boltz1',
    'boltz2': 'boltz2',
    'chai': 'chai',
    'bioemu': 'bioemu-structure'
}


def load_valid_pairs(valid_pairs_json):
    """Load valid_pairs.json as the base reference."""
    with open(valid_pairs_json, 'r') as f:
        return json.load(f)


def load_msa_pref(msa_pref_csv):
    """Load MSA preference data from CSV."""
    df = pd.read_csv(msa_pref_csv)
    
    # Create lookup dict: (set_name, cluster_id, conf1, conf2) -> row
    msa_lookup = {}
    for _, row in df.iterrows():
        key = (row['set_name'], row['cluster_id'], row['conf1_name'], row['conf2_name'])
        msa_lookup[key] = row.to_dict()
        # Also store reverse order for matching
        key_rev = (row['set_name'], row['cluster_id'], row['conf2_name'], row['conf1_name'])
        msa_lookup[key_rev] = {'_reversed': True, **row.to_dict()}
    
    return msa_lookup


def load_confbench(confbench_json):
    """Load ConfBench scores."""
    with open(confbench_json, 'r') as f:
        return json.load(f)


def load_confbench_distogram(confbench_distogram_json):
    """Load Distogram-based ConfBench scores."""
    with open(confbench_distogram_json, 'r') as f:
        return json.load(f)


def load_training_bias(training_bias_dir):
    """Load training bias data from tm_0.9_fident_0.8 ONLY."""
    bias_data = {}
    
    for model, filename in BIAS_FILE_MAP.items():
        if filename is None:
            continue
        filepath = os.path.join(training_bias_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                bias_data[model] = json.load(f)
            print(f"  Loaded bias for {model}: {filepath}")
        else:
            print(f"  Warning: Bias file not found for {model}: {filepath}")
    
    return bias_data


def load_survived_clusters(survived_clusters_json):
    """Load survived clusters (after training cutoff)."""
    with open(survived_clusters_json, 'r') as f:
        return json.load(f)


def create_bias_lookup(bias_data):
    """Create lookup dict for bias data.
    
    Extracts: ratio_difference, entry1_hits, entry2_hits
    """
    lookup = {}
    
    for model, data in bias_data.items():
        lookup[model] = {}
        for set_name, entries in data.items():
            for entry in entries:
                cluster = entry['cluster_name']
                e1, e2 = entry['entry1'], entry['entry2']
                
                # Store both orders
                key = (set_name, cluster, e1, e2)
                key_rev = (set_name, cluster, e2, e1)
                
                lookup[model][key] = {
                    'entry1_hits': entry['entry1_hits'],
                    'entry2_hits': entry['entry2_hits'],
                    'ratio_diff': entry['ratio_difference'],
                    '_reversed': False
                }
                
                # Reversed order (swap hits and negate ratio_diff)
                lookup[model][key_rev] = {
                    'entry1_hits': entry['entry2_hits'],
                    'entry2_hits': entry['entry1_hits'],
                    'ratio_diff': -entry['ratio_difference'],
                    '_reversed': True
                }
    
    return lookup


def create_survived_lookup(survived_data):
    """Create lookup dict for survived pairs."""
    lookup = {}
    
    for model_key, set_data in survived_data.items():
        lookup[model_key] = {}
        for set_name, cluster_data in set_data.items():
            for cluster_id, info in cluster_data.items():
                valid_pairs = info.get('valid_pairs', [])
                for pair in valid_pairs:
                    c1, c2 = pair[0], pair[1]
                    key = (set_name, cluster_id, c1, c2)
                    key_rev = (set_name, cluster_id, c2, c1)
                    lookup[model_key][key] = True
                    lookup[model_key][key_rev] = True
    
    return lookup


def get_confbench_data(confbench, set_name, cluster_id, conf1, conf2, model_key):
    """Get ConfBench data for a pair."""
    # Try to find the pair key in confbench
    pair_key_options = [
        f"{cluster_id}_{conf1}_{conf2}",
        f"{cluster_id}_{conf2}_{conf1}"
    ]
    
    entry = None
    for pk in pair_key_options:
        if pk in confbench.get(set_name, {}):
            entry = confbench[set_name][pk]
            break
    
    if entry is None:
        return None
    
    models = entry.get('models', {})
    model_data = models.get(model_key, {})
    
    result = {
        'rmsd_conf1_conf2': entry.get('rmsd_ref1_ref2') or entry.get('rmsd_apo_holo_ref')
    }
    
    if set_name == 'apo-monomers':
        # apo-monomers: single mean_confbench_score
        if 'predictions' in model_data:
            preds = model_data['predictions']
            if preds:
                result['confbench_mean'] = np.mean([p['confbench_score'] for p in preds])
        elif 'mean_confbench_score' in model_data:
            result['confbench_mean'] = model_data['mean_confbench_score']
    else:
        # induced sets: apo_predictions and holo_predictions
        apo_preds = model_data.get('apo_predictions', {})
        holo_preds = model_data.get('holo_predictions', {})
        
        result['confbench_apo_pred'] = apo_preds.get('mean_confbench_score')
        result['confbench_holo_pred'] = holo_preds.get('mean_confbench_score')
    
    return result


def get_confbench_distogram_data(confbench_distogram, set_name, cluster_id, conf1, conf2, model_key):
    """Get Distogram-based ConfBench data for a pair.
    
    Returns dict with:
    - For apo-monomers: distogram_confbench, distogram_dynamic_confbench
    - For induced sets: distogram_confbench_apo, distogram_confbench_holo,
                        distogram_dynamic_confbench_apo, distogram_dynamic_confbench_holo
    
    Also returns '_found_key' to indicate which key was found (for debugging order issues).
    """
    # Try to find the pair key in confbench_distogram
    # Format in distogram file: "{cluster_id}_{conf1}_{conf2}"
    pair_key_options = [
        f"{cluster_id}_{conf1}_{conf2}",
        f"{cluster_id}_{conf2}_{conf1}"
    ]
    
    entry = None
    found_key = None
    for pk in pair_key_options:
        if pk in confbench_distogram.get(set_name, {}):
            entry = confbench_distogram[set_name][pk]
            found_key = pk
            break
    
    if entry is None:
        return None, None
    
    models = entry.get('models', {})
    model_data = models.get(model_key, {})
    
    result = {}
    
    if set_name == 'apo-monomers':
        # apo-monomers: single mean_confbench_score, mean_dynamic_confbench_score
        result['distogram_confbench'] = model_data.get('mean_confbench_score')
        result['distogram_dynamic_confbench'] = model_data.get('mean_dynamic_confbench_score')
    else:
        # induced sets: apo_predictions and holo_predictions
        apo_preds = model_data.get('apo_predictions', {})
        holo_preds = model_data.get('holo_predictions', {})
        
        result['distogram_confbench_apo'] = apo_preds.get('mean_confbench_score')
        result['distogram_confbench_holo'] = holo_preds.get('mean_confbench_score')
        result['distogram_dynamic_confbench_apo'] = apo_preds.get('mean_dynamic_confbench_score')
        result['distogram_dynamic_confbench_holo'] = holo_preds.get('mean_dynamic_confbench_score')
    
    return result, found_key


def merge_all_data(valid_pairs_json, confbench_json, confbench_distogram_json,
                    msa_pref_csv, training_bias_dir, survived_clusters_json):
    """Merge all data sources into unified format."""
    print("Loading data sources...")
    print(f"  Training Bias from: {training_bias_dir}")
    
    valid_pairs = load_valid_pairs(valid_pairs_json)
    msa_lookup = load_msa_pref(msa_pref_csv)
    confbench = load_confbench(confbench_json)
    confbench_distogram = load_confbench_distogram(confbench_distogram_json)
    bias_data = load_training_bias(training_bias_dir)
    survived_data = load_survived_clusters(survived_clusters_json)
    
    bias_lookup = create_bias_lookup(bias_data)
    survived_lookup = create_survived_lookup(survived_data)
    
    print("\nMerging data...")
    
    # Output structures
    json_output = {model: {} for model in MODELS}
    csv_rows = []
    
    # Counters for validation
    total_pairs = 0
    missing_msa = 0
    missing_confbench = defaultdict(int)
    missing_bias = defaultdict(int)
    
    # NEW: Counters for distogram validation
    missing_distogram = defaultdict(int)
    missing_distogram_confbench = defaultdict(int)
    missing_distogram_dynamic = defaultdict(int)
    distogram_order_swapped = defaultdict(int)  # Track when pair order was swapped
    distogram_details = []  # Store detailed missing info
    
    for set_name in ['apo-monomers', 'ligand-induced', 'protein-induced']:
        for model in MODELS:
            if set_name not in json_output[model]:
                json_output[model][set_name] = {}
        
        clusters = valid_pairs.get(set_name, {})
        for cluster_id, pairs in clusters.items():
            for pair in pairs:
                conf1, conf2 = pair[0], pair[1]
                total_pairs += 1
                
                # Get MSA data
                msa_key = (set_name, cluster_id, conf1, conf2)
                msa_data = msa_lookup.get(msa_key)
                
                if msa_data is None:
                    missing_msa += 1
                    print(f"  Warning: Missing MSA data for {set_name}/{cluster_id}/{conf1}--{conf2}")
                    continue
                
                # Check if order was reversed in MSA
                if msa_data.get('_reversed'):
                    # Swap conf1/conf2 to match MSA order
                    conf1, conf2 = conf2, conf1
                    msa_key = (set_name, cluster_id, conf1, conf2)
                    msa_data = msa_lookup.get(msa_key)
                
                pair_key = f"{conf1}-{conf2}"
                
                # Process for each model
                for model in MODELS:
                    model_key = MODEL_KEY_MAP[model]
                    
                    # Initialize cluster dict if needed
                    if cluster_id not in json_output[model][set_name]:
                        json_output[model][set_name][cluster_id] = {}
                    
                    # Base data from MSA
                    pair_data = {
                        'msa_pref_sum': msa_data.get('msa_pref_sum'),
                        'msa_pref_avg': msa_data.get('msa_pref_avg'),
                        'same_sign_sum_avg': msa_data.get('same_sign_sum_avg'),
                        'over_coverage_0.1': msa_data.get('over_coverage_0.1')
                    }
                    
                    # ConfBench data (RMSD-based)
                    cb_data = get_confbench_data(confbench, set_name, cluster_id, conf1, conf2, model_key)
                    if cb_data:
                        pair_data.update(cb_data)
                        # Check if actual scores are missing (None values)
                        if set_name == 'apo-monomers':
                            if cb_data.get('confbench_mean') is None:
                                missing_confbench[model] += 1
                        else:
                            if cb_data.get('confbench_apo_pred') is None or cb_data.get('confbench_holo_pred') is None:
                                missing_confbench[model] += 1
                    else:
                        missing_confbench[model] += 1
                    
                    # NEW: Distogram-based ConfBench data
                    dg_data, found_key = get_confbench_distogram_data(
                        confbench_distogram, set_name, cluster_id, conf1, conf2, model_key
                    )
                    
                    if dg_data:
                        pair_data.update(dg_data)
                        
                        # Check for order swap (conf1_conf2 vs conf2_conf1)
                        expected_key = f"{cluster_id}_{conf1}_{conf2}"
                        if found_key and found_key != expected_key:
                            distogram_order_swapped[model] += 1
                        
                        # Check if specific values are missing
                        if set_name == 'apo-monomers':
                            if dg_data.get('distogram_confbench') is None:
                                missing_distogram_confbench[model] += 1
                            if dg_data.get('distogram_dynamic_confbench') is None:
                                missing_distogram_dynamic[model] += 1
                        else:
                            # Check apo predictions
                            if dg_data.get('distogram_confbench_apo') is None:
                                missing_distogram_confbench[model] += 1
                            if dg_data.get('distogram_dynamic_confbench_apo') is None:
                                missing_distogram_dynamic[model] += 1
                            # Check holo predictions
                            if dg_data.get('distogram_confbench_holo') is None:
                                missing_distogram_confbench[model] += 1
                            if dg_data.get('distogram_dynamic_confbench_holo') is None:
                                missing_distogram_dynamic[model] += 1
                    else:
                        missing_distogram[model] += 1
                        distogram_details.append({
                            'model': model,
                            'set_name': set_name,
                            'cluster_id': cluster_id,
                            'conf1': conf1,
                            'conf2': conf2,
                            'tried_keys': [f"{cluster_id}_{conf1}_{conf2}", f"{cluster_id}_{conf2}_{conf1}"]
                        })
                    
                    # Training Bias (from tm_0.9_fident_0.8 ONLY)
                    bias_key = (set_name, cluster_id, conf1, conf2)
                    
                    if model in bias_lookup:
                        bias = bias_lookup[model].get(bias_key)
                        if bias:
                            pair_data['bias_entry1_hits'] = bias['entry1_hits']
                            pair_data['bias_entry2_hits'] = bias['entry2_hits']
                            pair_data['bias_ratio_diff'] = bias['ratio_diff']
                        else:
                            missing_bias[model] += 1
                    
                    # Training Cutoff Survival
                    survived_model_key = SURVIVED_MODEL_MAP[model]
                    survived_key = (set_name, cluster_id, conf1, conf2)
                    pair_data['after_training_cutoff'] = survived_lookup.get(survived_model_key, {}).get(survived_key, False)
                    
                    # Store in JSON output
                    json_output[model][set_name][cluster_id][pair_key] = pair_data
                    
                    # Create CSV row
                    csv_row = {
                        'model': model,
                        'set_name': set_name,
                        'cluster_id': cluster_id,
                        'conf1_name': conf1,
                        'conf2_name': conf2,
                        **pair_data
                    }
                    csv_rows.append(csv_row)
    
    print(f"\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\nTotal pairs processed: {total_pairs}")
    print(f"Missing MSA data: {missing_msa}")
    
    print(f"\n[RMSD-based ConfBench - Missing]")
    print(f"  {dict(missing_confbench)}")
    
    print(f"\n[Distogram-based ConfBench]")
    print(f"  Missing entire entry: {dict(missing_distogram)}")
    print(f"  Missing mean_confbench_score: {dict(missing_distogram_confbench)}")
    print(f"  Missing mean_dynamic_confbench_score: {dict(missing_distogram_dynamic)}")
    print(f"  Order swapped (found with reversed conf1/conf2): {dict(distogram_order_swapped)}")
    
    print(f"\n[Training Bias - Missing]")
    print(f"  {dict(missing_bias)}")
    
    # Print detailed missing distogram entries (first 10)
    if distogram_details:
        print(f"\n[Distogram Missing Details - First 10 of {len(distogram_details)}]")
        for i, detail in enumerate(distogram_details[:10]):
            print(f"  {i+1}. {detail['model']}/{detail['set_name']}/{detail['cluster_id']}: "
                  f"{detail['conf1']} -- {detail['conf2']}")
            print(f"      Tried keys: {detail['tried_keys']}")
    
    return json_output, csv_rows


def save_outputs(json_output, csv_rows, output_dir):
    """Save JSON and CSV outputs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_path = os.path.join(output_dir, 'merged_valid_pairs_data.json')
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"\nSaved JSON: {json_path}")
    
    # Save CSV
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(output_dir, 'merged_valid_pairs_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Print summary
    print(f"\nOutput Summary:")
    print(f"  Total rows in CSV: {len(df)}")
    print(f"  Models: {df['model'].unique().tolist()}")
    print(f"  Sets: {df['set_name'].unique().tolist()}")
    print(f"  Unique pairs: {len(df) // len(MODELS)}")
    
    return df


@click.command()
@click.option('--valid-pairs-json', type=click.Path(exists=True),
              default='/home/bonjae02/projects/promise-bench/data_eval/valid_pairs.json',
              show_default=True, help='Path to valid_pairs.json')
@click.option('--confbench-json', type=click.Path(exists=True),
              default='/home/bonjae02/projects/promise-bench/data_eval/confbench_scores_valid_pairs.json',
              show_default=True, help='Path to confbench_scores_valid_pairs.json')
@click.option('--confbench-distogram-json', type=click.Path(exists=True),
              default='/home/bonjae02/projects/promise-bench/data_eval/confbench_scores_distogram.json',
              show_default=True, help='Path to confbench_scores_distogram.json')
@click.option('--msa-pref-csv', type=click.Path(exists=True),
              default='/home/bonjae02/projects/promise-bench/data_eval/per_pair_summary.csv',
              show_default=True, help='Path to per_pair_summary.csv (MSA preference)')
@click.option('--training-bias-dir', type=click.Path(exists=True, file_okay=False),
              default='/home/bonjae02/projects/promise-bench/data_eval/train/training_bias',
              show_default=True, help='Directory containing training bias JSON files')
@click.option('--survived-clusters-json', type=click.Path(exists=True),
              default='/home/bonjae02/projects/promise-bench/data_eval/survived_clusters_threshold60.json',
              show_default=True, help='Path to survived_clusters_threshold60.json')
@click.option('--output-dir', type=click.Path(),
              default='/home/bonjae02/projects/promise-bench/data_eval/output',
              show_default=True, help='Output directory for merged results')
def main(valid_pairs_json, confbench_json, confbench_distogram_json,
         msa_pref_csv, training_bias_dir, survived_clusters_json, output_dir):
    print("=" * 80)
    print("MERGE ALL VALID PAIR DATA (v3 - With Distogram ConfBench Scores)")
    print("=" * 80)
    
    json_output, csv_rows = merge_all_data(
        valid_pairs_json, confbench_json, confbench_distogram_json,
        msa_pref_csv, training_bias_dir, survived_clusters_json
    )
    df = save_outputs(json_output, csv_rows, output_dir)
    
    # Quick stats
    print("\n" + "=" * 80)
    print("QUICK STATS BY MODEL")
    print("=" * 80)
    
    for model in MODELS:
        model_df = df[df['model'] == model]
        cutoff_true = model_df['after_training_cutoff'].sum()
        total = len(model_df)
        print(f"{model:15s}: {cutoff_true:3d}/{total:3d} pairs survive training cutoff ({100*cutoff_true/total:.1f}%)")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

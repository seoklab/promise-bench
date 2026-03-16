import json
import os
import sys

from utils._config import eval_cfg as E

# Threshold parameter (can be passed as command line argument)
THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2

# Load the JSON data
input_file = str(E.file("merged_json"))
output_dir = str(E.dir("filtered_pairs"))

print(f"Using threshold: {THRESHOLD}")
print("Loading JSON file...")
with open(input_file, 'r') as f:
    data = json.load(f)

# Filter 1: |bias_ratio_diff| < THRESHOLD
filtered_bias = {}
total_pairs = 0
bias_filtered_pairs = 0

for model_name, model_data in data.items():
    filtered_bias[model_name] = {}
    
    for category, category_data in model_data.items():
        filtered_bias[model_name][category] = {}
        
        for cluster_id, cluster_data in category_data.items():
            filtered_bias[model_name][category][cluster_id] = {}
            
            for pair_name, pair_info in cluster_data.items():
                if model_name == data.keys().__iter__().__next__() and category == list(model_data.keys())[0] and cluster_id == list(category_data.keys())[0]:
                    total_pairs += 1
                
                # Get the value, handling None
                bias_ratio_diff = pair_info.get('bias_ratio_diff')
                
                # Check if value exists and meets criteria
                if bias_ratio_diff is not None and abs(bias_ratio_diff) < THRESHOLD:
                    filtered_bias[model_name][category][cluster_id][pair_name] = pair_info
                    if model_name == list(data.keys())[0] and category == list(model_data.keys())[0] and cluster_id == list(category_data.keys())[0]:
                        bias_filtered_pairs += 1
            
            # Remove empty clusters
            if not filtered_bias[model_name][category][cluster_id]:
                del filtered_bias[model_name][category][cluster_id]
        
        # Remove empty categories
        if not filtered_bias[model_name][category]:
            del filtered_bias[model_name][category]
    
    # Remove empty models
    if not filtered_bias[model_name]:
        del filtered_bias[model_name]

# Filter 2: |msa_pref_sum| < THRESHOLD
filtered_msa = {}
msa_filtered_pairs = 0

for model_name, model_data in data.items():
    filtered_msa[model_name] = {}
    
    for category, category_data in model_data.items():
        filtered_msa[model_name][category] = {}
        
        for cluster_id, cluster_data in category_data.items():
            filtered_msa[model_name][category][cluster_id] = {}
            
            for pair_name, pair_info in cluster_data.items():
                # Get the value, handling None
                msa_pref_sum = pair_info.get('msa_pref_sum')
                
                # Check if value exists and meets criteria
                if msa_pref_sum is not None and abs(msa_pref_sum) < THRESHOLD:
                    filtered_msa[model_name][category][cluster_id][pair_name] = pair_info
                    if model_name == list(data.keys())[0] and category == list(model_data.keys())[0] and cluster_id == list(category_data.keys())[0]:
                        msa_filtered_pairs += 1
            
            # Remove empty clusters
            if not filtered_msa[model_name][category][cluster_id]:
                del filtered_msa[model_name][category][cluster_id]
        
        # Remove empty categories
        if not filtered_msa[model_name][category]:
            del filtered_msa[model_name][category]
    
    # Remove empty models
    if not filtered_msa[model_name]:
        del filtered_msa[model_name]

# Count properly
total_pairs = 0
bias_filtered_pairs = 0
msa_filtered_pairs = 0

for model_name, model_data in data.items():
    for category, category_data in model_data.items():
        for cluster_id, cluster_data in category_data.items():
            total_pairs += len(cluster_data)

for model_name, model_data in filtered_bias.items():
    for category, category_data in model_data.items():
        for cluster_id, cluster_data in category_data.items():
            bias_filtered_pairs += len(cluster_data)

for model_name, model_data in filtered_msa.items():
    for category, category_data in model_data.items():
        for cluster_id, cluster_data in category_data.items():
            msa_filtered_pairs += len(cluster_data)

# Save filtered data
output_file_bias = os.path.join(output_dir, f'filtered_pairs_bias{THRESHOLD}.json')
output_file_msa = os.path.join(output_dir, f'filtered_pairs_msa{THRESHOLD}.json')

with open(output_file_bias, 'w') as f:
    json.dump(filtered_bias, f, indent=2)

with open(output_file_msa, 'w') as f:
    json.dump(filtered_msa, f, indent=2)

print(f"\nFiltering complete!")
print(f"Total pairs: {total_pairs}")
print(f"\nFilter 1 - |bias_ratio_diff| < {THRESHOLD}:")
print(f"  Filtered pairs: {bias_filtered_pairs}")
print(f"  Output: {output_file_bias}")
print(f"\nFilter 2 - |msa_pref_sum| < {THRESHOLD}:")
print(f"  Filtered pairs: {msa_filtered_pairs}")
print(f"  Output: {output_file_msa}")

# Print summary for bias filter
print(f"\n=== Summary for |bias_ratio_diff| < {THRESHOLD} ===")
for model_name, model_data in filtered_bias.items():
    for category, category_data in model_data.items():
        pair_count = sum(len(cluster_data) for cluster_data in category_data.values())
        cluster_count = len(category_data)
        print(f"  {model_name} - {category}: {pair_count} pairs in {cluster_count} clusters")

# Print summary for msa filter
print(f"\n=== Summary for |msa_pref_sum| < {THRESHOLD} ===")
for model_name, model_data in filtered_msa.items():
    for category, category_data in model_data.items():
        pair_count = sum(len(cluster_data) for cluster_data in category_data.values())
        cluster_count = len(category_data)
        print(f"  {model_name} - {category}: {pair_count} pairs in {cluster_count} clusters")

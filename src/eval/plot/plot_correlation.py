#!/usr/bin/env python3
"""
Plot correlation between distogram_confbench and confbench_mean
from filtered_pairs_bias0.2.json, colored by msa_pref_sum or bias_ratio_diff.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import matplotlib.cm as cm

from curation.utils._config import eval_cfg as E

# Mode: Set to True to only generate 4-panel plot, False to generate all plots
ONLY_4PANEL = True

# File paths
_filtered_dir = E.dir("filtered_pairs")
filtered_bias_file = str(_filtered_dir / "filtered_pairs_bias0.3.json")
filtered_msa_file = str(_filtered_dir / "filtered_pairs_msa0.3.json")
output_dir = E.dir("plots")
output_dir.mkdir(exist_ok=True)


def load_data(filepath):
    """Load filtered pairs data."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def extract_data_from_filtered(data):
    """
    Extract x (distogram_confbench), y (confbench_mean), and color values from filtered data.

    Structure: method -> pair_type -> cluster_id -> "ref1-ref2" -> entry
    
    For apo-monomers: confbench_mean, distogram_confbench
    For ligand/protein-induced: confbench_apo_pred, confbench_holo_pred, distogram_confbench_apo, distogram_confbench_holo

    Returns:
        dict with 'x', 'y', 'msa_pref_sum', 'bias_ratio_diff', 'labels', 'methods', 'pair_types'
    """
    result = {
        "x": [],  # distogram_confbench
        "y": [],  # confbench_mean
        "msa_pref_sum": [],
        "bias_ratio_diff": [],
        "total_hits": [],  # bias_entry1_hits + bias_entry2_hits
        "labels": [],
        "methods": [],
        "pair_types": [],
    }

    for method, pair_types_data in data.items():
        if not isinstance(pair_types_data, dict):
            continue
        for pair_type, clusters in pair_types_data.items():
            if not isinstance(clusters, dict):
                continue
            for cluster_id, pairs in clusters.items():
                if not isinstance(pairs, dict):
                    continue
                for pair_key, entry in pairs.items():
                    if not isinstance(entry, dict):
                        continue
                    
                    x_val = None
                    y_val = None
                    
                    # Check for apo-monomers style (single values)
                    if "confbench_mean" in entry and "distogram_confbench" in entry:
                        x_val = entry.get("distogram_confbench")
                        y_val = entry.get("confbench_mean")
                    # Check for induced style (apo/holo separate values)
                    elif "confbench_apo_pred" in entry and "confbench_holo_pred" in entry:
                        disto_apo = entry.get("distogram_confbench_apo")
                        disto_holo = entry.get("distogram_confbench_holo")
                        conf_apo = entry.get("confbench_apo_pred")
                        conf_holo = entry.get("confbench_holo_pred")
                        
                        # Skip if any value is None
                        if disto_apo is not None and disto_holo is not None:
                            if conf_apo is not None and conf_holo is not None:
                                # Skip NaN
                                if not (isinstance(disto_apo, float) and np.isnan(disto_apo)):
                                    if not (isinstance(disto_holo, float) and np.isnan(disto_holo)):
                                        if not (isinstance(conf_apo, float) and np.isnan(conf_apo)):
                                            if not (isinstance(conf_holo, float) and np.isnan(conf_holo)):
                                                x_val = (disto_apo + disto_holo) / 2
                                                y_val = (conf_apo + conf_holo) / 2

                    # Skip if we couldn't get valid values
                    if x_val is None or y_val is None:
                        continue
                    if isinstance(x_val, float) and np.isnan(x_val):
                        continue
                    if isinstance(y_val, float) and np.isnan(y_val):
                        continue

                    msa_pref = entry.get("msa_pref_sum", 0)
                    bias_ratio = entry.get("bias_ratio_diff", 0)
                    bias_entry1_hits = entry.get("bias_entry1_hits", 0)
                    bias_entry2_hits = entry.get("bias_entry2_hits", 0)
                    total_hits = bias_entry1_hits + bias_entry2_hits

                    # Handle NaN in color values
                    if msa_pref is None or (
                        isinstance(msa_pref, float) and np.isnan(msa_pref)
                    ):
                        msa_pref = 0
                    if bias_ratio is None or (
                        isinstance(bias_ratio, float) and np.isnan(bias_ratio)
                    ):
                        bias_ratio = 0
                    if total_hits is None:
                        total_hits = 0

                    result["x"].append(x_val)
                    result["y"].append(y_val)
                    result["msa_pref_sum"].append(msa_pref)
                    result["bias_ratio_diff"].append(bias_ratio)
                    result["total_hits"].append(total_hits)
                    result["labels"].append(
                        f"{method}_{pair_type}_{cluster_id}_{pair_key}"
                    )
                    result["methods"].append(method)
                    result["pair_types"].append(pair_type)

    return result


def filter_by_pair_type(data, target_pair_type):
    """Filter extracted data by pair_type."""
    result = {
        "x": [],
        "y": [],
        "msa_pref_sum": [],
        "bias_ratio_diff": [],
        "total_hits": [],
        "labels": [],
        "methods": [],
        "pair_types": [],
    }
    
    for i, pair_type in enumerate(data["pair_types"]):
        if pair_type == target_pair_type:
            result["x"].append(data["x"][i])
            result["y"].append(data["y"][i])
            result["msa_pref_sum"].append(data["msa_pref_sum"][i])
            result["bias_ratio_diff"].append(data["bias_ratio_diff"][i])
            result["total_hits"].append(data["total_hits"][i])
            result["labels"].append(data["labels"][i])
            result["methods"].append(data["methods"][i])
            result["pair_types"].append(data["pair_types"][i])
    
    return result


def filter_by_method(data, target_method):
    """Filter extracted data by method."""
    result = {
        "x": [],
        "y": [],
        "msa_pref_sum": [],
        "bias_ratio_diff": [],
        "total_hits": [],
        "labels": [],
        "methods": [],
        "pair_types": [],
    }
    
    for i, method in enumerate(data["methods"]):
        if method == target_method:
            result["x"].append(data["x"][i])
            result["y"].append(data["y"][i])
            result["msa_pref_sum"].append(data["msa_pref_sum"][i])
            result["bias_ratio_diff"].append(data["bias_ratio_diff"][i])
            result["total_hits"].append(data["total_hits"][i])
            result["labels"].append(data["labels"][i])
            result["methods"].append(data["methods"][i])
            result["pair_types"].append(data["pair_types"][i])
    
    return result


def extract_apo_holo_separated(data):
    """
    Extract apo and holo data points separately for ligand/protein-induced pairs.
    
    Returns:
        dict with separate 'apo' and 'holo' entries, each containing x, y, color values
    """
    result = {
        "apo": {
            "x": [], "y": [], "msa_pref_sum": [], "bias_ratio_diff": [],
            "total_hits": [], "labels": [], "methods": [], "pair_types": []
        },
        "holo": {
            "x": [], "y": [], "msa_pref_sum": [], "bias_ratio_diff": [],
            "total_hits": [], "labels": [], "methods": [], "pair_types": []
        }
    }
    
    for method, pair_types_data in data.items():
        if not isinstance(pair_types_data, dict):
            continue
        for pair_type, clusters in pair_types_data.items():
            if not isinstance(clusters, dict):
                continue
            # Only process ligand-induced and protein-induced
            if pair_type not in ["ligand-induced", "protein-induced"]:
                continue
                
            for cluster_id, pairs in clusters.items():
                if not isinstance(pairs, dict):
                    continue
                for pair_key, entry in pairs.items():
                    if not isinstance(entry, dict):
                        continue
                    
                    # Get apo and holo values separately
                    disto_apo = entry.get("distogram_confbench_apo")
                    disto_holo = entry.get("distogram_confbench_holo")
                    conf_apo = entry.get("confbench_apo_pred")
                    conf_holo = entry.get("confbench_holo_pred")
                    
                    msa_pref = entry.get("msa_pref_sum", 0)
                    bias_ratio = entry.get("bias_ratio_diff", 0)
                    bias_entry1_hits = entry.get("bias_entry1_hits", 0)
                    bias_entry2_hits = entry.get("bias_entry2_hits", 0)
                    total_hits = bias_entry1_hits + bias_entry2_hits
                    
                    # Handle NaN in color values
                    if msa_pref is None or (isinstance(msa_pref, float) and np.isnan(msa_pref)):
                        msa_pref = 0
                    if bias_ratio is None or (isinstance(bias_ratio, float) and np.isnan(bias_ratio)):
                        bias_ratio = 0
                    if total_hits is None:
                        total_hits = 0
                    
                    # Process apo
                    if disto_apo is not None and conf_apo is not None:
                        if not (isinstance(disto_apo, float) and np.isnan(disto_apo)):
                            if not (isinstance(conf_apo, float) and np.isnan(conf_apo)):
                                result["apo"]["x"].append(disto_apo)
                                result["apo"]["y"].append(conf_apo)
                                result["apo"]["msa_pref_sum"].append(msa_pref)
                                result["apo"]["bias_ratio_diff"].append(bias_ratio)
                                result["apo"]["total_hits"].append(total_hits)
                                result["apo"]["labels"].append(f"{method}_{pair_type}_{cluster_id}_{pair_key}_apo")
                                result["apo"]["methods"].append(method)
                                result["apo"]["pair_types"].append(pair_type)
                    
                    # Process holo
                    if disto_holo is not None and conf_holo is not None:
                        if not (isinstance(disto_holo, float) and np.isnan(disto_holo)):
                            if not (isinstance(conf_holo, float) and np.isnan(conf_holo)):
                                result["holo"]["x"].append(disto_holo)
                                result["holo"]["y"].append(conf_holo)
                                result["holo"]["msa_pref_sum"].append(msa_pref)
                                result["holo"]["bias_ratio_diff"].append(bias_ratio)
                                result["holo"]["total_hits"].append(total_hits)
                                result["holo"]["labels"].append(f"{method}_{pair_type}_{cluster_id}_{pair_key}_holo")
                                result["holo"]["methods"].append(method)
                                result["holo"]["pair_types"].append(pair_type)
    
    return result


def plot_correlation_colored(data, color_key, color_label, title_suffix, output_name, output_dir):
    """Create scatter plot with color based on specified key."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    x = np.array(data["x"])
    y = np.array(data["y"])
    colors = np.array(data[color_key])
    total_hits = np.array(data["total_hits"])

    if len(x) == 0:
        print(f"No data available for {output_name}")
        plt.close()
        return

    # Create colormap
    if color_key == "bias_ratio_diff":
        vmin, vmax = -1, 1  # Fixed range for bias_ratio_diff
    else:
        vmin, vmax = np.percentile(colors, [5, 95])  # Use percentiles to avoid outliers
    cmap = cm.coolwarm

    # Scale point sizes based on total_hits (min 10, max 200)
    sizes = np.clip(total_hits * 2, 10, 200)

    # Scatter plot
    scatter = ax.scatter(
        x,
        y,
        c=colors,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.7,
        s=sizes,
        edgecolors="black",
        linewidth=0.5,
    )

    # Calculate correlation
    if len(x) > 1:
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)

        # Add diagonal line
        min_val = min(min(x), min(y))
        max_val = max(max(x), max(y))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            alpha=0.3,
            linewidth=1,
        )

        # Add correlation info
        ax.text(
            0.05,
            0.05,
            f"Pearson r: {pearson_r:.3f} (p={pearson_p:.2e})\n"
            f"Spearman ρ: {spearman_r:.3f} (p={spearman_p:.2e})\n"
            f"N = {len(x)}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
    cbar.set_label(color_label, fontsize=12)

    ax.set_xlabel("Distogram ConfBench Score", fontsize=13)
    ax.set_ylabel("Structure ConfBench Mean Score", fontsize=13)
    ax.set_title(
        f"Distogram vs Structure ConfBench\n{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    output_file = output_dir / f"{output_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def main():
    pair_types = ["apo-monomers", "ligand-induced", "protein-induced"]
    
    if ONLY_4PANEL:
        print("=" * 80)
        print("ONLY 4-PANEL MODE - Skipping individual plots")
        print("=" * 80)
        
        # Load MSA data only
        msa_data = load_data(filtered_msa_file)
        
        # Create 4-panel subplot
        print("\n" + "=" * 80)
        print("Creating 4-panel subplot (AF3/Boltz2 x Ligand-Apo/Apo-monomers)")
        print("=" * 80)
        
        try:
            create_4panel_subplot(msa_data)
        except Exception as e:
            print(f"Error creating 4-panel subplot: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    print("=" * 80)
    print("Processing filtered_pairs_bias0.2.json (No Training Bias)")
    print("=" * 80)

    # Load and process bias file
    try:
        bias_data = load_data(filtered_bias_file)
        extracted_bias = extract_data_from_filtered(bias_data)
        print(f"Loaded {len(extracted_bias['x'])} valid pairs from bias file")

        # Get unique methods
        unique_methods = sorted(set(extracted_bias["methods"]))
        print(f"Found methods: {unique_methods}")
        
        # Process by method
        for method in unique_methods:
            method_data = filter_by_method(extracted_bias, method)
            if len(method_data["x"]) == 0:
                continue
                
            # Create method-specific output directory
            method_output_dir = output_dir / f"no_training_bias" / method
            method_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n  Method: {method} ({len(method_data['x'])} pairs)")
            
            # Plot ALL pair_types for this method
            plot_correlation_colored(
                method_data,
                color_key="msa_pref_sum",
                color_label="MSA Pref Sum",
                title_suffix=f"No Training Bias - {method}\n(colored by MSA Pref Sum)",
                output_name="correlation_all_pairs_msa_color",
                output_dir=method_output_dir,
            )
            
            # Plot by pair_type with msa_pref_sum only
            for pt in pair_types:
                filtered_data = filter_by_pair_type(method_data, pt)
                if len(filtered_data["x"]) > 0:
                    print(f"    {pt}: {len(filtered_data['x'])} pairs")
                    
                    plot_correlation_colored(
                        filtered_data,
                        color_key="msa_pref_sum",
                        color_label="MSA Pref Sum",
                        title_suffix=f"No Training Bias - {method} - {pt}\n(colored by MSA Pref Sum)",
                        output_name=f"correlation_{pt}_msa_color",
                        output_dir=method_output_dir,
                    )
                else:
                    print(f"    {pt}: No data")
        
        # Extract apo/holo separated data for ligand-induced and protein-induced
        print("\n  Processing apo/holo separated plots...")
        apo_holo_data = extract_apo_holo_separated(bias_data)
        
        # Filter by method and plot
        for method in unique_methods:
            method_output_dir = output_dir / f"no_training_bias" / method
            
            # Filter apo data by method
            apo_filtered = filter_by_method(apo_holo_data["apo"], method)
            # Filter holo data by method
            holo_filtered = filter_by_method(apo_holo_data["holo"], method)
            
            # Plot all apo/holo combined
            if len(apo_filtered["x"]) > 0:
                print(f"    {method} - Apo (all): {len(apo_filtered['x'])} pairs")
                plot_correlation_colored(
                    apo_filtered,
                    color_key="msa_pref_sum",
                    color_label="MSA Pref Sum",
                    title_suffix=f"No Training Bias - {method} - Apo\n(colored by MSA Pref Sum)",
                    output_name="correlation_apo_msa_color",
                    output_dir=method_output_dir,
                )
            
            if len(holo_filtered["x"]) > 0:
                print(f"    {method} - Holo (all): {len(holo_filtered['x'])} pairs")
                plot_correlation_colored(
                    holo_filtered,
                    color_key="msa_pref_sum",
                    color_label="MSA Pref Sum",
                    title_suffix=f"No Training Bias - {method} - Holo\n(colored by MSA Pref Sum)",
                    output_name="correlation_holo_msa_color",
                    output_dir=method_output_dir,
                )
            
            # Plot by pair_type (ligand-induced, protein-induced)
            for pt in ["ligand-induced", "protein-induced"]:
                # Filter apo by pair_type
                apo_pt_filtered = filter_by_pair_type(apo_filtered, pt)
                if len(apo_pt_filtered["x"]) > 0:
                    print(f"    {method} - {pt} - Apo: {len(apo_pt_filtered['x'])} pairs")
                    plot_correlation_colored(
                        apo_pt_filtered,
                        color_key="msa_pref_sum",
                        color_label="MSA Pref Sum",
                        title_suffix=f"No Training Bias - {method} - {pt} - Apo\n(colored by MSA Pref Sum)",
                        output_name=f"correlation_{pt}_apo_msa_color",
                        output_dir=method_output_dir,
                    )
                
                # Filter holo by pair_type
                holo_pt_filtered = filter_by_pair_type(holo_filtered, pt)
                if len(holo_pt_filtered["x"]) > 0:
                    print(f"    {method} - {pt} - Holo: {len(holo_pt_filtered['x'])} pairs")
                    plot_correlation_colored(
                        holo_pt_filtered,
                        color_key="msa_pref_sum",
                        color_label="MSA Pref Sum",
                        title_suffix=f"No Training Bias - {method} - {pt} - Holo\n(colored by MSA Pref Sum)",
                        output_name=f"correlation_{pt}_holo_msa_color",
                        output_dir=method_output_dir,
                    )
                
    except Exception as e:
        print(f"Error processing bias file: {e}")

    print("\n" + "=" * 80)
    print("Processing filtered_pairs_msa0.2.json (No MSA Bias)")
    print("=" * 80)

    # Load and process MSA file
    try:
        msa_data = load_data(filtered_msa_file)
        extracted_msa = extract_data_from_filtered(msa_data)
        print(f"Loaded {len(extracted_msa['x'])} valid pairs from MSA file")

        # Get unique methods
        unique_methods = sorted(set(extracted_msa["methods"]))
        print(f"Found methods: {unique_methods}")
        
        # Process by method
        for method in unique_methods:
            method_data = filter_by_method(extracted_msa, method)
            if len(method_data["x"]) == 0:
                continue
                
            # Create method-specific output directory
            method_output_dir = output_dir / f"no_msa_bias" / method
            method_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n  Method: {method} ({len(method_data['x'])} pairs)")
            
            # Plot ALL pair_types for this method
            plot_correlation_colored(
                method_data,
                color_key="bias_ratio_diff",
                color_label="Bias Ratio Diff",
                title_suffix=f"No MSA Bias - {method}\n(colored by Bias Ratio Diff)",
                output_name="correlation_all_pairs_bias_ratio_color",
                output_dir=method_output_dir,
            )
            
            # Plot by pair_type with bias_ratio_diff only
            for pt in pair_types:
                filtered_data = filter_by_pair_type(method_data, pt)
                if len(filtered_data["x"]) > 0:
                    print(f"    {pt}: {len(filtered_data['x'])} pairs")
                    
                    plot_correlation_colored(
                        filtered_data,
                        color_key="bias_ratio_diff",
                        color_label="Bias Ratio Diff",
                        title_suffix=f"No MSA Bias - {method} - {pt}\n(colored by Bias Ratio Diff)",
                        output_name=f"correlation_{pt}_bias_ratio_color",
                        output_dir=method_output_dir,
                    )
                else:
                    print(f"    {pt}: No data")
        
        # Extract apo/holo separated data for ligand-induced and protein-induced
        print("\n  Processing apo/holo separated plots...")
        apo_holo_data = extract_apo_holo_separated(msa_data)
        
        # Filter by method and plot
        for method in unique_methods:
            method_output_dir = output_dir / f"no_msa_bias" / method
            
            # Filter apo data by method
            apo_filtered = filter_by_method(apo_holo_data["apo"], method)
            # Filter holo data by method
            holo_filtered = filter_by_method(apo_holo_data["holo"], method)
            
            # Plot all apo/holo combined
            if len(apo_filtered["x"]) > 0:
                print(f"    {method} - Apo (all): {len(apo_filtered['x'])} pairs")
                plot_correlation_colored(
                    apo_filtered,
                    color_key="bias_ratio_diff",
                    color_label="Bias Ratio Diff",
                    title_suffix=f"No MSA Bias - {method} - Apo\n(colored by Bias Ratio Diff)",
                    output_name="correlation_apo_bias_ratio_color",
                    output_dir=method_output_dir,
                )
            
            if len(holo_filtered["x"]) > 0:
                print(f"    {method} - Holo (all): {len(holo_filtered['x'])} pairs")
                plot_correlation_colored(
                    holo_filtered,
                    color_key="bias_ratio_diff",
                    color_label="Bias Ratio Diff",
                    title_suffix=f"No MSA Bias - {method} - Holo\n(colored by Bias Ratio Diff)",
                    output_name="correlation_holo_bias_ratio_color",
                    output_dir=method_output_dir,
                )
            
            # Plot by pair_type (ligand-induced, protein-induced)
            for pt in ["ligand-induced", "protein-induced"]:
                # Filter apo by pair_type
                apo_pt_filtered = filter_by_pair_type(apo_filtered, pt)
                if len(apo_pt_filtered["x"]) > 0:
                    print(f"    {method} - {pt} - Apo: {len(apo_pt_filtered['x'])} pairs")
                    plot_correlation_colored(
                        apo_pt_filtered,
                        color_key="bias_ratio_diff",
                        color_label="Bias Ratio Diff",
                        title_suffix=f"No MSA Bias - {method} - {pt} - Apo\n(colored by Bias Ratio Diff)",
                        output_name=f"correlation_{pt}_apo_bias_ratio_color",
                        output_dir=method_output_dir,
                    )
                
                # Filter holo by pair_type
                holo_pt_filtered = filter_by_pair_type(holo_filtered, pt)
                if len(holo_pt_filtered["x"]) > 0:
                    print(f"    {method} - {pt} - Holo: {len(holo_pt_filtered['x'])} pairs")
                    plot_correlation_colored(
                        holo_pt_filtered,
                        color_key="bias_ratio_diff",
                        color_label="Bias Ratio Diff",
                        title_suffix=f"No MSA Bias - {method} - {pt} - Holo\n(colored by Bias Ratio Diff)",
                        output_name=f"correlation_{pt}_holo_bias_ratio_color",
                        output_dir=method_output_dir,
                    )
                
    except Exception as e:
        print(f"Error processing MSA file: {e}")

    print("\n" + "=" * 80)
    print(f"All plots saved to: {output_dir}")
    print("=" * 80)
    
    # Create 4-panel subplot
    print("\n" + "=" * 80)
    print("Creating 4-panel subplot (AF3/Boltz2 x Ligand-Apo/Apo-monomers)")
    print("=" * 80)
    
    try:
        create_4panel_subplot(msa_data)
    except Exception as e:
        print(f"Error creating 4-panel subplot: {e}")


def create_4panel_subplot(msa_data):
    """Create 4-panel subplot: AF3/Boltz2 x Ligand-Apo/Apo-monomers."""
    import matplotlib.gridspec as gridspec
    
    # Set paper figure style - larger elements for smaller print size
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['ytick.major.size'] = 6
    
    # Extract data
    extracted_msa = extract_data_from_filtered(msa_data)
    apo_holo_data = extract_apo_holo_separated(msa_data)
    
    # Check available methods
    print(f"\nAvailable methods in extracted_msa: {sorted(set(extracted_msa['methods']))}")
    print(f"Available methods in apo data: {sorted(set(apo_holo_data['apo']['methods']))}")
    
    # Get data for each panel - try different method names
    methods_to_try = ["af3", "AF3", "alphafold3", "AlphaFold3"]
    af3_method = None
    for m in methods_to_try:
        if m in extracted_msa['methods']:
            af3_method = m
            break
    
    boltz_methods_to_try = ["boltz2", "Boltz2", "boltz-2", "Boltz-2"]
    boltz_method = None
    for m in boltz_methods_to_try:
        if m in extracted_msa['methods']:
            boltz_method = m
            break
    
    print(f"Using AF3 method name: {af3_method}")
    print(f"Using Boltz2 method name: {boltz_method}")
    
    # Panel 1: AF3 ligand-induced apo
    if af3_method:
        af3_ligand_apo = filter_by_pair_type(filter_by_method(apo_holo_data["apo"], af3_method), "ligand-induced")
    else:
        af3_ligand_apo = {"x": [], "y": [], "bias_ratio_diff": [], "total_hits": []}
    
    # Panel 2: Boltz2 ligand-induced apo
    if boltz_method:
        boltz2_ligand_apo = filter_by_pair_type(filter_by_method(apo_holo_data["apo"], boltz_method), "ligand-induced")
    else:
        boltz2_ligand_apo = {"x": [], "y": [], "bias_ratio_diff": [], "total_hits": []}
    
    # Panel 3: AF3 apo-monomers
    if af3_method:
        af3_apo_mono = filter_by_pair_type(filter_by_method(extracted_msa, af3_method), "apo-monomers")
    else:
        af3_apo_mono = {"x": [], "y": [], "bias_ratio_diff": [], "total_hits": []}
    
    # Panel 4: Boltz2 apo-monomers
    if boltz_method:
        boltz2_apo_mono = filter_by_pair_type(filter_by_method(extracted_msa, boltz_method), "apo-monomers")
    else:
        boltz2_apo_mono = {"x": [], "y": [], "bias_ratio_diff": [], "total_hits": []}
    
    print(f"AF3 ligand-induced apo: {len(af3_ligand_apo['x'])} points")
    print(f"Boltz2 ligand-induced apo: {len(boltz2_ligand_apo['x'])} points")
    print(f"AF3 apo-monomers: {len(af3_apo_mono['x'])} points")
    print(f"Boltz2 apo-monomers: {len(boltz2_apo_mono['x'])} points")
    
    # Create figure - compact size for A4 half page
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05], 
                           hspace=0.16, wspace=0.24, right=0.94, left=0.08, top=0.96, bottom=0.06)
    
    # Create subplots
    axes = [
        fig.add_subplot(gs[0, 0]),  # AF3 ligand apo
        fig.add_subplot(gs[0, 1]),  # Boltz2 ligand apo
        fig.add_subplot(gs[1, 0]),  # AF3 apo-monomers
        fig.add_subplot(gs[1, 1]),  # Boltz2 apo-monomers
    ]
    
    # Colorbar axis
    cbar_ax = fig.add_subplot(gs[:, 2])
    
    # Data for each panel
    panel_data = [
        (af3_ligand_apo, "AF3 - Ligand-induced (Apo-conditioned)"),
        (boltz2_ligand_apo, "Boltz-2 - Ligand-induced (Apo-conditioned)"),
        (af3_apo_mono, "AF3 - Intrinsic Dynamics"),
        (boltz2_apo_mono, "Boltz-2 - Intrinsic Dynamics"),
    ]
    
    # Panel labels
    panel_labels = ["A", "B", "C", "D"]
    
    # Color settings
    vmin, vmax = -1, 1
    cmap = cm.coolwarm
    
    scatters = []
    
    # Plot each panel
    for idx, (ax, (data, title)) in enumerate(zip(axes, panel_data)):
        x = np.array(data["x"])
        y = np.array(data["y"])
        colors = np.array(data["bias_ratio_diff"])
        total_hits = np.array(data["total_hits"])
        
        if len(x) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                   fontsize=22, fontweight='bold')
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            # Add panel label
            ax.text(0.05, 0.95, panel_labels[idx], transform=ax.transAxes,
                   fontsize=26, fontweight='bold', va='top', ha='left')
            continue
        
        # Scale point sizes - much larger for print
        sizes = np.clip(total_hits * 4, 30, 400)
        
        # Scatter plot with thick edges
        scatter = ax.scatter(
            x, y, c=colors, cmap=cmap, vmin=vmin, vmax=vmax,
            alpha=0.75, s=sizes, edgecolors="black", linewidth=0.7,
        )
        scatters.append(scatter)
        
        # Calculate correlation
        if len(x) > 1:
            pearson_r, pearson_p = pearsonr(x, y)
            spearman_r, spearman_p = spearmanr(x, y)
            
            # Add diagonal line - full extent
            ax.plot([-1.05, 1.05], [-1.05, 1.05], "k--", alpha=0.5, linewidth=1.5)
        
        # Add panel label in top left
        ax.text(0.05, 0.95, panel_labels[idx], transform=ax.transAxes,
               fontsize=26, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8))
        
        # Add title
        ax.set_title(title, fontsize=18, fontweight='bold', pad=12)
        
        # Set labels and limits with larger fonts
        # Only show x-label on bottom row, y-label on left column
        if idx >= 2:  # Bottom row
            ax.set_xlabel(r"$\mathrm{Disto_{holo}}$", fontsize=20, fontweight='bold')
        if idx % 2 == 0:  # Left column
            ax.set_ylabel(r"$\mathrm{Struct_{holo}}$", fontsize=20, fontweight='bold')
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        
        # Set ticks at 0.5 intervals
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        # No grid, only center lines
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=1.2, alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=1.2, alpha=0.5)
        
        # Tick parameters - larger for readability
        ax.tick_params(axis='both', which='major', labelsize=17, length=6, width=1.5)
    
    # Add colorbar with better styling
    if scatters:
        cbar = fig.colorbar(scatters[0], cax=cbar_ax, orientation="vertical")
        cbar.set_label(r"$\mathrm{Train_{holo}}$", fontsize=20, labelpad=20)
        cbar.ax.tick_params(labelsize=17, width=1.5, length=6)
        cbar.outline.set_linewidth(1.5)
    
    output_path_png = output_dir / "correlation_4panel_comparison.png"
    output_path_pdf = output_dir / "correlation_4panel_comparison.pdf"
    plt.savefig(output_path_png, dpi=400, bbox_inches="tight", facecolor='white')
    plt.savefig(output_path_pdf, bbox_inches="tight", facecolor='white')
    print(f"Saved 4-panel plot: {output_path_png}")
    print(f"Saved 4-panel plot (PDF): {output_path_pdf}")
    plt.close()
    
    # Reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Create correspondence plot for AF3 vs Boltz2 ligand-induced apo
    create_correspondence_plot(af3_ligand_apo, boltz2_ligand_apo, "ligand-induced (Apo-conditioned)")


def create_correspondence_plot(af3_data, boltz2_data, title_suffix, min_distance=0.2):
    """Create direct comparison plots: AF3 vs Boltz2 for Dist and Struct separately."""
    import matplotlib.gridspec as gridspec
    
    # Set paper figure style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.linewidth'] = 1.5
    
    # Extract labels and create mapping
    af3_labels = np.array(af3_data["labels"])
    boltz2_labels = np.array(boltz2_data["labels"])
    
    # Create a key that removes the method name from labels
    # Format: {method}_{pair_type}_{cluster_id}_{pair_key}_apo
    def extract_pair_key(label):
        parts = label.split("_")
        # Skip first part (method name), join the rest
        return "_".join(parts[1:])
    
    af3_keys = [extract_pair_key(label) for label in af3_labels]
    boltz2_keys = [extract_pair_key(label) for label in boltz2_labels]
    
    # Find matching pairs
    matches = []
    af3_unmatched = []
    boltz2_unmatched = []
    
    for i, af3_key in enumerate(af3_keys):
        if af3_key in boltz2_keys:
            j = boltz2_keys.index(af3_key)
            matches.append((i, j))
        else:
            af3_unmatched.append(i)
    
    for j, boltz2_key in enumerate(boltz2_keys):
        if boltz2_key not in af3_keys:
            boltz2_unmatched.append(j)
    
    print(f"\n=== Correspondence Analysis for {title_suffix} ===")
    print(f"AF3 total points: {len(af3_data['x'])}")
    print(f"Boltz2 total points: {len(boltz2_data['x'])}")
    print(f"Matched pairs: {len(matches)}")
    print(f"AF3 unmatched: {len(af3_unmatched)}")
    print(f"Boltz2 unmatched: {len(boltz2_unmatched)}")
    
    if len(matches) == 0:
        print("No matching pairs found!")
        plt.rcParams.update(plt.rcParamsDefault)
        return
    
    # Prepare matched data for direct comparison
    af3_dist = []
    af3_struct = []
    boltz2_dist = []
    boltz2_struct = []
    bias_diffs = []
    total_hits_list = []
    
    for af3_idx, boltz2_idx in matches:
        af3_dist.append(af3_data["x"][af3_idx])
        af3_struct.append(af3_data["y"][af3_idx])
        boltz2_dist.append(boltz2_data["x"][boltz2_idx])
        boltz2_struct.append(boltz2_data["y"][boltz2_idx])
        # Average bias_ratio_diff
        bias_diffs.append((af3_data["bias_ratio_diff"][af3_idx] + 
                          boltz2_data["bias_ratio_diff"][boltz2_idx]) / 2)
        total_hits_list.append((af3_data["total_hits"][af3_idx] + 
                               boltz2_data["total_hits"][boltz2_idx]) / 2)
    
    af3_dist = np.array(af3_dist)
    af3_struct = np.array(af3_struct)
    boltz2_dist = np.array(boltz2_dist)
    boltz2_struct = np.array(boltz2_struct)
    bias_diffs = np.array(bias_diffs)
    total_hits_list = np.array(total_hits_list)
    
    # Create figure with 2 subplots (Dist and Struct comparison)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Color settings
    vmin, vmax = -1, 1
    cmap = cm.coolwarm
    
    # Scale point sizes
    sizes = np.clip(total_hits_list * 4, 30, 400)
    
    # Panel 1: Distogram comparison (AF3 vs Boltz2)
    ax1 = axes[0]
    scatter1 = ax1.scatter(af3_dist, boltz2_dist, c=bias_diffs, cmap=cmap, 
                          vmin=vmin, vmax=vmax, alpha=0.75, s=sizes,
                          edgecolors='black', linewidth=0.7)
    
    # Add diagonal line
    ax1.plot([-1.05, 1.05], [-1.05, 1.05], 'k--', alpha=0.5, linewidth=1.5, label='y=x')
    
    # Add center lines
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1.2, alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='-', linewidth=1.2, alpha=0.5)
    
    # Calculate correlation
    if len(af3_dist) > 1:
        pearson_r, _ = pearsonr(af3_dist, boltz2_dist)
        spearman_r, _ = spearmanr(af3_dist, boltz2_dist)
        
        # Add correlation text
        ax1.text(0.05, 0.95, f'Pearson r: {pearson_r:.3f}\nSpearman ρ: {spearman_r:.3f}',
                transform=ax1.transAxes, fontsize=14, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    ax1.set_xlabel(r'AlphaFold3 $\mathrm{Disto_{holo}}$', fontsize=18, fontweight='bold')
    ax1.set_ylabel(r'Boltz-2 $\mathrm{Disto_{holo}}$', fontsize=18, fontweight='bold')
    ax1.set_title(f'Distogram Score Comparison\n{title_suffix}', fontsize=18, fontweight='bold', pad=12)
    ax1.set_xlim(-1.05, 1.05)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax1.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax1.tick_params(axis='both', which='major', labelsize=15, length=6, width=1.5)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    
    # Panel 2: Structure comparison (AF3 vs Boltz2)
    ax2 = axes[1]
    scatter2 = ax2.scatter(af3_struct, boltz2_struct, c=bias_diffs, cmap=cmap,
                          vmin=vmin, vmax=vmax, alpha=0.75, s=sizes,
                          edgecolors='black', linewidth=0.7)
    
    # Add diagonal line
    ax2.plot([-1.05, 1.05], [-1.05, 1.05], 'k--', alpha=0.5, linewidth=1.5, label='y=x')
    
    # Add center lines
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1.2, alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='-', linewidth=1.2, alpha=0.5)
    
    # Calculate correlation
    if len(af3_struct) > 1:
        pearson_r, _ = pearsonr(af3_struct, boltz2_struct)
        spearman_r, _ = spearmanr(af3_struct, boltz2_struct)
        
        # Add correlation text
        ax2.text(0.05, 0.95, f'Pearson r: {pearson_r:.3f}\nSpearman ρ: {spearman_r:.3f}',
                transform=ax2.transAxes, fontsize=14, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    ax2.set_xlabel(r'AlphaFold3 $\mathrm{Struct_{holo}}$', fontsize=18, fontweight='bold')
    ax2.set_ylabel(r'Boltz-2 $\mathrm{Struct_{holo}}$', fontsize=18, fontweight='bold')
    ax2.set_title(f'Structure Score Comparison\n{title_suffix}', fontsize=18, fontweight='bold', pad=12)
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax2.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax2.tick_params(axis='both', which='major', labelsize=15, length=6, width=1.5)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)
    
    # Adjust layout before adding colorbar
    plt.tight_layout()
    
    # Add colorbar with better positioning
    cbar = fig.colorbar(scatter2, ax=axes, orientation='vertical', fraction=0.046, pad=0.04, aspect=30)
    cbar.set_label(r'$\mathrm{Train_{holo}}$', fontsize=18, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=15, width=1.5, length=6)
    cbar.outline.set_linewidth(1.5)
    
    filename_base = f"af3_vs_boltz2_direct_comparison_{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}"
    output_path_png = output_dir / f"{filename_base}.png"
    output_path_pdf = output_dir / f"{filename_base}.pdf"
    plt.savefig(output_path_png, dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"\nSaved direct comparison plot: {output_path_png}")
    print(f"Saved direct comparison plot (PDF): {output_path_pdf}")
    plt.close()
    
    # Reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)


if __name__ == "__main__":
    main()

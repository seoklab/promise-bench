#!/usr/bin/env python3
"""Compute per-pair training bias from weighted memorization intersection hits."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import click

from utils._config import eval_cfg as E

BETA_CHAIN = 0.5
BETA_INTERFACE = 1.0
SMOOTHING_CONSTANT = 0.0

CATEGORIES = (
    ("intrinsic", "intrinsic.csv"),
    ("ligand-induced", "ligand-induced.csv"),
    ("protein-induced", "protein-induced.csv"),
)
DEFAULT_MODELS = ("af3", "boltz_2", "chai_1", "bioemu")


def load_valid_pairs(valid_pairs_json: Path, category: str) -> dict:
    with valid_pairs_json.open() as handle:
        data = json.load(handle)
    return data.get(category, {})


def load_combinations_csv(csv_path: Path) -> dict[str, str]:
    entry_to_conf: dict[str, str] = {}
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entry_to_conf[f"{row['a_pdb']}_{row['a_assembly_id']}_{row['a_chain']}"] = row[
                "a_conf_label"
            ]
            entry_to_conf[f"{row['b_pdb']}_{row['b_assembly_id']}_{row['b_chain']}"] = row[
                "b_conf_label"
            ]
    return entry_to_conf


def parse_entry_name(entry_name: str) -> tuple[str | None, str | None, str | None, str | None]:
    parts = entry_name.rsplit("_", 1)
    if len(parts) != 2:
        return None, None, None, None
    core, suffix = parts
    core_parts = core.split("_")
    if len(core_parts) < 3:
        return None, None, None, None
    pdb = core_parts[0]
    assembly = core_parts[1]
    chain = "_".join(core_parts[2:])
    return pdb, assembly, chain, suffix


def chain_letter(chain: str) -> str:
    return "".join(char for char in chain if char.isalpha())


def lookup_chain_weight(chain_weights: dict[str, dict[str, float]], pdb: str, chain: str) -> float:
    record = chain_weights.get(pdb.lower()) or chain_weights.get(pdb.upper())
    if not record:
        return 0.0

    if chain in record:
        return float(record[chain])

    letter = chain_letter(chain)
    for key, weight in record.items():
        if chain_letter(str(key)) == letter:
            return float(weight)
    return 0.0


def interface_chain_letter_count(interface_key: str) -> int:
    """Return how many chains share the same chain letter in an interface key."""
    if "-" not in interface_key:
        return 1
    left, right = interface_key.split("-", 1)
    if chain_letter(left) == chain_letter(right):
        return 2
    return 1


def lookup_interface_weight(
    interface_weights: dict[str, dict[str, float]],
    pdb: str,
    interface_key: str,
) -> float:
    record = interface_weights.get(pdb.lower()) or interface_weights.get(pdb.upper())
    if not record:
        return 0.0
    if interface_key not in record:
        return 0.0
    weight = float(record[interface_key])
    return weight * interface_chain_letter_count(interface_key)


def parse_hit_target(target: str) -> tuple[str, str | None, str | None]:
    target = target.strip()
    if not target:
        return "", None, None
    if "_" not in target:
        return target.lower(), None, None
    pdb, chain = target.split("_", 1)
    if "-" in chain:
        return pdb.lower(), None, chain
    return pdb.lower(), chain, None


def hit_weight(
    chain_weights: dict[str, dict[str, float]],
    interface_weights: dict[str, dict[str, float]],
    target: str,
) -> float:
    pdb, chain, interface_key = parse_hit_target(target)
    if not pdb:
        return 0.0
    if interface_key is not None:
        return BETA_INTERFACE * lookup_interface_weight(
            interface_weights,
            pdb,
            interface_key,
        )
    if chain is None:
        return 0.0
    return BETA_CHAIN * lookup_chain_weight(chain_weights, pdb, chain)


def sum_conf_label_hits(
    cluster_dir: Path,
    conf_label: str,
    chain_weights: dict[str, dict[str, float]],
    interface_weights: dict[str, dict[str, float]],
) -> float:
    tsv_path = cluster_dir / f"conf_{conf_label}.tsv"
    if not tsv_path.exists():
        return 0.0

    total = 0.0
    with tsv_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            total += hit_weight(chain_weights, interface_weights, parts[1])
    return total


def calculate_smoothed_bias(hits1: float, hits2: float) -> float:
    total = hits1 + hits2 + SMOOTHING_CONSTANT
    if total == 0:
        return 0.0
    return (hits2 - hits1) / total


def calculate_bias_scores(
    valid_pairs_json: Path,
    csv_path: Path,
    memorization_base_dir: Path,
    model_name: str,
    category: str,
    chain_weights: dict[str, dict[str, float]],
    interface_weights: dict[str, dict[str, float]],
) -> list[dict]:
    valid_pairs_dict = load_valid_pairs(valid_pairs_json, category)
    entry_to_conf = load_combinations_csv(csv_path)
    memorization_dir = memorization_base_dir / model_name / category

    results: list[dict] = []
    total_pairs = sum(len(pairs) for pairs in valid_pairs_dict.values())
    processed = 0

    for cluster_name, pair_list in valid_pairs_dict.items():
        cluster_dir = memorization_dir / cluster_name
        if not cluster_dir.exists():
            continue

        for pair in pair_list:
            entry1, entry2 = pair[0], pair[1]
            pdb1, asm1, chain1, _ = parse_entry_name(entry1)
            pdb2, asm2, chain2, _ = parse_entry_name(entry2)
            if not pdb1 or not pdb2:
                click.echo(f"Warning: Could not parse entries: {entry1}, {entry2}")
                continue

            key1 = f"{pdb1}_{asm1}_{chain1}"
            key2 = f"{pdb2}_{asm2}_{chain2}"
            conf1 = entry_to_conf.get(key1)
            conf2 = entry_to_conf.get(key2)
            if conf1 is None or conf2 is None:
                click.echo(f"Warning: Conf label not found for {key1} or {key2}")
                continue

            hits1 = sum_conf_label_hits(cluster_dir, conf1, chain_weights, interface_weights)
            hits2 = sum_conf_label_hits(cluster_dir, conf2, chain_weights, interface_weights)
            smoothed_bias = calculate_smoothed_bias(hits1, hits2)
            total_hits = hits1 + hits2
            if total_hits == 0:
                ratio1 = ratio2 = ratio_diff = 0.0
            else:
                ratio1 = hits1 / total_hits
                ratio2 = hits2 / total_hits
                ratio_diff = ratio2 - ratio1

            results.append(
                {
                    "cluster_name": cluster_name,
                    "entry1": entry1,
                    "entry1_pdb": pdb1,
                    "entry1_conf_label": conf1,
                    "entry1_hits": hits1,
                    "hits_eff_a": hits1,
                    "entry1_ratio": round(ratio1, 4),
                    "entry2": entry2,
                    "entry2_pdb": pdb2,
                    "entry2_conf_label": conf2,
                    "entry2_hits": hits2,
                    "hits_eff_b": hits2,
                    "entry2_ratio": round(ratio2, 4),
                    "total_hits": total_hits,
                    "hits_eff": total_hits,
                    "ratio_difference": round(ratio_diff, 4),
                    "smoothed_bias": round(smoothed_bias, 4),
                    "train_bias_score": round(smoothed_bias, 4),
                }
            )
            processed += 1
            if processed % 100 == 0:
                click.echo(f"Processed {processed}/{total_pairs} pairs...")

    return results


def load_cluster_weights(weights_json: Path) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    with weights_json.open() as handle:
        data = json.load(handle)
    chain_weights = data.get("chain", {})
    interface_weights = data.get("interface", {})
    return chain_weights, interface_weights


@click.command()
@click.option(
    "--weights-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Precomputed cluster weights JSON (chain/interface sections).",
)
@click.option(
    "--valid-pairs-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=lambda: E.file("valid_pairs"),
    show_default="eval valid_pairs.json",
)
@click.option(
    "--combinations-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=lambda: Path(E.external("combinations_dir") or "data/combinations-final"),
    show_default="combinations-final",
)
@click.option(
    "--memorization-base-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=lambda: E.dir("memorization_hits_intersection"),
    show_default="eval memorization_hits_intersection",
)
@click.option("--tm-threshold", type=float, default=0.9, show_default=True)
@click.option("--fident-threshold", type=float, default=0.8, show_default=True)
@click.option(
    "--model",
    "models",
    multiple=True,
    type=click.Choice(DEFAULT_MODELS, case_sensitive=False),
    help="Model to process. Repeat to select multiple. Defaults to all models.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=lambda: E.dir("training_bias"),
    show_default="eval training_bias",
)
def main(
    weights_json: Path,
    valid_pairs_json: Path,
    combinations_dir: Path,
    memorization_base_dir: Path,
    tm_threshold: float,
    fident_threshold: float,
    models: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Sum weighted memorization hits per pair and write training-bias JSON."""
    selected_models = models or DEFAULT_MODELS
    threshold_dir = memorization_base_dir / f"hits_tm_{tm_threshold}_fident_{fident_threshold}"
    if not threshold_dir.exists():
        raise click.ClickException(f"Threshold directory not found: {threshold_dir}")

    chain_weights, interface_weights = load_cluster_weights(weights_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(
        f"Using chain beta={BETA_CHAIN}, interface beta={BETA_INTERFACE}, "
        f"smoothing constant={SMOOTHING_CONSTANT}"
    )

    for model in selected_models:
        click.echo(f"\nModel: {model}")
        model_results: dict[str, list[dict]] = {}

        for category, csv_filename in CATEGORIES:
            csv_path = combinations_dir / csv_filename
            if not csv_path.exists():
                click.echo(f"  Warning: CSV file not found: {csv_path}")
                continue

            click.echo(f"  Category: {category}")
            results = calculate_bias_scores(
                valid_pairs_json,
                csv_path,
                threshold_dir,
                model,
                category,
                chain_weights,
                interface_weights,
            )
            model_results[category] = results
            click.echo(f"    Total pairs processed: {len(results)}")

            if results:
                avg_bias = sum(row["smoothed_bias"] for row in results) / len(results)
                no_hits = sum(1 for row in results if row["total_hits"] == 0)
                click.echo(
                    f"    Smoothed bias: avg={avg_bias:.4f}, "
                    f"pairs with no hits: {no_hits} ({no_hits / len(results) * 100:.1f}%)"
                )

        output_path = output_dir / f"training_bias_per_pair_{model}.json"
        with output_path.open("w") as handle:
            json.dump(model_results, handle, indent=2)
        click.echo(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()

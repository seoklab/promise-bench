"""Generate figures from ``pair_level_results.json``.

Port of the core ``generate_unified_success_figure`` from
``dynamic_set/final/step8b_generate_figures.py``. The legacy script
bundled a dozen analyses (failure cases, intrinsic-dynamics
breakdowns, random-sampling Monte Carlo, etc.); this module starts
with the headline bar plot and exposes the supporting summary in
machine-readable form so other figures can be added incrementally.

Inputs
------
``pair_level_results.json`` produced by ``pair_results.py``. Promise-bench
labels the apo set as ``intrinsic`` (the legacy script used
``apo-monomers``).

Outputs
-------
- ``summary_statistics.json``
- ``figure_unified_success_rates.png`` + ``.pdf``

CLI
---
::

    PYTHONPATH=src uv run --project . \\
      python -m eval.plot.conformation_success.figures \\
        --results data_eval/plots/conformation_success/pair_level_results.json \\
        --output-dir data_eval/plots/conformation_success
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    METHODS,
    METHOD_DISPLAY_NAMES,
    SET_TYPES,
    SET_TYPE_COLORS,
    TM_THRESHOLD,
)

# Display labels for the figure x-axis groups. ``intrinsic`` keeps the
# canonical short label; the longer "Intrinsic Dynamics" string is used in
# the legend for visual continuity with the legacy figure.
SET_DISPLAY_NAMES: Dict[str, str] = {
    "intrinsic": "Intrinsic Dynamics",
    "ligand-induced": "Ligand-induced",
    "protein-induced": "Protein-Induced",
}


# ---------------------------------------------------------------------------
# IO + summary stats
# ---------------------------------------------------------------------------


def load_pair_level_results(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _intrinsic_success_rate(
    results: Dict[str, Any], method: str
) -> Tuple[float, int, int]:
    cmap = results.get("intrinsic", {}) or {}
    total = len(cmap)
    ok = sum(
        1
        for cid in cmap
        if (cmap[cid].get(method) or {}).get("success", False)
    )
    return (ok / total if total else 0.0, ok, total)


def _induced_cluster_avg_rate(
    results: Dict[str, Any], set_type: str, method: str
) -> Tuple[float, int, int, int]:
    """Return ``(cluster_avg_rate, n_clusters, total_pairs, total_successes)``.

    A cluster's rate = fraction of its pairs with ``overall_success``.
    The reported rate is the mean of per-cluster rates (matches legacy).
    """
    cmap = results.get(set_type, {}) or {}
    cluster_rates: List[float] = []
    total_pairs = 0
    total_successes = 0
    for cid in cmap:
        pairs = (cmap[cid] or {}).get(method) or []
        if not pairs:
            continue
        n_ok = sum(1 for p in pairs if p.get("overall_success"))
        cluster_rates.append(n_ok / len(pairs))
        total_pairs += len(pairs)
        total_successes += n_ok
    rate = float(np.mean(cluster_rates)) if cluster_rates else 0.0
    return rate, len(cluster_rates), total_pairs, total_successes


def compute_summary_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Build a serialisable summary mirroring the legacy structure.

    The returned dict has one entry per set_type, mapping method ->
    ``{success_rate, success_count, total_count, ...}``. Bioemu rows are
    filled with zeros for induced sets so the schema is rectangular.
    """
    out: Dict[str, Any] = {st: {} for st in SET_TYPES}

    for m in METHODS:
        rate, ok, total = _intrinsic_success_rate(results, m)
        out["intrinsic"][m] = {
            "success_rate": rate,
            "success_count": ok,
            "total_count": total,
        }

    for st in ("ligand-induced", "protein-induced"):
        for m in METHODS:
            if m == "bioemu":
                out[st][m] = {
                    "cluster_avg_rate": 0.0,
                    "n_clusters": 0,
                    "total_pairs": 0,
                    "total_successes": 0,
                }
                continue
            rate, nclu, tp, ts = _induced_cluster_avg_rate(results, st, m)
            out[st][m] = {
                "cluster_avg_rate": rate,
                "n_clusters": nclu,
                "total_pairs": tp,
                "total_successes": ts,
            }
    out["metadata"] = {
        "tm_threshold": TM_THRESHOLD,
        "methods": list(METHODS),
        "set_types": list(SET_TYPES),
    }
    return out


# ---------------------------------------------------------------------------
# Unified success-rate bar plot
# ---------------------------------------------------------------------------


def _rates_for_set(
    results: Dict[str, Any], set_type: str
) -> Tuple[List[float], List[Tuple[int, int]]]:
    """Return ``(rates, (numerator, denominator))`` for one set across METHODS.

    For intrinsic, ``numerator/denominator`` = clusters succeeded / total.
    For induced, ``numerator/denominator`` = (0, n_clusters_with_pairs) so
    the legend can show the cluster count even when individual successes
    are aggregated as a cluster-mean.
    """
    rates: List[float] = []
    counts: List[Tuple[int, int]] = []
    for m in METHODS:
        if set_type == "intrinsic":
            r, ok, total = _intrinsic_success_rate(results, m)
            rates.append(r)
            counts.append((ok, total))
        else:
            if m == "bioemu":
                rates.append(0.0)
                counts.append((0, 0))
                continue
            r, nclu, _, _ = _induced_cluster_avg_rate(results, set_type, m)
            rates.append(r)
            counts.append((0, nclu))
    return rates, counts


def unified_success_figure(
    results: Dict[str, Any],
    output_dir: Path,
    *,
    sampling_results: Optional[Dict[str, Any]] = None,
    stem: str = "figure_unified_success_rates",
) -> Path:
    """Emit a 3-group bar plot (intrinsic / ligand-induced / protein-induced).

    Returns the path to the saved PNG. PDF is saved alongside it.
    ``sampling_results`` is an optional dict shaped like the legacy
    ``analyze_random_sampling`` output: ``{<sampling_label>: {<set_type>:
    {<method>: rate}}}``. When provided, two marker types per bar are
    overlaid (legacy: ``10seeds_1model`` -> down-triangle red,
    ``1seed_10models`` -> diamond blue). Pass ``None`` (default) to omit
    sampling markers.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(METHODS))
    width = 0.25

    data_by_set = {st: _rates_for_set(results, st) for st in SET_TYPES}

    for i, st in enumerate(SET_TYPES):
        rates, counts = data_by_set[st]
        offset = width * (i - 1)
        bars = ax.bar(
            x + offset,
            rates,
            width,
            label=SET_DISPLAY_NAMES[st],
            color=SET_TYPE_COLORS[st],
            edgecolor="black",
            linewidth=1,
        )
        for j, (bar, rate, (_n_ok, n_total)) in enumerate(zip(bars, rates, counts)):
            if n_total <= 0:
                continue
            height = bar.get_height()
            bar_center = bar.get_x() + bar.get_width() / 2.0
            if i == 0 and j != 4:
                x_pos = bar_center - 0.02
            elif i == 2:
                x_pos = bar_center + 0.02
            else:
                x_pos = bar_center
            ax.text(
                x_pos,
                height + 0.02,
                f"{rate:.2f}",
                ha="center",
                va="bottom",
                fontsize=18,
                fontweight="bold",
            )

            if sampling_results:
                for key, marker, color, dx, size in (
                    ("10seeds_1model", "v", "red", -0.04, 60),
                    ("1seed_10models", "D", "blue", 0.04, 50),
                ):
                    r = (
                        (sampling_results.get(key) or {})
                        .get(st, {})
                        .get(METHODS[j])
                    )
                    if r is None or r < 0:
                        continue
                    y_pos = max(r, 0.02)
                    ax.scatter(
                        [bar_center + dx],
                        [y_pos],
                        marker=marker,
                        s=size,
                        c=color,
                        zorder=5,
                        edgecolors="black",
                        linewidth=0.5,
                    )

    ax.set_ylabel("Success Rate (Cluster Avg)", fontsize=25)
    ax.set_xticks(x)
    tick_labels = ax.set_xticklabels(
        [METHOD_DISPLAY_NAMES[m] for m in METHODS], fontsize=25
    )
    tick_labels[-1].set_horizontalalignment("right")
    tick_labels[-1].set_x(tick_labels[-1].get_position()[0] - 0.3)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.tick_params(axis="y", labelsize=20)

    # Legend
    n_intr = data_by_set["intrinsic"][1][0][1]
    n_lig = next((c[1] for c in data_by_set["ligand-induced"][1] if c[1] > 0), 0)
    n_pro = next((c[1] for c in data_by_set["protein-induced"][1] if c[1] > 0), 0)
    legend_elements = [
        mpatches.Patch(
            color=SET_TYPE_COLORS["intrinsic"],
            label=f"{SET_DISPLAY_NAMES['intrinsic']} (N={n_intr})",
        ),
        mpatches.Patch(
            color=SET_TYPE_COLORS["ligand-induced"],
            label=f"{SET_DISPLAY_NAMES['ligand-induced']}      (N={n_lig})",
        ),
        mpatches.Patch(
            color=SET_TYPE_COLORS["protein-induced"],
            label=f"{SET_DISPLAY_NAMES['protein-induced']}     (N={n_pro})",
        ),
    ]
    if sampling_results:
        legend_elements.extend(
            [
                plt.Line2D(
                    [0],
                    [0],
                    marker="v",
                    color="w",
                    markerfacecolor="red",
                    markersize=10,
                    label="10 seeds x 1 model",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="D",
                    color="w",
                    markerfacecolor="blue",
                    markersize=8,
                    label="1 seed x 10 models",
                ),
            ]
        )
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.80, 1),
        fontsize=25,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# Intrinsic failure breakdown (stacked bar of failure categories)
# ---------------------------------------------------------------------------

FAILURE_CATEGORY_COLORS: Dict[str, str] = {
    "Complete Failure": "#d3d3d3",
    "One-State Collapse": "#ffcccb",
    "Partial Collapse": "#ffeaa7",
    "Success": "#a8e6cf",
}


def categorise_intrinsic_cluster(cluster_info: Dict[str, Any]) -> str:
    """Return one of ``complete_failure / one_state / partial_collapse / success``.

    Categorisation is based on the number of covered conformation labels
    (``covered_confs``) vs the number expected (``expected_confs``):

    - ``complete_failure`` : ``n_covered == 0``
    - ``one_state``        : ``n_covered == 1``
    - ``partial_collapse`` : ``1 < n_covered < n_expected``
    - ``success``          : ``n_covered == n_expected`` (and >0)
    """
    exp = set(cluster_info.get("expected_confs") or [])
    cov = set(cluster_info.get("covered_confs") or [])
    n_exp = len(exp)
    n_cov = len(cov)
    if n_cov == 0:
        return "complete_failure"
    if n_cov == 1:
        return "one_state"
    if n_cov < n_exp:
        return "partial_collapse"
    return "success"


def intrinsic_failure_counts(results: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Per-method counts of each failure / success category for intrinsic clusters.

    Returns ``{method: {complete_failure, one_state, partial_collapse, success,
    n_failures, n_total}}``.
    """
    out: Dict[str, Dict[str, int]] = {}
    for method in METHODS:
        c_cf = c_one = c_par = c_suc = 0
        for cid, mmap in (results.get("intrinsic", {}) or {}).items():
            info = (mmap or {}).get(method) or {}
            if not info or info.get("no_data"):
                continue
            cat = categorise_intrinsic_cluster(info)
            if cat == "complete_failure":
                c_cf += 1
            elif cat == "one_state":
                c_one += 1
            elif cat == "partial_collapse":
                c_par += 1
            elif cat == "success":
                c_suc += 1
        out[method] = {
            "complete_failure": c_cf,
            "one_state": c_one,
            "partial_collapse": c_par,
            "success": c_suc,
            "n_failures": c_cf + c_one + c_par,
            "n_total": c_cf + c_one + c_par + c_suc,
        }
    return out


def intrinsic_failure_breakdown_figure(
    results: Dict[str, Any],
    output_dir: Path,
    *,
    stem: str = "figure_intrinsic_dynamics_failure",
) -> Path:
    """Emit a stacked bar of per-method failure categories (intrinsic only).

    Each bar's height is normalised to 1.0 using ``n_failures`` in the
    denominator (Success is shown above the bars as ``N=failures/total``,
    matching the legacy figure layout).
    """
    counts = intrinsic_failure_counts(results)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(METHODS))
    width = 0.6

    cf_rates: List[float] = []
    one_rates: List[float] = []
    par_rates: List[float] = []
    annotations: List[Tuple[int, int, int, int, int, int]] = []

    for m in METHODS:
        c = counts[m]
        nF = c["n_failures"]
        if nF > 0:
            cf_rates.append(c["complete_failure"] / nF)
            one_rates.append(c["one_state"] / nF)
            par_rates.append(c["partial_collapse"] / nF)
        else:
            cf_rates.append(0.0)
            one_rates.append(0.0)
            par_rates.append(0.0)
        annotations.append(
            (
                c["complete_failure"],
                c["one_state"],
                c["partial_collapse"],
                c["success"],
                c["n_total"],
                nF,
            )
        )

    ax.bar(
        x,
        cf_rates,
        width,
        label="Complete Failure",
        color=FAILURE_CATEGORY_COLORS["Complete Failure"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x,
        one_rates,
        width,
        bottom=cf_rates,
        label="One-State Collapse",
        color=FAILURE_CATEGORY_COLORS["One-State Collapse"],
        edgecolor="black",
        linewidth=0.5,
    )
    bottom_par = np.array(cf_rates) + np.array(one_rates)
    ax.bar(
        x,
        par_rates,
        width,
        bottom=bottom_par,
        label="Partial Collapse",
        color=FAILURE_CATEGORY_COLORS["Partial Collapse"],
        edgecolor="black",
        linewidth=0.5,
    )

    for j, (cf, one_, par, suc, total, nF) in enumerate(annotations):
        if nF == 0:
            ax.text(j, 0.5, "No\nFailures", ha="center", va="center", fontsize=12)
            continue
        ax.text(j, 1.02, f"N={nF}/{total}", ha="center", va="bottom", fontsize=12)
        if cf_rates[j] > 0.04:
            ax.text(
                j,
                cf_rates[j] / 2,
                f"{cf_rates[j]:.0%}\n({cf})",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
        if one_rates[j] > 0.04:
            ax.text(
                j,
                cf_rates[j] + one_rates[j] / 2,
                f"{one_rates[j]:.0%}\n({one_})",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

    ax.set_ylabel("Proportion of Failures", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY_NAMES[m] for m in METHODS], fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=14)

    plt.tight_layout()
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate conformation-success figures from pair_level_results.json"
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to pair_level_results.json (default: <conformation_success_dir>/pair_level_results.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dir (default: eval.dirs.conformation_success).",
    )
    parser.add_argument(
        "--sampling-results",
        type=str,
        default=None,
        help="Optional JSON with random-sampling overlay (legacy schema).",
    )
    parser.add_argument(
        "--skip-unified",
        action="store_true",
        help="Skip figure_unified_success_rates.",
    )
    parser.add_argument(
        "--skip-failure",
        action="store_true",
        help="Skip figure_intrinsic_dynamics_failure.",
    )
    args = parser.parse_args()

    from utils._config import eval_cfg as E

    output_dir = E.conformation_success_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = (
        Path(args.results)
        if args.results
        else (output_dir / "pair_level_results.json")
    )
    if not results_path.exists():
        raise FileNotFoundError(
            f"pair_level_results.json not found at {results_path}; run "
            "`python -m eval.plot.conformation_success.pair_results` first."
        )

    print(f"Loading {results_path}")
    results = load_pair_level_results(results_path)

    summary = compute_summary_statistics(results)
    summary_path = output_dir / "summary_statistics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path}")

    sampling = None
    if args.sampling_results:
        sampling = json.load(open(args.sampling_results))

    if not args.skip_unified:
        png = unified_success_figure(results, output_dir, sampling_results=sampling)
        print(f"Saved {png} (+.pdf)")

    if not args.skip_failure:
        png = intrinsic_failure_breakdown_figure(results, output_dir)
        print(f"Saved {png} (+.pdf)")
        # Also emit a small CSV summary mirroring legacy step8b.
        counts = intrinsic_failure_counts(results)
        import csv

        csv_path = output_dir / "intrinsic_dynamics_failure_summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "method",
                    "total_clusters",
                    "n_failures",
                    "n_success",
                    "n_complete_failure",
                    "n_one_state_collapse",
                    "n_partial_collapse",
                    "complete_failure_rate_among_failures",
                    "one_state_collapse_rate_among_failures",
                    "partial_collapse_rate_among_failures",
                ]
            )
            for m in METHODS:
                c = counts[m]
                nF = c["n_failures"]
                w.writerow(
                    [
                        m,
                        c["n_total"],
                        nF,
                        c["success"],
                        c["complete_failure"],
                        c["one_state"],
                        c["partial_collapse"],
                        c["complete_failure"] / nF if nF else 0.0,
                        c["one_state"] / nF if nF else 0.0,
                        c["partial_collapse"] / nF if nF else 0.0,
                    ]
                )
        print(f"Saved {csv_path}")

    print()
    print("Summary:")
    for st in SET_TYPES:
        print(f"  {st}:")
        for m in METHODS:
            row = summary[st][m]
            if st == "intrinsic":
                print(
                    f"    {m:<8} {row['success_rate']:.1%}  "
                    f"({row['success_count']}/{row['total_count']})"
                )
            else:
                print(
                    f"    {m:<8} cluster_avg={row['cluster_avg_rate']:.1%}  "
                    f"n_clusters={row['n_clusters']}  "
                    f"pairs={row['total_successes']}/{row['total_pairs']}"
                )


if __name__ == "__main__":
    _cli()

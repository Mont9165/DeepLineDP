"""
Aggregate LOPO fold results into per-scenario summary metrics.

Collects per-fold prediction CSVs, computes per-repo metrics, and
reports mean +/- SD with bootstrap 95% CIs for each scenario.

Usage:
    python lopo_aggregate.py \\
        --manifest ../output/lopo/manifest.json \\
        --output lopo_results.json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add project root to path for analyze_rq3_line imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analyze_rq3_line import (
    compute_bootstrap_ci,
    compute_line_level_metrics,
)


def _collect_fold_metrics(
    folds: List[Dict],
    output_dir: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load prediction CSVs and compute metrics for each fold.

    Returns:
        Dict mapping scenario name to list of per-fold result dicts.
    """
    scenario_results: Dict[str, List[Dict[str, Any]]] = {}

    for fold in folds:
        fold_id = fold["fold_id"]
        scenario = fold["scenario"]
        test_repo = fold["test_repo"]
        repo_slug = test_repo.replace("/", "__")

        pred_path = os.path.join(
            output_dir, "folds", scenario, repo_slug,
            "predictions", "predictions.csv",
        )

        # Check for skip marker
        skip_marker = os.path.join(
            output_dir, "folds", scenario, repo_slug, "SKIPPED_EMPTY_TEST",
        )

        if os.path.exists(skip_marker):
            continue

        if not os.path.exists(pred_path):
            print(f"  WARNING: Missing predictions for fold {fold_id} "
                  f"({scenario}/{test_repo})")
            continue

        pred_df = pd.read_csv(pred_path)
        if len(pred_df) == 0:
            print(f"  WARNING: Empty predictions for fold {fold_id}")
            continue

        metrics = compute_line_level_metrics(pred_df)
        if "error" in metrics:
            print(f"  WARNING: Fold {fold_id} error: {metrics['error']}")
            continue

        fold_result = {
            "fold_id": fold_id,
            "test_repo": test_repo,
            "n_test_files": metrics.get("total_files", 0),
            "n_test_lines": metrics.get("total_lines", 0),
            "n_buggy_lines": metrics.get("total_buggy_lines", 0),
            "metrics": {
                "recall_at_20_pct": metrics["recall_at_20_pct"],
                "effort_at_20_pct": metrics["effort_at_20_pct"],
                "ifa": metrics["ifa"],
                "file_level_auc": metrics.get("file_level_auc"),
                "file_level_balanced_accuracy": metrics.get("file_level_balanced_accuracy"),
                "file_level_mcc": metrics.get("file_level_mcc"),
            },
        }

        scenario_results.setdefault(scenario, []).append(fold_result)

    return scenario_results


def _aggregate_scenario(
    fold_results: List[Dict[str, Any]],
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """Aggregate per-fold metrics into summary statistics for one scenario.

    Computes mean, SD, median, and bootstrap 95% CI for each metric.
    """
    metric_keys = [
        "recall_at_20_pct", "effort_at_20_pct", "ifa",
        "file_level_auc", "file_level_balanced_accuracy", "file_level_mcc",
    ]

    # Collect per-repo metric values
    metric_values: Dict[str, List[float]] = {k: [] for k in metric_keys}
    for fr in fold_results:
        for k in metric_keys:
            v = fr["metrics"].get(k)
            if v is not None:
                metric_values[k].append(float(v))

    aggregated: Dict[str, Any] = {}
    for k in metric_keys:
        vals = metric_values[k]
        if not vals:
            aggregated[k] = {
                "mean": None, "sd": None, "median": None,
                "n": 0, "ci_lower": None, "ci_upper": None,
            }
            continue

        arr = np.array(vals)
        ci = compute_bootstrap_ci(vals, n_bootstrap=n_bootstrap)

        aggregated[k] = {
            "mean": round(float(np.mean(arr)), 4),
            "sd": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
            "median": round(float(np.median(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "n": len(arr),
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
        }

    return aggregated


def aggregate_lopo_results(
    manifest_path: str,
    output_dir: str = "",
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """Main aggregation: load manifest, compute per-fold metrics, aggregate.

    Returns:
        Full results dict suitable for JSON output.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    folds = manifest["folds"]
    summary = manifest["summary"]

    if not output_dir:
        output_dir = os.path.dirname(manifest_path)

    print(f"Aggregating LOPO results from {len(folds)} folds...")

    # Collect metrics
    scenario_results = _collect_fold_metrics(folds, output_dir)

    # Aggregate per scenario
    results: Dict[str, Any] = {}
    for scenario_name in sorted(scenario_results.keys()):
        fold_results = scenario_results[scenario_name]
        print(f"\n  {scenario_name}: {len(fold_results)} completed folds")

        agg = _aggregate_scenario(fold_results, n_bootstrap=n_bootstrap)

        results[scenario_name] = {
            "scenario": scenario_name,
            "n_folds_completed": len(fold_results),
            "n_folds_total": summary["per_scenario"].get(
                scenario_name, {}
            ).get("total", 0),
            "aggregated_metrics": agg,
            "per_fold": fold_results,
        }

    # Cross-scenario comparison using per-repo metrics
    results["cross_scenario_comparison"] = _cross_scenario_comparison(
        scenario_results
    )

    results["_lopo_summary"] = {
        **summary,
        "n_folds_with_results": sum(
            len(v) for v in scenario_results.values()
        ),
    }

    return results


def _cross_scenario_comparison(
    scenario_results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compare scenarios pairwise using per-repo metric distributions."""
    from itertools import combinations

    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return {"error": "scipy not available for statistical tests"}

    scenarios = sorted(scenario_results.keys())
    pairs = list(combinations(scenarios, 2))
    n_comparisons = max(len(pairs), 1)

    metric_keys = ["recall_at_20_pct", "effort_at_20_pct", "ifa"]

    comparisons: Dict[str, Dict[str, Any]] = {}
    for s_a, s_b in pairs:
        pair_key = f"{s_a} vs {s_b}"
        comparisons[pair_key] = {}

        for mk in metric_keys:
            vals_a = [
                fr["metrics"][mk]
                for fr in scenario_results[s_a]
                if fr["metrics"].get(mk) is not None
            ]
            vals_b = [
                fr["metrics"][mk]
                for fr in scenario_results[s_b]
                if fr["metrics"].get(mk) is not None
            ]

            if len(vals_a) < 2 or len(vals_b) < 2:
                comparisons[pair_key][mk] = {
                    "error": "Insufficient data",
                    "n_a": len(vals_a),
                    "n_b": len(vals_b),
                }
                continue

            stat, p_value = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            p_adjusted = min(p_value * n_comparisons, 1.0)

            # Cliff's delta
            n_a, n_b = len(vals_a), len(vals_b)
            more = sum(1 for a in vals_a for b in vals_b if a > b)
            less = sum(1 for a in vals_a for b in vals_b if a < b)
            delta = (more - less) / (n_a * n_b)
            abs_d = abs(delta)
            if abs_d < 0.147:
                interp = "negligible"
            elif abs_d < 0.33:
                interp = "small"
            elif abs_d < 0.474:
                interp = "medium"
            else:
                interp = "large"

            comparisons[pair_key][mk] = {
                "mann_whitney_u": round(float(stat), 4),
                "p_value": round(float(p_value), 6),
                "p_adjusted": round(float(p_adjusted), 6),
                "significant": p_adjusted < 0.05,
                "cliffs_delta": round(delta, 4),
                "effect_size": interp,
                "n_a": n_a,
                "n_b": n_b,
            }

    return {
        "pairwise": comparisons,
        "n_comparisons": n_comparisons,
        "correction": "bonferroni",
    }


def print_summary(results: Dict[str, Any]) -> None:
    """Print a summary table of aggregated LOPO results."""
    print("\n" + "=" * 80)
    print("LOPO Cross-Validation Results — RQ3 Line-Level Bug Prediction")
    print("=" * 80)

    header = (f"{'Scenario':<25} {'N':>4} "
              f"{'Recall@20%':>14} {'Effort@20%':>14} "
              f"{'IFA':>12} {'AUC':>12}")
    print(header)
    print("-" * 85)

    for name, result in results.items():
        if not isinstance(result, dict) or "aggregated_metrics" not in result:
            continue

        n = result.get("n_folds_completed", 0)
        agg = result["aggregated_metrics"]

        def _fmt(metric_dict):
            m = metric_dict.get("mean")
            s = metric_dict.get("sd")
            if m is None:
                return "N/A"
            if s is not None and s > 0:
                return f"{m:.4f}±{s:.4f}"
            return f"{m:.4f}"

        recall_s = _fmt(agg.get("recall_at_20_pct", {}))
        effort_s = _fmt(agg.get("effort_at_20_pct", {}))
        ifa_s = _fmt(agg.get("ifa", {}))
        auc_s = _fmt(agg.get("file_level_auc", {}))

        print(f"{name:<25} {n:>4} "
              f"{recall_s:>14} {effort_s:>14} "
              f"{ifa_s:>12} {auc_s:>12}")

    print()

    # Bootstrap CIs
    print("Bootstrap 95% CIs:")
    print(f"{'Scenario':<25} {'Metric':<16} {'Mean':>8} {'95% CI':>22}")
    print("-" * 75)

    for name, result in results.items():
        if not isinstance(result, dict) or "aggregated_metrics" not in result:
            continue
        agg = result["aggregated_metrics"]
        for mk, ml in [("recall_at_20_pct", "Recall@20%"),
                        ("effort_at_20_pct", "Effort@20%"),
                        ("ifa", "IFA"),
                        ("file_level_auc", "AUC")]:
            md = agg.get(mk, {})
            m = md.get("mean")
            lo = md.get("ci_lower")
            hi = md.get("ci_upper")
            if m is not None:
                ci_str = f"[{lo:.4f}, {hi:.4f}]" if lo is not None else "N/A"
                print(f"{name:<25} {ml:<16} {m:>8.4f} {ci_str:>22}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate LOPO fold results for RQ3"
    )
    parser.add_argument(
        "--manifest", "-m", required=True,
        help="Path to LOPO manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="LOPO output directory (default: manifest's parent dir)",
    )
    parser.add_argument(
        "--output", "-o",
        default="lopo_results.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Number of bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print JSON output",
    )

    args = parser.parse_args()

    results = aggregate_lopo_results(
        args.manifest,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
    )

    print_summary(results)

    indent = 2 if args.pretty else None
    with open(args.output, "w") as f:
        json.dump(results, f, indent=indent, default=str, ensure_ascii=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

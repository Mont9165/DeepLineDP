"""
Generate LOPO (Leave-One-Project-Out) fold manifest for RQ3 evaluation.

Reads existing preprocessed CSVs, extracts unique repositories per scenario,
and outputs a manifest JSON listing all (scenario, test_repo) pairs.

Cross-domain optimization:
  For human→agent: the training set is human_data. Leaving out a repo from
  agent_data's test set does NOT change the human training data unless that
  repo also appears in human_data. Repos that are agent-only share an
  identical training set → needs_retrain=false.

Usage:
    python lopo_generate_folds.py \\
        --data-dir ../datasets/preprocessed_data \\
        --output-dir ../output/lopo
"""

import argparse
import json
import os
from typing import Any, Dict, List, Set

import pandas as pd


# Scenario definitions — must match analyze_rq3_line.py
SCENARIOS = {
    "human_to_agent": {"train_data": "human_data", "test_data": "agent_data"},
    "agent_to_agent": {"train_data": "agent_data", "test_data": "agent_data"},
    "human_to_human": {"train_data": "human_data", "test_data": "human_data"},
    "agent_to_human": {"train_data": "agent_data", "test_data": "human_data"},
    "mixed_to_mixed": {"train_data": "mixed_data", "test_data": "mixed_data"},
}


def extract_repos(csv_path: str) -> Set[str]:
    """Extract unique repository names from a preprocessed CSV.

    Filenames are prefixed as ``owner/repo/path/to/file``.
    The repo is the first two path components.
    """
    df = pd.read_csv(csv_path, usecols=["filename"])
    repos = set()
    for fname in df["filename"].unique():
        parts = fname.split("/")
        if len(parts) >= 2:
            repos.add(f"{parts[0]}/{parts[1]}")
    return repos


def generate_manifest(data_dir: str) -> Dict[str, Any]:
    """Generate the LOPO fold manifest.

    Returns:
        Dict with keys: ``folds`` (list of fold dicts), ``summary``.
    """
    # Collect repos per data group
    group_repos: Dict[str, Dict[str, Set[str]]] = {}
    for group_name in ("agent_data", "human_data", "mixed_data"):
        group_repos[group_name] = {}
        for split in ("train", "eval", "test"):
            csv_path = os.path.join(data_dir, group_name, f"{split}.csv")
            if os.path.exists(csv_path):
                group_repos[group_name][split] = extract_repos(csv_path)
            else:
                group_repos[group_name][split] = set()

    # All repos present in each group (across all splits)
    all_repos_in_group: Dict[str, Set[str]] = {}
    for group_name in group_repos:
        all_repos_in_group[group_name] = set()
        for split_repos in group_repos[group_name].values():
            all_repos_in_group[group_name].update(split_repos)

    folds: List[Dict[str, Any]] = []
    fold_id = 0

    for scenario_name, scenario in SCENARIOS.items():
        train_source = scenario["train_data"]
        test_source = scenario["test_data"]

        # The test repos come from ALL splits of the test_source group
        # (in LOPO, every repo takes a turn as the test set)
        test_repos = sorted(all_repos_in_group.get(test_source, set()))

        if not test_repos:
            print(f"  WARNING: No repos found for scenario {scenario_name} "
                  f"(test_source={test_source})")
            continue

        # For cross-domain scenarios, determine which repos are shared
        # between train and test groups
        if train_source != test_source:
            train_all_repos = all_repos_in_group.get(train_source, set())
        else:
            train_all_repos = None  # same-domain: always needs retrain

        # Group folds by shared model (optimization)
        shared_group_id = 0
        shared_groups: Dict[str, int] = {}

        for repo in test_repos:
            if train_source != test_source:
                # Cross-domain: does this test repo appear in the training
                # data group? If not, removing it doesn't change training set.
                needs_retrain = repo in train_all_repos
            else:
                # Same-domain: leaving out a repo always changes the
                # training set
                needs_retrain = True

            # Assign shared model group for needs_retrain=False folds
            model_group = None
            if not needs_retrain:
                # All needs_retrain=False folds in this scenario share
                # the same training set
                group_key = f"{scenario_name}__shared"
                if group_key not in shared_groups:
                    shared_groups[group_key] = shared_group_id
                    shared_group_id += 1
                model_group = f"{scenario_name}_shared_{shared_groups[group_key]}"

            folds.append({
                "fold_id": fold_id,
                "scenario": scenario_name,
                "test_repo": repo,
                "train_source": train_source,
                "test_source": test_source,
                "needs_retrain": needs_retrain,
                "shared_model_group": model_group,
            })
            fold_id += 1

    # Summary statistics
    total = len(folds)
    retrain_count = sum(1 for f in folds if f["needs_retrain"])
    skip_count = total - retrain_count

    per_scenario = {}
    for f in folds:
        s = f["scenario"]
        if s not in per_scenario:
            per_scenario[s] = {"total": 0, "needs_retrain": 0, "skip": 0}
        per_scenario[s]["total"] += 1
        if f["needs_retrain"]:
            per_scenario[s]["needs_retrain"] += 1
        else:
            per_scenario[s]["skip"] += 1

    summary = {
        "total_folds": total,
        "needs_retrain": retrain_count,
        "shared_model_skip": skip_count,
        "estimated_training_runs": retrain_count + len(
            {f["shared_model_group"] for f in folds
             if f["shared_model_group"] is not None}
        ),
        "per_scenario": per_scenario,
        "data_dir": os.path.abspath(data_dir),
    }

    return {"folds": folds, "summary": summary}


def main():
    parser = argparse.ArgumentParser(
        description="Generate LOPO fold manifest for RQ3"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "datasets", "preprocessed_data",
        ),
        help="Base directory with agent_data/, human_data/, mixed_data/",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "output", "lopo",
        ),
        help="Output directory for manifest",
    )

    args = parser.parse_args()

    print("Generating LOPO fold manifest...")
    print(f"  Data dir: {args.data_dir}")

    manifest = generate_manifest(args.data_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    summary = manifest["summary"]
    print(f"\nManifest saved to {manifest_path}")
    print(f"  Total folds: {summary['total_folds']}")
    print(f"  Needs retrain: {summary['needs_retrain']}")
    print(f"  Shared model skip: {summary['shared_model_skip']}")
    print(f"  Estimated training runs: {summary['estimated_training_runs']}")

    print("\nPer-scenario breakdown:")
    for scenario, info in summary["per_scenario"].items():
        print(f"  {scenario}: {info['total']} folds "
              f"({info['needs_retrain']} retrain, {info['skip']} skip)")


if __name__ == "__main__":
    main()

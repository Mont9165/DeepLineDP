"""
Execute a single LOPO fold: build fold-specific CSVs and train/predict.

Reads the fold manifest, filters existing preprocessed CSVs to create
fold-specific train/eval/test splits, then calls ``train_pipeline.run_pipeline``
to train a model and generate predictions.

For ``needs_retrain=false`` folds (cross-domain optimization), expects
a pre-trained shared model and only runs the prediction stage.

Usage:
    # Single fold
    python lopo_run_fold.py --manifest ../output/lopo/manifest.json --fold-id 0

    # With shared Word2Vec
    python lopo_run_fold.py --manifest ../output/lopo/manifest.json --fold-id 0 \\
        --w2v-model ../output/lopo/shared_w2v/w2v-50dim.bin

    # Shared-model fold (prediction only)
    python lopo_run_fold.py --manifest ../output/lopo/manifest.json --fold-id 42 \\
        --shared-model-dir ../output/lopo/shared_models/human_to_agent_shared_0
"""

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from train_pipeline import run_pipeline


def _repo_from_filename(filename: str) -> str:
    """Extract ``owner/repo`` from a prefixed filename."""
    parts = filename.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return filename


def _load_full_csv(data_dir: str, group: str) -> pd.DataFrame:
    """Load and concatenate all split CSVs from a data group.

    Returns a single DataFrame with all rows from train + eval + test
    and an extra ``_original_split`` column.
    """
    dfs = []
    for split in ("train", "eval", "test"):
        path = os.path.join(data_dir, group, f"{split}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["_original_split"] = split
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No CSV files found in {os.path.join(data_dir, group)}"
        )
    return pd.concat(dfs, ignore_index=True)


def build_fold_csvs(
    fold: Dict,
    data_dir: str,
    output_dir: str,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[str, str]:
    """Build fold-specific train/eval/test CSVs.

    For same-domain scenarios (train_source == test_source):
      - test.csv: rows from repos matching test_repo
      - train.csv: ~(1-eval_ratio) of remaining rows
      - eval.csv: ~eval_ratio of remaining rows

    For cross-domain scenarios (train_source != test_source):
      - test.csv: rows from test_source matching test_repo
      - train.csv: all rows from train_source (excluding test_repo if present)
      - eval.csv: subset of train rows

    Returns:
        (fold_data_dir, fold_output_dir) paths.
    """
    scenario = fold["scenario"]
    test_repo = fold["test_repo"]
    train_source = fold["train_source"]
    test_source = fold["test_source"]

    # Create slug-safe directory name
    repo_slug = test_repo.replace("/", "__")
    fold_data_dir = os.path.join(output_dir, "folds", scenario, repo_slug, "data")
    fold_output_dir = os.path.join(output_dir, "folds", scenario, repo_slug)

    # Skip if predictions already exist
    pred_path = os.path.join(fold_output_dir, "predictions", "predictions.csv")
    if os.path.exists(pred_path):
        print(f"  Fold {fold['fold_id']}: predictions already exist, skipping CSV build")
        return fold_data_dir, fold_output_dir

    os.makedirs(fold_data_dir, exist_ok=True)

    rng = np.random.default_rng(seed=seed + fold["fold_id"])

    if train_source == test_source:
        # Same-domain LOPO
        full_df = _load_full_csv(data_dir, train_source)
        full_df["_repo"] = full_df["filename"].apply(_repo_from_filename)

        test_df = full_df[full_df["_repo"] == test_repo].drop(
            columns=["_original_split", "_repo"]
        )
        remaining = full_df[full_df["_repo"] != test_repo].copy()

        if len(test_df) == 0:
            print(f"  WARNING: No test data for repo {test_repo} in {test_source}")

        # Split remaining into train/eval by repo
        remaining_repos = sorted(remaining["_repo"].unique())
        if len(remaining_repos) < 2:
            print(f"  WARNING: Only {len(remaining_repos)} remaining repos "
                  f"after excluding {test_repo}. Using all for training.")
            eval_repo_set: set = set()
        else:
            n_eval_repos = max(1, int(len(remaining_repos) * eval_ratio))
            # Ensure at least 1 repo stays in train
            n_eval_repos = min(n_eval_repos, len(remaining_repos) - 1)
            eval_repo_set = set(
                rng.choice(remaining_repos, size=n_eval_repos, replace=False).tolist()
            )

        eval_df = remaining[remaining["_repo"].isin(eval_repo_set)].drop(
            columns=["_original_split", "_repo"]
        )
        train_df = remaining[~remaining["_repo"].isin(eval_repo_set)].drop(
            columns=["_original_split", "_repo"]
        )
    else:
        # Cross-domain LOPO
        test_full = _load_full_csv(data_dir, test_source)
        test_full["_repo"] = test_full["filename"].apply(_repo_from_filename)
        test_df = test_full[test_full["_repo"] == test_repo].drop(
            columns=["_original_split", "_repo"]
        )

        train_full = _load_full_csv(data_dir, train_source)
        train_full["_repo"] = train_full["filename"].apply(_repo_from_filename)

        # Exclude test_repo from training if it appears there
        remaining = train_full[train_full["_repo"] != test_repo].copy()

        remaining_repos = sorted(remaining["_repo"].unique())
        if len(remaining_repos) < 2:
            print(f"  WARNING: Only {len(remaining_repos)} remaining repos "
                  f"in {train_source}. Using all for training.")
            eval_repo_set = set()
        else:
            n_eval_repos = max(1, int(len(remaining_repos) * eval_ratio))
            n_eval_repos = min(n_eval_repos, len(remaining_repos) - 1)
            eval_repo_set = set(
                rng.choice(remaining_repos, size=n_eval_repos, replace=False).tolist()
            )

        eval_df = remaining[remaining["_repo"].isin(eval_repo_set)].drop(
            columns=["_original_split", "_repo"]
        )
        train_df = remaining[~remaining["_repo"].isin(eval_repo_set)].drop(
            columns=["_original_split", "_repo"]
        )

    # Write CSVs
    train_df.to_csv(os.path.join(fold_data_dir, "train.csv"), index=False)
    eval_df.to_csv(os.path.join(fold_data_dir, "eval.csv"), index=False)
    test_df.to_csv(os.path.join(fold_data_dir, "test.csv"), index=False)

    print(f"  Fold {fold['fold_id']} [{scenario}] test_repo={test_repo}: "
          f"train={len(train_df)} rows, eval={len(eval_df)} rows, "
          f"test={len(test_df)} rows")

    return fold_data_dir, fold_output_dir


def run_fold(
    fold: Dict,
    data_dir: str,
    output_dir: str,
    w2v_model: str = "",
    shared_model_dir: str = "",
    num_epochs: int = 10,
) -> None:
    """Run a single LOPO fold: build CSVs, train, predict.

    Args:
        fold: Fold dict from manifest.
        data_dir: Base preprocessed data directory.
        output_dir: LOPO output root (e.g., output/lopo).
        w2v_model: Path to pre-trained shared Word2Vec model.
        shared_model_dir: For needs_retrain=False folds, path to the
            pre-trained shared model directory containing model/ subdir.
        num_epochs: Training epochs.
    """
    fold_id = fold["fold_id"]
    scenario = fold["scenario"]
    test_repo = fold["test_repo"]
    needs_retrain = fold["needs_retrain"]

    # Check if predictions already exist
    repo_slug = test_repo.replace("/", "__")
    pred_path = os.path.join(
        output_dir, "folds", scenario, repo_slug,
        "predictions", "predictions.csv"
    )
    if os.path.exists(pred_path):
        print(f"Fold {fold_id}: predictions already exist at {pred_path}, skipping")
        return

    print(f"\n{'='*60}")
    print(f"Fold {fold_id}: {scenario} | test_repo={test_repo} "
          f"| retrain={needs_retrain}")
    print(f"{'='*60}")

    fold_data_dir, fold_output_dir = build_fold_csvs(
        fold, data_dir, output_dir,
    )

    # Verify test CSV has data
    test_csv = os.path.join(fold_data_dir, "test.csv")
    if not os.path.exists(test_csv):
        print(f"  ERROR: test.csv not found at {test_csv}")
        return
    test_df = pd.read_csv(test_csv)
    if len(test_df) == 0:
        print(f"  SKIP: Empty test set for repo {test_repo}")
        # Write empty marker
        marker = os.path.join(fold_output_dir, "SKIPPED_EMPTY_TEST")
        with open(marker, "w") as f:
            f.write(f"Empty test set for {test_repo}\n")
        return

    if not needs_retrain and not w2v_model:
        print(f"  ERROR: Fold {fold_id} has needs_retrain=False but no "
              f"--w2v-model was provided. Shared-model folds require a "
              f"pre-trained Word2Vec model.")
        sys.exit(1)

    if not needs_retrain and shared_model_dir:
        # Use shared model — only run prediction stage
        print(f"  Using shared model from {shared_model_dir}")

        # Symlink or copy model directory
        fold_model_dir = os.path.join(fold_output_dir, "model")
        shared_src = os.path.join(shared_model_dir, "model")
        if not os.path.exists(fold_model_dir) and os.path.exists(shared_src):
            os.makedirs(os.path.dirname(fold_model_dir), exist_ok=True)
            os.symlink(os.path.abspath(shared_src), fold_model_dir)

        # Run prediction only
        run_pipeline(
            data_dir=fold_data_dir,
            output_base=fold_output_dir,
            test_data_dir=fold_data_dir,
            stage="predict",
            w2v_model=w2v_model,
        )
    else:
        # Full training + prediction
        run_pipeline(
            data_dir=fold_data_dir,
            output_base=fold_output_dir,
            test_data_dir=fold_data_dir,
            stage="all",
            num_epochs=num_epochs,
            w2v_model=w2v_model,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run a single LOPO fold for RQ3"
    )
    parser.add_argument(
        "--manifest", "-m", required=True,
        help="Path to LOPO manifest JSON",
    )
    parser.add_argument(
        "--fold-id", "-f", type=int, required=True,
        help="Fold ID to run (from manifest)",
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="",
        help="Override base data directory (default: from manifest)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="",
        help="LOPO output directory (default: manifest's parent dir)",
    )
    parser.add_argument(
        "--w2v-model",
        default="",
        help="Path to pre-trained shared Word2Vec model",
    )
    parser.add_argument(
        "--shared-model-dir",
        default="",
        help="For needs_retrain=False folds: path to shared model dir",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=10,
        help="Training epochs (default: 10)",
    )

    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    folds = manifest["folds"]
    summary = manifest["summary"]

    # Find the fold
    fold = None
    for f in folds:
        if f["fold_id"] == args.fold_id:
            fold = f
            break

    if fold is None:
        print(f"ERROR: Fold ID {args.fold_id} not found in manifest "
              f"(total folds: {len(folds)})")
        sys.exit(1)

    data_dir = args.data_dir or summary.get("data_dir", "")
    if not data_dir:
        print("ERROR: No data directory specified (--data-dir or manifest)")
        sys.exit(1)
    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory does not exist: {data_dir}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(args.manifest)

    run_fold(
        fold,
        data_dir=data_dir,
        output_dir=output_dir,
        w2v_model=args.w2v_model,
        shared_model_dir=args.shared_model_dir,
        num_epochs=args.num_epochs,
    )

    print(f"\nFold {args.fold_id} completed.")


if __name__ == "__main__":
    main()

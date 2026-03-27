"""Generate predictions on eval (validation) data for each LOPO fold.

For each fold, loads the best-epoch checkpoint and runs inference on the
eval split.  The resulting predictions are saved alongside the existing
test predictions so that lopo_aggregate.py can use the eval predictions
to determine per-fold optimal thresholds without data leakage.

Usage:
    python generate_eval_predictions.py \
        --manifest ../output/lopo/manifest.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from DeepLineDP_model import HierarchicalAttentionNetwork
from my_util import prepare_code2d, get_x_vec


def load_split(data_dir: str, split: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{split}.csv")
    return pd.read_csv(path)


def generate_eval_predictions_for_fold(
    fold_dir: str,
    w2v_path: str,
    embed_dim: int = 50,
    device: torch.device = None,
) -> bool:
    """Generate eval predictions for a single fold. Returns True if successful."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(fold_dir, "data")
    model_dir = os.path.join(fold_dir, "model")
    pred_dir = os.path.join(fold_dir, "predictions")

    eval_path = os.path.join(data_dir, "eval.csv")
    output_path = os.path.join(pred_dir, "eval_predictions.csv")

    if os.path.exists(output_path):
        return True  # Already generated

    if not os.path.exists(eval_path):
        return False

    # Find best checkpoint (lowest validation loss)
    loss_path = os.path.join(model_dir, "loss", "loss_record.csv")
    if not os.path.exists(loss_path):
        return False

    loss_df = pd.read_csv(loss_path)
    best_epoch = int(loss_df.loc[loss_df["valid_loss"].idxmin(), "epoch"])
    model_path = os.path.join(model_dir, f"checkpoint_{best_epoch}epochs.pth")

    if not os.path.exists(model_path):
        return False

    # Load Word2Vec
    word2vec = Word2Vec.load(w2v_path)
    vocab_size = len(word2vec.wv) + 1

    # Load model
    model = HierarchicalAttentionNetwork(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        word_gru_hidden_dim=64,
        sent_gru_hidden_dim=64,
        word_gru_num_layers=1,
        sent_gru_num_layers=1,
        word_att_dim=64,
        sent_att_dim=64,
        use_layer_norm=True,
        dropout=0.2,
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load eval data
    eval_df = load_split(data_dir, "eval")
    eval_df["code_line"] = eval_df["code_line"].fillna("").astype(str)

    row_list = []
    for filename, df in eval_df.groupby("filename"):
        file_label = bool(df["file-label"].any())

        code = df["code_line"].tolist()
        code2d = prepare_code2d(code, to_lowercase=True)
        code3d = [code2d]
        codevec = get_x_vec(code3d, word2vec)

        try:
            with torch.no_grad():
                codevec_tensor = torch.tensor(codevec).to(device)
                output, _, __, ___ = model(codevec_tensor)
                file_prob = output.item()
        except RuntimeError:
            # GPU OOM/cuDNN error on large files — retry on CPU
            torch.cuda.empty_cache()
            try:
                cpu = torch.device("cpu")
                model_cpu = model.to(cpu)
                with torch.no_grad():
                    codevec_tensor = torch.tensor(codevec).to(cpu)
                    output, _, __, ___ = model_cpu(codevec_tensor)
                    file_prob = output.item()
                model.to(device)  # move back to GPU
            except Exception:
                continue  # skip if CPU also fails

        row_list.append({
            "filename": filename,
            "file-level-ground-truth": file_label,
            "prediction-prob": file_prob,
        })

    if row_list:
        result_df = pd.DataFrame(row_list)
        os.makedirs(pred_dir, exist_ok=True)
        result_df.to_csv(output_path, index=False)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--embed-dim", type=int, default=50)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    folds = manifest["folds"]
    output_dir = os.path.dirname(args.manifest)
    w2v_path = os.path.join(output_dir, "shared_w2v", "w2v-50dim.bin")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scenarios = {}
    for fold in folds:
        s = fold["scenario"]
        scenarios.setdefault(s, []).append(fold)

    total = 0
    success = 0
    skipped = 0

    for scenario, scenario_folds in scenarios.items():
        print(f"\n=== {scenario} ({len(scenario_folds)} folds) ===")
        for fold in tqdm(scenario_folds, desc=scenario):
            repo_slug = fold["test_repo"].replace("/", "__")
            fold_dir = os.path.join(output_dir, "folds", scenario, repo_slug)

            skip_marker = os.path.join(fold_dir, "SKIPPED_EMPTY_TEST")
            if os.path.exists(skip_marker):
                skipped += 1
                continue

            total += 1
            if generate_eval_predictions_for_fold(
                fold_dir, w2v_path, args.embed_dim, device,
            ):
                success += 1

    print(f"\nDone: {success}/{total} folds generated ({skipped} skipped)")


if __name__ == "__main__":
    main()

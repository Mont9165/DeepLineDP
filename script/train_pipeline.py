"""
Training pipeline for DeepLineDP with LLM4SZZ datasets.

Handles Word2Vec training, model training, and prediction generation
for cross-domain (Agent vs Human) experiments.

Unlike the original scripts which use hardcoded Java project releases,
this pipeline works with arbitrary dataset splits from build_dataset_from_llm4szz.py.

Usage:
    # Full pipeline: word2vec + train + predict
    python train_pipeline.py --data-dir ../datasets/preprocessed_data/mixed_data

    # Word2Vec only
    python train_pipeline.py --data-dir ../datasets/preprocessed_data/mixed_data --stage w2v

    # Train only (requires existing Word2Vec)
    python train_pipeline.py --data-dir ../datasets/preprocessed_data/mixed_data --stage train

    # Predict only (requires existing model)
    python train_pipeline.py --data-dir ../datasets/preprocessed_data/mixed_data --stage predict

    # Cross-domain: train on human data, predict on agent data
    python train_pipeline.py --data-dir ../datasets/preprocessed_data/human_data \\
        --test-data-dir ../datasets/preprocessed_data/agent_data
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
from sklearn.utils import compute_class_weight
from tqdm import tqdm

# Import model and utilities from original DeepLineDP
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from DeepLineDP_model import HierarchicalAttentionNetwork
from my_util import (
    get_w2v_weight_for_deep_learning_models,
    get_x_vec,
    prepare_code2d,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(data_dir: str, split: str) -> pd.DataFrame:
    """Load a preprocessed CSV split (train/eval/test)."""
    path = os.path.join(data_dir, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path)
    df = df.fillna("")
    df = df[~df["is_blank"]]
    df = df[~df["is_test_file"]]
    return df


def get_code3d_and_label(df: pd.DataFrame, to_lowercase: bool = False):
    """Convert DataFrame to 3D code representation + file labels."""
    code3d = []
    all_file_label = []

    for filename, group_df in df.groupby("filename"):
        file_label = bool(group_df["file-label"].any())

        code = list(group_df["code_line"])
        code2d = prepare_code2d(code, to_lowercase)
        code3d.append(code2d)
        all_file_label.append(file_label)

    return code3d, all_file_label


# ---------------------------------------------------------------------------
# Word2Vec
# ---------------------------------------------------------------------------

def train_word2vec(data_dir: str, output_dir: str, embed_dim: int = 50) -> str:
    """Train Word2Vec on the training split."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"w2v-{embed_dim}dim.bin")

    if os.path.exists(save_path):
        print(f"Word2Vec model already exists: {save_path}")
        return save_path

    print("Training Word2Vec model...")
    train_df = load_split(data_dir, "train")
    code3d, _ = get_code3d_and_label(train_df, to_lowercase=True)

    # Flatten to list of token lists
    all_texts = []
    for file_code in code3d:
        all_texts.extend(file_code)

    w2v = Word2Vec(all_texts, vector_size=embed_dim, min_count=1)
    w2v.save(save_path)
    print(f"Word2Vec saved: {save_path} (vocab={len(w2v.wv)})")
    return save_path


def train_shared_word2vec(
    data_dirs: list,
    output_dir: str,
    embed_dim: int = 50,
) -> str:
    """Train a shared Word2Vec model from multiple data directories.

    Combines training splits from all provided directories so that cross-domain
    scenarios share a common vocabulary. This reduces OOV rates when a model
    trained on one author group is tested on another.

    Word2Vec only learns distributional semantics (token co-occurrence) and does
    not use bug labels, so sharing the vocabulary across groups does not leak
    label information.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"w2v-{embed_dim}dim.bin")

    if os.path.exists(save_path):
        print(f"Shared Word2Vec model already exists: {save_path}")
        return save_path

    print("Training shared Word2Vec model...")
    all_texts = []
    dirs_loaded = 0
    for data_dir in data_dirs:
        train_path = os.path.join(data_dir, "train.csv")
        if not os.path.exists(train_path):
            print(f"  Warning: skipping {data_dir} (no train.csv)")
            continue
        train_df = load_split(data_dir, "train")
        code3d, _ = get_code3d_and_label(train_df, to_lowercase=True)
        for file_code in code3d:
            all_texts.extend(file_code)
        dirs_loaded += 1
        print(f"  Loaded {data_dir}: {len(code3d)} files")

    if not all_texts:
        raise ValueError("No training data found in any of the provided directories")
    if dirs_loaded < 2:
        print(f"  WARNING: Only {dirs_loaded} directory contributed data. "
              "The 'shared' model is effectively a single-group model.")

    w2v = Word2Vec(all_texts, vector_size=embed_dim, min_count=1)
    w2v.save(save_path)
    print(f"Shared Word2Vec saved: {save_path} (vocab={len(w2v.wv)})")
    return save_path


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(
    data_dir: str,
    w2v_path: str,
    model_dir: str,
    embed_dim: int = 50,
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 0.001,
    dropout: float = 0.2,
    max_train_loc: int = 900,
    device: torch.device = None,
) -> str:
    """Train DeepLineDP model. Returns path to best checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    os.makedirs(model_dir, exist_ok=True)
    loss_dir = os.path.join(model_dir, "loss")
    os.makedirs(loss_dir, exist_ok=True)

    # Load data
    print("Loading training data...")
    train_df = load_split(data_dir, "train")
    train_code3d, train_label = get_code3d_and_label(train_df, to_lowercase=True)

    print("Loading validation data...")
    eval_df = load_split(data_dir, "eval")
    eval_code3d, eval_label = get_code3d_and_label(eval_df, to_lowercase=True)

    print(f"Train: {len(train_code3d)} files, Eval: {len(eval_code3d)} files")
    print(f"Train defect ratio: {sum(train_label)}/{len(train_label)} "
          f"({sum(train_label)/max(len(train_label),1)*100:.1f}%)")

    # Class weights
    sample_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_label),
        y=train_label,
    )
    weight_dict = {
        "defect": float(np.max(sample_weights)),
        "clean": float(np.min(sample_weights)),
    }

    # Word2Vec
    word2vec = Word2Vec.load(w2v_path)
    vocab_size = len(word2vec.wv) + 1
    w2v_weights = get_w2v_weight_for_deep_learning_models(word2vec, embed_dim, device)

    # Vectorize
    x_train_vec = get_x_vec(train_code3d, word2vec)
    x_eval_vec = get_x_vec(eval_code3d, word2vec)

    max_sent_len = min(max(len(s) for s in x_train_vec), max_train_loc)

    # Dataloaders
    from my_util import get_dataloader
    train_dl = get_dataloader(x_train_vec, train_label, batch_size, max_sent_len, device)
    eval_dl = get_dataloader(x_eval_vec, eval_label, batch_size, max_sent_len, device)

    # Model
    word_gru_hidden_dim = 64
    sent_gru_hidden_dim = 64

    model = HierarchicalAttentionNetwork(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        word_gru_hidden_dim=word_gru_hidden_dim,
        sent_gru_hidden_dim=sent_gru_hidden_dim,
        word_gru_num_layers=1,
        sent_gru_num_layers=1,
        word_att_dim=64,
        sent_att_dim=64,
        use_layer_norm=True,
        dropout=dropout,
    ).to(device)

    # Check for existing checkpoints
    checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    start_epoch = 1
    train_losses_all = []
    val_losses_all = []

    if checkpoint_files:
        nums = [int(re.findall(r"\d+", f)[0]) for f in checkpoint_files]
        start_epoch = max(nums) + 1
        ckpt = torch.load(
            os.path.join(model_dir, f"checkpoint_{max(nums)}epochs.pth"),
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])

        loss_csv = os.path.join(loss_dir, "loss_record.csv")
        if os.path.exists(loss_csv):
            loss_df = pd.read_csv(loss_csv)
            train_losses_all = list(loss_df["train_loss"])
            val_losses_all = list(loss_df["valid_loss"])
        print(f"Resuming from epoch {start_epoch}")
    else:
        model.sent_attention.word_attention.init_embeddings(w2v_weights)

    model.sent_attention.word_attention.freeze_embeddings(False)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    if checkpoint_files:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="Training"):
        # Train
        model.train()
        train_losses = []
        for inputs, labels in train_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            output, _, __, ___ = model(inputs)

            # Apply class weights
            weight_list = [
                weight_dict["clean"] if lab == 0 else weight_dict["defect"]
                for lab in labels.cpu().numpy().squeeze().tolist()
            ]
            criterion.weight = torch.tensor(weight_list).reshape(-1, 1).to(device)

            loss = criterion(output, labels.reshape(-1, 1))
            train_losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()

        train_losses_all.append(np.mean(train_losses))

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            criterion.weight = None
            for inputs, labels in eval_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                output, _, __, ___ = model(inputs)
                val_loss = criterion(output, labels.reshape(-1, 1))
                val_losses.append(val_loss.item())

        val_loss_mean = np.mean(val_losses) if val_losses else float("inf")
        val_losses_all.append(val_loss_mean)

        # Save checkpoint
        ckpt_path = os.path.join(model_dir, f"checkpoint_{epoch}epochs.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)

        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            best_epoch = epoch

        # Save loss record
        loss_df = pd.DataFrame({
            "epoch": np.arange(1, len(train_losses_all) + 1),
            "train_loss": train_losses_all,
            "valid_loss": val_losses_all,
        })
        loss_df.to_csv(os.path.join(loss_dir, "loss_record.csv"), index=False)

        print(f"  Epoch {epoch}: train_loss={train_losses_all[-1]:.4f}, "
              f"val_loss={val_loss_mean:.4f}")

    print(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")
    return os.path.join(model_dir, f"checkpoint_{best_epoch}epochs.pth")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def generate_predictions(
    test_data_dir: str,
    model_path: str,
    w2v_path: str,
    output_dir: str,
    embed_dim: int = 50,
    device: torch.device = None,
) -> str:
    """Generate predictions on test data. Returns path to prediction CSV."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

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
    model.sent_attention.word_attention.freeze_embeddings(True)
    model = model.to(device)
    model.eval()

    # Load test data
    test_df = load_split(test_data_dir, "test")

    row_list = []

    for filename, df in tqdm(test_df.groupby("filename"), desc="Predicting"):
        file_label = bool(df["file-label"].any())
        line_labels = df["line-label"].tolist()
        line_numbers = df["line_number"].tolist()
        is_comments = df["is_comment"].tolist()
        code = df["code_line"].tolist()

        code2d = prepare_code2d(code, to_lowercase=True)
        code3d = [code2d]
        codevec = get_x_vec(code3d, word2vec)

        with torch.no_grad():
            codevec_tensor = torch.tensor(codevec).to(device)
            output, word_att_weights, line_att_weight, _ = model(codevec_tensor)
            file_prob = output.item()
            prediction = bool(round(output.item()))

        numpy_word_attn = word_att_weights[0].cpu().detach().numpy()
        numpy_line_attn = line_att_weight[0].cpu().detach().numpy()

        for i in range(len(code)):
            cur_line = code[i]
            token_list = cur_line.strip().split()
            max_len = min(len(token_list), 50)

            cur_line_attn = numpy_line_attn[i] if i < len(numpy_line_attn) else 0.0

            for j in range(max_len):
                word_attn = numpy_word_attn[i][j] if i < len(numpy_word_attn) and j < len(numpy_word_attn[i]) else 0.0
                row_list.append({
                    "filename": filename,
                    "file-level-ground-truth": file_label,
                    "prediction-prob": file_prob,
                    "prediction-label": prediction,
                    "line-number": line_numbers[i],
                    "line-level-ground-truth": line_labels[i],
                    "is-comment-line": is_comments[i],
                    "token": token_list[j],
                    "token-attention-score": float(word_attn),
                    "line-attention-score": float(cur_line_attn),
                })

    pred_df = pd.DataFrame(row_list)
    output_path = os.path.join(output_dir, "predictions.csv")
    pred_df.to_csv(output_path, index=False)
    print(f"Predictions saved: {output_path} ({len(pred_df)} rows, "
          f"{pred_df['filename'].nunique()} files)")
    return output_path


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data_dir: str,
    output_base: str,
    test_data_dir: str = "",
    stage: str = "all",
    embed_dim: int = 50,
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 0.001,
    dropout: float = 0.2,
    w2v_model: str = "",
):
    """Run the full training pipeline or a specific stage.

    Args:
        w2v_model: Path to a pre-trained Word2Vec model. If provided, skip
            W2V training and use this model instead (e.g. a shared model
            trained on all author groups).
    """
    if not test_data_dir:
        test_data_dir = data_dir

    w2v_dir = os.path.join(output_base, "Word2Vec_model")
    model_dir = os.path.join(output_base, "model")
    pred_dir = os.path.join(output_base, "predictions")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model_path = ""

    if w2v_model:
        w2v_path = w2v_model
        if stage in ("all", "w2v"):
            print(f"Using pre-trained Word2Vec: {w2v_model}")
    elif stage in ("all", "w2v"):
        w2v_path = train_word2vec(data_dir, w2v_dir, embed_dim)
    else:
        w2v_path = os.path.join(w2v_dir, f"w2v-{embed_dim}dim.bin")

    if stage in ("all", "train"):
        best_model_path = train_model(
            data_dir, w2v_path, model_dir,
            embed_dim=embed_dim,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            dropout=dropout,
            device=device,
        )

    if stage in ("all", "predict"):
        if not best_model_path:
            # Find best checkpoint
            ckpts = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
            if not ckpts:
                print("Error: No model checkpoints found")
                return
            # Use last epoch by default
            nums = [int(re.findall(r"\d+", f)[0]) for f in ckpts]
            best_model_path = os.path.join(model_dir, f"checkpoint_{max(nums)}epochs.pth")

        generate_predictions(
            test_data_dir, best_model_path, w2v_path, pred_dir,
            embed_dim=embed_dim,
            device=device,
        )


def main():
    parser = argparse.ArgumentParser(
        description="DeepLineDP training pipeline for LLM4SZZ datasets"
    )
    parser.add_argument(
        "--data-dir", "-d", required=True,
        help="Directory with train.csv and eval.csv",
    )
    parser.add_argument(
        "--test-data-dir", "-t", default="",
        help="Directory with test.csv (default: same as --data-dir)",
    )
    parser.add_argument(
        "--output", "-o", default="../output",
        help="Output base directory (default: ../output)",
    )
    parser.add_argument(
        "--stage", choices=["all", "w2v", "train", "predict"], default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument("--embed-dim", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--w2v-model",
        default="",
        help="Path to pre-trained Word2Vec model (skip W2V training)",
    )

    args = parser.parse_args()
    run_pipeline(
        args.data_dir, args.output, args.test_data_dir,
        stage=args.stage,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        dropout=args.dropout,
        w2v_model=args.w2v_model,
    )


if __name__ == "__main__":
    main()

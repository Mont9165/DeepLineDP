# DeepLineDP (fork for the *agent-vs-human-bugs* study)

This is a fork of [DeepLineDP](https://github.com/awsm-research/DeepLineDP) (Pornprasit &
Tantithamthavorn, *"DeepLineDP: Towards a Deep Learning Approach for Line-Level Defect Prediction,"*
TSE 2023). It is used as an off-the-shelf line-level defect-prediction model in the empirical study
**agent-vs-human-bugs** — *"Do AI Coding Agents Introduce Different Bugs than Humans?"* — under a
**leave-one-project-out (LOPO), cross-domain** protocol (agent vs. human code), **not** the original
within-release Java workflow.

> We do not modify the DeepLineDP model itself; this fork only adds multi-language support and the
> LOPO pipeline needed by the study.

## Environment and data

- **Environment:** managed by the parent repository's `uv` (`uv sync --group deeplinedp`, Python 3.12).
  There is no separate conda setup. Clone the parent with `git clone --recurse-submodules`
  (<https://github.com/Mont9165/agent-vs-human-bugs>); this repo is its `DeepLineDP/` submodule.
- **Data:** built from LLM4SZZ bug-inducing-commit output, not the original 9-project Java dataset.
  See the parent repository for how `save_logs/` is produced (its "Step 1").

## Running the LOPO pipeline (RQ3)

All commands use the parent's `uv` and run from `DeepLineDP/script/` unless noted. GPU is required for
training (step 3).

```bash
cd DeepLineDP/script

# 1. Build agent / human / mixed datasets from LLM4SZZ output (train/eval/test splits per group)
uv run python build_dataset_from_llm4szz.py --output ../datasets/preprocessed_data/

# 2. Generate the leave-one-project-out fold manifest (one fold per held-out repo x scenario)
uv run python lopo_generate_folds.py \
    --data-dir ../datasets/preprocessed_data/ --output-dir ../output/lopo

# 3. Train every fold (GPU; a single shared Word2Vec embedding is reused across folds).
#    Submit as a SLURM array (N = number of folds in ../output/lopo/manifest.json),
#    or use the cluster / single-node H100 drivers in ../../studies/icse/ (run_rq3_lopo_cc25.sh, run_rq3_h100.sh):
sbatch --array=0-<N> run_lopo_slurm.sh
#    Single fold (for debugging):
uv run python lopo_run_fold.py --manifest ../output/lopo/manifest.json --fold-id 0 \
    --output-dir ../output/lopo --num-epochs 10

# 4. Per-fold eval-split predictions (used for leak-free threshold selection)
uv run python generate_eval_predictions.py --manifest ../output/lopo/manifest.json

# 5. Aggregate per-fold metrics: file-level AUC; line-level Recall@20%, Effort@20%, IFA
uv run python lopo_aggregate.py --manifest ../output/lopo/manifest.json \
    --output ../../studies/icse/results/rq3_lopo_results.json --pretty
```

Then, from the parent repository root, compute the cross-scenario statistics
(Kruskal–Wallis, pairwise Mann–Whitney U with Holm correction, Cliff's δ):

```bash
uv run python rq3_median_analysis.py \
    --lopo-results studies/icse/results/rq3_lopo_results.json \
    --output studies/icse/results/rq3_median_results.json
```

The five cross-domain scenarios are Agent→Agent, Human→Human, Human→Agent, Agent→Human, and
Mixed→Mixed. The shared Word2Vec embedding is trained **unsupervised** on pooled training code (no
label leakage); defect labels train only the supervised model, fold by fold.

## What this fork adds

- **Multi-language parsing** via tree-sitter (13 languages), replacing the Java-only path.
- The **LOPO cross-domain scripts** used above: `build_dataset_from_llm4szz.py`,
  `lopo_generate_folds.py`, `lopo_run_fold.py` + `train_pipeline.py`, `generate_eval_predictions.py`,
  and `lopo_aggregate.py`.
- **Python 3.12 / `uv`** compatibility and SLURM / single-node H100 LOPO drivers.

## Model and hyper-parameters

`DeepLineDP_model.py` — a Hierarchical Attention Network with bidirectional GRU layers (unchanged from
upstream). Training uses: `batch_size` 32, `num_epochs` 10, word-embedding dim 50, word/sentence GRU
hidden dim 64 (1 layer each), `dropout` 0.2, `lr` 0.001.

## Original tool

For the original DeepLineDP documentation — the 9-project Java benchmark, conda setup, within-release
and cross-project (RQ1–RQ4) experiments, file-/line-level baselines, and the R-based evaluation — see
the upstream repository: <https://github.com/awsm-research/DeepLineDP>.

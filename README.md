# DeepLineDP (fork for the *agent-vs-human-bugs* study)

This is a fork of [DeepLineDP](https://github.com/awsm-research/DeepLineDP) (Pornprasit &
Tantithamthavorn, *"DeepLineDP: Towards a Deep Learning Approach for Line-Level Defect Prediction,"*
TSE 2023). It is used as an off-the-shelf line-level defect-prediction model in the empirical study
**agent-vs-human-bugs** — *"Do AI Coding Agents Introduce Different Bugs than Humans?"* — under a
**leave-one-project-out (LOPO), cross-domain** protocol (agent vs. human code), **not** the original
within-release Java workflow.

> We do not modify the DeepLineDP model itself; this fork only adds multi-language support and the
> LOPO pipeline needed by the study.

## Reproducing RQ3

The pipeline is driven from the **parent repository** — follow its README, not standalone steps here:

- **Repo / instructions:** <https://github.com/Mont9165/agent-vs-human-bugs> (section *"Step 4 — RQ3:
  Defect predictability"*).
- **Environment:** the parent's `uv` setup (`uv sync --group deeplinedp`, Python 3.12). No conda.
- **Data:** built from LLM4SZZ output by `script/build_dataset_from_llm4szz.py` (the original 9-project
  Java dataset is not used).

## Fork additions (branch `fix/python312-compat`)

- **Multi-language parsing** via tree-sitter (13 languages), replacing the Java-only path.
- **LOPO cross-domain pipeline** in `script/`:
  - `build_dataset_from_llm4szz.py` — build agent / human / mixed datasets from LLM4SZZ output.
  - `lopo_generate_folds.py` — generate the leave-one-project-out fold manifest.
  - `lopo_run_fold.py` + `train_pipeline.py` — per-fold train + predict (a single shared Word2Vec
    embedding is reused across folds).
  - `generate_eval_predictions.py` — per-fold threshold selection on the eval split (no test leakage).
  - `lopo_aggregate.py` — per-scenario metrics (file-level AUC; line-level Recall@20%, Effort@20%, IFA).
- **Python 3.12 / `uv`** compatibility and SLURM / single-node H100 LOPO drivers.

## Model and hyper-parameters

`DeepLineDP_model.py` — a Hierarchical Attention Network with bidirectional GRU layers (unchanged from
upstream). Training uses: `batch_size` 32, `num_epochs` 10, word-embedding dim 50, word/sentence GRU
hidden dim 64 (1 layer each), `dropout` 0.2, `lr` 0.001.

## Original tool

For the original DeepLineDP documentation — the 9-project Java benchmark, conda setup, within-release
and cross-project (RQ1–RQ4) experiments, file-/line-level baselines, and the R-based evaluation — see
the upstream repository: <https://github.com/awsm-research/DeepLineDP>.

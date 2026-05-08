# Improving Robustness in LLM Math Reasoners Under Input Perturbations

**Amy Kim | NYU | LLM Final Project 2026**

## Overview

This project investigates whether LLMs are truly reasoning on math problems or relying on surface patterns, and proposes a consistency loss training objective to improve robustness under input perturbations.

We evaluate Qwen2.5-Math-1.5B on [GSM-Plus](https://huggingface.co/datasets/qintongli/GSM-Plus) — a benchmark extending GSM8K with 8 perturbation types — and compare three training approaches:

1. **SFT-orig**: Fine-tune on original questions only
2. **SFT-both**: Fine-tune on original + perturbed questions
3. **SFT-cons**: Fine-tune with an additional consistency loss that aligns hidden representations at the most perturbation-sensitive layer

## Key Findings

- Applying consistency loss at layer `l*=23` (identified via hidden state similarity analysis as the point of maximum divergence) improves robustness on **4 of 7 perturbation types**
- Largest gains on **distraction insertion** (gap 0.20→0.10) and **numerical substitution** (gap 0.08→0.00)
- Full fine-tuning is essential — partial layer freezing causes catastrophic forgetting
- Symmetric gradient flow (both representations move toward each other) outperforms anchored (only perturbed moves) for robustness

## Results

| Model | Orig Acc | Pert Acc | Gap | Consistency |
|---|---|---|---|---|
| Base | 0.620 | 0.529 | 0.091 | 0.189 |
| SFT (a3) | 0.451 | 0.366 | 0.086 | 0.126 |
| GRPO | 0.743 | 0.666 | 0.077 | 0.263 |
| SFT-orig (ours) | 0.623 | 0.566 | 0.057 | 0.211 |
| SFT-both (ours) | 0.709 | 0.606 | 0.103 | 0.254 |
| SFT-cons l\*=8 sym | 0.689 | 0.597 | 0.091 | 0.243 |
| SFT-cons l\*=8 anchor | 0.720 | 0.594 | 0.126 | 0.243 |
| SFT-cons l\*=18 sym | 0.686 | 0.600 | 0.086 | 0.257 |
| SFT-cons l\*=23 sym | 0.671 | 0.589 | **0.083** | 0.234 |
| SFT-cons l\*=23 anchor | 0.703 | 0.600 | 0.103 | 0.251 |

## Repo Structure

```
llm-robustness/
├── src/
│   ├── train_robustness.py      # Main training + eval script
│   ├── eval_baselines.py        # Evaluate base, SFT, GRPO baselines
│   ├── analyze_consistency.py   # Per-type consistency analysis
│   ├── qualitative_eval.py      # Qualitative failure mode analysis
│   └── summary_table.py         # Print all model x type accuracy tables
├── results/                     # Saved evaluation CSVs
├── models/                      # Trained model checkpoints
├── logs/                        # Training logs
└── report/
    └── report.tex               # Final paper
```

## Setup

```bash
conda create -n llm-robustness python=3.11
conda activate llm-robustness
pip install torch transformers datasets accelerate sentencepiece tqdm pandas
```

## Training

```bash
# Full three-way ablation at default settings (lr=5e-6, l*=8)
python src/train_robustness.py --mode all --lr 5e-6 --lstar 8

# Consistency loss at layer 23 (best robustness)
python src/train_robustness.py --mode cons --lr 5e-6 --lstar 23
python src/train_robustness.py --mode eval --lr 5e-6 --lstar 23 --baseline

# Anchored gradient flow (only perturbed moves toward original)
python src/train_robustness.py --mode cons --lr 5e-6 --lstar 23 --anchor-orig
python src/train_robustness.py --mode eval --lr 5e-6 --lstar 23 --anchor-orig
```

**Key flags:**

| Flag | Description |
|---|---|
| `--mode` | `sft-orig`, `sft-both`, `cons`, `eval`, or `all` |
| `--lr` | Learning rate (default: `5e-6`) |
| `--lstar` | Layer to apply consistency loss at (default: `8`) |
| `--freeze` | Freeze all layers except 5 around `l*` |
| `--anchor-orig` | Detach original hidden state — only perturbed moves |
| `--baseline` | Also evaluate base model during eval |
| `--rerun-baseline` | Force re-evaluation of base model |

Models are saved to `models/<run_name>/` where run names encode all hyperparameters (e.g. `cons_lr5e-6_l23_full`), so different runs never overwrite each other.

Training is **resumable** — if a job dies, re-running the same command picks up from the last completed epoch checkpoint.

## Evaluation

```bash
# Evaluate base, SFT (a3), GRPO baselines
python src/eval_baselines.py --models base sft grpo

# Per-type consistency analysis (which types should/shouldn't be consistent)
python src/analyze_consistency.py

# Qualitative failure analysis
python src/qualitative_eval.py --from_csv results/base_results.csv
python src/qualitative_eval.py --from_csv results/cons_lr5e-6_l23_full_results.csv
```

## Hidden State Analysis

We compute layerwise cosine similarity between hidden states of original and perturbed inputs across 20 GSM-Plus pairs to identify the most perturbation-sensitive layer. For the base model, similarity drops steadily from layer 5 and reaches a minimum at **layer 23** (0.9081). Fine-tuned models (SFT, GRPO) show earlier divergence at layer 8, suggesting training amplifies sensitivity in middle layers. This analysis motivates using `l*=23` as the primary consistency loss target.

## Citation

If you use this code, please cite:

```
@misc{kim2026robustness,
  title={Improving Robustness in LLM Math Reasoners Under Input Perturbations},
  author={Kim, Jung Su Amy},
  year={2026},
  note={NYU LLM Final Project}
}
```

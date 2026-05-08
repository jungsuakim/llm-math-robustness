import argparse
import sys
import collections
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from logit_lens import (
    get_logit_lens_predictions,
    find_answer_commitment_layer,
    compare_logit_lens,
)

from utils import (
    extract_answer_token,
    build_eval_pairs,
    RESULTS_DIR, ROOT,
)

# Add src/ to path so we can import logit_lens.py
sys.path.insert(0, str(Path(__file__).parent))
from logit_lens import (
    get_logit_lens_predictions,
    find_answer_commitment_layer,
    compare_logit_lens,
)

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_START      = 0
EVAL_END        = 400
EVAL_PER_TYPE   = 50
EVAL_EXCLUDE    = {"critical thinking"}




def run_logit_lens_analysis(model, tokenizer, pairs, model_name):
    results = []
    device = next(model.parameters()).device

    for i, pair in enumerate(pairs):
        if i % 10 == 0:
            print(f"  [{i}/{len(pairs)}] {pair['perturbation_type']}")

        orig_prompt = f"Problem: {pair['original']}\nSolution:"
        pert_prompt = f"Problem: {pair['perturbed']}\nSolution:"

        orig_ans_tok = extract_answer_token(pair["original_answer"])
        pert_ans_tok = extract_answer_token(pair["perturbed_answer"])

        try:
            orig_preds = get_logit_lens_predictions(model, tokenizer, orig_prompt, top_k=5, device=device)
            pert_preds = get_logit_lens_predictions(model, tokenizer, pert_prompt, top_k=5, device=device)

            orig_commit = find_answer_commitment_layer(orig_preds, orig_ans_tok)
            pert_commit = find_answer_commitment_layer(pert_preds, pert_ans_tok)
            kl_divs     = compare_logit_lens(orig_preds, pert_preds)
            peak_kl_layer = int(np.argmax(kl_divs))
            peak_kl_val   = float(np.max(kl_divs))

            # Top prediction at each of the key layers
            orig_top_l8  = orig_preds[8]["top_token"]  if len(orig_preds) > 8  else ""
            orig_top_l23 = orig_preds[23]["top_token"] if len(orig_preds) > 23 else ""
            pert_top_l8  = pert_preds[8]["top_token"]  if len(pert_preds) > 8  else ""
            pert_top_l23 = pert_preds[23]["top_token"] if len(pert_preds) > 23 else ""
            orig_top_last = orig_preds[-1]["top_token"]
            pert_top_last = pert_preds[-1]["top_token"]

            results.append({
                "id":                  pair["id"],
                "perturbation_type":   pair["perturbation_type"],
                "original_answer":     pair["original_answer"],
                "perturbed_answer":    pair["perturbed_answer"],
                # Commitment layers
                "orig_commit_layer":   orig_commit,
                "pert_commit_layer":   pert_commit,
                "commit_layer_diff":   (pert_commit - orig_commit) if (orig_commit is not None and pert_commit is not None) else None,
                # KL divergence
                "peak_kl_layer":       peak_kl_layer,
                "peak_kl_val":         round(peak_kl_val, 4),
                "kl_at_l8":            round(float(kl_divs[8]),  4) if len(kl_divs) > 8  else None,
                "kl_at_l23":           round(float(kl_divs[23]), 4) if len(kl_divs) > 23 else None,
                # Top predictions at key layers
                "orig_top_l8":         orig_top_l8,
                "pert_top_l8":         pert_top_l8,
                "orig_top_l23":        orig_top_l23,
                "pert_top_l23":        pert_top_l23,
                "orig_top_last":       orig_top_last,
                "pert_top_last":       pert_top_last,
                "same_final_pred":     orig_top_last == pert_top_last,
            })

        except Exception as e:
            print(f"  ERROR on example {pair['id']}: {e}")
            continue

    return pd.DataFrame(results)


def print_summary(df, model_name, out_path):
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"LOGIT LENS ANALYSIS: {model_name}")
    lines.append(f"{'='*70}")
    lines.append(f"Total examples: {len(df)}")

    # Commitment layer stats
    orig_commits = df["orig_commit_layer"].dropna()
    pert_commits = df["pert_commit_layer"].dropna()
    lines.append(f"\n--- Answer Commitment Layers ---")
    lines.append(f"Original  — committed: {len(orig_commits)}/{len(df)} | mean layer: {orig_commits.mean():.1f} | median: {orig_commits.median():.0f}")
    lines.append(f"Perturbed — committed: {len(pert_commits)}/{len(df)} | mean layer: {pert_commits.mean():.1f} | median: {pert_commits.median():.0f}")
    lines.append(f"Original commits earlier: {(df['orig_commit_layer'] < df['pert_commit_layer']).sum()} examples")
    lines.append(f"Perturbed commits earlier: {(df['pert_commit_layer'] < df['orig_commit_layer']).sum()} examples")

    # KL divergence stats
    lines.append(f"\n--- KL Divergence ---")
    lines.append(f"Peak KL layer (mean): {df['peak_kl_layer'].mean():.1f}")
    lines.append(f"Peak KL value (mean): {df['peak_kl_val'].mean():.4f}")
    lines.append(f"KL at l*=8  (mean):  {df['kl_at_l8'].mean():.4f}")
    lines.append(f"KL at l*=23 (mean):  {df['kl_at_l23'].mean():.4f}")

    # Per perturbation type
    lines.append(f"\n--- Per Perturbation Type ---")
    type_summary = df.groupby("perturbation_type").agg(
        n=("id", "count"),
        orig_commit=("orig_commit_layer", "mean"),
        pert_commit=("pert_commit_layer", "mean"),
        peak_kl_layer=("peak_kl_layer", "mean"),
        peak_kl=("peak_kl_val", "mean"),
        kl_l8=("kl_at_l8", "mean"),
        kl_l23=("kl_at_l23", "mean"),
    ).round(2)
    lines.append(type_summary.to_string())

    # Examples where perturbed commits much later (robustness failures)
    late_pert = df[df["commit_layer_diff"].notna() & (df["commit_layer_diff"] > 5)]
    if len(late_pert) > 0:
        lines.append(f"\n--- Examples where perturbed commits 5+ layers later ({len(late_pert)} total) ---")
        for _, row in late_pert.head(3).iterrows():
            lines.append(f"  ID {row['id']} ({row['perturbation_type']}): orig commits at layer {row['orig_commit_layer']}, pert at {row['pert_commit_layer']}")

    output = "\n".join(lines)
    print(output)
    out_path.write_text(output)
    print(f"\nSaved to {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                        help="Model path or HuggingFace ID")
    parser.add_argument("--n_examples", type=int, default=None,
                        help="Limit number of examples (default: all 350)")
    args = parser.parse_args()

    model_stem = Path(args.model).name if Path(args.model).exists() else args.model.replace("/", "_")
    csv_out    = RESULTS_DIR / f"logit_lens_{model_stem}.csv"
    txt_out    = RESULTS_DIR / f"logit_lens_{model_stem}.txt"

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    print("Loading dataset...")
    dataset = load_dataset("qintongli/GSM-Plus")
    pairs   = build_eval_pairs(dataset, n_examples=args.n_examples)
    print(f"Running logit lens on {len(pairs)} examples...")

    df = run_logit_lens_analysis(model, tokenizer, pairs, model_stem)
    df.to_csv(csv_out, index=False)
    print(f"Results saved to {csv_out}")

    print_summary(df, model_stem, txt_out)
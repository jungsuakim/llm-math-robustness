"""
eval_baselines.py
-----------------
Evaluates base, SFT (MATH), and GRPO models on the same GSM-Plus eval
set used in train_robustness.py, so results are directly comparable.

Usage:
    python src/eval_baselines.py
    python src/eval_baselines.py --models base sft grpo 
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

from utils import (
    format_prompt,
    gsm_extract_answer,
    gsm_answers_match,
    build_eval_pairs,
    setup_logger,
    results_path_for,
    RESULTS_DIR, LOGS_DIR, ROOT,
)

# ── Model registry ────────────────────────────────────────────────────────────
# where my a3 models are saved
SFT_PATH  = Path("/scratch/jk8901/llm/nyu-llm-reasoners-a3/student/sft_checkpoints/tune_lr2e-5_bs16_ga4")
GRPO_PATH = Path("/scratch/jk8901/llm/nyu-llm-reasoners-a3/student/grpo_checkpoints/grpo_lr1e-5")

MODEL_REGISTRY = {
    "base": "Qwen/Qwen2.5-Math-1.5B",
    "sft":  str(SFT_PATH),
    "grpo": str(GRPO_PATH),
}

# ── Eval config ──────────────────────────────
EVAL_START      = 0
EVAL_END        = 400
EVAL_PER_TYPE   = 50
EVAL_BATCH_SIZE = 8
MAX_SEQ_LEN     = 512
MAX_NEW_TOKENS  = 1024
EVAL_EXCLUDE_TYPES = {"critical thinking"}

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model_id_or_path, model_name, eval_pairs, logger):
    out_path = RESULTS_DIR / f"{model_name}_results.csv"
    if out_path.exists():
        logger.info(f"Results already exist for {model_name} — loading from {out_path}")
        return pd.read_csv(out_path)

    logger.info("=" * 60)
    logger.info(f"Evaluating: {model_name}  ({model_id_or_path})")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    def run_batch(prompts):
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to("cuda")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        return [
            tokenizer.decode(ids[input_len:], skip_special_tokens=True)
            for ids in output_ids
        ]

    results = []
    for batch_start in tqdm(range(0, len(eval_pairs), EVAL_BATCH_SIZE),
                            desc=f"Eval [{model_name}]"):
        batch        = eval_pairs[batch_start : batch_start + EVAL_BATCH_SIZE]
        orig_prompts = [format_prompt(p["original"]) for p in batch]
        pert_prompts = [format_prompt(p["perturbed"]) for p in batch]

        orig_outs = run_batch(orig_prompts)
        pert_outs = run_batch(pert_prompts)

        for pair, orig_out, pert_out in zip(batch, orig_outs, pert_outs):
            orig_pred    = gsm_extract_answer(orig_out)
            pert_pred    = gsm_extract_answer(pert_out)
            orig_correct = gsm_answers_match(orig_pred, pair["original_answer"])
            pert_correct = gsm_answers_match(pert_pred, pair["perturbed_answer"])
            results.append({
                "id":                pair["id"],
                "perturbation_type": pair["perturbation_type"],
                "original_answer":   pair["original_answer"],
                "perturbed_answer":  pair["perturbed_answer"],
                "original_pred":     orig_pred,
                "perturbed_pred":    pert_pred,
                "original_correct":  orig_correct,
                "perturbed_correct": pert_correct,
                "consistent":        orig_pred == pert_pred,
                "pdr":               int(orig_correct) - int(pert_correct),
            })

    df       = pd.DataFrame(results)
    orig_acc = df["original_correct"].mean()
    pert_acc = df["perturbed_correct"].mean()

    logger.info(f"\n=== {model_name} ===")
    logger.info(f"Original accuracy:  {orig_acc:.3f}")
    logger.info(f"Perturbed accuracy: {pert_acc:.3f}")
    logger.info(f"Gap:                {orig_acc - pert_acc:.3f}")
    logger.info(f"Consistency:        {df['consistent'].mean():.3f}")

    type_summary = df.groupby("perturbation_type").agg(
        original_acc=("original_correct", "mean"),
        perturbed_acc=("perturbed_correct", "mean"),
        consistency=("consistent", "mean"),
        count=("id", "count"),
    ).reset_index()
    type_summary["gap"] = type_summary["original_acc"] - type_summary["perturbed_acc"]
    logger.info(f"\nPer perturbation type:\n{type_summary.sort_values('gap', ascending=False).to_string(index=False)}")

    df.to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        choices=["base", "sft", "grpo"],
        default=["base", "sft", "grpo"],
        help="Which baseline models to evaluate"
    )
    args   = parser.parse_args()
    logger = setup_logger("eval_baselines")

    dataset    = load_dataset("qintongli/GSM-Plus")
    eval_pairs = build_eval_pairs(dataset)

    dfs = {}
    for name in args.models:
        path = MODEL_REGISTRY[name]
        if not Path(path).exists() and not path.startswith("Qwen"):
            logger.info(f"WARNING: path not found for {name}: {path} — skipping")
            continue
        dfs[name] = evaluate_model(path, name, eval_pairs, logger)

    logger.info("\n\n=== BASELINE COMPARISON ===")
    rows = []

    for name, df in dfs.items():
        rows.append({
            "Model":       name,
            "Orig Acc":    round(df["original_correct"].mean(), 3),
            "Pert Acc":    round(df["perturbed_correct"].mean(), 3),
            "Gap":         round(df["original_correct"].mean() - df["perturbed_correct"].mean(), 3),
            "Consistency": round(df["consistent"].mean(), 3),
        })

    for trained_name, fname in [
        ("my-sft-orig",  "SFT-orig_results.csv"),
        ("my-sft-both",  "SFT-origpluspert_results.csv"),
        ("my-cons",      "SFTplusCons_results.csv"),
    ]:
        p = RESULTS_DIR / fname
        if p.exists():
            df = pd.read_csv(p)
            rows.append({
                "Model":       trained_name,
                "Orig Acc":    round(df["original_correct"].mean(), 3),
                "Pert Acc":    round(df["perturbed_correct"].mean(), 3),
                "Gap":         round(df["original_correct"].mean() - df["perturbed_correct"].mean(), 3),
                "Consistency": round(df["consistent"].mean(), 3),
            })

    comparison = pd.DataFrame(rows)
    logger.info(f"\n{comparison.to_string(index=False)}")
    comparison.to_csv(RESULTS_DIR / "full_comparison.csv", index=False)
    logger.info(f"Full comparison saved to {RESULTS_DIR / 'full_comparison.csv'}")
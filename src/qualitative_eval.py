import argparse
import io
import collections
import sys
from pathlib import Path

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import (
    format_prompt,
    gsm_extract_answer,
    gsm_answers_match,
    build_eval_pairs,
    RESULTS_DIR, ROOT,
)

EVAL_START      = 0
EVAL_END        = 400
EVAL_PER_TYPE   = 50
MAX_SEQ_LEN     = 512
MAX_NEW_TOKENS  = 512  

EVAL_EXCLUDE_TYPES = {"critical thinking"}

def run_model(model_path, pairs):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    results = []
    for pair in pairs:
        orig_prompt = format_prompt(pair["original"])
        pert_prompt = format_prompt(pair["perturbed"])

        with torch.no_grad():
            for prompt, key in [(orig_prompt, "orig"), (pert_prompt, "pert")]:
                inputs = tokenizer(prompt, return_tensors="pt",
                                   truncation=True, max_length=MAX_SEQ_LEN).to("cuda")
                output_ids = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id,
                )
                output = tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                pair[f"{key}_output"] = output
                pair[f"{key}_pred"]   = gsm_extract_answer(output)

        pair["orig_correct"] = gsm_answers_match(pair["orig_pred"], pair["original_answer"])
        pair["pert_correct"] = gsm_answers_match(pair["pert_pred"], pair["perturbed_answer"])
        pair["consistent"]   = pair["orig_pred"] == pair["pert_pred"]

        outcome = ("correct_both"   if pair["orig_correct"] and pair["pert_correct"] else
                   "correct_orig"   if pair["orig_correct"] and not pair["pert_correct"] else
                   "correct_pert"   if not pair["orig_correct"] and pair["pert_correct"] else
                   "wrong_both")
        pair["outcome"] = outcome
        results.append(pair)

    del model
    torch.cuda.empty_cache()
    return results


def load_from_csv(csv_path, dataset):
    """Load existing results CSV and merge with raw questions from dataset."""
    df      = pd.read_csv(csv_path)
    pairs   = build_eval_pairs(dataset)
    pairs_by_id = {p["id"]: p for p in pairs}

    results = []
    for _, row in df.iterrows():
        pair = pairs_by_id.get(int(row["id"]), {})
        pair.update({
            "orig_pred":     str(row.get("original_pred", "")),
            "pert_pred":     str(row.get("perturbed_pred", "")),
            "orig_correct":  bool(row.get("original_correct", False)),
            "pert_correct":  bool(row.get("perturbed_correct", False)),
            "consistent":    bool(row.get("consistent", False)),
            "orig_output":   "",  # not stored in CSV
            "pert_output":   "",
        })
        outcome = ("correct_both"  if pair["orig_correct"] and pair["pert_correct"] else
                   "correct_orig"  if pair["orig_correct"] and not pair["pert_correct"] else
                   "correct_pert"  if not pair["orig_correct"] and pair["pert_correct"] else
                   "wrong_both")
        pair["outcome"] = outcome
        results.append(pair)
    return results


OUTCOME_LABELS = {
    "correct_orig":  "✗ ROBUSTNESS FAILURE (correct on orig, wrong on pert)",
    "correct_pert":  "? PERTURB HELPS (wrong on orig, correct on pert)",
    "correct_both":  "✓ ROBUST (correct on both)",
    "wrong_both":    "✗ GENERAL FAILURE (wrong on both)",
}

def print_example(pair, show_output=True):
    print(f"  ID: {pair['id']}  |  Type: {pair['perturbation_type']}")
    print(f"  Original:  {pair['original']}")
    print(f"  Perturbed: {pair['perturbed']}")
    print(f"  Gold orig: {pair['original_answer']}  |  Gold pert: {pair['perturbed_answer']}")
    print(f"  Pred orig: {pair['orig_pred']}  |  Pred pert: {pair['pert_pred']}")
    if show_output and pair.get("orig_output"):
        print(f"  --- Original output ---")
        print(f"  {pair['orig_output'][:300].strip()}")
        print(f"  --- Perturbed output ---")
        print(f"  {pair['pert_output'][:300].strip()}")
    print()


def print_qualitative(results, n_per_type=2, show_output=True):
    by_type = collections.defaultdict(list)
    for r in results:
        by_type[r["perturbation_type"]].append(r) 
    outcomes = collections.Counter(r["outcome"] for r in results)
    print("\n" + "="*70)
    print("OVERALL OUTCOME DISTRIBUTION")
    print("="*70)
    total = len(results)
    for outcome, label in OUTCOME_LABELS.items():
        count = outcomes[outcome]
        print(f"  {label}: {count}/{total} ({100*count/total:.1f}%)")

    print("\n" + "="*70)
    print("PER PERTURBATION TYPE BREAKDOWN")
    print("="*70)
    type_rows = []
    for p_type, items in sorted(by_type.items()):
        type_outcomes = collections.Counter(r["outcome"] for r in items)
        type_rows.append({
            "type":          p_type,
            "robust":        type_outcomes["correct_both"],
            "fail_pert":     type_outcomes["correct_orig"],
            "fail_orig":     type_outcomes["correct_pert"],
            "fail_both":     type_outcomes["wrong_both"],
            "n":             len(items),
        })
    type_df = pd.DataFrame(type_rows)
    print(type_df.to_string(index=False))

    for p_type, items in sorted(by_type.items()):
        print(f"\n{'='*70}")
        print(f"PERTURBATION TYPE: {p_type.upper()}")
        print(f"{'='*70}")

        for outcome in ["correct_orig", "correct_pert", "correct_both", "wrong_both"]:
            matching = [r for r in items if r["outcome"] == outcome]
            if not matching:
                continue
            print(f"\n  {OUTCOME_LABELS[outcome]} ({len(matching)} examples)")
            print("  " + "-"*50)
            for ex in matching[:n_per_type]:
                print_example(ex, show_output=show_output)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_and_save(results, out_path, n_per_type, show_output):
    import io, sys
    buffer    = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    print_qualitative(results, n_per_type=n_per_type, show_output=show_output)
    sys.stdout = old_stdout
    output = buffer.getvalue()
    print(output)
    out_path.write_text(output)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Model path or HuggingFace ID to run inference on")
    parser.add_argument("--from_csv", type=str, default=None, help="Load existing results CSV instead of running inference")
    parser.add_argument("--n_per_type", type=int, default=2, help="Number of examples to show per outcome per type (default: 2)")
    parser.add_argument("--no_output", action="store_true", help="Don't show model output text, just questions and predictions")
    args = parser.parse_args()

    dataset = load_dataset("qintongli/GSM-Plus")

    if args.from_csv:
        csv_stem = Path(args.from_csv).stem.replace("_results", "")
        out_path = RESULTS_DIR / f"qualitative_{csv_stem}.txt"
        print(f"Loading from CSV: {args.from_csv}")
        results  = load_from_csv(args.from_csv, dataset)
        run_and_save(results, out_path, args.n_per_type, show_output=False)

    elif args.model:
        model_stem = Path(args.model).name if Path(args.model).exists() else args.model.replace("/", "_")
        out_path   = RESULTS_DIR / f"qualitative_{model_stem}.txt"
        print(f"Running inference on: {args.model}")
        pairs   = build_eval_pairs(dataset)
        results = run_model(args.model, pairs)
        run_and_save(results, out_path, args.n_per_type, show_output=not args.no_output)

    else:
        print("Please provide --model or --from_csv")
        print("Examples:")
        print("  python src/qualitative_eval.py --from_csv results/cons_lr5e-6_l23_full_results.csv")
        print("  python src/qualitative_eval.py --model models/cons_lr5e-6_l23_full")
        print("  python src/qualitative_eval.py --model Qwen/Qwen2.5-Math-1.5B --n_per_type 3")
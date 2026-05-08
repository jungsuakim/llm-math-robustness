"""
train_robustness.py
-------------------
Three-way ablation for robustness under input perturbations:

  1. sft-orig : CE loss on original questions only
  2. sft-both : CE loss on original + perturbed questions
  3. cons     : CE loss on original + perturbed + consistency loss

Flags:
  --freeze        freeze all layers except 5 around L_STAR
  --lr FLOAT      learning rate (default: 5e-6)
  --lstar INT     layer to apply consistency loss at (default: 8)
  --anchor-orig   detach original hidden state — only perturbed moves toward original
  --baseline      also evaluate base unfinetuned model during eval
  --rerun-baseline force re-evaluation of base model even if CSV exists

Usage:
    python src/train_robustness.py --mode cons --lr 5e-6 --lstar 23
    python src/train_robustness.py --mode cons --lr 5e-6 --lstar 23 --anchor-orig
    python src/train_robustness.py --mode eval --lr 5e-6 --lstar 23 --baseline
    python src/train_robustness.py --mode all --lr 5e-6 --lstar 8
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

from utils import (
    format_prompt,
    gsm_extract_answer,
    gsm_answers_match,
    build_eval_pairs,
    build_train_pairs,
    setup_logger,
    results_path_for,
    RESULTS_DIR, CKPT_DIR, LOGS_DIR, MODELS_DIR, ROOT,
    TRAIN_VALID_TYPES, EVAL_EXCLUDE_TYPES,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL_ID   = "Qwen/Qwen2.5-Math-1.5B"
L_STAR = 8 
LAMBDA = 0.01
N_EPOCHS = 2
MAX_SEQ_LEN = 512
MAX_NEW_TOKENS = 1024

TRAIN_START = 400
TRAIN_END = 10552
EVAL_START = 0
EVAL_END = 400
EVAL_PER_TYPE = 50
EVAL_BATCH_SIZE = 8

# ── Run naming ────────────────────────────────────────────────────────────────

def run_name(mode, lr, freeze, lstar, anchor_orig=False):
    lr_str     = f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    freeze_str = "frozen" if freeze else "full"
    lstar_str  = f"_l{lstar}" if mode == "cons" else ""
    anchor_str = "_anchor" if (mode == "cons" and anchor_orig) else ""
    return f"{mode.replace('-', '_')}_{lr_str}{lstar_str}{anchor_str}_{freeze_str}"


def save_path_for(mode, lr, freeze, lstar, anchor_orig=False):
    return MODELS_DIR / run_name(mode, lr, freeze, lstar, anchor_orig)


def ckpt_path_for(mode, lr, freeze, lstar, epoch, anchor_orig=False):
    return CKPT_DIR / f"{run_name(mode, lr, freeze, lstar, anchor_orig)}_epoch{epoch}"

# ── Checkpointing ─────────────────────────────────────────────────────────────

def get_start_epoch(mode, lr, freeze, lstar, anchor_orig=False):
    last = -1
    for epoch in range(N_EPOCHS):
        ckpt = ckpt_path_for(mode, lr, freeze, lstar, epoch, anchor_orig)
        if (ckpt / "config.json").exists():
            last = epoch
    return last + 1


def load_checkpoint_or_base(mode, lr, freeze, lstar, logger, anchor_orig=False):
    start_epoch = get_start_epoch(mode, lr, freeze, lstar, anchor_orig)
    if start_epoch > 0:
        ckpt = ckpt_path_for(mode, lr, freeze, lstar, start_epoch - 1, anchor_orig)
        logger.info(f"Resuming from {ckpt} (starting epoch {start_epoch + 1}/{N_EPOCHS})")
        tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
        )
    else:
        logger.info("No checkpoint found — loading base model.")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
        )
    return model, tokenizer, start_epoch


# ── Training ──────────────────────────────────────────────────────────────────

def apply_freeze(model, lstar, logger):
    for name, param in model.named_parameters():
        layer_nums = [f'layers.{i}.' for i in range(lstar - 2, lstar + 3)]
        param.requires_grad = any(l in name for l in layer_nums)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"FROZEN mode — Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")


def train(mode, lr, freeze, lstar, anchor_orig=False):
    assert mode in ("sft-orig", "sft-both", "cons")
    name      = run_name(mode, lr, freeze, lstar, anchor_orig)
    logger    = setup_logger(name)
    save_path = save_path_for(mode, lr, freeze, lstar, anchor_orig)

    logger.info("=" * 60)
    logger.info(f"Run: {name} | l*={lstar} | lambda={LAMBDA} | lr={lr} | epochs={N_EPOCHS} | freeze={freeze} | anchor_orig={anchor_orig}")
    logger.info("=" * 60)

    if (save_path / "config.json").exists():
        logger.info(f"Final model already exists at {save_path} — skipping.")
        return

    dataset     = load_dataset("qintongli/GSM-Plus")
    train_pairs = build_train_pairs(dataset)
    logger.info(f"Training pairs: {len(train_pairs)}")
    type_counts = pd.Series([p["perturbation_type"] for p in train_pairs]).value_counts()
    logger.info(f"Type breakdown:\n{type_counts.to_string()}")

    model, tokenizer, start_epoch = load_checkpoint_or_base(mode, lr, freeze, lstar, logger, anchor_orig)

    if start_epoch >= N_EPOCHS:
        logger.info("All epochs already completed.")
        return

    if freeze:
        apply_freeze(model, lstar, logger)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"FULL mode — Trainable: {total/1e6:.1f}M / {total/1e6:.1f}M (100.0%)")
        optimizer = AdamW(model.parameters(), lr=lr)

    model.train()

    for epoch in range(start_epoch, N_EPOCHS):
        total_loss = total_ce = total_cons = 0

        for step, pair in enumerate(tqdm(train_pairs, desc=f"[{name}] Epoch {epoch+1}/{N_EPOCHS}")):
            optimizer.zero_grad()

            # Original CE loss
            orig_prompt = format_prompt(pair["original"])
            orig_full   = orig_prompt + pair["original_solution"] + tokenizer.eos_token
            orig_inputs = tokenizer(orig_full, return_tensors="pt",
                                    truncation=True, max_length=MAX_SEQ_LEN).to("cuda")
            prompt_len  = len(tokenizer(orig_prompt, return_tensors="pt")["input_ids"][0])
            labels      = orig_inputs["input_ids"].clone()
            labels[:, :prompt_len] = -100

            need_hidden  = (mode == "cons")
            orig_outputs = model(**orig_inputs, output_hidden_states=need_hidden, labels=labels)
            loss_ce      = orig_outputs.loss

            # Perturbed CE loss (sft-both and cons)
            if mode in ("sft-both", "cons"):
                pert_prompt     = format_prompt(pair["perturbed"])
                pert_full       = pert_prompt + pair["perturbed_solution"] + tokenizer.eos_token
                pert_inputs     = tokenizer(pert_full, return_tensors="pt",
                                            truncation=True, max_length=MAX_SEQ_LEN).to("cuda")
                pert_prompt_len = len(tokenizer(pert_prompt, return_tensors="pt")["input_ids"][0])
                pert_labels     = pert_inputs["input_ids"].clone()
                pert_labels[:, :pert_prompt_len] = -100
                pert_outputs    = model(**pert_inputs, output_hidden_states=need_hidden,
                                        labels=pert_labels)
                loss_ce = (loss_ce + pert_outputs.loss) / 2

            # Consistency loss (cons only)
            if mode == "cons":
                h_orig = orig_outputs.hidden_states[lstar + 1][0, -1, :]
                h_pert = pert_outputs.hidden_states[lstar + 1][0, -1, :]
                if anchor_orig:
                    h_orig = h_orig.detach()  # original is fixed anchor, only pert moves
                loss_cons = 1.0 - F.cosine_similarity(
                    h_orig.unsqueeze(0).float(),
                    h_pert.unsqueeze(0).float(),
                )
                loss        = loss_ce + LAMBDA * loss_cons
                total_cons += loss_cons.item()
            else:
                loss = loss_ce

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce   += loss_ce.item()

            if step % 100 == 0:
                msg = f"  step {step} | loss={loss.item():.4f} | ce={loss_ce.item():.4f}"
                if mode == "cons":
                    msg += f" | cons={loss_cons.item():.4f}"
                logger.info(msg)

        n       = len(train_pairs)
        summary = f"Epoch {epoch+1} done | avg_loss={total_loss/n:.4f} | avg_ce={total_ce/n:.4f}"
        if mode == "cons":
            summary += f" | avg_cons={total_cons/n:.4f}"
        logger.info(summary)

        ckpt = ckpt_path_for(mode, lr, freeze, lstar, epoch, anchor_orig)
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        logger.info(f"Epoch checkpoint saved to {ckpt}")

    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Final model saved to {save_path}")

    del model
    torch.cuda.empty_cache()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model_path, model_name, eval_pairs):
    logger   = setup_logger("eval")
    out_path = results_path_for(model_name)

    if out_path.exists():
        logger.info(f"Results already exist for {model_name} — loading.")
        return pd.read_csv(out_path)

    logger.info("=" * 60)
    logger.info(f"Evaluating: {model_name}")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    def run_batch(prompts):
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LEN,
        ).to("cuda")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        return [
            tokenizer.decode(ids[input_len:], skip_special_tokens=True)
            for ids in output_ids
        ]

    results = []
    for batch_start in tqdm(range(0, len(eval_pairs), EVAL_BATCH_SIZE),
                            desc=f"Eval [{model_name}]"):
        batch     = eval_pairs[batch_start : batch_start + EVAL_BATCH_SIZE]
        orig_outs = run_batch([format_prompt(p["original"]) for p in batch])
        pert_outs = run_batch([format_prompt(p["perturbed"]) for p in batch])

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
    logger.info(f"\nPer type:\n{type_summary.sort_values('gap', ascending=False).to_string(index=False)}")

    df.to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return df


def run_eval(lr, freeze, lstar, anchor_orig=False, baseline=False, rerun_baseline=False):
    logger     = setup_logger("eval")
    dataset    = load_dataset("qintongli/GSM-Plus")
    eval_pairs = build_eval_pairs(dataset)

    models = [
        (run_name("sft-orig", lr, freeze, lstar),               str(save_path_for("sft-orig", lr, freeze, lstar))),
        (run_name("sft-both", lr, freeze, lstar),               str(save_path_for("sft-both", lr, freeze, lstar))),
        (run_name("cons",     lr, freeze, lstar, anchor_orig),  str(save_path_for("cons",     lr, freeze, lstar, anchor_orig))),
    ]

    if baseline:
        if rerun_baseline:
            base_csv = results_path_for("base")
            if base_csv.exists():
                base_csv.unlink()
                logger.info("Deleted cached base results — will re-evaluate.")
        models.insert(0, ("base", BASE_MODEL_ID))

    dfs = {}
    for name, path_str in models:
        if path_str != BASE_MODEL_ID and not Path(path_str).exists():
            logger.info(f"WARNING: {path_str} not found — skipping {name}")
            continue
        dfs[name] = evaluate_model(path_str, name, eval_pairs)

    if dfs:
        logger.info("\n\n=== COMPARISON ===")
        rows = []
        for name, df in dfs.items():
            rows.append({
                "Model":       name,
                "Orig Acc":    round(df["original_correct"].mean(), 3),
                "Pert Acc":    round(df["perturbed_correct"].mean(), 3),
                "Gap":         round(df["original_correct"].mean() - df["perturbed_correct"].mean(), 3),
                "Consistency": round(df["consistent"].mean(), 3),
            })
        comparison = pd.DataFrame(rows)
        logger.info(f"\n{comparison.to_string(index=False)}")
        out = RESULTS_DIR / f"comparison_{run_name('all', lr, freeze, lstar, anchor_orig)}.csv"
        comparison.to_csv(out, index=False)
        logger.info(f"Saved to {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sft-orig", "sft-both", "cons", "eval", "all"],
                        default="all")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate (default: 5e-6)")
    parser.add_argument("--lstar", type=int, default=8,
                        help="Layer to apply consistency loss at (default: 8)")
    parser.add_argument("--freeze", action="store_true",
                        help="Freeze all layers except 5 around L_STAR")
    parser.add_argument("--anchor-orig", action="store_true",
                        help="Detach original hidden state — only perturbed moves toward original")
    parser.add_argument("--baseline", action="store_true",
                        help="Also evaluate the base unfinetuned model during eval")
    parser.add_argument("--rerun-baseline", action="store_true",
                        help="Force re-evaluation of base model even if CSV exists")
    args = parser.parse_args()

    if args.mode in ("sft-orig", "all"):
        train(mode="sft-orig", lr=args.lr, freeze=args.freeze,
              lstar=args.lstar, anchor_orig=args.anchor_orig)
    if args.mode in ("sft-both", "all"):
        train(mode="sft-both", lr=args.lr, freeze=args.freeze,
              lstar=args.lstar, anchor_orig=args.anchor_orig)
    if args.mode in ("cons", "all"):
        train(mode="cons", lr=args.lr, freeze=args.freeze,
              lstar=args.lstar, anchor_orig=args.anchor_orig)
    if args.mode in ("eval", "all"):
        run_eval(lr=args.lr, freeze=args.freeze, lstar=args.lstar,
                 anchor_orig=args.anchor_orig, baseline=args.baseline,
                 rerun_baseline=args.rerun_baseline)
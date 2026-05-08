"""
src/utils.py
------------
Shared utilities used across train_robustness.py, eval_baselines.py,
qualitative_eval.py, and run_logit_lens.py.

Includes:
  - Answer extraction and normalization
  - GSM-Plus data loading
  - Logging setup
"""

import re
import sys
import logging
import collections
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
LOGS_DIR    = ROOT / "logs"
CKPT_DIR    = ROOT / "checkpoints"
MODELS_DIR  = ROOT / "models"

for d in [RESULTS_DIR, LOGS_DIR, CKPT_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Eval config ───────────────────────────────────────────────────────────────

EVAL_START         = 0
EVAL_END           = 400
EVAL_PER_TYPE      = 50
EVAL_EXCLUDE_TYPES = {"critical thinking"}

TRAIN_VALID_TYPES = {
    "digit expansion",
    "integer-decimal-fraction conversion",
    "numerical substitution",
    "distraction insertion",
    "problem understanding",
    "adding operation",
}

# ── Prompting ─────────────────────────────────────────────────────────────────

def format_prompt(problem: str) -> str:
    return f"Problem: {problem}\nSolution:"


# ── Answer extraction ─────────────────────────────────────────────────────────

def gsm_extract_answer(text: str) -> str:
    """
    Extract the final numeric answer from model output.
    Tries #### format (GSM8K), then \\boxed{} (MATH), then last number in text.
    """
    if "####" in text:
        ans  = text.split("####")[-1].strip()
        nums = re.findall(r'-?\d+\.?\d*', ans.replace(",", ""))
        return nums[0] if nums else ans.strip()
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        nums = re.findall(r'-?\d+\.?\d*', boxed[-1].replace(",", ""))
        return nums[0] if nums else boxed[-1].strip()
    nums = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
    return nums[-1] if nums else ""


def normalize_num(s: str) -> str:
    """Normalize a numeric string for comparison (strip commas, trailing dots, etc.)."""
    s = str(s).strip().replace(",", "").rstrip(".")
    try:
        return str(float(s)) if "." in s else str(int(float(s)))
    except Exception:
        return s.lower()


def gsm_answers_match(pred: str, gold: str) -> bool:
    """Check if predicted answer matches gold after normalization."""
    return normalize_num(pred) == normalize_num(gold)


def extract_answer_token(answer_str: str) -> str:
    """Get the first numeric token from an answer string (for logit lens)."""
    nums = re.findall(r'-?\d+\.?\d*', str(answer_str).replace(",", ""))
    return nums[0] if nums else answer_str.strip()


# ── Data loading ──────────────────────────────────────────────────────────────

def build_eval_pairs(dataset, split: str = "test") -> list:
    """
    Build evaluation pairs from GSM-Plus.
    Samples exactly EVAL_PER_TYPE examples per perturbation type
    from indices EVAL_START to EVAL_END, excluding EVAL_EXCLUDE_TYPES.
    """
    buckets = collections.defaultdict(list)
    for i, ex in enumerate(dataset[split]):
        if i < EVAL_START:
            continue
        if i >= EVAL_END:
            break
        p_type = ex.get("perturbation_type", "unknown")
        if p_type in EVAL_EXCLUDE_TYPES:
            continue
        if len(buckets[p_type]) < EVAL_PER_TYPE:
            buckets[p_type].append({
                "id":                 i,
                "original":           ex["seed_question"],
                "perturbed":          ex["question"],
                "original_answer":    str(ex["seed_answer"]).strip(),
                "perturbed_answer":   str(ex["answer"]).strip(),
                "original_solution":  ex.get("seed_solution", ""),
                "perturbed_solution": ex.get("solution", ""),
                "perturbation_type":  p_type,
            })

    pairs = [p for items in buckets.values() for p in items]
    return pairs


def build_train_pairs(dataset, split: str = "test",
                      train_start: int = 400,
                      train_end: int = 10552) -> list:
    """
    Build training pairs from GSM-Plus.
    Filtered to TRAIN_VALID_TYPES only.
    """
    pairs = []
    for i, ex in enumerate(dataset[split]):
        if i < train_start:
            continue
        if i >= train_end:
            break
        p_type = ex.get("perturbation_type", "unknown")
        if p_type not in TRAIN_VALID_TYPES:
            continue
        pairs.append({
            "id":                 i,
            "original":           ex["seed_question"],
            "perturbed":          ex["question"],
            "original_answer":    str(ex["seed_answer"]).strip(),
            "perturbed_answer":   str(ex["answer"]).strip(),
            "original_solution":  ex.get("seed_solution", ""),
            "perturbed_solution": ex.get("solution", ""),
            "perturbation_type":  p_type,
        })
    return pairs


def print_eval_pair_counts(pairs: list) -> None:
    """Print the number of eval pairs per perturbation type."""
    counts = collections.Counter(p["perturbation_type"] for p in pairs)
    print("Eval pairs per type:")
    for t, n in sorted(counts.items()):
        print(f"  {t}: {n}")
    print(f"Total eval pairs: {len(pairs)}")


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger that writes to both stdout and logs/<name>.log.
    Doesn't overwrite previous runs.
    """
    log_path = LOGS_DIR / f"{name}.log"
    logger   = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Logging to {log_path}")
    return logger


# ── Results helpers ───────────────────────────────────────────────────────────

def results_path_for(model_name: str) -> Path:
    """Return the path for a model's results CSV."""
    return RESULTS_DIR / f"{model_name.replace(' ', '_').replace('+', 'plus')}_results.csv"
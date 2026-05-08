"""
src/logit_lens.py
Logit lens analysis: decode hidden states at each layer using the model's
unembedding matrix to track how the model's "prediction" evolves through layers.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


def get_logit_lens_predictions(
    model,
    tokenizer,
    text: str,
    top_k: int = 5,
    device: Optional[str] = None,
    token_position: int = -1,
):
    """
    Apply the logit lens: project each layer's hidden state through the
    unembedding matrix (lm_head) to get token predictions at each layer.

    Args:
        model: loaded CausalLM (must have model.lm_head or model.embed_out)
        tokenizer: corresponding tokenizer
        text: input string
        top_k: number of top predicted tokens to return per layer
        token_position: which input token position to analyze (-1 = last)

    Returns:
        layer_predictions: list of dicts (one per layer), each with:
            - 'layer': layer index
            - 'top_tokens': list of (token_str, probability) tuples
            - 'top_token': the single most likely token string
            - 'top_prob': probability of most likely token
    """
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get the unembedding matrix (works for Qwen2 / LLaMA style models)
    if hasattr(model, "lm_head"):
        unembed = model.lm_head
    elif hasattr(model, "embed_out"):
        unembed = model.embed_out
    else:
        raise AttributeError("Cannot find unembedding layer (lm_head or embed_out)")

    # Apply layer norm if present (Qwen2 has model.model.norm)
    try:
        norm = model.model.norm
    except AttributeError:
        norm = None

    layer_predictions = []
    for layer_idx, hs in enumerate(outputs.hidden_states):
        h = hs[0, token_position, :].unsqueeze(0)  # (1, hidden_dim)

        # Apply final layer norm (only makes full sense at the last layer,
        # but applying at each layer gives useful signal — standard logit lens)
        if norm is not None:
            h = norm(h)

        with torch.no_grad():
            logits = unembed(h).squeeze(0)  # (vocab_size,)

        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = probs.topk(top_k)

        top_tokens = [
            (tokenizer.decode([tid.item()]).strip(), prob.item())
            for tid, prob in zip(top_ids, top_probs)
        ]

        layer_predictions.append({
            "layer":      layer_idx,
            "top_tokens": top_tokens,
            "top_token":  top_tokens[0][0],
            "top_prob":   top_tokens[0][1],
        })

    return layer_predictions


def find_answer_commitment_layer(layer_predictions: list, answer_token: str) -> Optional[int]:
    """
    Find the earliest layer where the model's top prediction matches
    the expected answer token. Returns None if never committed.

    Args:
        layer_predictions: output of get_logit_lens_predictions
        answer_token: the expected answer string (e.g. "150")

    Returns:
        Layer index of first commitment, or None
    """
    for pred in layer_predictions:
        if answer_token.strip() in pred["top_token"]:
            return pred["layer"]
    return None


def compare_logit_lens(
    orig_preds: list,
    pert_preds: list,
) -> np.ndarray:
    """
    Compute KL divergence between original and perturbed layer distributions.
    Higher KL = more divergent at that layer.

    Args:
        orig_preds, pert_preds: outputs of get_logit_lens_predictions for a pair

    Returns:
        kl_divs: np.ndarray of shape (n_layers,)
    """
    # We only have top_k probs, so this is approximate
    kl_divs = []
    for o, p in zip(orig_preds, pert_preds):
        p_orig = np.array([t[1] for t in o["top_tokens"]])
        p_pert = np.array([t[1] for t in p["top_tokens"]])
        # Clip for numerical stability
        p_orig = np.clip(p_orig, 1e-10, 1)
        p_pert = np.clip(p_pert, 1e-10, 1)
        kl = np.sum(p_orig * np.log(p_orig / p_pert))
        kl_divs.append(kl)
    return np.array(kl_divs)

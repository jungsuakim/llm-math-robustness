"""
src/hidden_states.py
Extract layerwise hidden states and compute cosine similarity between
original and perturbed inputs.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


def get_hidden_states(
    model,
    tokenizer,
    text: str,
    device: Optional[str] = None,
    token_position: int = -1,   # -1 = last token
):
    """
    Run a forward pass and return the hidden states at every layer.

    Args:
        model: loaded HuggingFace CausalLM
        tokenizer: corresponding tokenizer
        text: input string
        device: torch device string
        token_position: which token's hidden state to extract (-1 = last input token)

    Returns:
        hidden_states: tensor of shape (n_layers + 1, hidden_dim)
                       index 0 = embedding layer, 1..n = transformer layers
    """
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )

    # outputs.hidden_states is a tuple of (n_layers + 1) tensors
    # each tensor: (batch=1, seq_len, hidden_dim)
    # Extract the hidden state at token_position for each layer
    hidden_states = torch.stack(
        [layer_hs[0, token_position, :] for layer_hs in outputs.hidden_states]
    )  # shape: (n_layers + 1, hidden_dim)

    return hidden_states.cpu().float()


def layerwise_cosine_similarity(hs_orig: torch.Tensor, hs_pert: torch.Tensor) -> np.ndarray:
    """
    Compute cosine similarity between original and perturbed hidden states at each layer.

    Args:
        hs_orig: (n_layers + 1, hidden_dim)
        hs_pert: (n_layers + 1, hidden_dim)

    Returns:
        similarities: np.ndarray of shape (n_layers + 1,), values in [-1, 1]
    """
    assert hs_orig.shape == hs_pert.shape, "Hidden state shapes must match"
    sims = F.cosine_similarity(hs_orig, hs_pert, dim=-1)
    return sims.numpy()


def find_critical_layer(similarities: np.ndarray, window: int = 3) -> int:
    """
    Find the layer where cosine similarity drops most sharply (the "critical layer" l*).
    Uses the largest negative gradient in a smoothed similarity profile.

    Args:
        similarities: array of per-layer cosine similarities
        window: smoothing window size

    Returns:
        Index of the critical layer
    """
    # Smooth with a simple moving average
    kernel = np.ones(window) / window
    smoothed = np.convolve(similarities, kernel, mode="same")
    gradients = np.diff(smoothed)
    return int(np.argmin(gradients)) + 1   # +1 because diff reduces length by 1


def batch_cosine_similarities(
    model,
    tokenizer,
    problem_pairs,
    device: Optional[str] = None,
):
    """
    Compute layerwise cosine similarities for a list of problem pairs.

    Args:
        model, tokenizer: loaded model
        problem_pairs: list of dicts with keys 'original', 'perturbed'
        device: torch device

    Returns:
        all_sims: np.ndarray of shape (n_pairs, n_layers + 1)
        critical_layers: list of int, one per pair
    """
    all_sims = []
    critical_layers = []

    for pair in problem_pairs:
        hs_orig = get_hidden_states(model, tokenizer, pair["original"], device)
        hs_pert = get_hidden_states(model, tokenizer, pair["perturbed"], device)
        sims = layerwise_cosine_similarity(hs_orig, hs_pert)
        all_sims.append(sims)
        critical_layers.append(find_critical_layer(sims))

    return np.array(all_sims), critical_layers

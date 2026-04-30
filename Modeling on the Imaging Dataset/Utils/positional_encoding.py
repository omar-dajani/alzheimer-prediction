"""
positional_encoding — Temporal and spatial positional encodings for longitudinal MRI.

Pipeline position:
    Phase 3 utility module. Consumed by LongformerSequence and, if implemented,
    any other sequence model that requires explicit temporal positional encoding.
    Mamba (Phase 4) does NOT use this module — it encodes time implicitly via ZOH
    discretization. That distinction must be preserved.

Components:
    CTLPE  — Continuous-Time Learnable Positional Embedding (purely linear).
             Formula: PE(t) = k*t + b  where k, b in R^d_model are learned.
             Source: arXiv:2409.20092, Theorem 3.1.
             NOT the same as mTAN which adds periodic terms.
             NOT injected as an attention bias — injected additively to token embeddings.

    SpatialSinusoidalPE — Fixed (non-learned) sinusoidal encoding for the 512 spatial
                          tokens within a visit. Distinguishes voxel positions in the
                          8x8x8 BrainIAC spatial grid. Shared across all visits and
                          all subjects — no temporal dependence.


Dependencies:
    - Transformer/config/model_config.py
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class CTLPE(nn.Module):
    """Continuous-Time Learnable Positional Embedding (purely linear).

    Implements the positional embedding:
    any positional embedding satisfying both monotonicity and translation
    invariance must be linear. The formula is:

        PE(t) = k * t + b

    where k in R^d_model is a learnable slope vector and b in R^d_model
    is a learnable bias vector.

    This is NOT mTAN, which adds periodic sin/cos
    terms for cyclical pattern expressivity. CTLPE contains zero
    trigonometric operations — it is purely linear.

    This is NOT ALiBi, which injects time as an
    attention bias matrix. CTLPE is injected additively to token
    embeddings: x = x + PE(t).

    All 512 spatial tokens within a visit share the same temporal PE
    value because they all originate from the same scan time.

    Tensor shapes:
        Input: visit_times — [B, V] (absolute times in months)
        Output: [B, V, d_model]

    Args:
        d_model: Embedding dimension (default 512 from ModelConfig).
    """

    def __init__(self, d_model: int) -> None:
        """Initialize CTLPE with learnable slope and bias vectors.

        Args:
            d_model: Embedding dimension. The slope and bias vectors
                will each have d_model elements.
        """
        super().__init__()
        # Learnable slope k — initialized small (std=0.02) following
        # standard transformer parameter initialization
        self.slope = nn.Parameter(torch.randn(d_model) * 0.02)
        # Learnable bias b — initialized to zero so PE(0) = b = 0
        # at the start of training
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, visit_times: Tensor) -> Tensor:
        """Compute temporal positional embeddings from absolute visit times.

        IMPORTANT take note: This method expects absolute visit times (e.g.,
        [0, 6, 18, 30] months from baseline), NOT inter-visit deltas
        (e.g., [0, 6, 12, 12]). If only deltas are available, compute
        cumulative sum first: visit_times = time_deltas.cumsum(dim=1).

        Args:
            visit_times: Absolute visit times in months of shape [B, V].
                Example: [[0, 6, 12], [0, 12, 24]] for two subjects with
                3 visits each at different intervals.

        Returns:
            Temporal embeddings of shape [B, V, d_model]. One embedding
            vector per visit, to be broadcast across all 512 tokens in
            that visit before additive injection.
        """
        # PE(t) = k * t + b — the complete CTLPE formula in one line.
        # unsqueeze(-1) broadcasts [B, V] to [B, V, 1] so multiplication
        # with self.slope [d_model] produces [B, V, d_model].
        return visit_times.unsqueeze(-1) * self.slope + self.bias


class SpatialSinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding for spatial token positions.

    Encodes the 512 distinct spatial positions within the 8x8x8 BrainIAC
    grid using the standard transformer sinusoidal formula:

        PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

    This encoding is:
        - Fixed (not learned) — registered as a buffer, not a parameter
        - Shared across all visits and all subjects — no temporal dependence
        - Deterministic — the same position always gets the same encoding

    Each token within a visit gets a unique spatial PE, but all tokens at
    the same spatial position across different visits and subjects share
    the same spatial PE. This complements the temporal PE from CTLPE.

    Callers should unsqueeze and broadcast as needed:
        pe.unsqueeze(0).unsqueeze(0) gives [1, 1, n_tokens, d_model]
        for broadcasting over batch and visit dimensions.

    Tensor shapes:
        Output: [n_tokens, d_model] (default [512, 512])

    Args:
        d_model: Embedding dimension.
        n_tokens: Number of spatial positions (default 512 = 8^3).
    """

    def __init__(self, d_model: int, n_tokens: int = 512) -> None:
        """Initialize fixed sinusoidal spatial positional encoding.

        Args:
            d_model: Embedding dimension.
            n_tokens: Number of spatial tokens per visit (512 for 8x8x8).
        """
        super().__init__()

        pe = torch.zeros(n_tokens, d_model) # [n_tokens, d_model]
        position = torch.arange(
            0, n_tokens, dtype=torch.float
        ).unsqueeze(1) # [n_tokens, 1]
        # Compute the divisor term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        ) # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term) # even indices
        pe[:, 1::2] = torch.cos(position * div_term) # odd indices

        # Register as buffer — not a learnable parameter, but moves
        # with the model to the correct device and dtype
        self.register_buffer("pe", pe)

    def forward(self) -> Tensor:
        """Return the fixed spatial positional encoding.

        No input required — this encoding is independent of input data.
        Callers should unsqueeze and broadcast as needed:
            pe = spatial_pe()  # [n_tokens, d_model]
            pe = pe.unsqueeze(0).unsqueeze(0)  # [1, 1, n_tokens, d_model]

        Returns:
            Tensor of shape [n_tokens, d_model] with fixed sinusoidal
            encodings for each spatial position.
        """
        return self.pe

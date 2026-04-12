"""
pooling — Pooling by Multi-Head Attention (PMA) for survival-relevant feature extraction.

Pipeline position:
    Phase 5 of the ADNI Advanced Survival Pipeline. Sits between the sequence
    model (Phase 3 or 4) and the survival head (Phase 6). Compresses the full
    contextualized token sequence into a fixed-size multi-aspect summary vector.

Inputs:
    Z: [B, N, d_model] — contextualized token sequence from LongformerSequence
       or Mamba3DSequence, where N = V * n_tokens_per_visit (e.g., 2560 for V=5).

Outputs:
    [B, m * d_model] — flattened multi-seed summary.
    For default m=8, d_model=512: output shape is [B, 4096].
    This tensor feeds directly into TraCeRSurvivalHead (Phase 6).

Why PMA over alternatives:
    CLS token: single bottleneck — cannot represent diverse atrophy patterns
    Average pool: uniform weights — background voxels dilute atrophy signal
    Max pool: discards magnitude — misses joint spatial progression patterns
    PMA: m independent learned queries extract m distinct morphological
         aspects simultaneously, with data-dependent importance weighting.

Seed vector interpretation (emerges during training, not fixed by design):
    Each of the m=8 seed vectors learns to attend to a different aspect of
    morphological progression: regional atrophy patterns, bilateral asymmetry,
    ventricular dilation trajectory, temporal acceleration of decline, etc.

Dependencies:
    - Transformer/config/model_config.py
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

import sys
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)
from Transformer.config.model_config import ModelConfig


logger = logging.getLogger(__name__)

# Default number of attention heads for MAB (d_model=512 must be divisible by this)
MAB_DEFAULT_NUM_HEADS = 8


class MAB(nn.Module):
    """Multi-head Attention Block

    A general building block that performs cross-attention between queries Q
    and keys/values K, followed by a residual feedforward. Used inside PMA
    where Q = learned seed vectors and K = V = contextualized token sequence.

    Formulation:
        H = LayerNorm(Q + MultiheadAttention(Q, K, K))
        out = LayerNorm(H + rFF(H))

    The cross-attention mechanism allows each query (seed) to attend to
    different parts of the key sequence with data-dependent importance
    weighting. This is fundamentally different from:
        - CLS pooling: single bottleneck query (m=1, learned from scratch)
        - Average pooling: uniform attention weights (no data dependence)
        - Max pooling: winner-take-all (no magnitude or joint patterns)

    MAB with m>1 queries captures m distinct morphological aspects
    simultaneously, each with independently learned attention patterns.

    Tensor shapes:
        Input: Q — [B, m, d_model] (seed vectors or intermediate queries)
        Input: K — [B, N, d_model] (token sequence, used as both K and V)
        Output: [B, m, d_model]

    Args:
        d_model: Embedding dimension.
        num_heads: Number of attention heads. Must divide d_model evenly.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = MAB_DEFAULT_NUM_HEADS,
    ) -> None:
        """Initialize the Multi-head Attention Block.

        Args:
            d_model: Embedding dimension.
            num_heads: Number of attention heads for multi-head attention.

        Raises:
            ValueError: If d_model is not divisible by num_heads.
        """
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads "
                f"({num_heads}). Current d_model % num_heads = "
                f"{d_model % num_heads}."
            )

        self.attention = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # rFF inside MAB: applied to H after attention + norm
        # Paper specifies Linear + ReLU — no GELU, no second layer, no LN
        self.rff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

    def forward(self, Q: Tensor, K: Tensor) -> Tensor:
        """Apply multi-head cross-attention with residual feedforward.

        Implements: H = LN(Q + Attn(Q, K, K)), out = LN(H + rFF(H))

        Args:
            Q: Query tensor [B, m, d_model]. In PMA, these are the
                learned seed vectors expanded across the batch.
            K: Key/Value tensor [B, N, d_model]. The contextualized
                token sequence from the sequence model. Used as both
                keys and values in the cross-attention.

        Returns:
            Attended output [B, m, d_model]. Same query count as input Q.
        """
        # Cross-attention: seeds attend to token sequence
        # nn.MultiheadAttention returns (attn_output, attn_weights) — use only output
        attn_output, _ = self.attention(Q, K, K) # [B, m, d_model]

        # Residual connection + LayerNorm
        H = self.norm1(Q + attn_output) # [B, m, d_model]

        # Row-wise feedforward with residual + LayerNorm
        rff_out = self.rff(H) # [B, m, d_model]
        out = self.norm2(H + rff_out) # [B, m, d_model]

        return out


class PMA(nn.Module):
    """Pooling by Multi-Head Attention

    Compresses a variable-length contextualized token sequence [B, N, d_model]
    into a fixed-size multi-aspect summary vector [B, m * d_model] using m
    learned seed vectors as attention queries.

    Formulation:
        PMA_m(Z) = MAB(S, rFF(Z))
        where S in R^{m x d_model} are learned seed vectors.

    Tensor shapes:
        Input:  Z — [B, N, d_model] where N is variable (V * 512)
        Output: [B, m * d_model] = [B, 4096] for default config

    Args:
        config: ModelConfig providing d_model and pma_seeds.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize PMA with learned seed vectors and attention block.

        Args:
            config: ModelConfig providing d_model (embedding dim) and
                pma_seeds (number of seed query vectors m).
        """
        super().__init__()
        self.m = config.pma_seeds
        self.d_model = config.d_model

        # Learned seed vectors — m independent queries that each extract
        # a different aspect of morphological progression
        # Leading dim=1 for batch broadcasting in forward()
        self.seeds = nn.Parameter(
            torch.randn(1, config.pma_seeds, config.d_model) * 0.02
        )

        # rFF applied to input Z BEFORE the MAB
        # Ablations show this pre-MAB rFF improves performance
        # Paper specifies Linear + ReLU — do not substitute GELU
        self.rff = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
        )

        # Multi-head Attention Block (separate module for testability)
        self.mab = MAB(config.d_model)

    def forward(self, Z: Tensor) -> Tensor:
        """Compress variable-length token sequence to fixed-size summary.

        Args:
            Z: Contextualized token sequence [B, N, d_model] from the
                sequence model (LongformerSequence or Mamba3DSequence).
                N = V * n_tokens_per_visit (variable across subjects).

        Returns:
            Flattened multi-seed summary [B, m * d_model].
            For default config: [B, 4096]. This tensor feeds directly
            into TraCeRSurvivalHead (Phase 6).
        """
        B = Z.size(0)

        # Step 1: Apply rFF to input before MAB
        # This pre-MAB feedforward is part of the PMA formulation and
        # should not be skipped or merged with the rFF inside MAB
        Z_prime = self.rff(Z) # [B, N, d_model]

        # Step 2: Expand seed vectors across batch dimension
        S = self.seeds.expand(B, -1, -1) # [B, m, d_model]

        # Step 3: MAB with seeds as queries, transformed input as keys/values
        # Each seed attends to the full token sequence independently
        out = self.mab(S, Z_prime) # [B, m, d_model]

        # Step 4: Flatten seed outputs into a single vector per subject
        # The survival head (Phase 6) expects [B, m * d_model]
        out = out.reshape(B, self.m * self.d_model) # [B, m * d_model]

        assert out.shape == (B, self.m * self.d_model), (
            f"PMA output shape mismatch: got {tuple(out.shape)}, "
            f"expected ({B}, {self.m * self.d_model}). Phase 6 "
            f"(TraCeRSurvivalHead) requires [B, m * d_model] input."
        )

        return out

    def output_dim(self) -> int:
        """Return the flattened output dimension.

        Pass this to TraCeRSurvivalHead as d_input to avoid hardcoded
        shape dependencies between Phase 5 and Phase 6.

        Returns:
            Integer m * d_model (default: 8 * 512 = 4096).
        """
        return self.m * self.d_model


# Smoke test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    print("Phase 5 — PMA Pooling Smoke Test")

    config = ModelConfig()
    pass_count = 0
    fail_count = 0

    # Test 1: num_heads divisibility check
    try:
        _ = MAB(d_model=512, num_heads=8)
        try:
            _ = MAB(d_model=512, num_heads=7)
            print("FAIL: divisibility check — no error raised")
            fail_count += 1
        except ValueError as e:
            assert len(str(e)) > 0
            print(f"PASS: divisibility check — {e}")
            pass_count += 1
    except Exception as e:
        print(f"FAIL: MAB instantiation — {e}")
        fail_count += 1

    # Test 2: MAB forward pass
    try:
        mab = MAB(d_model=512, num_heads=8)
        Q = torch.randn(2, 8, 512) # B=2, m=8 seeds
        K = torch.randn(2, 2560, 512) # B=2, N=2560 tokens
        with torch.no_grad():
            mab_out = mab(Q, K)
        assert mab_out.shape == (2, 8, 512), (
            f"MAB shape: {mab_out.shape}"
        )
        print(f"PASS: MAB shape — {tuple(mab_out.shape)}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: MAB forward — {e}")
        fail_count += 1

    # Test 3: PMA forward pass with default config
    try:
        pma = PMA(config)
        pma.eval()
        Z = torch.randn(2, 2560, 512) # B=2, N=5*512
        with torch.no_grad():
            pma_out = pma(Z)
        assert pma_out.shape == (2, 4096), f"PMA shape: {pma_out.shape}"
        print(f"PASS: PMA shape — {tuple(pma_out.shape)}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: PMA forward — {e}")
        fail_count += 1

    # Test 4: output_dim
    try:
        assert pma.output_dim() == 4096, (
            f"output_dim: {pma.output_dim()}"
        )
        print(f"PASS: output_dim — {pma.output_dim()}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: output_dim — {e}")
        fail_count += 1

    # Test 5: Variable N (V=3 visits instead of V=5)
    try:
        Z_short = torch.randn(2, 1536, 512) # N=3*512
        with torch.no_grad():
            out_short = pma(Z_short)
        assert out_short.shape == (2, 4096), (
            f"Variable N shape: {out_short.shape}"
        )
        print(f"PASS: variable N — {tuple(out_short.shape)}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: variable N — {e}")
        fail_count += 1

    # Test 6: Batch size 1 (inference edge case)
    try:
        Z_single = torch.randn(1, 2560, 512)
        with torch.no_grad():
            out_single = pma(Z_single)
        assert out_single.shape == (1, 4096), (
            f"Batch 1 shape: {out_single.shape}"
        )
        print(f"PASS: batch size 1 — {tuple(out_single.shape)}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: batch size 1 — {e}")
        fail_count += 1

    # Test 7: rFF bypass shape stability
    try:
        pma_bypass = PMA(config)
        pma_bypass.eval()
        pma_bypass.rff = nn.Identity() # monkey-patch rFF out
        Z_test = torch.randn(2, 2560, 512)
        with torch.no_grad():
            out_bypass = pma_bypass(Z_test)
        assert out_bypass.shape == (2, 4096), (
            f"rFF bypass shape: {out_bypass.shape}"
        )
        print(f"PASS: rFF bypass shape stability — {tuple(out_bypass.shape)}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: rFF bypass — {e}")
        fail_count += 1

    # Summary
    print(f"\nResults: {pass_count} PASS, {fail_count} FAIL")
    if fail_count == 0:
        print("PASS — All assertions hold.")
    else:
        print("FAIL — Some tests did not pass.")

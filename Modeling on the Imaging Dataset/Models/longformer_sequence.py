"""
longformer_sequence — Longformer-style longitudinal sequence model using
PyTorch-native Scaled Dot Product Attention (SDPA) with sliding window masks.

Pipeline position:
    Phase 3 of the ADNI Advanced Survival Pipeline. Receives the flattened spatial
    token sequence from Phase 1 (BrainIAC) and returns a contextualized sequence
    of the same shape for consumption by Phase 5 (PMA pooling).
Inputs:
    x: [B, V * 512, d_model] — flat token sequence from BrainIAC extractor
    time_deltas: [B, V] — inter-visit intervals in months (delta_t[0] = 0)
    missing_mask: [B, V] — 1 = visit present, 0 = missing or padded

Outputs:
    [B, V * 512, d_model] — contextualized token sequence, same shape as input x.

Architecture sequence (in forward() order):
    1. validate_shapes() — runtime contract enforcement
    2. Reshape x to [B, V, 512, d_model] — restore visit structure
    3. ModalityDropout on content — replace missing/dropped visits
    4. Compute temporal PE via CTLPE — [B, V, d_model]
    5. Add spatial PE — [512, d_model] broadcast
    6. Flatten back to [B, V*512, d_model] — attention expects flat sequence
    7. Prepend V global separator tokens — [B, V*512 + V, d_model]
    8. Build sliding window + global mask — boolean attn_mask for SDPA
    9. Run SDPA transformer layers — native PyTorch attention
    10. Strip separator tokens — restore [B, V*512, d_model]
    11. Validate output shape before return

"""

import logging
from pathlib import Path

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)
from Transformer.config.model_config import ModelConfig
from Transformer.models.base import BaseSequenceModel, ModalityDropout
from Transformer.utils.positional_encoding import CTLPE, SpatialSinusoidalPE


logger = logging.getLogger(__name__)

# Longformer hyperparameters not exposed in ModelConfig
# (architecture-internal, not tuned across experiments)
NUM_ATTENTION_HEADS = 8
NUM_HIDDEN_LAYERS = 6
INTERMEDIATE_MULTIPLIER = 4


def _build_sliding_window_mask(
    seq_len: int,
    window_size: int,
    global_positions: list,
    device: torch.device,
) -> Tensor:
    """Build a boolean attention mask with sliding window + global tokens.

    Creates a [seq_len, seq_len] boolean mask where:
        - True = ALLOW attention (can attend)
        - False = BLOCK attention (masked out)
    Within the sliding window, tokens can attend to neighbors within
    ±(window_size // 2). Global positions can attend to ALL tokens
    and ALL tokens can attend to global positions.

    This replaces HuggingFace's Longformer attention implementation with
    native PyTorch boolean masking compatible with SDPA on all backends
    (MPS, CUDA, CPU).

    Args:
        seq_len: Total sequence length including separators.
        window_size: Sliding window width (tokens attend to ±w/2).
        global_positions: List of token indices with global attention.
        device: Device for the output tensor.

    Returns:
        Boolean tensor [seq_len, seq_len] for use as attn_mask in SDPA.
    """
    # Start with no attention allowed
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # Sliding window: token i can attend to [i - w/2, i + w/2]
    half_w = window_size // 2
    for i in range(seq_len):
        lo = max(0, i - half_w)
        hi = min(seq_len, i + half_w + 1)
        mask[i, lo:hi] = True

    # Global positions: bidirectional attention to/from all tokens
    for pos in global_positions:
        mask[pos, :] = True # Global token attends to all
        mask[:, pos] = True # All tokens attend to global token

    return mask # [seq_len, seq_len]


class SDPATransformerLayer(nn.Module):
    """Single transformer layer using PyTorch-native SDPA.

    Replaces HuggingFace LongformerLayer with a clean implementation that
    works on MPS, CUDA, and CPU. Uses pre-norm (LayerNorm before attention
    and FFN) for training stability with deep networks.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward intermediate dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Pre-norm LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """Forward pass with masked SDPA attention.

        Args:
            x: Input tensor [B, N, d_model].
            attn_mask: Boolean mask [N, N] (True = allow attention).

        Returns:
            Output tensor [B, N, d_model].
        """
        B, N, D = x.shape

        # Pre-norm + multi-head attention
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [B, n_heads, N, head_dim]

        # SDPA with boolean mask — hardware-accelerated on MPS and CUDA
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,  # [N, N] broadcast to [B, n_heads, N, N]
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )  # [B, n_heads, N, head_dim]

        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        attn_out = self.out_proj(attn_out)
        x = x + attn_out  # Residual

        # Pre-norm + FFN
        x = x + self.ffn(self.norm2(x))  # Residual

        return x


class LongformerSequence(BaseSequenceModel):
    """Longformer-style sequence model using native PyTorch SDPA.

    Processes a flattened multi-visit spatial token sequence using sparse
    sliding-window attention with global visit-separator tokens. Returns
    a contextualized sequence of the same shape for downstream PMA pooling.

    Phase 3/4 modernization:
        Replaced HuggingFace LongformerModel with native PyTorch SDPA
        (torch.nn.functional.scaled_dot_product_attention). The sliding
        window sparsity is achieved via a boolean mask constructed natively.

    Window size strategy:
        w = 512 = n_tokens_per_visit. Every token can attend to every
        other token within its own visit (full intra-visit attention).
        Cross-visit information flows through global separator tokens.

    Tensor shapes:
        Input: x — [B, V * 512, d_model]
        Output: [B, V * 512, d_model] (same shape — immutable contract)

    Args:
        config: ModelConfig providing d_model, n_tokens_per_visit,
            longformer_window, and modality_dropout_p.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the Longformer sequence model.

        Args:
            config: ModelConfig with all pipeline hyperparameters.
        """
        super().__init__(config)

        # Positional encoding modules
        self.ctlpe = CTLPE(config.d_model)
        self.spatial_pe = SpatialSinusoidalPE(
            config.d_model, config.n_tokens_per_visit
        )

        # Visit-level modality dropout 
        self.modality_dropout = ModalityDropout(config)

        # Global separator token embedding — one learned vector shared
        # across all V separator positions.
        self.separator_embedding = nn.Parameter(
            torch.randn(config.d_model) * 0.02
        )

        # Native SDPA transformer stack (replaces HuggingFace Longformer)
        d_ff = config.d_model * INTERMEDIATE_MULTIPLIER
        self.layers = nn.ModuleList([
            SDPATransformerLayer(
                d_model=config.d_model,
                n_heads=NUM_ATTENTION_HEADS,
                d_ff=d_ff,
                dropout=0.1,
            )
            for _ in range(NUM_HIDDEN_LAYERS)
        ])

        # Final LayerNorm (post-transformer normalization)
        self.final_norm = nn.LayerNorm(config.d_model)

        self._config = config
        self._window_size = config.longformer_window

        logger.info(
            "Built native SDPA LongformerSequence (heads=%d, layers=%d, "
            "window=%d). No HuggingFace dependency.",
            NUM_ATTENTION_HEADS,
            NUM_HIDDEN_LAYERS,
            config.longformer_window,
        )

    def _build_global_attention_mask(
        self,
        batch_size: int,
        v: int,
        n: int,
        device: torch.device,
    ) -> Tensor:
        """Build SDPA boolean attention mask with sliding window + global tokens.

        Creates a [total_len, total_len] boolean mask for SDPA where:
            - True = ALLOW attention
            - False = BLOCK attention

        Separator layout in the interleaved sequence:
            [sep_0, tok_0..511, sep_1, tok_512..1023, ...]
        Separator positions: [0, 513, 1026, ...] — every (512 + 1)th position.

        Args:
            batch_size: Batch size B (unused, mask is shared).
            v: Number of visits V.
            n: Number of content tokens (V * 512).
            device: Device for the output tensor.

        Returns:
            Boolean tensor [total_len, total_len] for SDPA attn_mask.
        """
        total_len = n + v  # content tokens + separator tokens
        sep_stride = self.n_tokens_per_visit + 1
        separator_positions = [i * sep_stride for i in range(v)]

        return _build_sliding_window_mask(
            seq_len=total_len,
            window_size=self._window_size,
            global_positions=separator_positions,
            device=device,
        )

    def _interleave_separators(
        self,
        x: Tensor,
        sep_tokens: Tensor,
        v: int,
    ) -> Tensor:
        """Interleave separator tokens before each visit's token block.

        Takes the flat content sequence and inserts a separator token
        before each visit's 512 tokens, producing:
            [sep_0, tokens_visit_0, sep_1, tokens_visit_1, ...]

        Args:
            x: Flat content tokens [B, V * 512, d_model].
            sep_tokens: Separator embeddings [B, V, d_model] (with PE).
            v: Number of visits.

        Returns:
            Interleaved sequence [B, V * 512 + V, d_model].
        """
        b = x.shape[0]
        # Reshape content to per-visit blocks
        x_visits = x.reshape(
            b, v, self.n_tokens_per_visit, self.d_model
        )  # [B, V, 512, d_model]

        # Insert separator before each visit block
        # sep_tokens: [B, V, d_model] -> [B, V, 1, d_model]
        sep_expanded = sep_tokens.unsqueeze(2)  # [B, V, 1, d_model]

        # Concatenate sep + tokens per visit: [B, V, 513, d_model]
        interleaved = torch.cat(
            [sep_expanded, x_visits], dim=2
        )  # [B, V, 513, d_model]

        # Flatten visit dimension: [B, V * 513, d_model]
        return interleaved.reshape(
            b, v * (self.n_tokens_per_visit + 1), self.d_model
        )

    def _strip_separators(
        self,
        x: Tensor,
        v: int,
    ) -> Tensor:
        """Remove separator tokens from the interleaved sequence.

        Reverses the interleaving: extracts only the content token
        positions, discarding the V separator tokens.

        Args:
            x: Interleaved sequence [B, V * 513, d_model].
            v: Number of visits.

        Returns:
            Content-only sequence [B, V * 512, d_model].
        """
        b = x.shape[0]
        # Reshape to per-visit blocks of 513 (1 sep + 512 content)
        x = x.reshape(
            b, v, self.n_tokens_per_visit + 1, self.d_model
        ) # [B, V, 513, d_model]
        # Slice out the content tokens (skip index 0 = separator)
        content = x[:, :, 1:, :] # [B, V, 512, d_model]
        # Flatten back to [B, V * 512, d_model]
        return content.reshape(
            b, v * self.n_tokens_per_visit, self.d_model
        )

    def forward(
        self,
        x: Tensor,
        time_deltas: Tensor,
        missing_mask: Tensor,
    ) -> Tensor:
        """Process flattened longitudinal token sequence through SDPA Longformer.

        Args:
            x: Flattened token sequence [B, V * 512, d_model].
            time_deltas: Inter-visit intervals [B, V] in months.
                Baseline visit has time_delta = 0.
            missing_mask: Visit presence mask [B, V].
                1 = present, 0 = missing/padded.

        Returns:
            Contextualized token sequence [B, V * 512, d_model],
            same shape as input x.

        Raises:
            ValueError: If input shapes violate the contract.
            RuntimeError: If output shape does not match input shape.
        """
        # Step 1: Runtime contract enforcement
        self.validate_shapes(x, time_deltas, missing_mask)
        b = x.shape[0]
        v = time_deltas.shape[1]
        n = v * self.n_tokens_per_visit  # total content tokens

        # Step 2: Restore visit structure for per-visit operations
        x = x.reshape(
            b, v, self.n_tokens_per_visit, self.d_model
        )  # [B, V, 512, d_model]

        # Step 3: ModalityDropout — replace missing/dropped visits
        # Applied to content BEFORE adding positional encoding so that
        # temporal PE is preserved even for dropped visits
        x = self.modality_dropout(x, missing_mask)
        # x: [B, V, 512, d_model]

        # Step 4a: Compute absolute visit times from inter-visit deltas
        visit_times = time_deltas.cumsum(dim=1) # [B, V]

        # Step 4b: Compute temporal PE via CTLPE (purely linear)
        temporal_pe = self.ctlpe(visit_times) # [B, V, d_model]

        # Step 4c: Expand temporal PE from per-visit to per-token
        # All 512 tokens within a visit share the same temporal PE
        temporal_pe_expanded = temporal_pe.unsqueeze(2).expand(
            b, v, self.n_tokens_per_visit, self.d_model
        ) # [B, V, 512, d_model]

        # Step 5: Add spatial PE — distinguishes voxel positions within visit
        # spatial_pe: [512, d_model] -> [1, 1, 512, d_model] for broadcast
        spatial_pe = self.spatial_pe() # [512, d_model]
        spatial_pe = spatial_pe.unsqueeze(0).unsqueeze(0) # [1, 1, 512, d_model]

        # Additive injection: x = content + temporal_pe + spatial_pe
        # This is the CTLPE injection method
        x = x + temporal_pe_expanded + spatial_pe
        # x: [B, V, 512, d_model]

        # Step 6: Flatten back to [B, V*512, d_model] for attention
        x = x.reshape(b, n, self.d_model) # [B, V*512, d_model]

        # Step 7: Prepare global separator tokens with temporal PE
        # Separator content is shared; temporal PE differentiates them
        sep = self.separator_embedding.unsqueeze(0).unsqueeze(0).expand(
            b, v, self.d_model
        ) # [B, V, d_model]
        sep_pe = self.ctlpe(visit_times) # [B, V, d_model]
        sep = sep + sep_pe # [B, V, d_model]

        # Interleave separators into the token sequence
        x = self._interleave_separators(x, sep, v)
        # x: [B, V*513, d_model]

        # Step 8: Build SDPA boolean attention mask (sliding window + global)
        attn_mask = self._build_global_attention_mask(
            b, v, n, x.device
        )  # [total_len, total_len]

        # Step 9: Run native SDPA transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.final_norm(x)
        # x: [B, V*513, d_model]

        # Step 10: Strip separator tokens to restore content-only sequence
        x = self._strip_separators(x, v)
        # x: [B, V*512, d_model]

        # Step 11: Validate output shape matches contract
        expected_shape = (b, n, self.d_model)
        if x.shape != expected_shape:
            raise RuntimeError(
                f"Output shape mismatch: got {tuple(x.shape)}, "
                f"expected {expected_shape}. The sequence model must "
                f"preserve [B, V*512, d_model] shape."
            )

        return x


# Smoke test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    print("Phase 3 — LongformerSequence (SDPA) Smoke Test")

    config = ModelConfig()
    pass_count = 0
    fail_count = 0

    # Test 1: Instantiate LongformerSequence
    try:
        model = LongformerSequence(config)
        model.eval()
        print("PASS: LongformerSequence instantiated (native SDPA)")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: LongformerSequence instantiation — {e}")
        fail_count += 1

    # Test 2: Forward pass with valid input (B=2, V=3)
    b, v = 2, 3
    n = v * config.n_tokens_per_visit  # 1536
    x = torch.randn(b, n, config.d_model)
    time_deltas = torch.tensor([[0.0, 6.0, 12.0], [0.0, 12.0, 24.0]])
    missing_mask = torch.ones(b, v)

    try:
        with torch.no_grad():
            output = model(x, time_deltas, missing_mask)
        assert output.shape == (b, n, config.d_model), (
            f"Shape mismatch: {output.shape} != {(b, n, config.d_model)}"
        )
        print(f"PASS: forward pass — output shape {tuple(output.shape)}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: forward pass — {e}")
        fail_count += 1

    # Test 3: Output shape equals input shape (immutable contract)
    try:
        assert output.shape == x.shape, (
            f"Contract violation: output {output.shape} != input {x.shape}"
        )
        print("PASS: output shape == input shape (contract preserved)")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: shape contract — {e}")
        fail_count += 1

    # Test 4: Missing visit handling
    try:
        missing_mask_partial = torch.ones(b, v)
        missing_mask_partial[0, 2] = 0.0  # subject 0, visit 2 missing
        with torch.no_grad():
            output_missing = model(x, time_deltas, missing_mask_partial)
        assert output_missing.shape == (b, n, config.d_model)
        print("PASS: missing visit handled")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: missing visit — {e}")
        fail_count += 1

    # Test 5: Sliding window + global attention mask
    try:
        mask = model._build_global_attention_mask(
            batch_size=2, v=3, n=1536, device=torch.device("cpu")
        )
        total_len = 1536 + 3  # 1539
        assert mask.shape == (total_len, total_len), (
            f"Mask shape: {mask.shape} != ({total_len}, {total_len})"
        )
        # Separator positions: 0, 513, 1026 — should have global attention
        assert mask[0, :].all(), "Position 0 (separator) should attend to all"
        assert mask[:, 0].all(), "All should attend to position 0 (separator)"
        assert mask[513, :].all(), "Position 513 (separator) should attend to all"
        print(f"PASS: SDPA attention mask — shape {tuple(mask.shape)}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: attention mask — {e}")
        fail_count += 1

    # Test 6: CTLPE output shape and linearity
    try:
        ctlpe = CTLPE(config.d_model)
        visit_times = torch.tensor([[0.0, 6.0, 18.0], [0.0, 12.0, 24.0]])
        pe_out = ctlpe(visit_times)
        assert pe_out.shape == (2, 3, config.d_model), (
            f"CTLPE shape: {pe_out.shape}"
        )

        # Test linearity: with slope=1 and bias=0, PE(2t) = 2*PE(t)
        ctlpe_test = CTLPE(config.d_model)
        ctlpe_test.slope.data.fill_(1.0)
        ctlpe_test.bias.data.fill_(0.0)
        t1 = torch.tensor([[1.0]])
        t2 = torch.tensor([[2.0]])
        pe1 = ctlpe_test(t1)
        pe2 = ctlpe_test(t2)
        assert torch.allclose(pe2, 2.0 * pe1), "CTLPE is not linear"
        print(f"PASS: CTLPE — shape {tuple(pe_out.shape)}, linearity verified")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: CTLPE — {e}")
        fail_count += 1

    # Summary
    print(f"\nResults: {pass_count} PASS, {fail_count} FAIL")
    if fail_count == 0:
        print("PASS — All assertions hold.")
    else:
        print("FAIL — Some tests did not pass.")

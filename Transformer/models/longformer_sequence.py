"""
longformer_sequence — Longformer-based longitudinal sequence model for DVF token streams.

Pipeline position:
    Phase 3 of the ADNI Advanced Survival Pipeline. Receives the flattened spatial
    token sequence from Phase 1 (BrainIAC) and returns a contextualized sequence
    of the same shape for consumption by Phase 5 (PMA pooling).

Inputs:
    x: [B, V * 512, d_model] — flat token sequence from BrainIAC extractor
    time_deltas: [B, V] — inter-visit intervals in months (delta_t[0] = 0 for baseline)
    missing_mask: [B, V] — 1 = visit present, 0 = missing or padded

Outputs:
    [B, V * 512, d_model] — contextualized token sequence, same shape as input x.

Architecture sequence (in forward() order):
    1. validate_shapes()                      — runtime contract enforcement
    2. Reshape x to [B, V, 512, d_model]      — restore visit structure for PE injection
    3. ModalityDropout on content             — replace missing/dropped visits
    4. Compute temporal PE via CTLPE          — [B, V, d_model] -> expand to [B, V, 512, d_model]
    5. Add spatial PE                         — [512, d_model] broadcast across B and V
    6. Flatten back to [B, V*512, d_model]    — Longformer expects flat sequence
    7. Prepend V global separator tokens      — [B, V*512 + V, d_model]
    8. Build global attention mask            — 1 at separator positions, 0 elsewhere
    9. Run Longformer attention layers        — full sparse + global attention
    10. Strip separator tokens                — restore [B, V*512, d_model]
    11. validate output shape before return
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
from Transformer.models.base import BaseSequenceModel, ModalityDropout
from Transformer.utils.positional_encoding import CTLPE, SpatialSinusoidalPE


logger = logging.getLogger(__name__)

# Longformer hyperparameters not exposed in ModelConfig
# (architecture-internal, not tuned across experiments)
NUM_ATTENTION_HEADS = 8
NUM_HIDDEN_LAYERS = 6
INTERMEDIATE_MULTIPLIER = 4


class LongformerSequence(BaseSequenceModel):
    """Longformer-based sequence model for longitudinal DVF token streams.

    Processes a flattened multi-visit spatial token sequence using sparse
    sliding-window attention with global visit-separator tokens. Returns
    a contextualized sequence of the same shape for downstream PMA pooling.

    Design rationale over alternatives:
        - Standard Transformer: O(N^2) is infeasible for N=2560 with 128^3
          volumetric inputs already consuming most GPU memory.
        - Performer/LinFormer: approximate attention trades accuracy for
          speed. Longformer's exact sparse attention is preferable when the
          sparsity pattern (visit-aligned windows) has semantic meaning.
        - Mamba (Phase 4): SSM-based, handles irregular time natively via
          ZOH discretization. Longformer + CTLPE is the baseline; Mamba is
          the upgrade path.

    Window size strategy:
        w = 512 = n_tokens_per_visit. This is not a coincidence — it ensures
        every token can attend to every other token within its own visit
        (full intra-visit attention). Cross-visit information flows
        exclusively through global separator tokens, making the attention
        pattern interpretable: window boundary = visit boundary.

    Global separator tokens:
        V learned separator tokens are prepended to the sequence before the
        Longformer layers. These tokens have global attention (attend to all
        N tokens bidirectionally), creating information bridges between
        visits. A global token at visit boundary k enables 2-hop information
        flow: Visit_1 -> Global -> Visit_V. Separators are stripped after
        the Longformer layers to restore the original sequence shape.

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

        # Visit-level modality dropout (shared component from base.py)
        self.modality_dropout = ModalityDropout(config)

        # Global separator token embedding — one learned vector shared
        # across all V separator positions. Separators are differentiated
        # by their temporal PE (each gets the PE of its corresponding
        # visit), not by their content embedding.
        self.separator_embedding = nn.Parameter(
            torch.randn(config.d_model) * 0.02
        )

        # Longformer attention stack via HuggingFace
        self._build_longformer(config)

        self._config = config

    def _build_longformer(self, config: ModelConfig) -> None:
        """Build the HuggingFace Longformer attention stack.

        Uses LongformerModel with custom configuration matching the
        pipeline's d_model and attention window. The Longformer's own
        positional embeddings are disabled (we inject CTLPE + spatial PE
        externally), and the pooler is disabled.

        Args:
            config: ModelConfig providing d_model and longformer_window.
        """
        try:
            from transformers import (
                LongformerConfig,
                LongformerModel,
            )

            longformer_config = LongformerConfig(
                hidden_size=config.d_model,
                num_attention_heads=NUM_ATTENTION_HEADS,
                num_hidden_layers=NUM_HIDDEN_LAYERS,
                intermediate_size=config.d_model * INTERMEDIATE_MULTIPLIER,
                attention_window=[
                    config.longformer_window
                ] * NUM_HIDDEN_LAYERS,
                # Must be large enough for max possible sequence length:
                # v_max * n_tokens_per_visit + v_max separators
                max_position_embeddings=(
                    config.v_max * config.n_tokens_per_visit
                    + config.v_max + 2
                ),
                # We handle positional encoding externally via CTLPE
                # and SpatialSinusoidalPE — disable HuggingFace's
                # internal position embeddings by zeroing them after init
            )
            self.longformer = LongformerModel(longformer_config)
            # Zero out the internal positional embeddings — we use CTLPE
            # and SpatialSinusoidalPE instead
            self.longformer.embeddings.position_embeddings.weight.data.zero_()
            self.longformer.embeddings.position_embeddings.weight.requires_grad = False
            logger.info(
                "Built HuggingFace LongformerModel (heads=%d, layers=%d, "
                "window=%d).",
                NUM_ATTENTION_HEADS,
                NUM_HIDDEN_LAYERS,
                config.longformer_window,
            )
        except ImportError:
            raise ImportError(
                "HuggingFace transformers is required for LongformerSequence. "
                "Install via: pip install transformers"
            )

    def _build_global_attention_mask(
        self,
        batch_size: int,
        v: int,
        n: int,
        device: torch.device,
    ) -> Tensor:
        """Build the global attention mask for Longformer.

        Creates a binary mask where separator token positions receive
        value 1 (global attention — attends to all tokens bidirectionally)
        and content token positions receive value 0 (local sliding-window
        attention only).

        Global tokens create 2-hop information paths between visits:
        Visit_1 tokens -> Global_1 separator -> Visit_V tokens.
        This is how cross-visit context propagates without every token
        attending to the full sequence.

        Separator layout in the interleaved sequence:
            [sep_0, tok_0..511, sep_1, tok_512..1023, sep_2, tok_1024..1535, ...]
        Separator positions: [0, 513, 1026, ...] — every (512 + 1)th position.

        Args:
            batch_size: Batch size B.
            v: Number of visits V.
            n: Number of content tokens (V * 512).
            device: Device for the output tensor.

        Returns:
            Long tensor of shape [B, N + V] with 1 at separator positions
            and 0 elsewhere.
        """
        total_len = n + v  # content tokens + separator tokens
        mask = torch.zeros(
            batch_size, total_len, dtype=torch.long, device=device
        )  # [B, N + V]
        # Separator positions: 0, 513, 1026, ... (every 512+1 tokens)
        sep_stride = self.n_tokens_per_visit + 1
        separator_positions = [i * sep_stride for i in range(v)]
        mask[:, separator_positions] = 1
        return mask

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
        """Process flattened longitudinal token sequence through Longformer.

        Follows the 11-step architecture sequence documented in the
        module docstring. Each step has an inline comment showing the
        tensor shape after that operation.

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
            ValueError: If input shapes violate the contract (via
                validate_shapes).
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
        # This is the CTLPE injection method (Algorithm 1, arXiv:2409.20092)
        x = x + temporal_pe_expanded + spatial_pe
        # x: [B, V, 512, d_model]

        # Step 6: Flatten back to [B, V*512, d_model] for Longformer
        x = x.reshape(b, n, self.d_model)  # [B, V*512, d_model]

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

        # Step 8: Build global attention mask
        global_attention_mask = self._build_global_attention_mask(
            b, v, n, x.device
        )  # [B, V*513]

        # Step 9: Run Longformer attention layers
        # inputs_embeds bypasses the embedding layer — we provide
        # already-embedded tokens with our custom positional encoding
        longformer_output = self.longformer(
            inputs_embeds=x,
            global_attention_mask=global_attention_mask,
        )
        x = longformer_output.last_hidden_state # [B, V*513, d_model]

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

    print("Phase 3 — LongformerSequence Smoke Test")

    config = ModelConfig()
    pass_count = 0
    fail_count = 0

    # Test 1: Instantiate LongformerSequence
    try:
        model = LongformerSequence(config)
        model.eval()
        print("PASS: LongformerSequence instantiated")
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

    # Test 5: Global attention mask
    try:
        mask = model._build_global_attention_mask(
            batch_size=2, v=3, n=1536, device=torch.device("cpu")
        )
        assert mask.shape == (2, 1539), (
            f"Mask shape: {mask.shape} != (2, 1539)"
        )
        # Separator positions: 0, 513, 1026
        assert mask[0, 0].item() == 1, "Position 0 should be global"
        assert mask[0, 513].item() == 1, "Position 513 should be global"
        assert mask[0, 1026].item() == 1, "Position 1026 should be global"
        # Non-separator positions should be 0
        assert mask[0, 1].item() == 0, "Position 1 should be local"
        assert mask[0, 512].item() == 0, "Position 512 should be local"
        assert mask[0, 514].item() == 0, "Position 514 should be local"
        # Total global tokens should equal V
        assert mask[0].sum().item() == 3, "Should have exactly V=3 global tokens"
        print(f"PASS: global attention mask — shape {tuple(mask.shape)}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: global attention mask — {e}")
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

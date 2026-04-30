"""
base — Abstract base class and shared components for longitudinal sequence models.

Pipeline position:
    Phase 2 of the ADNI Advanced Survival Pipeline. This defines the
    interface contract that all sequence models (Longformer, Mamba) must satisfy.
    It has no runtime dependency on any other Transformer module except config.

Inputs (contract defined here, enforced in subclasses):
    x: [B, V * n_tokens_per_visit, d_model] — flat token sequence
    time_deltas: [B, V] — inter-visit intervals in months (0 for baseline visit)
    missing_mask: [B, V] — 1 = visit present, 0 = visit missing or padded

Outputs (contract defined here, enforced in subclasses):
    [B, V * n_tokens_per_visit, d_model] — contextualized token sequence,
    same shape as input x.

Components:
    BaseSequenceModel — ABC enforcing the forward() contract and shape validation.
    ModalityDropout — Shared visit-level dropout with learned default embeddings.

Design rationale:
    Longformer (Phase 3) and Mamba (Phase 4) are drop-in interchangeable via this
    interface. Swapping sequence models requires zero changes to pooling (Phase 5),
    survival head (Phase 6), or the trainer (Phase 9).

Dependencies:
    - Transformer/config/model_config.py
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

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


# BaseSequenceModel — Abstract Base Class

class BaseSequenceModel(ABC, nn.Module):
    """
    This ABC defines the interface that both the Longformer (Phase 3) and
    Mamba (Phase 4) sequence models must satisfy. Using an ABC over a simple
    base class provides two enforcement mechanisms:

    1. Static enforcement — Python's ABC machinery raises TypeError at
       instantiation time if a subclass does not implement forward().
       The error surfaces immediately, not after a full training loop.
    2. Runtime enforcement — validate_shapes() checks tensor dimensions
       at the start of every forward() call, catching shape mismatches
       before they propagate into silent broadcast bugs in attention layers.

    Direct instantiation of this class raises TypeError.
    Subclasses must implement forward().

    Tensor shapes:
        Input:  x — [B, V * n_tokens_per_visit, d_model]
        Input:  time_deltas — [B, V]
        Input:  missing_mask — [B, V]
        Output: [B, V * n_tokens_per_visit, d_model]
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the base sequence model with pipeline configuration.
        Direct instantiation of this class raises TypeError.
        Subclasses must implement forward().
        Args:
            config: A ModelConfig instance providing d_model and
                n_tokens_per_visit for shape validation.
        """
        super().__init__()
        self.d_model = config.d_model
        self.n_tokens_per_visit = config.n_tokens_per_visit

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        time_deltas: Tensor,
        missing_mask: Tensor,
    ) -> Tensor:
        """Process a flattened longitudinal token sequence.

        This contract is immutable. Subclasses must not change input
        or output shapes. Violating this will silently break Phase 5
        (PMA pooling) because it expects [B, N, d_model] output.

        Args:
            x: Flattened token sequence of shape
                [B, V * n_tokens_per_visit, d_model] where V is the
                number of visits and each visit contributes
                n_tokens_per_visit (512) spatial tokens from Phase 1.
            time_deltas: Inter-visit time intervals of shape [B, V],
                in months. The baseline visit has time_delta = 0.
            missing_mask: Binary visit presence mask of shape [B, V].
                1 = visit present, 0 = visit missing or padded.

        Returns:
            Contextualized token sequence of shape
            [B, V * n_tokens_per_visit, d_model], same shape as input x.
        """
        ...

    def validate_shapes(
        self,
        x: Tensor,
        time_deltas: Tensor,
        missing_mask: Tensor,
    ) -> None:
        """Validate input tensor shapes against the I/O contract.

        Should be called at the top of every subclass forward() during
        development. Can be guarded with an if-flag for production to
        avoid overhead.

        Args:
            x: Token sequence, expected [B, N, d_model].
            time_deltas: Visit time intervals, expected [B, V].
            missing_mask: Visit presence mask, expected [B, V].

        Raises:
            ValueError: If any tensor shape does not match the contract,
                with a descriptive message showing actual vs expected.
        """
        # Check 1: x must be 3-dimensional
        if x.ndim != 3:
            raise ValueError(
                f"x must be a 3D tensor [B, N, d_model], "
                f"got {x.ndim}D tensor with shape {tuple(x.shape)}."
            )

        b, n, d = x.shape

        # Check 2: embedding dimension must match config
        if d != self.d_model:
            raise ValueError(
                f"x has d_model={d} but this model was configured with "
                f"d_model={self.d_model}. Check ModelConfig.d_model."
            )

        # Check 3: batch sizes must match across all inputs
        if time_deltas.shape[0] != b:
            raise ValueError(
                f"Batch size mismatch: x has batch={b} but "
                f"time_deltas has batch={time_deltas.shape[0]}."
            )

        v = time_deltas.shape[1]

        # Check 4: token count must equal V * n_tokens_per_visit
        expected_n = v * self.n_tokens_per_visit
        if n != expected_n:
            raise ValueError(
                f"Token count mismatch: x has N={n} tokens but "
                f"time_deltas implies V={v} visits, so expected "
                f"N = V * n_tokens_per_visit = {v} * "
                f"{self.n_tokens_per_visit} = {expected_n}."
            )

        # Check 5: missing_mask shape must align with visits
        expected_mask_shape = (b, v)
        if missing_mask.shape != expected_mask_shape:
            raise ValueError(
                f"missing_mask shape mismatch: got "
                f"{tuple(missing_mask.shape)}, expected "
                f"{expected_mask_shape} (matching batch and visit dims)."
            )

    def expand_visit_to_tokens(
        self,
        visit_tensor: Tensor,
        v: int,
    ) -> Tensor:
        """Broadcast a per-visit tensor to per-token resolution.

        Each visit in the longitudinal sequence has n_tokens_per_visit
        (512) spatial tokens that share the same visit-level metadata
        (e.g., temporal positional encoding). This utility repeats each
        visit's values across all its tokens so the result can be added
        directly to the flattened token sequence.

        Use case: broadcasting per-visit CTLPE temporal embeddings
        [B, V, d_model] down to token-level [B, V * 512, d_model] so
        every token in a visit shares the same temporal embedding.

        Args:
            visit_tensor: Per-visit tensor of shape [B, V, F] where F
                is an arbitrary feature dimension (typically d_model).
            v: Number of visits. Must match visit_tensor.shape[1].

        Returns:
            Expanded tensor of shape [B, V * n_tokens_per_visit, F]
            where each visit's row is repeated n_tokens_per_visit times.
        """
        b, v_actual, f = visit_tensor.shape  # [B, V, F]
        expanded = visit_tensor.unsqueeze(2)  # [B, V, 1, F]
        expanded = expanded.expand(
            b, v_actual, self.n_tokens_per_visit, f
        )  # [B, V, n_tokens_per_visit, F]
        return expanded.reshape(
            b, v_actual * self.n_tokens_per_visit, f
        )  # [B, V * n_tokens_per_visit, F]


# ModalityDropout — shared visit-level dropout

class ModalityDropout(nn.Module):
    """Visit-level dropout with learned default embeddings for missing data.

    This module handles two distinct scenarios under a single code path:

    1. Structural missingness — visits absent from the ADNI record because
       the patient missed an appointment or dropped out of the study.
       These are indicated by missing_mask == 0 and are always replaced.
    2. Training augmentation — visits randomly dropped during training via
       Bernoulli sampling to improve robustness. These are only dropped
       during training (self.training == True).

    Both cases replace the visit's content (DVF-derived spatial tokens) with
    a learned default embedding that represents "no scan available at this
    time." The default embedding is trained end-to-end with the rest of
    the model.

    IMPORTANT: Temporal positional encoding is NOT touched by this module.
    ModalityDropout only replaces content tokens. The positional encoding
    is added AFTER ModalityDropout in the sequence model's forward(). The
    model must know when a missing visit should have occurred even if it
    has no content for that visit.

    The dropout probability defaults to ModelConfig.modality_dropout_p
    (0.25). The lower end of [0.2, 0.3] is recommended for medical data
    because each ADNI visit carries irreplaceable diagnostic information.
    Aggressive augmentation risks destroying signal that the model needs
    to learn disease progression patterns from.

    Tensor shapes:
        Input:  x — [B, V, n_tokens_per_visit, d_model] (pre-flatten)
        Input:  missing_mask — [B, V] (1=present, 0=missing)
        Output: [B, V, n_tokens_per_visit, d_model] (same shape)
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize ModalityDropout with learned default embeddings.

        Args:
            config: A ModelConfig instance providing n_tokens_per_visit,
                d_model, and modality_dropout_p.
        """
        super().__init__()
        self.n_tokens = config.n_tokens_per_visit
        self.p = config.modality_dropout_p

        # Learned embedding representing "no scan available at this time."
        # Initialized with small random values (std=0.02) following the
        # standard transformer embedding initialization convention.
        self.default_embedding = nn.Parameter(
            torch.randn(config.n_tokens_per_visit, config.d_model) * 0.02
        )

    def forward(self, x: Tensor, missing_mask: Tensor) -> Tensor:
        """Replace missing or dropped visits with learned default embeddings.

        During training, combines structural missingness (missing_mask == 0)
        with stochastic dropout (Bernoulli(p)) to form a combined mask.
        During inference, only structurally missing visits are replaced.

        Args:
            x: Token tensor of shape [B, V, n_tokens_per_visit, d_model]
                with the visit dimension still intact (pre-flatten).
            missing_mask: Binary mask of shape [B, V] where
                1 = visit present, 0 = visit missing/padded.

        Returns:
            Tensor of shape [B, V, n_tokens_per_visit, d_model] with
            missing/dropped visits replaced by self.default_embedding.
        """
        b, v, t, d = x.shape  # [B, V, n_tokens_per_visit, d_model]

        if self.training:
            # Stochastic dropout: randomly drop present visits
            drop_mask = torch.bernoulli(
                torch.full((b, v), self.p, device=x.device)
            )  # [B, V] — 1 = drop this visit, 0 = keep
            # Combine: replace if structurally missing OR stochastically dropped
            # (1 - missing_mask) is 1 where visits are missing
            combined_mask = (1.0 - missing_mask.float()) + drop_mask  # [B, V]
            combined_mask = combined_mask.clamp(max=1.0)  # [B, V] — avoid >1
        else:
            # Inference: replace only structurally missing visits
            combined_mask = 1.0 - missing_mask.float()  # [B, V]

        # Expand default embedding to match batch and visit dims
        # default_embedding: [n_tokens, d_model] → [B, V, n_tokens, d_model]
        default = self.default_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D]
        default = default.expand(b, v, -1, -1)  # [B, V, T, D]

        # Expand combined_mask to broadcast over token and feature dims
        # combined_mask: [B, V] → [B, V, 1, 1] → [B, V, T, D]
        mask_expanded = combined_mask.unsqueeze(-1).unsqueeze(-1) # [B, V, 1, 1]
        mask_expanded = mask_expanded.expand_as(x)  # [B, V, T, D]

        # Replace: where mask is 1, use default; where 0, keep original
        x = x * (1.0 - mask_expanded) + default * mask_expanded  # [B, V, T, D]

        return x


# Smoke test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    print("Phase 2 — BaseSequenceModel & ModalityDropout Smoke Test")

    config = ModelConfig()
    pass_count = 0
    fail_count = 0

    # Test 1: Direct ABC instantiation raises TypeError
    try:
        _ = BaseSequenceModel(config)
        print("FAIL: direct instantiation did NOT raise TypeError")
        fail_count += 1
    except TypeError:
        print("PASS: direct instantiation raises TypeError")
        pass_count += 1

    # Test 2: Concrete subclass instantiation succeeds
    class PassthroughSequenceModel(BaseSequenceModel):
        """Minimal concrete subclass for testing the ABC contract."""

        def forward(self, x, time_deltas, missing_mask):
            self.validate_shapes(x, time_deltas, missing_mask)
            return x

    try:
        model = PassthroughSequenceModel(config)
        print("PASS: concrete subclass instantiation succeeds")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: concrete subclass instantiation — {e}")
        fail_count += 1

    # Test 3: Valid input passes validate_shapes
    b, v, d = 2, 3, config.d_model
    n = v * config.n_tokens_per_visit  # 3 * 512 = 1536
    x = torch.randn(b, n, d)
    time_deltas = torch.tensor([[0.0, 6.0, 12.0], [0.0, 12.0, 24.0]])
    missing_mask = torch.ones(b, v)

    try:
        output = model(x, time_deltas, missing_mask)
        assert output.shape == x.shape
        print("PASS: valid input passes validate_shapes")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: valid input — {e}")
        fail_count += 1

    # Test 4: Wrong d_model raises ValueError
    try:
        x_bad = torch.randn(2, 1536, 256)  # d_model=256, expected 512
        model(x_bad, time_deltas, missing_mask)
        print("FAIL: wrong d_model did NOT raise ValueError")
        fail_count += 1
    except ValueError as e:
        assert len(str(e)) > 0, "Error message should be non-empty"
        print(f"PASS: wrong d_model raises ValueError — {e}")
        pass_count += 1

    # Test 5: Mismatched N raises ValueError
    try:
        x_bad = torch.randn(2, 1000, 512)  # N=1000, expected 1536
        model(x_bad, time_deltas, missing_mask)
        print("FAIL: mismatched N did NOT raise ValueError")
        fail_count += 1
    except ValueError as e:
        assert len(str(e)) > 0, "Error message should be non-empty"
        print(f"PASS: mismatched N raises ValueError — {e}")
        pass_count += 1

    # Test 6: Mismatched batch raises ValueError
    try:
        td_bad = torch.zeros(3, 3)  # batch=3, expected 2
        model(x, td_bad, missing_mask)
        print("FAIL: mismatched batch did NOT raise ValueError")
        fail_count += 1
    except ValueError as e:
        assert len(str(e)) > 0, "Error message should be non-empty"
        print(f"PASS: mismatched batch raises ValueError — {e}")
        pass_count += 1

    # Test 7: expand_visit_to_tokens
    try:
        visit_input = torch.ones(2, 3, 8)  # [B=2, V=3, F=8]
        expanded = model.expand_visit_to_tokens(visit_input, v=3)
        expected_shape = (2, 3 * 512, 8)  # [B, V*512, F]
        assert expanded.shape == expected_shape, (
            f"Shape mismatch: {expanded.shape} != {expected_shape}"
        )
        # Check that each block of 512 rows within a visit is identical
        reshaped = expanded.reshape(2, 3, 512, 8)
        for visit_idx in range(3):
            visit_block = reshaped[:, visit_idx, :, :]  # [B, 512, 8]
            first_row = visit_block[:, 0:1, :]  # [B, 1, 8]
            assert torch.allclose(
                visit_block, first_row.expand_as(visit_block)
            ), f"Rows within visit {visit_idx} are not identical"
        print(f"PASS: expand_visit_to_tokens — {tuple(expanded.shape)}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: expand_visit_to_tokens — {e}")
        fail_count += 1

    # Test 8: ModalityDropout training and inference
    try:
        dropout = ModalityDropout(config)
        x_4d = torch.randn(2, 3, 512, 512)  # [B, V, T, D]
        # Visit 1 of subject 0 is missing
        mask = torch.ones(2, 3)
        mask[0, 1] = 0.0 # subject 0, visit 1 is structurally missing

        # Training mode
        dropout.train()
        out_train = dropout(x_4d.clone(), mask)
        assert out_train.shape == x_4d.shape, "Shape changed during training"
        # The structurally missing visit must be replaced
        default_expanded = dropout.default_embedding.unsqueeze(0)  # [1, T, D]
        assert torch.allclose(out_train[0, 1], default_expanded[0]), (
            "Structurally missing visit was not replaced in training"
        )

        # Eval mode
        dropout.eval()
        # Run multiple times — stochastic dropout should NOT occur
        x_clone = x_4d.clone()
        out_eval = dropout(x_clone, mask)
        assert out_eval.shape == x_4d.shape, "Shape changed during eval"
        # Structurally missing visit should still be replaced
        assert torch.allclose(out_eval[0, 1], default_expanded[0]), (
            "Structurally missing visit was not replaced in eval"
        )
        # Present visits should be unchanged in eval mode
        assert torch.allclose(out_eval[0, 0], x_clone[0, 0]), (
            "Present visit was modified during eval"
        )
        assert torch.allclose(out_eval[0, 2], x_clone[0, 2]), (
            "Present visit was modified during eval"
        )
        print("PASS: ModalityDropout training/inference")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: ModalityDropout training/inference — {e}")
        fail_count += 1

    # Summary
    print(f"\nResults: {pass_count} PASS, {fail_count} FAIL")
    if fail_count == 0:
        print("PASS — All assertions hold.")
    else:
        print("FAIL — Some tests did not pass.")

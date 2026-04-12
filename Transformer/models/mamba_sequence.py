"""
mamba_sequence — Dt-Mamba3D selective state-space sequence model for DVF token streams.

Pipeline position:
    Phase 4 of the ADNI Advanced Survival Pipeline. Drop-in alternative to
    LongformerSequence (Phase 3). Implements the same BaseSequenceModel interface
    and produces identically shaped outputs. Swap by passing Mamba3DSequence
    instead of LongformerSequence to ADNISurvivalPipeline (Phase 9).

Inputs:
    x:            [B, V * 512, d_model] — flat token sequence from BrainIAC extractor
    time_deltas:  [B, V] — inter-visit intervals in months (0 for baseline visit)
    missing_mask: [B, V] — 1 = visit present, 0 = missing or padded

Outputs:
    [B, V * 512, d_model] — contextualized token sequence, same shape as input x.

Architecture sequence (in forward() order):
    1.  validate_shapes()
    2.  Expand time_deltas [B, V] -> dt_tokens [B, N] (repeat each delta for 512 tokens)
    3.  Input projection: x [B, N, d_model] -> xz [B, N, 2 * d_inner]
    4.  Split xz into x_ssm [B, N, d_inner] and z (gate) [B, N, d_inner]
    5.  Depthwise conv1d for local context, then SiLU activation
    6.  Selective projections: B_sel, C_sel from x_proj
    7.  Dt-aware discretization: delta = softplus(dt_proj(x_ssm)) * dt_tokens
    8.  Compute A_bar = exp(A * delta), B_bar = delta * B_sel (ZOH + Euler)
    9.  Selective scan (CUDA kernel or PyTorch fallback)
    10. Output gating: y * SiLU(z)
    11. Output projection: [B, N, d_inner] -> [B, N, d_model]
    12. validate output shape before return

Key advantage over Longformer (Phase 3):
    Clinical time is encoded in the SSM physics (steps 7-8), not via a separate
    positional encoding layer. Longer inter-visit gaps produce proportionally more
    state forgetting via exp(A * delta) -> 0 as delta increases. Missing visits are
    absorbed as larger deltas — no special handling required.

No explicit positional encoding:
    This module does NOT import or use CTLPE or SpatialSinusoidalPE. The absence
    of explicit PE is intentional and is a design advantage, not an omission.

Dependencies:
    - Transformer/models/base.py (BaseSequenceModel)
    - Transformer/config/model_config.py
    - mamba_ssm package (CUDA) OR internal PyTorch fallback (CPU)
"""

import logging
from pathlib import Path

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
from Transformer.models.base import BaseSequenceModel


logger = logging.getLogger(__name__)

# Mamba architecture constants (from Gu & Dao, 2023)
MAMBA_D_CONV = 4 # Depthwise conv kernel size — local context window
MAMBA_DT_RANK_DIVISOR = 16 # dt_rank = d_model // MAMBA_DT_RANK_DIVISOR


def _try_import_mamba_ssm():
    """Attempt to import the CUDA-accelerated mamba_ssm selective scan.

    The mamba_ssm package provides a fused CUDA kernel for the parallel
    associative scan that underpins Mamba's O(N) complexity. If the
    package is unavailable (CPU-only environment, Apple Silicon, etc.),
    we fall back to a sequential pure-PyTorch implementation.

    Returns:
        Tuple of (available: bool, scan_fn: callable or None).
    """
    try:
        from mamba_ssm import selective_scan_fn
        logger.info("mamba_ssm CUDA kernel available.")
        return True, selective_scan_fn
    except ImportError:
        logger.warning(
            "mamba_ssm package not found — using sequential PyTorch "
            "fallback for selective scan. For production use on CUDA, "
            "install via: pip install mamba-ssm causal-conv1d"
        )
        return False, None


MAMBA_SSM_AVAILABLE, _selective_scan_fn = _try_import_mamba_ssm()


class Mamba3DSequence(BaseSequenceModel):
    """Dt-Mamba3D selective state-space sequence model.

    Processes longitudinal DVF token sequences using a selective SSM with
    clinical time injection via ZOH discretization. This is a drop-in
    replacement for LongformerSequence (Phase 3) — it implements the same
    BaseSequenceModel interface and produces identically shaped outputs.

    Design rationale over Longformer:
        - Clinical time is encoded in the SSM physics (ZOH discretization),
          not as a separate positional encoding layer. This removes the need
          for CTLPE or SpatialSinusoidalPE.
        - Missing visits are absorbed as larger dt values — the state simply
          decays more (exp(A * large_dt) -> 0). No ModalityDropout or learned
          default embeddings needed for temporal handling.
        - Linear O(N) complexity via parallel associative scan, vs O(N*w)
          for Longformer sliding-window attention.
        - Selective (input-dependent) B, C, dt projections allow the model
          to remember informative visits (rapid atrophy) and forget noise
          (stable plateaus).

    Design rationale over simpler SSMs (S4):
        S4 uses fixed (input-independent) B, C, dt parameters, making it a
        Linear Time-Invariant (LTI) system. Mamba (S6) makes these selective
        (input-dependent), which is in our case helps because visit informativeness varies.

    Tensor shapes:
        Input: x — [B, V * 512, d_model]
        Output: [B, V * 512, d_model] (same shape — immutable contract)

    Args:
        config: ModelConfig providing d_model, n_tokens_per_visit,
            mamba_d_state, and mamba_expand.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the Mamba3D sequence model.

        Args:
            config: ModelConfig with all pipeline hyperparameters.
        """
        super().__init__(config)

        # Derived dimensions
        self.d_inner = config.d_model * config.mamba_expand  # 1024
        self.d_state = config.mamba_d_state  # 16
        self.d_conv = MAMBA_D_CONV  # 4
        self.dt_rank = config.d_model // MAMBA_DT_RANK_DIVISOR  # 32

        # Input / Output Projections
        # in_proj maps d_model -> 2*d_inner (split into x_ssm and gate z)
        self.in_proj = nn.Linear(
            config.d_model, self.d_inner * 2, bias=False
        )
        # out_proj maps d_inner back to d_model
        self.out_proj = nn.Linear(
            self.d_inner, config.d_model, bias=False
        )

        # Local Context (depthwise conv before SSM)
        # Captures local token relationships before the selective scan.
        # padding = d_conv - 1 for causal convolution (no future leakage)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, self.d_conv,
            padding=self.d_conv - 1, groups=self.d_inner,
        )

        # SSM State Parameters
        # A_log stores log(-A); A is negative diagonal for stable decay.
        # Negative A guarantees exp(A * delta) in (0, 1): bounded decay,
        # not explosion. With positive A, exp(A * delta) grows unboundedly,
        # making the hidden state numerically unstable.
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, self.d_state + 1, dtype=torch.float32)
            )
            .unsqueeze(0)
            .expand(self.d_inner, -1)
            .clone()
        )  # [d_inner, d_state]
        # D is the skip connection coefficient (direct input -> output)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Selective (Input-Dependent) Projections
        # B and C are projected from the SSM input, not fixed parameters.
        # This is what makes Mamba "selective" (S6) vs S4 (LTI).
        self.x_proj = nn.Linear(
            self.d_inner, self.d_state * 2, bias=False
        )  # projects to B_sel [d_state] + C_sel [d_state]
        # dt_proj projects x_ssm to per-channel step sizes
        self.dt_proj = nn.Linear(
            self.d_inner, self.d_inner, bias=True
        )

        self._config = config

    def forward(
        self,
        x: Tensor,
        time_deltas: Tensor,
        missing_mask: Tensor,
    ) -> Tensor:
        """Process flattened longitudinal token sequence through Mamba SSM.

        Follows the 12-step architecture sequence documented in the module
        docstring. Clinical time is encoded directly in the SSM discretization
        step (step 7) — no explicit positional encoding is used.

        Missing visits are handled implicitly: the missing visit's time gap
        is absorbed into a larger dt value for the next present visit.
        The SSM state decays proportionally (exp(A * large_dt) -> 0).
        This is a fundamental advantage over the Longformer, which requires
        ModalityDropout and learned default embeddings.

        This module does NOT use CTLPE or SpatialSinusoidalPE. The absence
        of explicit positional encoding is intentional.

        Args:
            x: Flattened token sequence [B, V * 512, d_model].
            time_deltas: Inter-visit intervals [B, V] in months.
                Baseline visit has time_delta = 0.
            missing_mask: Visit presence mask [B, V].
                1 = present, 0 = missing/padded. Used only for
                contract compliance (validate_shapes). Missing visits
                are handled by larger dt values, not by this mask.

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
        n = v * self.n_tokens_per_visit  # total tokens

        # Step 2: Expand time_deltas from per-visit to per-token
        # Each visit has 512 tokens — all tokens share the same dt
        dt_tokens = (
            time_deltas
            .unsqueeze(-1)  # [B, V, 1]
            .expand(-1, -1, self.n_tokens_per_visit)  # [B, V, 512]
            .reshape(b, n)  # [B, N] where N = V * 512
        )

        # Step 3: Input projection
        xz = self.in_proj(x)  # [B, N, 2 * d_inner]

        # Step 4: Split into SSM branch and gating branch
        x_ssm, z = xz.chunk(2, dim=-1)
        # x_ssm: [B, N, d_inner] — feeds into SSM
        # z: [B, N, d_inner] — gating signal

        # Step 5: Depthwise conv1d for local context
        # Conv1d expects [B, C, N] layout
        x_ssm = x_ssm.transpose(1, 2)  # [B, d_inner, N]
        x_ssm = self.conv1d(x_ssm)[:, :, :n]  # causal: trim padding
        x_ssm = x_ssm.transpose(1, 2)  # [B, N, d_inner]
        x_ssm = F.silu(x_ssm)  # [B, N, d_inner]

        # Step 6: Selective projections — B and C from input
        # This is the "selective" part: B, C are input-dependent (S6)
        bc = self.x_proj(x_ssm)  # [B, N, 2 * d_state]
        B_sel, C_sel = bc.split(self.d_state, dim=-1)
        # B_sel: [B, N, d_state] — selective input matrix
        # C_sel: [B, N, d_state] — selective output matrix

        # Step 7: Dt-aware discretization
        # ZOH discretization for A_bar and B_bar:
        # Theoretical: A_bar = exp(A * delta), B_bar = (A_bar - I) * A^-1 * B
        # Mamba codebase uses Euler approximation for B_bar: B_bar ~ delta * B
        # Justification: first-order Taylor expansion, accurate when delta is small.
        # For A diagonal: A^-1 = diag(1/a_1, ..., 1/a_n), so exact formula
        # is tractable, but Euler is used for computational efficiency and
        # numerical stability.
        # L'Hopital: as a_i -> 0, (exp(a_i*delta) - 1)/a_i -> delta,
        # matching the Euler result.

        # Selective step size from input content
        delta_learned = F.softplus(
            self.dt_proj(x_ssm)
        )  # [B, N, d_inner]

        # Option B (modulation — default, preserves input selectivity)
        # Scales the learned step size by clinical inter-visit time.
        # The model can still adapt dt to token content while respecting
        # clinical timing.
        delta = delta_learned * dt_tokens.unsqueeze(-1)  # [B, N, d_inner]

        # Option A
        # Commented out: replaces learned dt entirely with clinical time.
        # More biologically principled (pure physics-based discretization)
        # but removes input selectivity. Use for ablation experiments.
        # delta = dt_tokens.unsqueeze(-1).expand_as(delta_learned)

        # Step 8: Compute discretized matrices
        # A is guaranteed negative by construction: A = -exp(A_log)
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state], all < 0

        # Step 9: Selective scan (CUDA or fallback)
        y = self._selective_scan(x_ssm, delta, A, B_sel, C_sel)
        # y: [B, N, d_inner]

        # Skip connection: y = y + D * x_ssm
        y = y + x_ssm * self.D.unsqueeze(0).unsqueeze(0)  # [B, N, d_inner]

        # Step 10: Output gating
        y = y * F.silu(z)  # [B, N, d_inner]

        # Step 11: Output projection back to d_model
        y = self.out_proj(y)  # [B, N, d_model]

        # Step 12: Validate output shape
        expected_shape = (b, n, self.d_model)
        if y.shape != expected_shape:
            raise RuntimeError(
                f"Output shape mismatch: got {tuple(y.shape)}, "
                f"expected {expected_shape}. The sequence model must "
                f"preserve [B, V*512, d_model] shape."
            )

        return y

    def _selective_scan(
        self,
        u: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
    ) -> Tensor:
        """Dispatch selective scan to CUDA kernel or PyTorch fallback.

        The parallel associative scan exploits:
            (A_bar_2, v_2) @ (A_bar_1, v_1) = (A_bar_2 * A_bar_1, A_bar_2 * v_1 + v_2)
        This operator is associative, enabling O(log N) parallel depth
        with O(N) total work on GPU.

        Args:
            u: SSM input [B, N, d_inner].
            delta: Discretization step size [B, N, d_inner].
            A: State matrix [d_inner, d_state] (negative diagonal).
            B: Selective input matrix [B, N, d_state].
            C: Selective output matrix [B, N, d_state].

        Returns:
            SSM output [B, N, d_inner].
        """
        if MAMBA_SSM_AVAILABLE and u.is_cuda:
            return _selective_scan_fn(
                u.contiguous(),
                delta.contiguous(),
                A.contiguous(),
                B.contiguous().unsqueeze(1),
                C.contiguous().unsqueeze(1),
                None, # D handled separately
                None, # z handled separately
                None, # delta_bias
                True, # delta_softplus already applied
            )
        return self._sequential_scan_fallback(u, delta, A, B, C)

    def _sequential_scan_fallback(
        self,
        u: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
    ) -> Tensor:
        """Sequential fallback for CPU/testing environments.

        Implements the SSM recurrence step-by-step:
            h_k = A_bar_k * h_{k-1} + B_bar_k * u_k
            y_k = C_k * h_k

        Produces identical results to the CUDA parallel scan but is O(N)
        sequential rather than O(log N) parallel. Use only for development
        and smoke tests — unacceptably slow for production training.

        Args:
            u: SSM input [B, N, d_inner].
            delta: Discretization step size [B, N, d_inner].
            A: State matrix [d_inner, d_state] (negative diagonal).
            B: Selective input matrix [B, N, d_state].
            C: Selective output matrix [B, N, d_state].

        Returns:
            SSM output [B, N, d_inner].
        """
        b, n, d_inner = u.shape
        d_state = A.shape[1]

        # Initialize hidden state to zeros
        h = torch.zeros(
            b, d_inner, d_state, device=u.device, dtype=u.dtype
        )  # [B, d_inner, d_state]

        outputs = []
        for k in range(n):
            # Current step values
            delta_k = delta[:, k, :] # [B, d_inner]
            u_k = u[:, k, :] # [B, d_inner]
            B_k = B[:, k, :] # [B, d_state]
            C_k = C[:, k, :] # [B, d_state]

            # Discretize A and B for this step
            # A_bar = exp(A * delta_k) — exact ZOH for state transition
            # A: [d_inner, d_state], delta_k: [B, d_inner]
            A_bar = torch.exp(
                A.unsqueeze(0) * delta_k.unsqueeze(-1)
            )  # [B, d_inner, d_state]

            # B_bar = delta_k * B_k — Euler approximation
            # delta_k: [B, d_inner] -> [B, d_inner, 1]
            # B_k: [B, d_state] -> [B, 1, d_state]
            B_bar = (
                delta_k.unsqueeze(-1) * B_k.unsqueeze(1)
            )  # [B, d_inner, d_state]

            # Recurrence: h_k = A_bar * h_{k-1} + B_bar * u_k
            # u_k: [B, d_inner] -> [B, d_inner, 1]
            h = A_bar * h + B_bar * u_k.unsqueeze(-1)
            # h: [B, d_inner, d_state]

            # Output: y_k = sum over d_state of (C_k * h_k)
            # C_k: [B, d_state] -> [B, 1, d_state]
            y_k = torch.sum(
                h * C_k.unsqueeze(1), dim=-1
            )  # [B, d_inner]
            outputs.append(y_k)

        # Stack along sequence dimension
        return torch.stack(outputs, dim=1)  # [B, N, d_inner]

    def get_param_groups(self):
        """Return optimizer parameter groups with configured learning rate.

        Returns:
            List with one dict containing all parameters and the
            lr_sequence_model learning rate from config.
        """
        return [
            {
                "params": list(self.parameters()),
                "lr": self._config.lr_sequence_model,
            }
        ]


# Smoke test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    print("Phase 4 — Mamba3DSequence Smoke Test")

    config = ModelConfig()
    pass_count = 0
    fail_count = 0

    # Test 1: Instantiate Mamba3DSequence
    try:
        model = Mamba3DSequence(config)
        model.eval()
        print("PASS: Mamba3DSequence instantiated")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: Mamba3DSequence instantiation — {e}")
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
            f"Shape mismatch: {output.shape}"
        )
        print(f"PASS: forward pass — output shape {tuple(output.shape)}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: forward pass — {e}")
        fail_count += 1

    # Test 3: Output shape equals input shape
    try:
        assert output.shape == x.shape, (
            f"Contract violation: {output.shape} != {x.shape}"
        )
        print("PASS: output shape == input shape (contract preserved)")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: shape contract — {e}")
        fail_count += 1

    # Test 4: Large delta (simulating skipped visit)
    try:
        td_gap = torch.tensor([[0.0, 6.0, 36.0], [0.0, 12.0, 24.0]])
        with torch.no_grad():
            output_gap = model(x, td_gap, missing_mask)
        assert output_gap.shape == (b, n, config.d_model)
        print("PASS: large delta handled (36-month gap)")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: large delta — {e}")
        fail_count += 1

    # Test 5: dt token expansion
    try:
        td_test = torch.tensor([[0.0, 6.0, 12.0]])  # [1, 3]
        dt_tokens = (
            td_test
            .unsqueeze(-1)  # [1, 3, 1]
            .expand(-1, -1, config.n_tokens_per_visit)  # [1, 3, 512]
            .reshape(1, 3 * config.n_tokens_per_visit)  # [1, 1536]
        )
        assert dt_tokens.shape == (1, 1536), f"Shape: {dt_tokens.shape}"
        # First 512 values = 0.0 (visit 0)
        assert torch.all(dt_tokens[0, :512] == 0.0), "Visit 0 dt != 0"
        # Values 512-1023 = 6.0 (visit 1)
        assert torch.all(dt_tokens[0, 512:1024] == 6.0), "Visit 1 dt != 6"
        # Values 1024-1535 = 12.0 (visit 2)
        assert torch.all(dt_tokens[0, 1024:] == 12.0), "Visit 2 dt != 12"
        print("PASS: dt token expansion")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: dt token expansion — {e}")
        fail_count += 1

    # Test 6: A stability — negative diagonal, bounded decay
    try:
        A = -torch.exp(model.A_log.detach())
        assert (A < 0).all(), "A must be all negative"
        # Large dt should drive exp(A * dt) toward zero
        large_dt = torch.full_like(A, 100.0)
        decay = torch.exp(A * large_dt)
        assert decay.max().item() < 1.0, (
            f"exp(A * 100) should be < 1, got max={decay.max().item()}"
        )
        assert decay.max().item() < 1e-10, (
            f"exp(A * 100) should decay near zero, got max={decay.max().item()}"
        )
        print("PASS: A stability — all negative, bounded decay")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: A stability — {e}")
        fail_count += 1

    # Test 7: Sequential scan fallback shape
    try:
        # Small input for direct scan test
        d_inner = config.d_model * config.mamba_expand
        d_state = config.mamba_d_state
        u_small = torch.randn(1, 4, d_inner)
        delta_small = torch.ones(1, 4, d_inner) * 0.1
        A_small = -torch.exp(model.A_log.detach())
        B_small = torch.randn(1, 4, d_state)
        C_small = torch.randn(1, 4, d_state)
        scan_out = model._sequential_scan_fallback(
            u_small, delta_small, A_small, B_small, C_small
        )
        assert scan_out.shape == (1, 4, d_inner), (
            f"Scan shape: {scan_out.shape}"
        )
        print(f"PASS: fallback scan — shape {tuple(scan_out.shape)}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: fallback scan — {e}")
        fail_count += 1

    # Summary
    print(f"\nResults: {pass_count} PASS, {fail_count} FAIL")
    if fail_count == 0:
        print("PASS — All assertions hold.")
    else:
        print("FAIL — Some tests did not pass.")

"""
Transformer/models/pipeline.py
================================
ADNISurvivalPipeline — end-to-end assembly for the advanced longitudinal
competing-risks survival model.

Data flow (with tensor shapes at each stage):
  DVF volumes       [B, V, 3, 128, 128, 128]
  → BrainIAC        [B, V, 512, d_model]   (per-visit, no_grad auto-applied when frozen)
  → flatten         [B, V*512, d_model]
  → sequence model  [B, V*512, d_model]    (LongformerSequence OR Mamba3DSequence — injected)
  → PMA pooling     [B, pma_seeds * d_model]   i.e. [B, 4096] for defaults
  → survival head   hazards [B, n_risks, n_grid]  i.e. [B, 2, 5] for defaults

Returns a dict with hazards AND intermediate representations so ablation
studies can probe any intermediate layer without re-running the full pipeline.

Design notes
------------
- `sequence_model_cls` is injected at construction time so either
  LongformerSequence or Mamba3DSequence can be dropped in with zero
  changes to this file.
- BrainIACFeatureExtractor.forward() auto-detects its own freeze state and
  applies torch.no_grad() internally when frozen. No manual context
  switching is needed here.
- PMA(config) takes the config directly and exposes output_dim() so that
  the d_input passed to TraCeRSurvivalHead is always correct — never a
  hardcoded literal.
- Freeze/unfreeze helpers keep all grad-state logic in one place so the
  trainer can call a single method rather than touching module internals.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Type, Optional

from Transformer.models.brainiac_extractor import BrainIACFeatureExtractor
from Transformer.models.pooling import PMA
from Transformer.models.survival_head import TraCeRSurvivalHead
from Transformer.models.base import BaseSequenceModel
from Transformer.config.model_config import ModelConfig


class ADNISurvivalPipeline(nn.Module):
    """
    End-to-end pipeline from raw DVF volumes to competing-risks hazards.

    Args:
        config (ModelConfig):
            Central hyperparameter store. All architectural dimensions
            are read from here — never hardcode shapes.
        sequence_model_cls (Type[BaseSequenceModel]):
            Class (not instance) of the sequence model to use.
            Must satisfy the BaseSequenceModel ABC contract:
              forward(x, time_deltas, missing_mask) -> [B, N, d_model]
            Pass either LongformerSequence or Mamba3DSequence.
        pretrained_brainiac_path (str | None):
            Path to BrainIAC pretrained weights (.pt / .pth).
            If None, BrainIAC initialises with random weights (smoke
            testing only — do not use for real training runs).
    """

    def __init__(
        self,
        config: ModelConfig,
        sequence_model_cls: Type[BaseSequenceModel],
        pretrained_brainiac_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config

        # ── Phase 1: Spatial feature extractor ───────────────────────────────
        # BrainIACFeatureExtractor(config, pretrained_path) is the correct
        # constructor signature. It freezes backbone by default at init.
        # Its forward() auto-applies no_grad when backbone is frozen.
        self.brainiac = BrainIACFeatureExtractor(
            config=config,
            pretrained_path=pretrained_brainiac_path,
        )

        # ── Phase 3/4: Sequence model (injected — never hardcoded) ───────────
        self.sequence_model: BaseSequenceModel = sequence_model_cls(config)

        # ── Phase 5: Pooling by Multi-Head Attention ──────────────────────────
        # PMA(config) reads config.pma_seeds and config.d_model internally.
        # Use pma.output_dim() for downstream shape queries — never hardcode 4096.
        self.pma = PMA(config)

        # ── Phase 6: Survival head ────────────────────────────────────────────
        # d_input = pma.output_dim() — this is the inter-phase shape contract.
        # TraCeRSurvivalHead(d_input, config) is the correct constructor.
        self.survival_head = TraCeRSurvivalHead(
            d_input=self.pma.output_dim(),
            config=config,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def forward(
        self,
        dvf_sequence: torch.Tensor,
        time_deltas: torch.Tensor,
        missing_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass: DVF volumes → competing-risks hazards.

        Args:
            dvf_sequence  [B, V, 3, 128, 128, 128]  Padded DVF stack.
            time_deltas   [B, V]                     Inter-visit intervals (months).
            missing_mask  [B, V]                     1=real visit, 0=padding.

        Returns:
            dict with keys:
              'hazards'      [B, n_risks, n_grid]     Cause-specific discrete hazards.
              'pma_features' [B, pma_seeds * d_model] Fixed-size subject representation.
              'tokens'       [B, V*512, d_model]      Contextualised sequence tokens.
        """
        B, V = dvf_sequence.shape[:2]

        # ── Stage 1: BrainIAC extraction — loop over visits ───────────────────
        # Memory note: 128³ volumes are the GPU RAM bottleneck. Looping over V
        # (not reshaping to [B*V, ...]) keeps peak memory bounded.
        # BrainIACFeatureExtractor.forward() applies no_grad internally when
        # frozen — no context manager needed here.
        spatial_tokens = self._extract_per_visit(dvf_sequence, B, V)
        # spatial_tokens: [B, V*512, d_model]

        # ── Stage 2: Sequence modelling ───────────────────────────────────────
        sequence_out = self.sequence_model(
            x=spatial_tokens,
            time_deltas=time_deltas,
            missing_mask=missing_mask,
        )
        # sequence_out: [B, V*512, d_model]  — shape-preserving (ABC contract)

        # ── Stage 3: PMA pooling → fixed-size subject vector ──────────────────
        # PMA.forward() returns the already-flattened [B, pma_seeds * d_model]
        pma_flat = self.pma(sequence_out)
        # pma_flat: [B, pma_seeds * d_model]

        # ── Stage 4: Survival head ─────────────────────────────────────────────
        hazards = self.survival_head(pma_flat)
        # hazards: [B, n_risks, n_grid]

        return {
            "hazards": hazards,
            "pma_features": pma_flat,
            "tokens": sequence_out,
        }

    def param_groups(self) -> list[dict]:
        """
        Return named parameter groups for differential learning rates.

        Called by LPFTTrainer to build the Stage 2 optimizer. LRs are set
        to None here; the trainer assigns them from ModelConfig values.

        Returns:
            [
                {'name': 'brainiac_backbone',   'params': [...], 'lr': None},
                {'name': 'brainiac_projection', 'params': [...], 'lr': None},
                {'name': 'sequence_model',      'params': [...], 'lr': None},
                {'name': 'pma',                 'params': [...], 'lr': None},
                {'name': 'survival_head',       'params': [...], 'lr': None},
            ]
        """
        return [
            {
                "name": "brainiac_backbone",
                "params": list(self.brainiac.backbone.parameters()),
                "lr": None,
            },
            {
                "name": "brainiac_projection",
                "params": list(self.brainiac.projection.parameters()),
                "lr": None,
            },
            {
                "name": "sequence_model",
                "params": list(self.sequence_model.parameters()),
                "lr": None,
            },
            {
                "name": "pma",
                "params": list(self.pma.parameters()),
                "lr": None,
            },
            {
                "name": "survival_head",
                "params": list(self.survival_head.parameters()),
                "lr": None,
            },
        ]

    def freeze_for_linear_probe(self) -> None:
        """
        Stage 1 setup: freeze BrainIAC (backbone + projection) and sequence model.
        Only PMA and survival head remain trainable.

        Called by LPFTTrainer.run_stage1() before the first epoch.
        """
        # BrainIAC: freeze backbone and projection
        for p in self.brainiac.parameters():
            p.requires_grad_(False)

        # Sequence model: freeze
        for p in self.sequence_model.parameters():
            p.requires_grad_(False)

        # PMA + survival head: trainable
        for p in self.pma.parameters():
            p.requires_grad_(True)
        for p in self.survival_head.parameters():
            p.requires_grad_(True)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"  [Stage 1] Frozen {n_frozen:,} params | "
            f"Trainable {n_train:,} params (PMA + head only)"
        )

    def unfreeze_for_finetuning(self) -> None:
        """
        Stage 2 setup: unfreeze sequence model + BrainIAC projection.
        BrainIAC backbone stays frozen to prevent catastrophic forgetting.

        Called by LPFTTrainer.run_stage2() before the first Stage 2 epoch.
        """
        # Sequence model: unfreeze
        for p in self.sequence_model.parameters():
            p.requires_grad_(True)

        # BrainIAC projection: unfreeze
        for p in self.brainiac.projection.parameters():
            p.requires_grad_(True)

        # BrainIAC backbone: keep frozen
        for p in self.brainiac.backbone.parameters():
            p.requires_grad_(False)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"  [Stage 2] Trainable {n_train:,} params "
            f"(sequence model + BrainIAC projection + PMA + head)"
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_per_visit(
        self,
        dvf_sequence: torch.Tensor,
        B: int,
        V: int,
    ) -> torch.Tensor:
        """
        Run BrainIAC on each visit independently, stack, then flatten to sequence.

        Args:
            dvf_sequence [B, V, 3, 128, 128, 128]
            B: batch size
            V: number of visits (padded to v_max within this batch)

        Returns:
            [B, V * n_tokens_per_visit, d_model]  — flat longitudinal token sequence
        """
        visit_tokens = []
        for v in range(V):
            dvf_v = dvf_sequence[:, v]        # [B, 3, 128, 128, 128]
            tokens_v = self.brainiac(dvf_v)   # [B, 512, d_model]
            visit_tokens.append(tokens_v)

        stacked = torch.stack(visit_tokens, dim=1)      # [B, V, 512, d_model]
        flat = stacked.flatten(start_dim=1, end_dim=2)  # [B, V*512, d_model]
        return flat


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from Transformer.config.model_config import ModelConfig
    from Transformer.models.longformer_sequence import LongformerSequence

    print("=" * 65)
    print("ADNISurvivalPipeline — smoke test")
    print("=" * 65)

    cfg = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── Test 1: Construction ──────────────────────────────────────────────────
    print("\n[1] Construction (LongformerSequence, no pretrained weights)...")
    pipeline = ADNISurvivalPipeline(
        config=cfg,
        sequence_model_cls=LongformerSequence,
        pretrained_brainiac_path=None,
    ).to(device)
    print("  PASS")

    # ── Test 2: param_groups() structure ─────────────────────────────────────
    print("\n[2] param_groups()...")
    groups = pipeline.param_groups()
    expected_names = {
        "brainiac_backbone", "brainiac_projection",
        "sequence_model", "pma", "survival_head",
    }
    assert {g["name"] for g in groups} == expected_names
    print(f"  PASS — {[g['name'] for g in groups]}")

    # ── Test 3: freeze_for_linear_probe ──────────────────────────────────────
    print("\n[3] freeze_for_linear_probe()...")
    pipeline.freeze_for_linear_probe()
    assert all(not p.requires_grad for p in pipeline.sequence_model.parameters())
    assert all(p.requires_grad for p in pipeline.survival_head.parameters())
    print("  PASS — sequence model frozen, head trainable")

    # ── Test 4: Full forward pass ─────────────────────────────────────────────
    B, V = 2, 3
    print(f"\n[4] Forward pass [B={B}, V={V}, 3, 128, 128, 128]...")
    dvf          = torch.randn(B, V, 3, 128, 128, 128, device=device)
    time_deltas  = torch.rand(B, V, device=device) * 12
    missing_mask = torch.ones(B, V, device=device)
    missing_mask[0, 2] = 0

    with torch.no_grad():
        out = pipeline(dvf, time_deltas, missing_mask)

    assert out["hazards"].shape      == (B, cfg.n_risks, cfg.n_grid)
    assert out["pma_features"].shape == (B, pipeline.pma.output_dim())
    assert out["tokens"].shape       == (B, V * cfg.n_tokens_per_visit, cfg.d_model)
    print(
        f"  PASS — hazards {tuple(out['hazards'].shape)}, "
        f"pma_features {tuple(out['pma_features'].shape)}, "
        f"tokens {tuple(out['tokens'].shape)}"
    )

    # ── Test 5: unfreeze_for_finetuning ──────────────────────────────────────
    print("\n[5] unfreeze_for_finetuning()...")
    pipeline.unfreeze_for_finetuning()
    assert any(p.requires_grad for p in pipeline.sequence_model.parameters())
    assert all(not p.requires_grad for p in pipeline.brainiac.backbone.parameters())
    print("  PASS — sequence model unfrozen, backbone still frozen")

    # ── Test 6: Output dict keys stable ──────────────────────────────────────
    print("\n[6] Output keys stable after unfreeze...")
    with torch.no_grad():
        out2 = pipeline(dvf, time_deltas, missing_mask)
    assert set(out2.keys()) == {"hazards", "pma_features", "tokens"}
    print("  PASS")

    print("\n" + "=" * 65)
    print("All 6 smoke tests PASSED")
    print("=" * 65)
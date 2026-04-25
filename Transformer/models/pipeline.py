"""
pipeline — End-to-end assembly of the ADNI Advanced Survival Pipeline.

Phase 9a: wires BrainIAC → parallel sequence models → gated fusion →
PMA → TraCeR into a single nn.Module with LP-FT parameter group helpers.

Phase 3/4 update: Parallel gated fusion architecture. Longformer and Mamba
run simultaneously on the same BrainIAC embeddings. Their outputs are
concatenated and a learned sigmoid gate `alpha` blends them:
    output = alpha * longformer_out + (1 - alpha) * mamba_out

Forward contract:
    Input:  dvf_sequence [B, V, 3, 128, 128, 128]
            time_deltas  [B, V]
            missing_mask [B, V]
    Output: dict with keys hazards, pma_features, tokens
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Type

import torch
import torch.nn as nn
from torch import Tensor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Transformer.config.model_config import ModelConfig
from Transformer.models.brainiac_extractor import BrainIACFeatureExtractor
from Transformer.models.base import BaseSequenceModel
from Transformer.models.longformer_sequence import LongformerSequence
from Transformer.models.mamba_sequence import Mamba3DSequence
from Transformer.models.pooling import PMA
from Transformer.models.survival_head import TraCeRSurvivalHead

logger = logging.getLogger(__name__)


class GatedFusion(nn.Module):
    """Learned gated fusion of two sequence model outputs.

    Concatenates the outputs of two sequence models along the feature
    dimension, projects through a linear layer + sigmoid to learn a
    per-feature gating weight alpha, then blends:
        output = alpha * branch_a + (1 - alpha) * branch_b

    This replaces sequential chaining with a parallel architecture
    where both models process the same input simultaneously and their
    representations are fused adaptively.

    Args:
        d_model: Feature dimension of each branch's output.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Project concatenated features [2*d_model] → [d_model] → sigmoid
        self.gate_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute gated fusion of two branch outputs.

        Args:
            a: Longformer output [B, N, d_model].
            b: Mamba output [B, N, d_model].

        Returns:
            Fused output [B, N, d_model].
        """
        concat = torch.cat([a, b], dim=-1)  # [B, N, 2*d_model]
        alpha = torch.sigmoid(self.gate_proj(concat))  # [B, N, d_model]
        return alpha * a + (1.0 - alpha) * b  # [B, N, d_model]


class ADNISurvivalPipeline(nn.Module):
    """End-to-end pipeline: DVF volumes → cause-specific hazard rates.

    Phase 3/4 architecture: Parallel gated fusion of Longformer and Mamba.
    Both sequence models process the same BrainIAC embeddings in parallel
    and their outputs are blended via a learned sigmoid gate.


    When sequence_model_cls is provided (backward compat), only that single
    model runs (no parallel fusion). For parallel fusion, pass
    sequence_model_cls=None (default).
    """

    def __init__(
        self,
        config: ModelConfig,
        sequence_model_cls: Optional[Type[BaseSequenceModel]] = None,
        pretrained_brainiac_path: Optional[str] = None,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.config = config

        # Phase 1 — BrainIAC spatial feature extractor
        # Phase 2: GroupNorm + gradient checkpointing integrated
        self.extractor = BrainIACFeatureExtractor(
            config,
            pretrained_path=pretrained_brainiac_path,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Latent dimensionality alignment 
        extractor_d = config.d_model
        seq_d = config.d_model
        if extractor_d != seq_d:
            self.align_projection = nn.Linear(extractor_d, seq_d)
            logger.info("Alignment projection: %d → %d", extractor_d, seq_d)
        else:
            self.align_projection = nn.Identity()

        # Phase 3/4: Parallel gated fusion
        # If a specific sequence_model_cls is given, use single-model
        # mode (backward compatible). Otherwise, instantiate both
        # Longformer and Mamba and fuse their outputs.
        self._parallel_mode = (sequence_model_cls is None)

        if self._parallel_mode:
            self.longformer = LongformerSequence(config)
            self.mamba = Mamba3DSequence(config)
            self.gated_fusion = GatedFusion(config.d_model)
            self.sequence_model = None  # Not used in parallel mode
            logger.info("Parallel gated fusion: Longformer + Mamba")
        else:
            self.sequence_model = sequence_model_cls(config)
            self.longformer = None
            self.mamba = None
            self.gated_fusion = None
            logger.info("Single sequence model: %s", sequence_model_cls.__name__)

        # Phase 5 — PMA pooling
        self.pma = PMA(config)

        # Phase 6 — TraCeR survival head
        self.survival_head = TraCeRSurvivalHead(
            d_input=self.pma.output_dim(),
            config=config,
        )

        logger.info(
            "ADNISurvivalPipeline assembled: extractor=%s, "
            "parallel=%s, pma_seeds=%d, d_pma=%d, grad_ckpt=%s",
            type(self.extractor).__name__,
            self._parallel_mode,
            config.pma_seeds,
            self.pma.output_dim(),
            use_gradient_checkpointing,
        )


    def forward(
        self,
        dvf_sequence: Tensor,
        time_deltas: Tensor,
        missing_mask: Tensor,
        tabular: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Full forward pass from raw DVF volumes to hazard rates.

        Args:
            dvf_sequence: [B, V, 3, 128, 128, 128] padded DVF stack.
            time_deltas: [B, V] inter-visit intervals in months.
            missing_mask: [B, V] 1=present, 0=padded.
            tabular: reserved for future tabular fusion.

        Returns:
            Dict with keys:
                hazards: [B, K, G] cause-specific discrete hazards
                pma_features: [B, m*d_model] pooled summary
                tokens: [B, V*512, d_model] fused sequence output
        """
        B, V = dvf_sequence.shape[:2]

        # Phase 1: per-visit BrainIAC extraction
        # The extractor may live on a different device (CPU) when Conv3D
        # backward is not supported on MPS (torch < 2.5). We detect this
        # via parameter device and route tensors accordingly.
        extractor_device = next(self.extractor.parameters()).device
        main_device = dvf_sequence.device

        visit_tokens = []
        for v in range(V):
            dvf_v = dvf_sequence[:, v]  # [B, 3, 128, 128, 128]
            # Use no_grad when backbone is frozen
            if not any(
                p.requires_grad for p in self.extractor.backbone.parameters()
            ):
                with torch.no_grad():
                    tok = self.extractor(dvf_v.to(extractor_device))
            else:
                tok = self.extractor(dvf_v.to(extractor_device))
            # Move tokens back to main device if mixed-device
            if extractor_device != main_device:
                tok = tok.to(main_device)
            visit_tokens.append(tok)

        # Stack and flatten: [B, V, 512, d_model] → [B, V*512, d_model]
        x = torch.stack(visit_tokens, dim=1)  # [B, V, 512, d_model]
        x = x.reshape(B, V * self.config.n_tokens_per_visit, self.config.d_model)

        # Phase 2: alignment projection (Identity when dims match)
        x = self.align_projection(x)  # [B, V*512, d_model]

        # Phase 3/4: sequence modeling
        if self._parallel_mode:
            # Parallel: both models process the same embeddings
            longformer_out = self.longformer(x, time_deltas, missing_mask)
            mamba_out = self.mamba(x, time_deltas, missing_mask)
            # Gated fusion: alpha * longformer + (1-alpha) * mamba
            tokens = self.gated_fusion(longformer_out, mamba_out)
        else:
            # Single model (backward compatible)
            tokens = self.sequence_model(x, time_deltas, missing_mask)

        # Phase 5: PMA pooling
        pma_features = self.pma(tokens)  # [B, m*d_model]

        # Phase 6: TraCeR survival head
        hazards = self.survival_head(pma_features)  # [B, K, G]

        return {
            "hazards": hazards,
            "pma_features": pma_features,
            "tokens": tokens,
        }

    # LP-FT parameter groups

    def get_stage1_param_groups(self) -> list:
        """Stage 1 (linear probe): train only PMA + survival head.

        Backbone and sequence models are frozen.

        Returns:
            List of param group dicts for AdamW.
        """
        # Freeze backbone (already frozen by default)
        for p in self.extractor.parameters():
            p.requires_grad = False
        # Freeze sequence model(s)
        if self._parallel_mode:
            for p in self.longformer.parameters():
                p.requires_grad = False
            for p in self.mamba.parameters():
                p.requires_grad = False
            for p in self.gated_fusion.parameters():
                p.requires_grad = False
        else:
            for p in self.sequence_model.parameters():
                p.requires_grad = False

        # Train PMA + head at survival head LR
        lr = self.config.lr_survival_head
        return [
            {"params": list(self.pma.parameters()), "lr": lr},
            {"params": list(self.survival_head.parameters()), "lr": lr},
        ]

    def get_stage2_param_groups(self) -> list:
        """Stage 2: unfreeze sequence model(s) with differential LRs.

        Backbone stays frozen unless dataset is large.

        Returns:
            List of param group dicts with differential learning rates.
        """
        # Unfreeze projection head in extractor (not backbone)
        for p in self.extractor.projection.parameters():
            p.requires_grad = True

        cfg = self.config
        groups = [
            {
                "params": list(self.extractor.projection.parameters()),
                "lr": cfg.lr_projection_head,
            },
        ]

        # Unfreeze sequence model(s)
        if self._parallel_mode:
            for p in self.longformer.parameters():
                p.requires_grad = True
            for p in self.mamba.parameters():
                p.requires_grad = True
            for p in self.gated_fusion.parameters():
                p.requires_grad = True
            groups.extend([
                {
                    "params": list(self.longformer.parameters()),
                    "lr": cfg.lr_sequence_model,
                },
                {
                    "params": list(self.mamba.parameters()),
                    "lr": cfg.lr_sequence_model,
                },
                {
                    "params": list(self.gated_fusion.parameters()),
                    "lr": cfg.lr_sequence_model,
                },
            ])
        else:
            for p in self.sequence_model.parameters():
                p.requires_grad = True
            groups.append({
                "params": list(self.sequence_model.parameters()),
                "lr": cfg.lr_sequence_model,
            })

        groups.extend([
            {
                "params": list(self.pma.parameters()),
                "lr": cfg.lr_survival_head,
            },
            {
                "params": list(self.survival_head.parameters()),
                "lr": cfg.lr_survival_head,
            },
        ])

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Stage 2 param groups: %d trainable parameters", trainable)
        return groups

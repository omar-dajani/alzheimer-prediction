"""
pipeline — End-to-end assembly of the ADNI Advanced Survival Pipeline.

Phase 9a: wires BrainIAC → sequence model → PMA → TraCeR into a single
nn.Module with LP-FT parameter group helpers.

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
from Transformer.models.pooling import PMA
from Transformer.models.survival_head import TraCeRSurvivalHead

logger = logging.getLogger(__name__)


class ADNISurvivalPipeline(nn.Module):
    """End-to-end pipeline: DVF volumes → cause-specific hazard rates.

    The sequence model is injected via ``sequence_model_cls`` so that
    Longformer and Mamba can be swapped without touching any other code.

    Tensor flow:
        dvf [B, V, 3, 128, 128, 128]
        per-visit BrainIAC [B, V, 512, d_model]
        flatten [B, V*512, d_model]
        sequence model [B, V*512, d_model]
        PMA pooling [B, m*d_model]
        TraCeR head [B, K, G]
    """

    def __init__(
        self,
        config: ModelConfig,
        sequence_model_cls: Type[BaseSequenceModel] = LongformerSequence,
        pretrained_brainiac_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config

        # Phase 1 — BrainIAC spatial feature extractor
        self.extractor = BrainIACFeatureExtractor(
            config,
            pretrained_path=pretrained_brainiac_path,
        )

        # Phase 3/4 — sequence model (injected)
        self.sequence_model = sequence_model_cls(config)

        # Phase 5 — PMA pooling
        self.pma = PMA(config)

        # Phase 6 — TraCeR survival head
        self.survival_head = TraCeRSurvivalHead(
            d_input=self.pma.output_dim(),
            config=config,
        )

        logger.info(
            "ADNISurvivalPipeline assembled: extractor=%s, "
            "sequence=%s, pma_seeds=%d, d_pma=%d",
            type(self.extractor).__name__,
            type(self.sequence_model).__name__,
            config.pma_seeds,
            self.pma.output_dim(),
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
                tokens: [B, V*512, d_model] sequence model output
        """
        B, V = dvf_sequence.shape[:2]

        # Phase 1: per-visit BrainIAC extraction
        visit_tokens = []
        for v in range(V):
            dvf_v = dvf_sequence[:, v]  # [B, 3, 128, 128, 128]
            # Use no_grad when backbone is frozen
            if not any(
                p.requires_grad for p in self.extractor.backbone.parameters()
            ):
                with torch.no_grad():
                    tok = self.extractor(dvf_v)  # [B, 512, d_model]
            else:
                tok = self.extractor(dvf_v)
            visit_tokens.append(tok)

        # Stack and flatten: [B, V, 512, d_model] → [B, V*512, d_model]
        x = torch.stack(visit_tokens, dim=1)  # [B, V, 512, d_model]
        x = x.reshape(B, V * self.config.n_tokens_per_visit, self.config.d_model)

        # Phase 3/4: sequence model
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

        Backbone and sequence model are frozen.

        Returns:
            List of param group dicts for AdamW.
        """
        # Freeze backbone (already frozen by default)
        for p in self.extractor.parameters():
            p.requires_grad = False
        # Freeze sequence model
        for p in self.sequence_model.parameters():
            p.requires_grad = False

        # Train PMA + head at survival head LR
        lr = self.config.lr_survival_head
        return [
            {"params": list(self.pma.parameters()), "lr": lr},
            {"params": list(self.survival_head.parameters()), "lr": lr},
        ]

    def get_stage2_param_groups(self) -> list:
        """Stage 2: unfreeze sequence model with differential LRs.

        Backbone stays frozen unless dataset is large.

        Returns:
            List of param group dicts with differential learning rates.
        """
        # Unfreeze sequence model
        for p in self.sequence_model.parameters():
            p.requires_grad = True

        # Unfreeze projection head in extractor (not backbone)
        for p in self.extractor.projection.parameters():
            p.requires_grad = True

        cfg = self.config
        groups = [
            {
                "params": list(self.extractor.projection.parameters()),
                "lr": cfg.lr_projection_head,
            },
            {
                "params": list(self.sequence_model.parameters()),
                "lr": cfg.lr_sequence_model,
            },
            {
                "params": list(self.pma.parameters()),
                "lr": cfg.lr_survival_head,
            },
            {
                "params": list(self.survival_head.parameters()),
                "lr": cfg.lr_survival_head,
            },
        ]

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Stage 2 param groups: %d trainable parameters", trainable)
        return groups

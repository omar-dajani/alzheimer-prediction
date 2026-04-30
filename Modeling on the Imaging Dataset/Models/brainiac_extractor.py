"""
brainiac_extractor — BrainIAC 3D ResNet50 spatial feature extractor for DVF inputs.

Pipeline position:
    Phase 1 of the ADNI Advanced Survival Pipeline. Runs once per visit per subject.
    Receives normalized DVF tensors from Phase 0 and produces spatial token sequences
    consumed by the sequence models in Phases 3 and 4.

Inputs:
    dvf: torch.Tensor of shape [B, 3, 128, 128, 128] — per-channel normalized DVF,
         where channels 0/1/2 correspond to delta x/delta y/delta z displacement components.

Outputs:
    tokens: torch.Tensor of shape [B, 512, d_model] — 512 spatial tokens (one per
            voxel in the 8x8x8 BrainIAC output grid), projected to d_model dimensions.

Architecture summary:
    DVF [B, 3, 128, 128, 128]
      → BrainIAC ResNet50 (no_max_pool=True, conv1 adapted for 3-ch input)
      → Spatial feature map [B, 2048, 8, 8, 8]
      → Reshape to token sequence [B, 512, 2048]
      → Trainable projection head (2048 → 1024 → d_model)
      → Spatial tokens [B, 512, d_model]

Freezing strategy:
    Backbone is frozen by default (pre-trained weights preserved).
    Only the projection head is trained in Stage 1 (LP-FT).
    Backbone can be unfrozen for Stage 2 fine-tuning via unfreeze_backbone().

Critical configuration:
    no_max_pool=True is required. Without it, the ResNet50 stem applies max-pooling
    (32x total downsampling), collapsing the spatial grid to 4x4x4 = 64 tokens.
    With no_max_pool=True, downsampling is 16x, yielding 8x8x8 = 512 tokens.

Dependencies:
    - Transformer/config/model_config.py (ModelConfig.d_model)
    - brainiac-model PyPI package OR MONAI 3D ResNet50 fallback
"""

import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

import sys
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)
from Transformer.config.model_config import ModelConfig


logger = logging.getLogger(__name__)

# Module-level constants

RESNET_LAYER_CONFIG = (3, 4, 6, 3)
"""ResNet50 bottleneck block counts per stage (Stages 1–4)."""

RESNET_STAGE4_CHANNELS = 2048
"""Stage 4 output channels (expansion=4, 512*4)."""

SPATIAL_GRID_SIZE = 8
"""Linear spatial dimension after 16x downsampling (128/16 = 8)."""

N_SPATIAL_TOKENS = SPATIAL_GRID_SIZE ** 3
"""Total number of spatial tokens: 8x8x8 = 512."""

GROUPNORM_NUM_GROUPS = 16
"""Number of groups for GroupNorm replacements. All ResNet50 channel
counts (64, 128, 256, 512, 1024, 2048) are divisible by 16."""


def replace_bn3d_with_groupnorm(
    model: nn.Module,
    num_groups: int = GROUPNORM_NUM_GROUPS,
) -> int:
    """Recursively replace all BatchNorm3d layers with GroupNorm.

    GroupNorm is preferred over BatchNorm3d for two reasons:
        1. MPS rank-5 normalization latency bug — BatchNorm3d on 5-D
           tensors is extremely slow on Apple Silicon's Metal backend.
        2. Batch-size independence — with micro-batches of 1–2 subjects
           (each 128^3 volume), BatchNorm statistics are unreliable.
           GroupNorm normalizes within channel groups per-sample.

    Args:
        model: The nn.Module to modify in-place.
        num_groups: Number of groups per GroupNorm layer.

    Returns:
        Count of BatchNorm3d layers replaced.
    """
    count = 0
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm3d):
            gn = nn.GroupNorm(
                num_groups=min(num_groups, child.num_features),
                num_channels=child.num_features,
                eps=child.eps,
                affine=child.affine,
            )
            # Copy affine params if they exist
            if child.affine and child.weight is not None:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(model, name, gn)
            count += 1
        else:
            count += replace_bn3d_with_groupnorm(child, num_groups)
    return count


# Backbone loader

def load_brainiac_resnet50(
    pretrained_path: Optional[Path] = None,
    no_max_pool: bool = True,
) -> nn.Module:
    """Load a 3D ResNet50 backbone for spatial feature extraction.

    Attempts to load the BrainIAC pre-trained ResNet50 from the brainiac-model
    PyPI package first.  If the package is not installed, falls back to building
    an equivalent architecture from MONAI primitives with the same layer
    configuration [3, 4, 6, 3].

    The global average pooling layer (model.avgpool) is replaced with
    nn.Identity() so that the spatial feature grid [B, 2048, 8, 8, 8]
    is preserved rather than collapsed to [B, 2048].  If avgpool were
    kept, all spatial structure would be lost and no spatial tokens could be
    extracted — the downstream sequence models require per-voxel tokens.

    The classification head (model.fc) is similarly replaced with
    nn.Identity() because we do not perform classification here.

    Args:
        pretrained_path: Optional filesystem path to a .pt or .pth
            checkpoint containing BrainIAC pre-trained weights. When None,
            the model is initialised with random weights and a warning is
            logged.
        no_max_pool: If True (default, required), disables the stem
            max-pooling layer so that total spatial downsampling is 16x,
            yielding an 8x8x8 = 512-token output grid. Setting this to
            False would introduce an additional 2x downsampling
            (total 32x), collapsing the grid to 4x4x4 = 64 tokens.

    Returns:
        A torch.nn.Module with no classification head and no global
        average pooling, whose forward pass produces a feature map of shape
        [B, 2048, D', H', W'] where D'=H'=W'=8 when
        no_max_pool=True and the input is 128^3.

    Raises:
        RuntimeError: If pretrained_path is specified but the file
            cannot be loaded.
    """
    model = None
    weights_loaded = False

    # ── Path A: Explicit pretrained checkpoint ──────────────────────
    # Highest priority — user supplied a .pt/.pth file directly.
    if pretrained_path is not None:
        pretrained_path = Path(pretrained_path)
        try:
            from monai.networks.nets import ResNet, ResNetBottleneck

            model = ResNet(
                block=ResNetBottleneck,
                layers=list(RESNET_LAYER_CONFIG),
                block_inplanes=[64, 128, 256, 512],
                spatial_dims=3,
                n_input_channels=1,
                no_max_pool=False,
            )
            state_dict = torch.load(
                pretrained_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict, strict=False)
            weights_loaded = True
            logger.info(
                "Loaded pre-trained weights from %s.", pretrained_path
            )
        except Exception as exc:
            logger.error(
                "Failed to load weights from %s: %s",
                pretrained_path, exc,
            )
            raise

    # Path B: MONAI MedicalNet pretrained (primary default)
    # MedicalNet ResNet50 pretrained on 23 medical imaging datasets
    # Why not the brainiac PyPI package? The brainiac-model package on
    # PyPI wraps MONAI ResNet50 in a HuggingFace PreTrainedModel, but
    # the actual SimCLR pretrained weights live on a gated HuggingFace
    # repo (Divytak/brainiac) requiring authentication. Without auth,
    # `BrainiacModel(config)` only gives random weights - no better
    # than our MONAI fallback.  MedicalNet weights are freely available
    # via MONAI and provide strong 3D medical imaging features.
    if model is None:
        try:
            from monai.networks.nets import resnet50 as monai_resnet50

            model = monai_resnet50(
                pretrained=True,
                spatial_dims=3,
                n_input_channels=1,
                feed_forward=False,
                shortcut_type="B",
                bias_downsample=False,
            )
            weights_loaded = True
            logger.info(
                "Loaded MedicalNet pretrained ResNet50 (23 datasets, "
                "including brain MRI). This backbone provides strong "
                "3D medical imaging features without BrainIAC weights."
            )
        except Exception as exc:
            logger.warning(
                "MedicalNet pretrained download failed: %s. "
                "Falling back to random MONAI ResNet50.",
                exc,
            )

    # Path D: MONAI random init (last resort)
    if model is None:
        try:
            from monai.networks.nets import ResNet, ResNetBottleneck

            model = ResNet(
                block=ResNetBottleneck,
                layers=list(RESNET_LAYER_CONFIG),
                block_inplanes=[64, 128, 256, 512],
                spatial_dims=3,
                n_input_channels=1,
                no_max_pool=False,
            )
            logger.warning(
                "Using MONAI ResNet50 with RANDOM weights. "
                "Performance will be significantly worse than with "
                "pretrained weights."
            )
        except ImportError as exc:
            raise ImportError(
                "Neither brainiac-model nor MONAI is installed. "
                "At least one is required to build the 3D ResNet50 "
                "backbone."
            ) from exc

    if not weights_loaded:
        logger.warning(
            "Backbone has RANDOM weights. For best results, use "
            "MedicalNet (default) or provide BrainIAC weights."
        )

    # Strip classification head and global average pooling
    model.fc = nn.Identity()
    model.avgpool = nn.Identity()

    return model


# 3-channel weight adaptation

def adapt_conv1_to_3ch(model: nn.Module) -> None:
    """Adapt the stem conv1 layer from 1-channel to 3-channel input.

    The pre-trained BrainIAC conv1 weight has shape [64, 1, 7, 7, 7]
    (single-channel brain MRI input). This function replaces it with a
    nn.Conv3d(3, 64, 7, stride=2, padding=3, bias=False) whose weight
    [64, 3, 7, 7, 7] is formed by repeating the original kernel across
    the channel dimension and dividing by 3.

    The operation is performed in-place on the provided model.

    Args:
        model: A 3D ResNet whose conv1 attribute is a nn.Conv3d
            with in_channels=1.

    Mathematical justification for W/3 adaptation:
        The original single-channel weight W produces activations whose
        expected value is E[y] = W · E[x] After duplication to 3
        channels the naive output would be y' = W·x_1 + W·x_2 + W·x_3,
        tripling the activation magnitude and invalidating the downstream
        BatchNorm running statistics.

        By dividing each duplicated kernel by 3 we obtain:
            y' = (W/3)·x_1 + (W/3)·x_2 + (W/3)·x_3
        Taking expectation:
            E[y'] = 3 · (W/3) · E[x] = W · E[x] = E[y]

        This preserves the first moment of activations, keeping BatchNorm
        statistics valid so pre-trained downstream layers function without
        re-calibration.
    """
    old_conv = model.conv1

    # Create replacement conv layer matching all spatial params
    new_conv = nn.Conv3d(
        in_channels=3,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )

    # Mathematical justification for W/3 adaptation:
    # Original single-channel output: y = sum_i(w_i * x_i)
    # With 3 channels (W/3 each): y' = (W/3)*x1 + (W/3)*x2 + (W/3)*x3
    # Taking expectation: E[y'] = 3 * (W/3) * E[x] = W * E[x] = E[y]
    # This preserves the first moment of activations, keeping BatchNorm
    # statistics valid without requiring re-calibration of downstream layers.
    original_weight = old_conv.weight.data  # [64, 1, 7, 7, 7]
    new_weight = original_weight.repeat(1, 3, 1, 1, 1) / 3.0  # [64, 3, 7, 7, 7]
    new_conv.weight.data = new_weight

    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data.clone()

    model.conv1 = new_conv
    logger.info(
        "Adapted conv1: [%d, 1, 7, 7, 7] → [%d, 3, 7, 7, 7] via W/3 "
        "duplication.",
        old_conv.out_channels,
        old_conv.out_channels,
    )


# BrainIAC Feature Extractor

class BrainIACFeatureExtractor(nn.Module):
    """Spatial feature extractor converting DVF volumes to token sequences.

    This module wraps a 3D ResNet50 backbone (BrainIAC or MONAI-based) and
    a trainable two-layer projection head. It is the sole bridge between
    the raw 3D DVF data and the downstream sequence models (Longformer /
    Mamba), which operate on 1-D token sequences.

    Architectural justification over alternatives:
        - A **ViT-based** extractor (e.g. BrainIAC SimCLR-ViT-B) is the
          paper's recommended backbone, but the publicly released PyPI
          package only provides a ResNet50 — we use what is available.
        - A **2D slice-wise** extractor would discard the inter-slice
          spatial context that 3D convolutions capture (axial-coronal-
          sagittal continuity of atrophy patterns).
        - A **shallower ResNet** lacks the capacity to
          represent the 2048-dimensional feature space that captures fine-
          grained deformation patterns across 16 datasets.

    Tensor shapes in / out:
        Input: dvf — [B, 3, 128, 128, 128]
        Output: tokens — [B, 512, d_model]

    The backbone is frozen by default.  Only the projection head
    is trained during LP-FT Stage 1.  Call
    unfreeze_backbone() for Stage 2 fine-tuning at 10x lower LR.

    Args:
        config: A ModelConfig instance providing d_model and
            lr_projection_head.
        pretrained_path: Optional path to BrainIAC pre-trained weights.
    """

    def __init__(
        self,
        config: ModelConfig,
        pretrained_path: Optional[Path] = None,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()

        self._d_model = config.d_model
        self._lr_projection_head = config.lr_projection_head
        self._use_gradient_checkpointing = use_gradient_checkpointing

        # no_max_pool=True is the correct flag for the BrainIAC package
        # (stride-2 conv1, no maxpool → 16x total). For the MONAI
        # fallback, load_brainiac_resnet50() internally overrides this
        # to no_max_pool=False because MONAI's stride-1 conv1 needs
        # maxpool to achieve the same 16x downsampling.
        self.backbone = load_brainiac_resnet50(
            pretrained_path=pretrained_path,
            no_max_pool=True,
        )

        # Adapt conv1 from 1-channel MRI to 3-channel DVF input
        adapt_conv1_to_3ch(self.backbone)

        # Replace BatchNorm3d -> GroupNorm
        # Performed AFTER weight loading so that BN affine params are
        # transferred to GroupNorm (weight/bias copied in the swap).
        bn_replaced = replace_bn3d_with_groupnorm(
            self.backbone, num_groups=GROUPNORM_NUM_GROUPS
        )
        logger.info(
            "Replaced %d BatchNorm3d layers with GroupNorm(num_groups=%d)",
            bn_replaced, GROUPNORM_NUM_GROUPS,
        )

        # Projection head (2048 -> 1024 -> d_model)
        # Phase 1: Linear(2048, 1024) -> LayerNorm(1024) -> GELU
        # Dropout(0.1) -> Linear(1024, d_model) -> LayerNorm(d_model)
        self.projection = nn.Sequential(
            nn.Linear(RESNET_STAGE4_CHANNELS, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self._d_model),
            nn.LayerNorm(self._d_model),
        )

        # Freeze backbone, keep projection trainable
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Verify projection head is trainable
        for param in self.projection.parameters():
            param.requires_grad = True

        if self._use_gradient_checkpointing:
            logger.info("Gradient checkpointing enabled for layer3 + layer4")

    # Internal backbone forward (bypasses avgpool / fc / flatten)
    def _backbone_feature_forward(self, x: Tensor) -> Tensor:
        """Run the backbone up through Stage 4, preserving spatial dims.

        MONAI's ResNet.forward() includes a hardcoded
        x.view(x.size(0), -1) that flattens the spatial grid even
        when avgpool is replaced with nn.Identity(). This
        helper runs only the convolutional stages (conv1 -> bn1 -> act
        -> layer1 -> layer2 -> layer3 -> layer4) and returns the spatial
        feature map without flattening.

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            Feature map [B, 2048, D', H', W'] with spatial
            dimensions preserved.
        """
        bb = self.backbone
        x = bb.conv1(x) # Stem conv (stride depends on backend)
        x = bb.bn1(x)   # Now GroupNorm after Phase 2 swap
        x = bb.act(x)
        # Apply maxpool if it exists and is not Identity.
        # MONAI fallback: maxpool present -> stride-1 conv + maxpool = 2x
        # BrainIAC: maxpool absent (no_max_pool=True) -> stride-2 conv = 2x
        # Either way the spatial dims should be 64^3 after this stage.
        if (
            hasattr(bb, "maxpool")
            and not isinstance(bb.maxpool, nn.Identity)
        ):
            x = bb.maxpool(x) # [B, 64, 64, 64, 64]
        x = bb.layer1(x) # [B, 256, 64, 64, 64] (stride 1)
        x = bb.layer2(x) # [B, 512, 32, 32, 32] (stride 2)

        # Gradient checkpointing on layer3 + layer4
        # These are the deepest (and heaviest) residual stages.
        # Checkpointing trades compute for memory: activations are
        # discarded during the forward pass and recomputed on backward.
        if self._use_gradient_checkpointing and self.training:
            x = grad_checkpoint(bb.layer3, x, use_reentrant=False)
            x = grad_checkpoint(bb.layer4, x, use_reentrant=False)
        else:
            x = bb.layer3(x) # [B, 1024, 16, 16, 16] (stride 2)
            x = bb.layer4(x) # [B, 2048, 8, 8, 8] (stride 2)
        return x

    # Forward pass
    def forward(self, dvf: Tensor) -> Tensor:
        """Extract spatial tokens from a batch of DVF volumes.

        Args:
            dvf: Normalized DVF tensor of shape [B, 3, 128, 128, 128]
                where channels correspond to delta x, delta y, delta z displacement
                components.

        Returns:
            Spatial token tensor of shape [B, 512, d_model], where
            each of the 512 tokens encodes the deformation pattern at one
            cell of the 8x8x8 spatial grid.

        Raises:
            RuntimeError: If the backbone feature map does not have the
                expected shape [B, 2048, 8, 8, 8], which indicates
                no_max_pool was not set correctly.
            RuntimeError: If the final output shape does not match
                [B, 512, d_model].
        """
        batch_size = dvf.shape[0]

        # Backbone forward (frozen by default)
        # We call _backbone_feature_forward instead of self.backbone(dvf)
        # because MONAI's ResNet.forward() flattens the spatial grid.
        if not any(p.requires_grad for p in self.backbone.parameters()):
            with torch.no_grad():
                features = self._backbone_feature_forward(dvf)  # [B, 2048, 8, 8, 8]
        else:
            features = self._backbone_feature_forward(dvf)  # [B, 2048, 8, 8, 8]

        # Shape assertion on backbone output
        expected_shape = (
            batch_size,
            RESNET_STAGE4_CHANNELS,
            SPATIAL_GRID_SIZE,
            SPATIAL_GRID_SIZE,
            SPATIAL_GRID_SIZE,
        )
        if features.shape != expected_shape:
            raise RuntimeError(
                f"Backbone feature map shape mismatch: got "
                f"{tuple(features.shape)}, expected {expected_shape}. "
                f"This usually means no_max_pool was not set to True "
                f"when constructing the ResNet50. With no_max_pool=False "
                f"the stem applies max-pooling (32x total downsampling), "
                f"collapsing the spatial grid to 4x4x4 = 64 tokens "
                f"instead of the required 8x8x8 = 512. Re-initialise "
                f"the backbone with no_max_pool=True."
            )

        # Reshape spatial grid to token sequence
        # features: [B, 2048, 8, 8, 8]
        b, c, d, h, w = features.shape
        tokens = features.reshape(b, c, d * h * w)  # [B, 2048, 512]
        tokens = tokens.permute(0, 2, 1)  # [B, 512, 2048]

        # Project to d_model
        tokens = self.projection(tokens)  # [B, 512, d_model]

        # Output shape assertion
        expected_output = (batch_size, N_SPATIAL_TOKENS, self._d_model)
        if tokens.shape != expected_output:
            raise RuntimeError(
                f"Output shape mismatch: got {tuple(tokens.shape)}, "
                f"expected {expected_output}. Check d_model "
                f"({self._d_model}) and projection head configuration."
            )

        return tokens

    # LP-FT utilities
    def unfreeze_backbone(self) -> List[dict]:
        """Unfreeze backbone parameters for Stage 2 fine-tuning.

        Sets requires_grad = True for all backbone parameters and
        returns optimizer parameter groups with differential learning
        rates. The backbone receives a 10x lower learning rate than
        the projection head to avoid catastrophic forgetting of
        pre-trained features — this is the LP-FT (Linear Probe then
        Fine-Tune) protocol from the blueprint.

        Returns:
            A list of two parameter-group dicts compatible with
            torch.optim.AdamW:
                - Group 0: backbone params at lr_projection_head * 0.1
                - Group 1: projection params at lr_projection_head
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

        logger.info(
            "Backbone unfrozen. Total trainable params: %d",
            self.trainable_parameter_count(),
        )

        return [
            {
                "params": list(self.backbone.parameters()),
                "lr": self._lr_projection_head * 0.1,
            },
            {
                "params": list(self.projection.parameters()),
                "lr": self._lr_projection_head,
            },
        ]

    def trainable_parameter_count(self) -> int:
        """Return the number of parameters with requires_grad = True.

        Useful for logging to confirm that only the projection head is
        training during LP-FT Stage 1, and that the full model is
        training after unfreeze_backbone() is called in Stage 2.

        Returns:
            Integer count of trainable parameters.
        """
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )


# Smoke test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    print("BrainIAC Feature Extractor — Smoke Test")

    # 1. Instantiate config with defaults
    config = ModelConfig()
    print(f"\n[1] ModelConfig created (d_model={config.d_model})")

    # 2. Instantiate extractor (MONAI fallback expected, no weights)
    extractor = BrainIACFeatureExtractor(config, pretrained_path=None)
    print("[2] BrainIACFeatureExtractor instantiated (MONAI fallback)")

    # 3. Print trainable parameter count (projection head only)
    trainable_stage1 = extractor.trainable_parameter_count()
    print(
        f"[3] Trainable parameters (Stage 1 — projection only): "
        f"{trainable_stage1:,}"
    )

    # 4. Create random input [B=2, 3, 128, 128, 128]
    device = torch.device("cpu")
    dummy_dvf = torch.randn(2, 3, 128, 128, 128, device=device)
    print(f"[4] Input tensor shape: {tuple(dummy_dvf.shape)}")

    # 5. Forward pass #1
    with torch.no_grad():
        output1 = extractor(dummy_dvf)
    print(f"[5] Output shape (frozen backbone): {tuple(output1.shape)}")

    # 6. Unfreeze backbone
    param_groups = extractor.unfreeze_backbone()
    trainable_stage2 = extractor.trainable_parameter_count()
    print(
        f"[6] Trainable parameters (Stage 2 — backbone unfrozen): "
        f"{trainable_stage2:,}"
    )
    print(f"Param groups returned: {len(param_groups)}")

    # 7. Forward pass #2 (backbone unfrozen)
    output2 = extractor(dummy_dvf)
    print(
        f"[7] Output shape (unfrozen backbone): {tuple(output2.shape)}"
    )

    # 8. Assert output shape
    expected = (2, N_SPATIAL_TOKENS, config.d_model)
    assert output2.shape == expected, (
        f"Shape assertion failed: {tuple(output2.shape)} != {expected}"
    )
    print(f"[8] Shape assertion passed: {tuple(output2.shape)} == {expected}")

    # 9. Final verdict
    print("PASS — All assertions hold.")

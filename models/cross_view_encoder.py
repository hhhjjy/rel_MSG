import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from torch.utils.checkpoint import checkpoint

from .vggt_layers import (
    Block,
    RotaryPositionEmbedding2D,
    PositionGetter,
)


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, :, :].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, :, :].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.reshape(B * S, *combined.shape[2:])
    return combined


class CrossViewEncoder(nn.Module):
    """
    Cross-View Encoder based on VGGT's Alternating Attention mechanism.

    This encoder applies alternating frame-level and global-level attention
    to fuse information across multiple views. It preserves the original VGGT
    architecture with necessary adaptations for RelationalMSG.

    Args:
        dim: Dimension of the token embeddings.
        num_heads: Number of attention heads.
        num_layers: Number of alternating attention layers.
        num_img_patches: Number of image patches (for position embedding init).
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: Whether to include bias in QKV projections.
        proj_bias: Whether to include bias in the output projection.
        ffn_bias: Whether to include bias in MLP layers.
        qk_norm: Whether to apply QK normalization.
        rope_freq: Base frequency for rotary embedding. -1 to disable.
        init_values: Init scale for layer scale.
        num_register_tokens: Number of register tokens.
        aa_order: The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size: How many blocks to group under each attention type before switching.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 2,
        num_img_patches: int = 256,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        qk_norm: bool = True,
        rope_freq: float = 100.0,
        init_values: float = 0.01,
        num_register_tokens: int = 4,
        aa_order: List[str] = ["frame", "global"],
        aa_block_size: int = 1,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_img_patches = num_img_patches
        self.num_register_tokens = num_register_tokens
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # Frame-level and Global-level attention blocks
        self.frame_blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(num_layers)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(num_layers)
            ]
        )

        # Validate that depth is divisible by aa_block_size
        if self.num_layers % self.aa_block_size != 0:
            raise ValueError(f"num_layers ({num_layers}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.num_layers // self.aa_block_size

        # Camera and register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # For compatibility with original CrossViewEncoder interface
        # Box embedding (kept for interface compatibility)
        self.box_emb = nn.Linear(4, dim, bias=False)
        # Input adaptors (kept for interface compatibility)
        self.object_proj = nn.Linear(dim, dim, bias=False)
        self.place_proj = nn.Linear(dim, dim, bias=False)

        # Scene-level decoder (kept for interface compatibility)
        self.scene_decoder = None

        self.use_reentrant = False  # hardcoded to False

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B * S, P, 2)

        intermediates = []

        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B, S * P, 2)

        intermediates = []

        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        return tokens, global_idx, intermediates

    def forward_vggt_style(
        self,
        patch_tokens: torch.Tensor,
        num_views: int,
        patch_grid_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        VGGT-style forward pass with alternating attention.

        Args:
            patch_tokens: Patch tokens with shape [B, S, N, C], where S is number of views.
            num_views: Number of views (S).
            patch_grid_size: Optional grid size (H, W) of patches.

        Returns:
            Tuple of (output_list, patch_start_idx), where:
                output_list: List of outputs from each attention layer.
                patch_start_idx: Index where patch tokens start (after special tokens).
        """
        B, S, N, C = patch_tokens.shape

        # Reshape to [B*S, N, C] for processing
        patch_tokens = patch_tokens.reshape(B * S, N, C)

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None and patch_grid_size is not None:
            Hp, Wp = patch_grid_size
            pos = self.position_getter(B * S, Hp, Wp, device=patch_tokens.device)

            if self.patch_start_idx > 0:
                # Do not use position embedding for special tokens
                pos = pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=patch_tokens.device, dtype=pos.dtype)
                pos = torch.cat([pos_special, pos], dim=1)

        # Update P because we added special tokens
        _, P, _ = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # Concat frame and global intermediates
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        return output_list, self.patch_start_idx

    def forward(
        self,
        bbox_feats: torch.Tensor,
        img_feats: Optional[torch.Tensor] = None,
        bboxes: Optional[torch.Tensor] = None,
        bbox_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for CrossViewEncoder with bbox features.

        This module applies alternating frame-level (intra-view) and global-level
        (cross-view) attention to bbox features from multiple views.

        Args:
            bbox_feats: [B, V, N, C], N is number of bboxes per view.
            img_feats: [B, V, H, W, C] or [B, V, L, C], image features (optional, unused).
            bboxes: [B, V, N, 4], bbox coordinates (optional, for positional embedding).
            bbox_masks: [B, V, N], valid bbox masks, True=valid, False=padding.

        Returns:
            refined_obj_bank: [B, V, N, C] fused object features.
        """
        B, V, N, C = bbox_feats.shape

        # bbox_masks: True means valid, False means padding
        # Convert to attention mask format: True (attention) / False (mask out)
        if bbox_masks is None:
            bbox_masks = torch.ones(B, V, N, dtype=torch.bool, device=bbox_feats.device)
        else:
            bbox_masks = bbox_masks.bool()

        # Flatten to [B*V, N, C] for frame-level processing
        bbox_feats_flat = bbox_feats.reshape(B * V, N, C)
        masks_flat = bbox_masks.reshape(B * V, N)

        # ===== Frame-level (Intra-view) Attention =====
        # Each view processes its own bbox features independently
        frame_tokens = bbox_feats_flat

        # Create frame-level positional embedding (view index as position)
        # Use learnable view embeddings for each view
        # pos_dim = C // 2
        # if not hasattr(self, 'view_embedding'):
        #     self.view_embedding = nn.Parameter(
        #         torch.randn(1, V, 1, pos_dim, device=bbox_feats.device, dtype=bbox_feats.dtype)
        #     )
        # view_emb = self.view_embedding.expand(B, -1, N, -1).reshape(B * V, N, pos_dim)

        # if bboxes is not None:
        #     bboxes_flat = bboxes.reshape(B * V, N, 4)
        #     bbox_pos = self._encode_bbox_pos(bboxes_flat)
        #     pos_emb = torch.cat([view_emb, bbox_pos], dim=-1)
        # else:
        #     pos_emb = view_emb

        # frame_tokens = frame_tokens + pos_emb

        # Apply frame-level attention blocks
        frame_idx = 0
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    frame_tokens, frame_idx, _ = self._process_frame_attention_with_mask(
                        frame_tokens, B, V, N, C, frame_idx, mask=masks_flat
                    )
                elif attn_type == "global":
                    # Global attention on flattened sequence [B, V*N, C]
                    frame_tokens_2d = frame_tokens.reshape(B, V * N, C)
                    masks_2d = masks_flat.reshape(B, V * N)
                    frame_tokens_2d, _, _ = self._process_global_attention_with_mask(
                        frame_tokens_2d, B, V, N, C, 0, mask=masks_2d
                    )
                    frame_tokens = frame_tokens_2d.reshape(B * V, N, C)

        # Reshape back to [B, V, N, C]
        refined_obj_bank = frame_tokens.reshape(B, V, N, C)

        # ===== Optional: Cross-attention with image features =====
        if img_feats is not None and hasattr(self, 'img_cross_attn'):
            if len(img_feats.shape) == 5:
                _, _, H, W, _ = img_feats.shape
                img_feats_2d = img_feats.reshape(B, V, H * W, C)
            else:
                img_feats_2d = img_feats

            # Cross-attention: bbox -> image
            img_cross_attn = self.img_cross_attn(
                query=refined_obj_bank.reshape(B * V, N, C),
                key=img_feats_2d.reshape(B * V, -1, C),
                value=img_feats_2d.reshape(B * V, -1, C),
            ).reshape(B, V, N, C)

            # Create valid mask for image tokens (all valid)
            img_masks = torch.ones(B * V, img_feats_2d.shape[2], dtype=torch.bool, device=bbox_feats.device)

            # Apply cross-attention with mask
            refined_obj_bank = refined_obj_bank + img_cross_attn

        return refined_obj_bank

    def _encode_bbox_pos(self, bboxes: torch.Tensor) -> torch.Tensor:
        """Encode bbox coordinates to positional embedding.

        Args:
            bboxes: [B*V, N, 4] in format [x1, y1, x2, y2] normalized by 224

        Returns:
            pos_emb: [B*V, N, C//2] positional embedding
        """
        B, N, _ = bboxes.shape
        pos_dim = self.dim // 2

        bboxes_norm = bboxes / 224.0

        x1 = bboxes_norm[..., 0:1]
        y1 = bboxes_norm[..., 1:2]
        x2 = bboxes_norm[..., 2:3]
        y2 = bboxes_norm[..., 3:4]

        freqs = torch.logspace(0, 0.5, pos_dim // 4, device=bboxes.device, dtype=bboxes.dtype)

        x1_enc = torch.sin(x1 * freqs)
        y1_enc = torch.cos(y1 * freqs)
        x2_enc = torch.sin(x2 * freqs)
        y2_enc = torch.cos(y2 * freqs)

        return torch.cat([x1_enc, y1_enc, x2_enc, y2_enc], dim=-1)

    def _process_frame_attention_with_mask(
        self, tokens: torch.Tensor, B: int, V: int, N: int, C: int,
        frame_idx: int, mask=None
    ):
        """
        Process frame attention blocks with masking support.

        Args:
            tokens: [B*V, N, C] or [B, V, N, C]
            B, V, N, C: batch, views, num_bboxes, channels
            frame_idx: current frame block index
            mask: attention mask [B*V, N], True=valid, False=mask out

        Returns:
            tokens: processed tokens
            frame_idx: updated block index
            intermediates: list of intermediate outputs
        """
        if tokens.shape != (B * V, N, C):
            tokens = tokens.reshape(B, V, N, C).reshape(B * V, N, C)

        # Handle mask: convert to key padding mask format [B*V, N]
        # True in mask means "mask out", False means "keep"
        key_padding_mask = ~mask if mask is not None else None

        intermediates = []

        for _ in range(self.aa_block_size):
            block = self.frame_blocks[frame_idx]
            if self.training:
                tokens = checkpoint(block, tokens, use_reentrant=self.use_reentrant)
            else:
                tokens = block(tokens)

            # Apply mask to output tokens (zero out invalid positions)
            if key_padding_mask is not None:
                tokens = tokens.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

            frame_idx += 1
            intermediates.append(tokens.reshape(B, V, N, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention_with_mask(
        self, tokens: torch.Tensor, B: int, V: int, N: int, C: int,
        global_idx: int, mask=None
    ):
        """
        Process global attention blocks with masking support.

        Args:
            tokens: [B, V*N, C]
            B, V, N, C: batch, views, num_bboxes, channels
            global_idx: current global block index
            mask: attention mask [B, V*N], True=valid, False=mask out

        Returns:
            tokens: processed tokens
            global_idx: updated block index
            intermediates: list of intermediate outputs
        """
        if tokens.shape != (B, V * N, C):
            tokens = tokens.reshape(B, V, N, C).reshape(B, V * N, C)

        # Handle mask
        key_padding_mask = ~mask if mask is not None else None

        intermediates = []

        for _ in range(self.aa_block_size):
            block = self.global_blocks[global_idx]
            if self.training:
                tokens = checkpoint(block, tokens, use_reentrant=self.use_reentrant)
            else:
                tokens = block(tokens)

            # Apply mask
            if key_padding_mask is not None:
                tokens = tokens.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

            global_idx += 1
            intermediates.append(tokens.reshape(B, V, N, C))

        return tokens, global_idx, intermediates

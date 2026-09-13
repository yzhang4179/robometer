#!/usr/bin/env python3
"""PreciseTransformer: block-causal progress prediction over Wan-2.2 VAE latents.

One frozen VAE encodes both streams; a small transformer reads the latents and a
per-frame progress token, and the stock RBM progress/success heads read those
tokens back out. Same `(ModelOutput, timing_raw)` contract as ReWiND, so the
existing trainer and loss code apply unchanged.

Three signals are kept deliberately separate (Step 2 §6 of the plan):

* **position** -- one shared 3-D RoPE over `(t, h, w)` latent-grid coordinates. The
  RGB token and the pointmap token at the same `(t, h, w)` describe the same
  physical patch at the same instant, so they get the *same* position. That is
  what lets attention bind them for free. Flattening the two streams into one long
  1-D sequence would instead tell the model the pointmap is 64 steps later, which
  is simply false.
* **modality** -- one additive learned vector per stream. Because rgb and pointmap
  now share positions, this is the only thing that distinguishes them, so it is
  required rather than optional.
* **identity of a progress token** -- a learned per-frame vector from a
  `(1, max_len, D)` table. Inside one progress block, `prog_2..prog_5` see
  identical context; their own embedding is the only thing telling them which
  frame they are predicting. ReWiND does the same with `prog_token_A`.

Note the leakage this inherits from the VAE, which no mask can fix: latent frame 2
is built from input frames 2-5, so `prog_2` cannot see frame 2 without also seeing
frames 3-5. Only the last frame of a clip is leak-free. `progress_logits` is
supervised on all frames; `output.last_token_index` marks the one that matches the
Step 4 eval protocol, so training can log a leak-free metric alongside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

from robometer.models.heads import PredictionHeadsMixin
from robometer.models.precise.layout import (
    POINTMAP,
    PROG,
    RGB,
    TokenLayout,
    assert_shared_positions,
    build_block_causal_mask,
    build_token_layout,
    visual_streams,
)
from robometer.models.precise.wan_tokenizer import WanLatentTokenizer
from robometer.models.utils import ModelOutput


@dataclass
class PreciseModelOutput(ModelOutput):
    """`ModelOutput` plus the bookkeeping the dev commands and Step 3 metrics need."""

    layout: Optional[TokenLayout] = None
    latents: Optional[Dict[str, torch.Tensor]] = None
    prog_hidden: Optional[torch.Tensor] = None
    last_token_index: Optional[int] = None  # the only leak-free frame; see module docstring


class PreciseTransformerConfig(PretrainedConfig):
    model_type = "precise_transformer"

    def __init__(
        self,
        modality: str = "rgb_pointmap",
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_attention_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_len: int = 9,
        latent_patch_size: int = 2,
        prog_block_causal: bool = False,
        rope_theta: float = 10000.0,
        vae_model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        vae_dtype: str = "bfloat16",
        vae_z_dim: int = 48,
        vae_spatial_ratio: int = 16,
        vae_temporal_ratio: int = 4,
        scale_latents: bool = True,
        vae_keep_decoder: bool = False,
        pointmap_norm_bounds: Optional[Dict[str, Any]] = None,
        frame_size: int = 256,
        progress_loss_type: str = "l2",
        progress_discrete_bins: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.modality = modality
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.max_len = max_len
        self.latent_patch_size = latent_patch_size
        self.prog_block_causal = prog_block_causal
        self.rope_theta = rope_theta
        self.vae_model_id = vae_model_id
        self.vae_dtype = vae_dtype
        self.vae_z_dim = vae_z_dim
        self.vae_spatial_ratio = vae_spatial_ratio
        self.vae_temporal_ratio = vae_temporal_ratio
        self.scale_latents = scale_latents
        self.vae_keep_decoder = vae_keep_decoder
        self.pointmap_norm_bounds = pointmap_norm_bounds
        self.frame_size = frame_size
        self.progress_loss_type = progress_loss_type
        self.progress_discrete_bins = progress_discrete_bins


# --------------------------------------------------------------------- 3-D RoPE


def split_rope_dims(head_dim: int) -> Tuple[int, int, int]:
    """Divide the head's rotation pairs across the (t, h, w) axes.

    `head_dim // 2` pairs are split as evenly as possible; the remainder goes to
    the spatial axes, which have more distinct values to resolve than time does.
    """
    if head_dim % 2:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    pairs = head_dim // 2
    if pairs < 3:
        raise ValueError(f"head_dim {head_dim} is too small to split across 3 axes")
    pairs_t = pairs // 3
    pairs_h = (pairs - pairs_t) // 2
    pairs_w = pairs - pairs_t - pairs_h
    return pairs_t, pairs_h, pairs_w


class RotaryEmbedding3D(nn.Module):
    """Axial RoPE over `(t, h, w)`; positions come from the token layout."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.splits = split_rope_dims(head_dim)
        for name, pairs in zip(("t", "h", "w"), self.splits):
            inv_freq = theta ** (-torch.arange(pairs, dtype=torch.float32) / max(pairs, 1))
            self.register_buffer(f"inv_freq_{name}", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """`[L, 3]` int positions -> `(cos, sin)`, each `[L, head_dim // 2]`."""
        angles = []
        for axis, name in enumerate(("t", "h", "w")):
            inv_freq = getattr(self, f"inv_freq_{name}")
            angles.append(positions[:, axis].to(inv_freq.dtype)[:, None] * inv_freq[None, :])
        angle = torch.cat(angles, dim=-1)
        return angle.cos(), angle.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channel pairs of `[B, heads, L, head_dim]`."""
    cos = cos[None, None].to(x.dtype)
    sin = sin[None, None].to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
    return rotated.flatten(-2)


# --------------------------------------------------------------- encoder blocks


class PreciseAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError(f"hidden_dim {hidden_dim} is not divisible by num_attention_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, cos, sin, keep_mask):
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=keep_mask, dropout_p=self.dropout if self.training else 0.0
        )
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.out(attended)


class PreciseEncoderLayer(nn.Module):
    """Pre-norm block. Written out rather than reusing `nn.TransformerEncoderLayer`
    because that one routes through `nn.MultiheadAttention`, which has no hook for
    applying rotary embeddings to q and k."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = PreciseAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin, keep_mask):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin, keep_mask))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------- model


class PreciseTransformer(PredictionHeadsMixin, PreTrainedModel):
    """Progress + success prediction over Wan-2.2 latents of RGB and/or pointmaps."""

    config_class = PreciseTransformerConfig
    supports_gradient_checkpointing = True

    def __init__(self, config, latent_tokenizer: Optional[WanLatentTokenizer] = None):
        # Accept either the outer model config (training builds it that way, mirroring
        # ReWiND) or a bare PreciseTransformerConfig, which is what `from_pretrained`
        # hands back after a checkpoint round-trip.
        precise_config = config.precise if hasattr(config, "precise") else config
        if isinstance(precise_config, dict):
            # `self.config_class`, not the literal, so a subclass keeps its own extra
            # fields instead of having them dropped on the way in.
            precise_config = self.config_class(**precise_config)
        if not isinstance(precise_config, PreciseTransformerConfig):
            raise ValueError(
                "model.precise is missing -- PreciseTransformer needs its sub-config "
                f"(got {type(precise_config).__name__})"
            )

        super().__init__(
            config=precise_config,
            model_config=config if hasattr(config, "precise") else precise_config,
            hidden_dim=precise_config.hidden_dim,
            dropout=precise_config.dropout,
        )

        self.streams = visual_streams(precise_config.modality)
        patch = precise_config.latent_patch_size
        latent_side = precise_config.frame_size // precise_config.vae_spatial_ratio
        if latent_side % patch:
            raise ValueError(
                f"latent grid {latent_side}x{latent_side} is not divisible by latent_patch_size {patch}. "
                f"frame_size={precise_config.frame_size}, vae_spatial_ratio={precise_config.vae_spatial_ratio}."
            )
        self.grid_h = self.grid_w = latent_side // patch
        patch_dim = precise_config.vae_z_dim * patch * patch

        # Separate projections per stream: rgb and pointmap latents have very
        # different statistics, so a shared projection would fight itself.
        self.stream_proj = nn.ModuleDict(
            {stream: nn.Linear(patch_dim, precise_config.hidden_dim) for stream in self.streams}
        )
        self.modality_embedding = nn.ParameterDict(
            {
                stream: nn.Parameter(torch.randn(1, 1, precise_config.hidden_dim) * 0.02)
                for stream in (*self.streams, PROG)
            }
        )
        # The per-frame identity of each progress token; without it, prog_2..prog_5
        # are indistinguishable inside their block.
        self.prog_token = nn.Parameter(torch.randn(1, precise_config.max_len, precise_config.hidden_dim) * 0.02)

        self.input_norm = nn.LayerNorm(precise_config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                PreciseEncoderLayer(
                    precise_config.hidden_dim,
                    precise_config.num_attention_heads,
                    precise_config.mlp_ratio,
                    precise_config.dropout,
                )
                for _ in range(precise_config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(precise_config.hidden_dim)
        self.rope = RotaryEmbedding3D(
            precise_config.hidden_dim // precise_config.num_attention_heads, precise_config.rope_theta
        )

        # `PredictionHeadsMixin` builds a preference head, but this model has no
        # preference path -- `forward` rejects preference samples outright. Removing it
        # keeps dead parameters out of every checkpoint and avoids the FSDP hazard the
        # ReWiND docstring warns about ("make sure the forward pass uses all of the heads").
        del self.preference_head

        # Startup validation: a bounds typo should be an error here, not a silent skew.
        if POINTMAP in self.streams:
            from robometer.models.precise.wan_tokenizer import validate_bounds

            validate_bounds(precise_config.pointmap_norm_bounds)

        # Plain attribute, not a submodule -- see wan_tokenizer.py for why.
        self.latent_tokenizer = latent_tokenizer
        self._layout_cache: Dict[Tuple, TokenLayout] = {}
        self._mask_cache: Dict[Tuple, torch.Tensor] = {}

    # ------------------------------------------------------------------ helpers

    def attach_tokenizer(self, latent_tokenizer: WanLatentTokenizer) -> None:
        self.latent_tokenizer = latent_tokenizer

    def get_layout(self, num_frames: int) -> TokenLayout:
        key = (num_frames, self.config.modality, self.grid_h, self.grid_w)
        if key not in self._layout_cache:
            layout = build_token_layout(
                num_frames,
                self.config.modality,
                self.grid_h,
                self.grid_w,
                self.config.vae_temporal_ratio,
            )
            assert_shared_positions(layout)
            self._layout_cache[key] = layout
        return self._layout_cache[key]

    def _rope_and_mask(self, layout: TokenLayout, device, dtype):
        """`(cos, sin, keep_mask)` for this layout, cached per (layout, device)."""
        key = (layout.seq_len, layout.modality, layout.num_frames, str(device))
        if key not in self._mask_cache:
            positions = torch.as_tensor(layout.positions, device=device)
            cos, sin = self.rope(positions)
            # `build_block_causal_mask` uses ReWiND's convention (True = blocked);
            # scaled_dot_product_attention wants True = allowed, so invert once, here.
            blocked = build_block_causal_mask(layout, self.config.prog_block_causal)
            keep = torch.as_tensor(~blocked, device=device)
            self._mask_cache[key] = (cos, sin, keep)
        cos, sin, keep = self._mask_cache[key]
        return cos.to(dtype), sin.to(dtype), keep

    def _tokenize_stream(self, stream: str, latents: torch.Tensor) -> torch.Tensor:
        """`[B, z, T_lat, H, W]` -> `[B, T_lat, grid_h * grid_w, D]`."""
        patch = self.config.latent_patch_size
        tokens = einops.rearrange(
            latents, "b c t (h p1) (w p2) -> b t (h w) (c p1 p2)", p1=patch, p2=patch
        )
        tokens = self.stream_proj[stream](tokens.to(self.stream_proj[stream].weight.dtype))
        return tokens + self.modality_embedding[stream][:, None]

    def _prefix_tokens(self, reference_tokens: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        """Hook: `[B, n, D]` tokens to prepend ahead of every visual block, or None.

        The base model has none. `LangPreciseTransformer` returns one language token
        here and widens the layout to match; nothing else in `forward` changes.
        """
        return None

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        pixel_values_videos: Optional[torch.Tensor] = None,
        pointmap_values: Optional[torch.Tensor] = None,
        sample_type: str = "progress",
        timing_raw: Optional[dict] = None,
        return_latents: bool = False,
        **kwargs,
    ):
        """`pixel_values_videos` `[B, T, 3, H, W]` uint8, `pointmap_values` `[B, T, 3, H, W]` raw metres."""
        if timing_raw is None:
            timing_raw = {}
        if sample_type != "progress":
            raise ValueError(
                f"PreciseTransformer only handles progress samples, got sample_type={sample_type!r}. "
                "Set data.sample_type_ratio=[0, 1, 0]."
            )
        if self.latent_tokenizer is None:
            raise ValueError("No WanLatentTokenizer attached; build the model via build_precise_model()")

        reference = pixel_values_videos if RGB in self.streams else pointmap_values
        if reference is None:
            raise ValueError(f"modality={self.config.modality} requires inputs for streams {self.streams}")
        batch, num_frames = reference.shape[0], reference.shape[1]
        if num_frames > self.config.max_len:
            raise ValueError(f"{num_frames} frames exceeds max_len={self.config.max_len}")
        height, width = reference.shape[-2], reference.shape[-1]
        if (height, width) != (self.config.frame_size, self.config.frame_size):
            # The latent grid is derived from frame_size alone, so a pixel mismatch would
            # otherwise surface much later as an unexplained token-count assertion.
            raise ValueError(
                f"frames are {height}x{width} but model.precise.frame_size={self.config.frame_size} "
                "(square). Re-render the data or set frame_size to match."
            )

        self.latent_tokenizer.ensure_device(reference.device)

        latents: Dict[str, torch.Tensor] = {}
        if RGB in self.streams:
            latents[RGB] = self.latent_tokenizer.encode_rgb(pixel_values_videos)
        if POINTMAP in self.streams:
            if pointmap_values is None:
                raise ValueError("modality includes pointmaps but pointmap_values is None")
            latents[POINTMAP] = self.latent_tokenizer.encode_pointmap(pointmap_values)

        layout = self.get_layout(num_frames)
        stream_tokens = {stream: self._tokenize_stream(stream, latents[stream]) for stream in self.streams}

        prog_tokens = self.prog_token[:, :num_frames] + self.modality_embedding[PROG]
        prog_tokens = prog_tokens.to(stream_tokens[self.streams[0]].dtype).expand(batch, -1, -1)

        # Assemble in block order: [visual(l), prog(frames of l)] for each latent frame.
        chunks = []
        for latent_frame in range(layout.num_latent_frames):
            for stream in self.streams:
                chunks.append(stream_tokens[stream][:, latent_frame])
            frames = torch.as_tensor(
                (layout.frame_to_latent == latent_frame).nonzero()[0], device=prog_tokens.device
            )
            chunks.append(prog_tokens[:, frames])
        prefix = self._prefix_tokens(chunks[0], **kwargs)
        if prefix is not None:
            chunks.insert(0, prefix)
        sequence = torch.cat(chunks, dim=1)
        if sequence.shape[1] != layout.seq_len:
            raise AssertionError(f"assembled {sequence.shape[1]} tokens but the layout says {layout.seq_len}")

        cos, sin, keep_mask = self._rope_and_mask(layout, sequence.device, sequence.dtype)

        hidden = self.input_norm(sequence)
        for layer in self.layers:
            if getattr(self, "gradient_checkpointing", False) and self.training:
                hidden = self._gradient_checkpointing_func(layer.__call__, hidden, cos, sin, keep_mask)
            else:
                hidden = layer(hidden, cos, sin, keep_mask)
        hidden = self.final_norm(hidden)

        prog_index = torch.as_tensor(layout.prog_index, device=hidden.device)
        prog_hidden = hidden[:, prog_index]  # [B, T, D]
        width = prog_hidden.shape[-1]

        progress_logits = self.progress_head(prog_hidden.reshape(-1, width))
        if self.use_discrete_progress:
            progress_logits = einops.rearrange(progress_logits, "(b t) ... -> b t ...", b=batch, t=num_frames)
        else:
            progress_logits = einops.rearrange(progress_logits, "(b t) 1 -> b t", b=batch, t=num_frames)

        success_logits = self.success_head(prog_hidden.reshape(-1, width))
        success_logits = einops.rearrange(success_logits, "(b t) 1 -> b t", b=batch, t=num_frames)

        output = PreciseModelOutput(
            progress_logits={"A": progress_logits, "B": None},
            success_logits={"A": success_logits, "B": None},
            hidden_states=hidden,
            layout=layout,
            prog_hidden=prog_hidden,
            last_token_index=num_frames - 1,
        )
        if return_latents:
            output.latents = latents
        return output, timing_raw

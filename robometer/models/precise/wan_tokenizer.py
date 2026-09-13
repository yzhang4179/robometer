#!/usr/bin/env python3
"""The frozen Wan-2.2 VAE, wrapped as a tokenizer for RGB and pointmaps.

`WanLatentTokenizer` is deliberately **not** an `nn.Module`. Two reasons:

* it must never appear in `state_dict()` -- it is 2.8 GB of frozen weights that
  would otherwise land in every checkpoint;
* attaching it to a `PreciseTransformer` attribute therefore does not register it,
  so `model.parameters()` stays clean and the optimizer cannot touch it.

The price is that `model.to(device)` does not move it, so the model calls
`ensure_device()` at the top of `forward` instead. That is one line and it is
always right, including under `.cuda()`, DDP and `accelerate`.

**This file holds the only clamp in the pipeline.** `normalize_pointmap` clamps
raw camera-frame metres to `pointmap_norm_bounds` and maps them to [-1, 1].
Everything upstream -- the hdf5 converter, the cache, the sampler, the collator --
carries raw metres, so the bounds stay a CLI flag rather than a re-conversion.
See "Where the clamp lives in the code" in precise_robometer_research.md.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from robometer.models.precise.layout import latent_frame_count

WAN_TI2V_5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Metres, camera frame. Frozen for the square task -- see Step 2 §7 of the plan.
DEFAULT_POINTMAP_BOUNDS: Dict[str, Tuple[float, float]] = {
    "x": (-0.6, 0.6),
    "y": (-1.0, 0.25),
    "z": (0.2, 1.3),
}
CHANNELS = ("x", "y", "z")


def validate_bounds(bounds: Optional[Dict[str, Sequence[float]]]) -> Dict[str, Tuple[float, float]]:
    """Fail at startup on a bad bounds config rather than silently skewing training."""
    if bounds is None:
        raise ValueError(
            "pointmap_norm_bounds is required when the modality includes pointmaps. "
            f"The frozen square-task values are {DEFAULT_POINTMAP_BOUNDS}."
        )
    checked = {}
    for channel in CHANNELS:
        if channel not in bounds:
            raise ValueError(f"pointmap_norm_bounds is missing channel {channel!r}; got keys {sorted(bounds)}")
        pair = bounds[channel]
        if len(pair) != 2:
            raise ValueError(f"pointmap_norm_bounds.{channel} must be [lo, hi], got {pair!r}")
        lo, hi = float(pair[0]), float(pair[1])
        if not lo < hi:
            raise ValueError(f"pointmap_norm_bounds.{channel} needs lo < hi, got [{lo}, {hi}]")
        checked[channel] = (lo, hi)
    return checked


class WanLatentTokenizer:
    """Frozen Wan-2.2 VAE encoder, shared by the RGB and pointmap streams.

    Flex-pi reports that this VAE reconstructs 3D pointmaps almost losslessly with
    no pointmap-specific training, which is why one frozen encoder serves both.
    """

    def __init__(
        self,
        model_id: str = WAN_TI2V_5B,
        subfolder: str = "vae",
        dtype: torch.dtype = torch.bfloat16,
        pointmap_norm_bounds: Optional[Dict[str, Sequence[float]]] = None,
        scale_latents: bool = True,
        keep_decoder: bool = False,
        device: torch.device | str = "cpu",
        vae: Optional[torch.nn.Module] = None,
    ):
        from diffusers import AutoencoderKLWan

        self.model_id = model_id
        self.dtype = dtype
        self.scale_latents = scale_latents

        self.vae = vae if vae is not None else AutoencoderKLWan.from_pretrained(
            model_id, subfolder=subfolder, torch_dtype=dtype
        )
        self.vae.eval()
        self.vae.requires_grad_(False)

        # The decoder is 555 M of the VAE's 705 M parameters (~1.1 GB in bf16) and we
        # never call it -- this is an encoder-only use. Dropping it is most of the VAE's
        # memory back. Keep it only to eyeball what the VAE loses on a reconstruction.
        self.has_decoder = keep_decoder
        if not keep_decoder and getattr(self.vae, "decoder", None) is not None:
            self.vae.decoder = None

        self.bounds = validate_bounds(pointmap_norm_bounds) if pointmap_norm_bounds is not None else None
        self._device = torch.device(device)

        config = self.vae.config
        self.z_dim = int(config.z_dim)
        self.spatial_ratio = int(config.scale_factor_spatial)
        self.temporal_ratio = int(config.scale_factor_temporal)

        # Per-channel latent statistics that Wan's own pipeline uses; the encode side
        # is (z - mean) / std, the decode side multiplies back.
        self._latents_mean = torch.tensor(config.latents_mean, dtype=torch.float32).view(1, self.z_dim, 1, 1, 1)
        self._latents_std = torch.tensor(config.latents_std, dtype=torch.float32).view(1, self.z_dim, 1, 1, 1)
        self._lo: Optional[torch.Tensor] = None
        self._hi: Optional[torch.Tensor] = None
        if self.bounds is not None:
            self._lo = torch.tensor([self.bounds[c][0] for c in CHANNELS], dtype=torch.float32).view(1, 3, 1, 1, 1)
            self._hi = torch.tensor([self.bounds[c][1] for c in CHANNELS], dtype=torch.float32).view(1, 3, 1, 1, 1)

        self.to(self._device)

    # ------------------------------------------------------------------ plumbing

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "WanLatentTokenizer":
        device = torch.device(device)
        self.vae.to(device)
        self._latents_mean = self._latents_mean.to(device)
        self._latents_std = self._latents_std.to(device)
        if self._lo is not None:
            self._lo = self._lo.to(device)
            self._hi = self._hi.to(device)
        self._device = device
        return self

    def ensure_device(self, device: torch.device | str) -> None:
        """Move onto `device` if we are not already there. Called from `forward`."""
        device = torch.device(device)
        if device != self._device:
            self.to(device)

    def latent_frame_count(self, num_frames: int) -> int:
        return latent_frame_count(num_frames, self.temporal_ratio)

    def latent_grid(self, height: int, width: int) -> Tuple[int, int]:
        if height % self.spatial_ratio or width % self.spatial_ratio:
            raise ValueError(
                f"frame size {height}x{width} is not divisible by the VAE spatial ratio {self.spatial_ratio}"
            )
        return height // self.spatial_ratio, width // self.spatial_ratio

    # --------------------------------------------------------------- normalisers

    def normalize_rgb(self, frames: torch.Tensor) -> torch.Tensor:
        """`[B, T, 3, H, W]` uint8 (or float 0-255) -> `[B, 3, T, H, W]` in [-1, 1]."""
        frames = frames.to(self._device)
        frames = frames.permute(0, 2, 1, 3, 4).to(torch.float32)
        return frames / 127.5 - 1.0

    def normalize_pointmap(self, pointmap: torch.Tensor) -> torch.Tensor:
        """`[B, T, 3, H, W]` raw camera-frame metres -> `[B, 3, T, H, W]` in [-1, 1].

        **The only clamp in the pipeline.** Out-of-range geometry (the back wall,
        the floor) saturates rather than compressing the workspace, which is the
        whole point of hand-set bounds -- see Flex-pi's `_normalize_pointmap`.
        """
        if self._lo is None:
            raise ValueError("normalize_pointmap needs pointmap_norm_bounds; none were configured")
        pointmap = pointmap.to(self._device)
        pointmap = pointmap.permute(0, 2, 1, 3, 4).to(torch.float32)
        clamped = torch.clamp(pointmap, self._lo, self._hi)
        return 2.0 * (clamped - self._lo) / (self._hi - self._lo) - 1.0

    def clip_fraction(self, pointmap: torch.Tensor) -> Dict[str, float]:
        """Share of pixels the bounds saturate, per channel. Diagnostics only."""
        if self._lo is None:
            return {}
        pointmap = pointmap.to(self._device).permute(0, 2, 1, 3, 4).to(torch.float32)
        out = {}
        for axis, channel in enumerate(CHANNELS):
            values = pointmap[:, axis]
            lo, hi = self._lo[0, axis, 0, 0, 0], self._hi[0, axis, 0, 0, 0]
            out[channel] = float(((values < lo) | (values > hi)).float().mean())
        return out

    # ------------------------------------------------------------------- encoder

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """`[B, 3, T, H, W]` in [-1, 1] -> `[B, z_dim, T_latent, H/16, W/16]`.

        Wan's encoder is causal in time: latent frame 0 comes from input frame 0,
        latent frame i>0 from input frames [1+4(i-1), 1+4i). The whole clip goes in
        one call because the encoder carries a feature cache across those chunks.
        """
        video = video.to(device=self._device, dtype=self.dtype)
        latent = self.vae.encode(video).latent_dist.mode()
        latent = latent.to(torch.float32)
        if self.scale_latents:
            latent = (latent - self._latents_mean) / self._latents_std
        return latent

    def encode_rgb(self, frames: torch.Tensor) -> torch.Tensor:
        """`[B, T, 3, H, W]` uint8 -> normalised latents."""
        return self.encode(self.normalize_rgb(frames))

    def encode_pointmap(self, pointmap: torch.Tensor) -> torch.Tensor:
        """`[B, T, 3, H, W]` raw metres -> normalised latents (clamped on the way in)."""
        return self.encode(self.normalize_pointmap(pointmap))

    def describe(self) -> str:
        bounds = "none" if self.bounds is None else ", ".join(f"{c}:{list(self.bounds[c])}" for c in CHANNELS)
        return (
            f"WanLatentTokenizer({self.model_id})  z_dim={self.z_dim}  "
            f"spatial/{self.spatial_ratio}x  temporal/{self.temporal_ratio}x  dtype={self.dtype}  "
            f"device={self._device}  scale_latents={self.scale_latents}  "
            f"decoder={'kept' if self.has_decoder else 'dropped (encoder-only, saves ~1.1 GB)'}\n"
            f"  pointmap_norm_bounds: {bounds}"
        )

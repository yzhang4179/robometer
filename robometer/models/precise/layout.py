#!/usr/bin/env python3
"""Where every token sits, and who is allowed to look at whom.

This module is pure bookkeeping -- no torch modules, no weights -- so the model,
the causal mask and the "print the token order" dev command all read the *same*
description of the sequence instead of three hand-written copies that can drift.

The layout, for `modality="rgb_pointmap"`, 9 input frames, `latent_patch_size=2`:

    B1 = [t1_rgb (64 tok), t1_pointmap (64 tok)]   <- latent frame 1
    B2 = [prog_1]
    B3 = [t2_rgb, t2_pointmap]                     <- latent frame 2
    B4 = [prog_2, prog_3, prog_4, prog_5]
    B5 = [t3_rgb, t3_pointmap]                     <- latent frame 3
    B6 = [prog_6, prog_7, prog_8, prog_9]

Attention is block-causal: block i sees blocks 1..i. Visual blocks are
bidirectional inside themselves (patches of one frame must see each other, and
rgb must see its pointmap, which is the whole point of giving them the same
position). Progress blocks are **bidirectional inside themselves** too: the tokens in
one block predict frames the VAE already merged into a single latent, so there is no
ordering left to enforce between them, and each one already knows which frame it is from
its learned `prog_token[k]` embedding. `prog_block_causal=True` restores the ordering
constraint if you want to compare.

Which frames land in which block is decided by the VAE, not by us. Wan's encoder
builds latent frame 0 from input frame 0 and latent frame `i>0` from input frames
`[1+4(i-1), 1+4i)` -- see `_encode` in `diffusers/models/autoencoders/
autoencoder_kl_wan.py`. `frame_to_latent_index` is that rule and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

RGB = "rgb"
POINTMAP = "pointmap"
PROG = "prog"

MODALITIES = ("rgb", "pointmap", "rgb_pointmap")


def visual_streams(modality: str) -> Tuple[str, ...]:
    """The visual token streams a modality contributes, in sequence order."""
    if modality == "rgb":
        return (RGB,)
    if modality == "pointmap":
        return (POINTMAP,)
    if modality == "rgb_pointmap":
        return (RGB, POINTMAP)
    raise ValueError(f"Unknown modality {modality!r}, expected one of {MODALITIES}")


def check_frame_count(num_frames: int, temporal_ratio: int = 4) -> None:
    """Wan encodes frame 0, then whole groups of `temporal_ratio`, and **silently drops
    the remainder**.

    `_encode` runs `1 + (T - 1) // 4` chunks, the last of which ends at input frame
    `1 + 4 * (iter - 1)`. Feed it 16 frames and frames 14, 15 and 16 never reach the
    encoder at all -- no error, no warning, just three frames of supervision predicted
    from data the model was never shown. So only `4k + 1` lengths are accepted.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if (num_frames - 1) % temporal_ratio:
        usable = 1 + ((num_frames - 1) // temporal_ratio) * temporal_ratio
        raise ValueError(
            f"num_frames={num_frames} is not 1 + a multiple of the VAE temporal ratio {temporal_ratio}. "
            f"Wan's encoder would silently drop the last {num_frames - usable} frame(s), so use "
            f"{usable} or {usable + temporal_ratio} instead (data.max_frames)."
        )


def latent_frame_count(num_frames: int, temporal_ratio: int = 4) -> int:
    """Wan's temporal rule: 9 frames -> 3 latent frames, 5 -> 2, 1 -> 1."""
    check_frame_count(num_frames, temporal_ratio)
    return 1 + (num_frames - 1) // temporal_ratio


def frame_to_latent_index(num_frames: int, temporal_ratio: int = 4) -> np.ndarray:
    """`out[f]` = the latent frame that first contains input frame `f`.

    Frame 0 gets its own latent frame; every later group of `temporal_ratio`
    frames shares one. This is the source of the leakage the plan warns about:
    `prog_2` must attend to latent frame 1 to see frame 2 at all, and latent
    frame 1 is also built from frames 3, 4 and 5.
    """
    check_frame_count(num_frames, temporal_ratio)
    frames = np.arange(num_frames)
    return np.where(frames == 0, 0, 1 + (frames - 1) // temporal_ratio).astype(np.int64)


@dataclass
class Block:
    """One attention block: a contiguous span of the sequence."""

    index: int  # 0-based; printed as B{index+1}
    kind: str  # "visual" or "prog"
    latent_frame: int
    start: int  # inclusive token index
    end: int  # exclusive token index
    members: List[str] = field(default_factory=list)  # e.g. ["t1_rgb (64)", "t1_pointmap (64)"]

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def label(self) -> str:
        return f"B{self.index + 1} = [{', '.join(self.members)}]"


@dataclass
class TokenLayout:
    """Everything the model and the dev commands need to know about the sequence."""

    num_frames: int
    num_latent_frames: int
    modality: str
    grid_h: int
    grid_w: int
    kinds: np.ndarray  # [L] str, one of rgb / pointmap / prog
    block_of: np.ndarray  # [L] int, which block each token belongs to
    positions: np.ndarray  # [L, 3] int, the (t, h, w) fed to 3-D RoPE
    prog_index: np.ndarray  # [num_frames] int, where prog_k sits in the sequence
    frame_to_latent: np.ndarray  # [num_frames] int
    blocks: List[Block]

    @property
    def seq_len(self) -> int:
        return int(self.kinds.shape[0])

    @property
    def tokens_per_visual_frame(self) -> int:
        return self.grid_h * self.grid_w

    def count(self, kind: str) -> int:
        return int(np.sum(self.kinds == kind))


def build_token_layout(
    num_frames: int,
    modality: str,
    grid_h: int,
    grid_w: int,
    temporal_ratio: int = 4,
) -> TokenLayout:
    """Lay the sequence out once: [visual(l), prog(frames of l)] for each latent frame l."""
    streams = visual_streams(modality)
    num_latent = latent_frame_count(num_frames, temporal_ratio)
    frame_to_latent = frame_to_latent_index(num_frames, temporal_ratio)

    kinds: List[str] = []
    positions: List[Tuple[int, int, int]] = []
    block_of: List[int] = []
    blocks: List[Block] = []
    prog_index = np.full(num_frames, -1, dtype=np.int64)

    grid_positions = [(h, w) for h in range(grid_h) for w in range(grid_w)]

    for latent in range(num_latent):
        # ---- visual block: rgb and pointmap share positions, on purpose -------------
        start = len(kinds)
        members = []
        for stream in streams:
            for h, w in grid_positions:
                kinds.append(stream)
                positions.append((latent, h, w))
                block_of.append(len(blocks))
            members.append(f"t{latent + 1}_{stream} ({len(grid_positions)})")
        blocks.append(Block(len(blocks), "visual", latent, start, len(kinds), members))

        # ---- progress block: the frames this latent frame is responsible for --------
        start = len(kinds)
        members = []
        for frame in np.flatnonzero(frame_to_latent == latent):
            prog_index[frame] = len(kinds)
            kinds.append(PROG)
            # A slot one step outside the visual grid, so a progress token can never
            # collide with a patch position no matter how the grid is sized.
            positions.append((latent, grid_h, grid_w))
            block_of.append(len(blocks))
            members.append(f"prog_{int(frame) + 1}")
        blocks.append(Block(len(blocks), "prog", latent, start, len(kinds), members))

    return TokenLayout(
        num_frames=num_frames,
        num_latent_frames=num_latent,
        modality=modality,
        grid_h=grid_h,
        grid_w=grid_w,
        kinds=np.asarray(kinds),
        block_of=np.asarray(block_of, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.int64),
        prog_index=prog_index,
        frame_to_latent=frame_to_latent,
        blocks=blocks,
    )


def build_block_causal_mask(layout: TokenLayout, prog_block_causal: bool = False) -> np.ndarray:
    """`[L, L]` bool mask. **True means blocked**, matching ReWiND / `nn.TransformerEncoder`.

    (`F.scaled_dot_product_attention` uses the opposite convention, so the model
    inverts this once at the attention call. The convention lives here, in one place.)
    """
    block_of = layout.block_of
    # query i may not look at key j when j sits in a later block
    blocked = block_of[None, :] > block_of[:, None]

    if prog_block_causal:
        same_block = block_of[None, :] == block_of[:, None]
        is_prog = layout.kinds == PROG
        both_prog = is_prog[:, None] & is_prog[None, :]
        index = np.arange(layout.seq_len)
        later = index[None, :] > index[:, None]
        # inside a progress block, prog_2 -> prog_3 -> ... stays ordered
        blocked = blocked | (same_block & both_prog & later)

    return blocked


def format_layout(layout: TokenLayout, prog_block_causal: bool = False) -> str:
    """Human-readable dump of the token order -- what Step 2's print command shows."""
    lines = []
    lines.append(
        f"modality={layout.modality}  frames={layout.num_frames} -> latent_frames={layout.num_latent_frames}  "
        f"latent grid={layout.grid_h}x{layout.grid_w} ({layout.tokens_per_visual_frame} tokens/frame/stream)"
    )
    lines.append("")
    counts = {
        "num_rgb_tokens": layout.count(RGB),
        "num_pointmap_tokens": layout.count(POINTMAP),
        "num_prog_tokens": layout.count(PROG),
    }
    total = layout.seq_len
    for name, value in counts.items():
        share = 100.0 * value / total if total else 0.0
        lines.append(f"  {name:<22} {value:>5}   ({share:5.1f}% of the sequence)")
    lines.append(f"  {'sequence_length':<22} {total:>5}")
    lines.append("")
    lines.append("  sequence order (attention is block-causal: block i sees blocks 1..i)")
    for block in layout.blocks:
        inside = "bidirectional" if block.kind == "visual" else ("causal" if prog_block_causal else "bidirectional")
        lines.append(f"    [{block.start:>4}:{block.end:<4}] {block.label}")
        lines.append(f"{'':>18}latent_frame={block.latent_frame + 1}  size={block.size}  inside={inside}")
    lines.append("")
    lines.append("  which input frames each latent frame carries (Wan compresses 4x in time)")
    for latent in range(layout.num_latent_frames):
        frames = [int(f) + 1 for f in np.flatnonzero(layout.frame_to_latent == latent)]
        leak = "leak-free" if len(frames) == 1 else f"prog_{frames[0]} also sees frames {frames[1:]}"
        lines.append(f"    latent {latent + 1} <- input frames {frames}   ({leak})")
    return "\n".join(lines)


def describe_positions(layout: TokenLayout, limit: int = 6) -> str:
    """A few tokens' (t, h, w) RoPE coordinates, to show rgb and pointmap really do match."""
    lines = ["  RoPE positions (t, h, w) -- rgb and its pointmap share a position by design"]
    shown: List[int] = []
    for kind in (RGB, POINTMAP, PROG):
        found = np.flatnonzero(layout.kinds == kind)
        shown.extend(found[: limit // 2].tolist())
        shown.extend(found[-1:].tolist())
    for index in sorted(set(shown)):
        t, h, w = layout.positions[index]
        lines.append(f"    token {index:>5}  {layout.kinds[index]:<9} pos=(t={t}, h={h}, w={w})")
    return "\n".join(lines)


def assert_shared_positions(layout: TokenLayout) -> None:
    """Guard the property the whole design leans on: rgb[i] and pointmap[i] have equal positions."""
    if layout.modality != "rgb_pointmap":
        return
    rgb = layout.positions[layout.kinds == RGB]
    pointmap = layout.positions[layout.kinds == POINTMAP]
    if rgb.shape != pointmap.shape or not np.array_equal(rgb, pointmap):
        raise AssertionError("rgb and pointmap tokens must occupy identical (t, h, w) positions")


def visual_stream_slices(layout: TokenLayout) -> Sequence[Tuple[str, np.ndarray]]:
    """Token indices per visual stream, handy for probing activations."""
    return [(stream, np.flatnonzero(layout.kinds == stream)) for stream in visual_streams(layout.modality)]

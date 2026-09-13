#!/usr/bin/env python3
"""`PreciseTransformer` plus one language token, for the lift-to-N-cm study.

"lift the cube to 5 cm" and "lift the cube to 20 cm" produce *pixel-identical*
frames with different progress labels, so the target height has to enter the input
or the model is being asked to guess. One token carrying the instruction embedding
is the smallest thing that does it.

The token is prepended as block **B0**, ahead of every visual block:

    B0 = [lang]                                    <- the instruction
    B1 = [t1_rgb, t1_pointmap]                     <- latent frame 1
    B2 = [prog_1]
    ...

Attention is already block-causal on `block_of`, so shifting every existing block
up by one and putting the language token in block 0 means every later block sees
it and it sees nothing -- no mask code changes at all. Its RoPE position is
`(0, grid_h + 1, grid_w + 1)`, one step outside both the patch grid (`h < grid_h`)
and the progress slot (`h == grid_h`), so it can never collide with a real token.

`use_lang_token` defaults to **false**, in which case this class constructs, loads
and runs bit-identically to `PreciseTransformer` -- the exp-1 checkpoints stay valid.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from robometer.models.precise.layout import Block, TokenLayout
from robometer.models.precise.precise_transformer import (
    PreciseTransformer,
    PreciseTransformerConfig,
)
from robometer.models.precise.wan_tokenizer import WanLatentTokenizer

LANG = "lang"

# all-MiniLM-L6-v2, which is what the dataset's `lang_vector` column holds.
LANG_VECTOR_DIM = 384


class LangPreciseTransformerConfig(PreciseTransformerConfig):
    model_type = "lang_precise_transformer"

    def __init__(
        self,
        use_lang_token: bool = False,
        lang_vector_dim: int = LANG_VECTOR_DIM,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_lang_token = use_lang_token
        self.lang_vector_dim = lang_vector_dim


def prepend_lang_block(layout: TokenLayout) -> TokenLayout:
    """Shift every token up by one and insert the language token as block 0."""
    blocks = [Block(0, LANG, -1, 0, 1, [LANG])]
    blocks.extend(
        Block(block.index + 1, block.kind, block.latent_frame, block.start + 1, block.end + 1, list(block.members))
        for block in layout.blocks
    )
    lang_position = np.asarray([[0, layout.grid_h + 1, layout.grid_w + 1]], dtype=np.int64)
    return TokenLayout(
        num_frames=layout.num_frames,
        num_latent_frames=layout.num_latent_frames,
        modality=layout.modality,
        grid_h=layout.grid_h,
        grid_w=layout.grid_w,
        kinds=np.concatenate([np.asarray([LANG]), layout.kinds]),
        block_of=np.concatenate([np.zeros(1, dtype=np.int64), layout.block_of + 1]),
        positions=np.concatenate([lang_position, layout.positions]),
        prog_index=layout.prog_index + 1,
        frame_to_latent=layout.frame_to_latent,
        blocks=blocks,
    )


class LangPreciseTransformer(PreciseTransformer):
    """Progress + success over Wan latents, conditioned on one instruction token."""

    config_class = LangPreciseTransformerConfig

    def __init__(self, config, latent_tokenizer: Optional[WanLatentTokenizer] = None):
        super().__init__(config, latent_tokenizer=latent_tokenizer)
        self.use_lang_token = bool(getattr(self.config, "use_lang_token", False))
        if self.use_lang_token:
            self.lang_proj = nn.Linear(self.config.lang_vector_dim, self.config.hidden_dim)
            self.lang_embedding = nn.Parameter(torch.randn(1, 1, self.config.hidden_dim) * 0.02)
        self._lang_layout_cache: Dict[Tuple, TokenLayout] = {}

    def get_layout(self, num_frames: int) -> TokenLayout:
        layout = super().get_layout(num_frames)
        if not self.use_lang_token:
            return layout
        key = (num_frames, self.config.modality, self.grid_h, self.grid_w)
        if key not in self._lang_layout_cache:
            self._lang_layout_cache[key] = prepend_lang_block(layout)
        return self._lang_layout_cache[key]

    def _prefix_tokens(
        self,
        reference_tokens: torch.Tensor,
        lang_vector: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        if not self.use_lang_token:
            return None
        if lang_vector is None:
            raise ValueError(
                "model.precise.use_lang_token=true but the batch has no 'lang_vector'. "
                "The converter must write a real sentence embedding into that column."
            )
        weight = self.lang_proj.weight
        token = self.lang_proj(lang_vector.to(device=weight.device, dtype=weight.dtype))
        token = token[:, None] + self.lang_embedding
        return token.to(reference_tokens.dtype)

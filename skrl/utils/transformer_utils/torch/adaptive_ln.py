import torch
import torch.nn as nn

# https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/models/normalization.py#L130
class AdaptiveLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, do_zero: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = nn.LayerNorm(embedding_dim)
        self.silu = nn.SiLU()
        # If doing normal adaptive layer norm multiply by 2. If doing zero adaptive layer norm use 6.
        self.dim_increase = 2 if not do_zero else 6
        self.scale_shift = nn.Linear(embedding_dim, self.dim_increase * embedding_dim)

    
    def forward(self, x, condition):
        # This assumes condition is already in the embedding space
        out = self.scale_shift(condition)
        # TODO: This dimension might need to be 2
        scale, shift = out.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale) + shift
        return x
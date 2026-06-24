import torch
import torch.nn as nn
import copy

class EncoderNetwork(nn.Module):
    def __init__(self, model_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_params['d_model'], 
                                                   nhead=model_params['nhead'],
                                                   dim_feedforward=model_params['dim_feedforward'],
                                                   batch_first=model_params['batch_first'],
                                                   norm_first=model_params['norm_first'])
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=model_params['num_layers'], )
    
    def forward(self, x):
        return self.encoder(x)

class AdaptiveEncoderNetwork(nn.Module):
    def __init__(self, model_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        encoder_layer = AdaptiveEncoderLayer(d_model=model_params['d_model'], 
                                                   nhead=model_params['nhead'],
                                                   dim_feedforward=model_params['dim_feedforward'],
                                                   batch_first=model_params['batch_first'],
                                                   norm_first=model_params['norm_first'],
                                                   scale=model_params['scale'],)
        self.encoder = AdaptiveEncoder(encoder_layer, num_layers=model_params['num_layers'], )
    
    def forward(self, x):
        # TODO: Need to extract the condition here B x 1 x d_model
        c = torch.rand((x.shape[0], 1, x.shape[2]), device=x.device)
        return self.encoder(x, c)

class AdaptiveEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x, c):
        output = x
        for mod in self.layers:
            output = mod(
                output,
                c
            )

        return output

# https://github.com/facebookresearch/DiT/blob/main/models.py
class AdaptiveEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU, layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None, scale=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, elementwise_affine=False, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, elementwise_affine=False, **factory_kwargs)
        if scale not in [4, 6]:
            raise ValueError(f"Scale must be 4 or 6. Got {scale}.")
        self.scale = scale
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(d_model, self.scale * d_model))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()
    
    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift
    
    def forward(self, x, c):
        # adaLN
        if self.scale == 4:
            shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaln(c).chunk(self.scale, dim=2)
            if self.norm_first:
                x_norm = self.modulate(self.norm1(x), shift_msa, scale_msa)
                x = x + self.dropout1(self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0])
                x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.modulate(self.norm2(x), shift_mlp, scale_mlp))))))
            else:
                x = self.modulate(self.norm1(x + self.dropout1(self.self_attn(x, x, x, need_weights=False)[0])), shift_msa, scale_msa)
                x = self.modulate(self.norm2(x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))), shift_mlp, scale_mlp)
        # adaLN zero
        elif self.scale == 6:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(c).chunk(self.scale, dim=2)
            if self.norm_first:
                x_norm = self.modulate(self.norm1(x), shift_msa, scale_msa)
                x = x + gate_msa * self.dropout1(self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0])
                x = x + gate_mlp * self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.modulate(self.norm2(x), shift_mlp, scale_mlp))))))
            else:
                x = gate_msa * self.modulate(self.norm1(x + self.dropout1(self.self_attn(x, x, x, need_weights=False)[0])), shift_msa, scale_msa)
                x = gate_mlp * self.modulate(self.norm2(x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))), shift_mlp, scale_mlp)
        return x


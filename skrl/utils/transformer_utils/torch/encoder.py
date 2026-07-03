import torch
from torch import Tensor
import torch.nn as nn
import copy
from typing import Optional

# Transformer Encoder with GRU Gate option from (https://opendilab.github.io/DI-engine/_modules/ding/torch_utils/network/gtrxl.html#GTrXL)
# Transformer Encoder with Adaptive Layer Norm option
class EncoderNetwork(nn.Module):
    def __init__(self, model_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if model_params.get('layer_norm_type', None) is None and model_params.get('gate', None) is None:
            encoder_layer = nn.TransformerEncoderLayer(d_model=model_params['d_model'], 
                                                   nhead=model_params['nhead'],
                                                   dim_feedforward=model_params.get('dim_feedforward', 2048),
                                                   batch_first=model_params.get('batch_first', True),
                                                   norm_first=model_params.get('norm_first', True),
                                                   dropout=model_params.get('dropout', 0.0))
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=model_params['num_layers'], )
        
        else:
            encoder_layer = EncoderLayer(d_model=model_params['d_model'], 
                                         nhead=model_params['nhead'],
                                         bias_g=model_params.get('bias_g', None),
                                         layer_norm_type=model_params.get('layer_norm_type', None),
                                         scale=model_params.get('scale', None),
                                         gate=model_params.get('gate', None),
                                         dim_feedforward=model_params.get('dim_feedforward', 2048),
                                         batch_first=model_params.get('batch_first', True),
                                         norm_first=model_params.get('norm_first', True),
                                         dropout=model_params.get('dropout', 0.0))
            self.encoder = Encoder(encoder_layer, num_layers=model_params['num_layers'], )

    def forward(self, x):
        return self.encoder(x)

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x):
        output = x
        for mod in self.layers:
            output = mod(
                output,
                attn_mask=None,
                src_key_padding_mask=None,
                is_causal=False,
            )
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, bias_g=2.0, layer_norm_type=None, scale=None, gate=None, dim_feedforward=2048, dropout=0.0, activation=nn.ReLU, layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None, *args, **kwargs):
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
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.layer_norm_type = layer_norm_type
        self.do_adaptive = True if self.layer_norm_type == 'adaptive' else False
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, elementwise_affine=not self.do_adaptive, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, elementwise_affine=not self.do_adaptive, **factory_kwargs)
        if self.do_adaptive:
            if scale not in [4, 6]:
                raise ValueError(f"Scale must be 4 or 6. Got {scale}.")
            self.scale = scale
            self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(d_model, self.scale * d_model))

        self.use_residual = True
        if gate == 'GRU':
            self.use_residual = False
            self.gate1 = GRUGate(d_model, bias_g)
            self.gate2 = GRUGate(d_model, bias_g)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()
    
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False,):
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _get_ada_params(self):
        if not self.do_adaptive:
            return None
        
        c = torch.rand((x.shape[0], 1, x.shape[2]))
        params = self.adaln(c).chunk(self.scale, dim=2)

        if self.scale == 4:
            return {
                'msa': {'shift': params[0], 'scale': params[1]},
                'mlp': {'shift': params[2], 'scale': params[3]}
            }
        elif self.scale == 6:
            return {
                'msa': {'shift': params[0], 'scale': params[1], 'gate': params[2]},
                'mlp': {'shift': params[3], 'scale': params[4], 'gate': params[5]}
            }
    
    def _norm(self, x, layer_norm, ada_params):
        x_norm = layer_norm(x)
        if self.do_adaptive and ada_params is not None:
            return x_norm * (1 + ada_params['scale']) + ada_params['shift']
        return x_norm
    
    def _apply_block(self, x, block_output, gate_fn, ada_params):
        gate_scale = ada_params.get('gate', None) if (self.do_adaptive and ada_params) else None
        block_output = block_output if gate_scale is None else gate_scale * block_output
        
        if gate_fn:
            return gate_fn(x, block_output)
        return x + block_output

    def adaptive_norm(self, x, shift, scale):
        return x * (1 + scale) + shift

    def get_norm(self, x, norm, shift=None, scale=None):
        if self.layer_norm_type is None:
            return norm(x)
        elif self.layer_norm_type == 'adaptive':
            return norm(x) * (1 + scale) + shift

    def forward(self, x, attn_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, is_causal: bool = False):
        ada_p = self._get_ada_params()
        p_msa, p_mlp = (ada_p['msa'], ada_p['mlp']) if ada_p else (None, None)

        if self.norm_first:
            # Pre-LN
            sa_out = self._sa_block(self._norm(x, self.norm1, p_msa), attn_mask, src_key_padding_mask, is_causal)
            x = self._apply_block(x, sa_out, None if self.use_residual else self.gate1, p_msa)
            
            ff_out = self._ff_block(self._norm(x, self.norm2, p_mlp))
            x = self._apply_block(x, ff_out, None if self.use_residual else self.gate2, p_mlp)
        else:
            # Post-LN
            sa_out = self._sa_block(x, attn_mask, src_key_padding_mask, is_causal)
            x = self._norm(self._apply_block(x, sa_out, None if self.use_residual else self.gate1, p_msa), self.norm1, p_msa)
            
            ff_out = self._ff_block(x)
            x = self._norm(self._apply_block(x, ff_out, None if self.use_residual else self.gate2, p_mlp), self.norm2, p_mlp)

        # if self.do_adaptive:
        #         # TODO: extract c one I know how to pass it in
        #         c = torch.rand((x.shape[0], 1, x.shape[2]))
        #         if self.scale == 4:
        #             shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaln(c).chunk(self.scale, dim=2)
        #             gate_msa, gate_mlp = [1., 1.]
        #         elif self.scale == 6:
        #             shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(c).chunk(self.scale, dim=2)
        # else:
        #     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [None, None, 1., None, None, 1.]

        # if self.norm_first:
        #     if self.use_residual:
        #         x = x + gate_msa * self._sa_block(
        #             self.get_norm(x, self.norm1, shift_msa, scale_msa), attn_mask, key_padding_mask, is_causal=is_causal
        #         )
        #         x = x + gate_mlp * self._ff_block(self.get_norm(x, self.norm2, shift_mlp, scale_mlp))
        #     else:
        #         x = self.gate1(x, gate_msa * self._sa_block(
        #                 self.get_norm(x, self.norm1, shift_msa, scale_msa), attn_mask, key_padding_mask, is_causal=is_causal
        #             )
        #         )
        #         x = self.gate2(x, gate_mlp * self._ff_block(self.get_norm(x, self.norm2, shift_mlp, scale_mlp)))
            
        # else:
        #     if self.use_residual:
        #         x = self.get_norm(
        #             x
        #             + self._ff_block(x, attn_mask, key_padding_mask, is_causal=is_causal),
        #             self.norm1,
        #             shift_msa,
        #             scale_msa
        #         )
        #         x = self.get_norm(x + self.ff_block(x), self.norm2, shift_mlp, scale_mlp)
        #     else:
        #         x = self.get_norm(self.gate1(x, self.ff_block(x, attn_mask, key_padding_mask, is_causal=is_causal)), self.norm1, shift_msa, scale_msa)
        #         x = self.get_norm(self.gate2(x, self.ff_block(x)), self.norm2, shift_mlp, scale_mlp)
            

        return x

class GRUGate(nn.Module):
    def __init__(self, d_model, bias_g=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Wr = torch.nn.Linear(d_model, d_model, bias=False)
        self.Ur = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wz = torch.nn.Linear(d_model, d_model, bias=False)
        self.Uz = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wg = torch.nn.Linear(d_model, d_model, bias=False)
        self.Ug = torch.nn.Linear(d_model, d_model, bias=False)
        self.bg = nn.Parameter(torch.full([d_model], bias_g))  # bias
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
    
    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))  # element wise multiplication
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g


# Transformer Encoder with Adaptive Layer Norm
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


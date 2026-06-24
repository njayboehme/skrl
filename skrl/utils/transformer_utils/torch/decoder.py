import torch
import torch.nn as nn

class DecoderNetwork(nn.Module):
    def __init__(self, model_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_params['d_model'], 
                                                   nhead=model_params['nhead'],
                                                   dim_feedforward=model_params['dim_feedforward'],
                                                   batch_first=model_params['batch_first'],
                                                   norm_first=model_params['norm_first'])
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=model_params['num_layers'], )
    
    def forward(self, x):
        return self.decoder(x, x)
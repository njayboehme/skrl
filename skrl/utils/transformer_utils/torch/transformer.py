import torch
import torch.nn as nn
from typing import Any, Literal, Union
from skrl.utils.transformer_utils.torch.decoder import DecoderNetwork
from skrl.utils.transformer_utils.torch.encoder import EncoderNetwork, AdaptiveEncoderNetwork
from skrl.utils.transformer_utils.torch.embedder import Embedder
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor, AutoImageProcessor


class TransformerNetwork(nn.Module):
    def __init__(self, input_size, model_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Embeddings for proprioception
        self.proprioception_embeds = Embedder(input_size, model_params)

        # Pooling method
        self.pool_method: Literal['mean', 'max', 'last', 'CLS'] = model_params['pooling_method']
        if self.pool_method == 'CLS':
            self.cls_token = nn.Parameter(
                torch.zeros(size=(1, model_params['d_model'],), dtype=torch.float32), requires_grad=True
            )

        # Project pooled token into action space
        # self.output_layer = nn.Linear(model_params['d_model'], output_size)

        # Frozen Network
        self.preprocessor = None
        if model_params['model_path'] is not None:
            self.config = AutoConfig.from_pretrained(model_params['model_path'])
            if model_params['use_text'] and model_params['use_images']:
                self.preprocessor = AutoProcessor.from_pretrained(model_params['model_path'])
            elif model_params['use_text']:
                self.preprocessor = AutoTokenizer.from_pretrained(model_params['model_path'])
            elif model_params['use_images']:
                self.preprocessor = AutoImageProcessor.from_pretrained(model_params['model_path'])
            self.pretrained_network = AutoModel.from_config(self.config)
        
        # Trainable Network
        use_adaptive_encoder = model_params['use_adaptive']
        zero_out = model_params['zero_out']
        # Encoder
        if model_params['use_encoder']:
            if not use_adaptive_encoder:
                self.enc = EncoderNetwork(model_params)
            else:
                self.enc = AdaptiveEncoderNetwork(model_params)
                if zero_out:
                    for l in self.enc.encoder.layers:
                        nn.init.constant_(l.adaln[-1].weight, 0)
                        nn.init.constant_(l.adaln[-1].bias, 0)
                    # nn.init.constant_(self.output_layer.linear.weight, 0)
                    # nn.init.constant_(self.output_layer.linear.bias, 0)
        # Decoder
        if model_params['use_decoder']:
            self.dec = DecoderNetwork(model_params)

    
    def pool(self, x):
        # x is B x S x d_model
        # Returns x as B x d_model
        if self.pool_method == 'mean':
            return torch.mean(x, dim=1)
        elif self.pool_method == 'max':
            return torch.max(x, dim=1)[0]
        elif self.pool_method in ['last', 'CLS']:
            # This assumes the CLS token is the last token
            return x[:, -1, :]
        return x

    def split_input(self, x):
        # TODO: split the input into proprioception, text, images, and anything else
        return [x, None, None, None]
    
    def get_inputs(self, x):
        # x is the state split into images, proprioception, and anything else
        # return the tokens from each part of x
        prop, text, img, extra = x
        prop_embeds = self.proprioception_embeds(prop)
        if self.pool_method == 'CLS':
            prop_embeds = torch.cat((prop_embeds, self.cls_token.unsqueeze(0).expand(prop_embeds.shape[0], -1, -1)), dim=1)
        pretrained_inputs = None
        if self.preprocessor is not None:
            # Process text and img
            if text is not None and img is not None:
                pretrained_inputs = self.preprocessor(text=text, images=img, return_tensors="pt")
            # Process text
            elif text is not None:
                pretrained_inputs = self.preprocessor(text=text, return_tensors="pt")
            # Process image
            elif img is not None:
                pretrained_inputs = self.preprocessor(images=img, return_tensors="pt")
        return prop_embeds, pretrained_inputs
    
    def forward(self, inputs):
        inps = self.split_input(inputs)
        x, pretrained_inputs = self.get_inputs(inps)
        if pretrained_inputs is not None:
            with torch.no_grad():
                outputs = self.pretrained_network(**pretrained_inputs)
                # Concat the output embeddings to the proprioceptive embeddings
                x = torch.cat([outputs.last_hidden_state, x], dim=1)
        # Go through the Encoder
        if hasattr(self, 'enc'):
            x = self.enc(x)
        # Go through the Decoder
        if hasattr(self, 'dec'):
            x = self.dec(x)
        
        x = self.pool(x)
        # return self.output_layer(x)
        return x

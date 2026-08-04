import torch
import torch.nn as nn
from typing import Any, Literal, Union
import math
from skrl.utils.transformer_utils.torch.decoder import DecoderNetwork
from skrl.utils.transformer_utils.torch.encoder import EncoderNetwork
from skrl.utils.transformer_utils.torch.embedder import Embedder
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor, AutoImageProcessor


class TransformerNetwork(nn.Module):
    def __init__(self, input_size, model_params, shared=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If the transformer is a shared backbone
        self.shared = shared
        # Only used if the transformer is a shared backbone
        self.state_token = None

        # Embeddings for proprioception
        self.proprioception_embeds = Embedder(input_size, model_params)

        # Pooling method
        self.pool_method: Literal['mean', 'max', 'last', 'CLS'] = model_params.get('pooling_method', None)
        if self.pool_method == 'CLS':
            self.cls_token = nn.Parameter(
                torch.zeros(size=(1, model_params['d_model'],), dtype=torch.float32), requires_grad=True
            )
        
        # Number of action steps to predict
        self.action_chunk_size = model_params.get('action_chunk_size', 1)
        self.action_pred_type = model_params.get('action_pred_type', None)
        if self.action_pred_type == 'action_tokens':
            self.action_tokens = nn.Parameter(
                nn.init.normal_(torch.empty(self.action_chunk_size, model_params['d_model'], dtype=torch.float32), 
                                std=1/math.sqrt(model_params['d_model'])), 
                requires_grad=True
            )
            if self.shared:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.mask = torch.full(size=(self.action_chunk_size + 1, self.action_chunk_size + 1), fill_value=False).to(device=device)
                # Make sure the first token (value token) only sees itself
                self.mask[0, 1:] = True

        # Frozen Network
        self.preprocessor = None
        if model_params.get('model_path', None) is not None:
            self.config = AutoConfig.from_pretrained(model_params['model_path'])
            if model_params['use_text'] and model_params['use_images']:
                self.preprocessor = AutoProcessor.from_pretrained(model_params['model_path'])
            elif model_params['use_text']:
                self.preprocessor = AutoTokenizer.from_pretrained(model_params['model_path'])
            elif model_params['use_images']:
                self.preprocessor = AutoImageProcessor.from_pretrained(model_params['model_path'])
            self.pretrained_network = AutoModel.from_config(self.config)
        
        # Trainable Network
        zero_out = model_params.get('zero_out', False)
        # Encoder
        if model_params.get('use_encoder', False):
            self.enc = EncoderNetwork(model_params)
            if zero_out:
                for l in self.enc.encoder.layers:
                    nn.init.constant_(l.adaln[-1].weight, 0)
                    nn.init.constant_(l.adaln[-1].bias, 0)
                # nn.init.constant_(self.output_layer.linear.weight, 0)
                # nn.init.constant_(self.output_layer.linear.bias, 0)
        # Decoder
        if model_params.get('use_decoder', False):
            self.dec = DecoderNetwork(model_params)


    def get_enc_mask(self):
        # Only for an encoder
        if hasattr(self, 'mask'):
            return self.mask
        else:
            return None

    def get_state_token(self):
        return self.state_token

    
    def pool(self, x):
        # x is B x S x d_model
        # Returns x as B x num_pred_acts x d_model
        if self.action_pred_type == 'action_tokens':
            p = x[:, -self.action_chunk_size:, :]
        # Everything below returns x as B x d_model
        elif self.pool_method == 'mean':
            p = torch.mean(x, dim=1)
        elif self.pool_method == 'max':
            p = torch.max(x, dim=1)[0]
        elif self.pool_method in ['last', 'CLS']:
            # This assumes the CLS token is the last token
            p = x[:, -1, :]
        else:
            p = x[:, 0]
        if self.shared:
            if self.action_pred_type == 'action_tokens':
                # This assumes the state token is the first token
                self.state_token = x[:, 0]
            else:
                self.state_token = p
        return p

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
        if self.action_pred_type == 'action_tokens':
            prop_embeds = torch.cat((prop_embeds, self.action_tokens.unsqueeze(0).expand(prop_embeds.shape[0], -1, -1)), dim=1)
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
            mask = self.get_enc_mask()
            x = self.enc(x, mask)
        # Go through the Decoder
        if hasattr(self, 'dec'):
            x = self.dec(x)
        x = self.pool(x)
        return x

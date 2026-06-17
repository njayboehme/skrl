import torch
import torch.nn as nn
from skrl.utils.transformer_utils.torch.decoder import DecoderNetwork
from skrl.utils.transformer_utils.torch.encoder import EncoderNetwork
from skrl.utils.transformer_utils.torch.tokenizer import Tokenizer

# TODO: Define Tokenizer, Encoder, Decoder and test it can work as expected
# TODO: Add in different pooling methods, layer norms, tokenization methods
class TransformerNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Embeddings
        self.embeds = Tokenizer()
        # Encoder
        self.enc = EncoderNetwork()
        # Decoder
        self.dec = DecoderNetwork()
    
    def forward(self, x):
        inp = self.embeds(x)
        out = self.enc(inp)
        out = self.dec(out)
        return out
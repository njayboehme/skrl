import torch
import torch.nn as nn
from typing import Any, Literal, Union

class Embedder(nn.Module):
    def __init__(self, inp_size, model_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_params = model_params
        # tokenization_method: 'all' puts the entire state into a single token, 
                            # 'single' puts each input into a token, 
                            # 'bin' uses predefined bins to create tokens for each input, 
                            # 'groups' groups parts of the input together into a single token
        self.tokenization_method: Literal['all', 'single', 'bin', 'groups',] = model_params['tokenization_method']

        if self.tokenization_method == 'all':
            # Put the entire state into a single token
            self.embeds = nn.Linear(inp_size, model_params['d_model'])
        elif self.tokenization_method == 'single':
            # Each state has its own embedding
            self.embeds = nn.ModuleList([nn.Linear(1, model_params['d_model']) for _ in range(inp_size)])
        elif self.tokenization_method == 'bin':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Bin each state into num_bins
            self.embeds = nn.Embedding(model_params['num_bins'] * (inp_size), model_params['d_model'])
            self.boundaries = [torch.linspace(model_params['bin_start'][i], model_params['bin_end'][i], model_params['num_bins'], device=device) for i in range(inp_size)]
        elif self.tokenization_method == 'groups':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Group states together into a single embedding based on groups
            # groups should be of the form [0, 1, 2, 0, 0, 2] and begin at zero for indexes
            unique_vals, inds = torch.unique(torch.tensor(model_params['groups'], device=device), return_inverse=True)
            self.embeds = nn.ModuleList()
            # Holds the group to the indices to be in the same group
            self.groups = {}
            for i, val in enumerate(unique_vals):
                matching_inds = (inds == i).nonzero(as_tuple=True)[0]
                self.groups[val.item()] = matching_inds
                self.embeds.append(nn.Linear(len(matching_inds), model_params['d_model']))


    
    def forward(self, x: torch.Tensor):
        # x is B x inp_dim
        # Return B x S x d_model
        if self.tokenization_method == 'all':
            # Treat the entire state as a token
            return self.embeds(x).unsqueeze(1)
        elif self.tokenization_method == 'single':
            # Each state has its own linear embedding
            out = torch.zeros(x.shape[0], len(self.embeds), self.model_params['d_model'], device=x.device)
            for i, l in enumerate(self.embeds):
                out[:, i] = l(x[:, i:i+1])
            return out
        elif self.tokenization_method == 'bin':
            # Bin each state
            binned_states = torch.zeros((x.shape), dtype=torch.int, device=x.device)
            # TODO: Need to shift each part of the binned states by the number of bins
            for i in range(x.shape[1]):
                binned_states[:, i] = torch.bucketize(x[:, i].contiguous(), boundaries=self.boundaries[i])
            return self.embeds(binned_states)
        elif self.tokenization_method == 'groups':
            # Each group has a linear embedding
            out = torch.zeros(x.shape[0], len(self.embeds), self.model_params['d_model'], device=x.device)
            for i, (k, l) in enumerate(zip(self.groups.keys(), self.embeds)):
                inds = self.groups[k]
                out[:, i] = l(x[:, inds])
            return out
        
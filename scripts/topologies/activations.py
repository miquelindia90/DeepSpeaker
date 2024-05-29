from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


ACTIVATIONS = [nn.PReLU(), nn.ReLU(), nn.RReLU(), nn.SELU(), nn.CELU(), nn.GELU(), nn.LeakyReLU(), nn.SiLU(), nn.Mish()]

def _new_parameter(*size):
    out = torch.nn.Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out


class AttentionActivation(nn.Module):
    def __init__(self, embedding_size):
        super(AttentionActivation, self).__init__()
        self.embedding_size = embedding_size
        self.pooling_attention = _new_parameter(self.embedding_size, 1)
        self.activations = nn.ModuleList(ACTIVATIONS)
        self.activations_embeddings = _new_parameter(self.embedding_size, len(ACTIVATIONS))

    def _get_pooling_embedding(self, input_tensor):
        attention_score = torch.matmul(input_tensor, self.pooling_attention).squeeze()
        attention_score = F.softmax(attention_score, dim=-1).view(
            input_tensor.size(0), input_tensor.size(1), input_tensor.size(2), 1
        )
        ct = torch.sum(input_tensor * attention_score, dim=2)

        return ct
    
    def _calculate_activation(self, input_tensor, pooled_vector):
        
        attention_score = torch.matmul(pooled_vector, self.activations_embeddings).squeeze()
        attention_score = F.softmax(attention_score, dim=-1)
        
        output_tensor = torch.zeros(input_tensor.size()).to(input_tensor.device)
        for activation_index, activation in enumerate(self.activations):
            activated_tensor = activation(input_tensor)
            weighted_activation = activated_tensor * attention_score[:, :, activation_index].unsqueeze(-1).unsqueeze(-1)
            output_tensor += weighted_activation

        return output_tensor

    def forward(self, ht):

        pooled_vector = self._get_pooling_embedding(ht)
        output_tensor = self._calculate_activation(ht, pooled_vector)
        return output_tensor
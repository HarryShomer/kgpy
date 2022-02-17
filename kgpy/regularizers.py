import torch 
import torch.nn as nn


class N3(nn.Module):
    """
    Nuclear 3 norm introduced here - https://arxiv.org/abs/1806.07297
    """
    def __init__(self):
        super(N3).__init__()
    
    def forward(triples):
        """
        Parameters:
        -----------
            triples: torch.Tensor
        """
        pass
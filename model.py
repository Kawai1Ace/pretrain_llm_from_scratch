import math
import struct
import inspect

import torch
import torch.nn.functional as F

import numpy as np

from torch import nn
from dataclasses import dataclass
from typing import Any, Optional, Tuple

@dataclass
class ModelArgs:
    dim: int=4096
    n_layers: int=32
    n_heads: int=32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 #define by tokenizer
    multiple_of: int = 256 # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float=1e-5
    max_seq_len: int = 2048
    dropout: float=0.0

class RMSNorm(torch.nn.Module):
    def __init__(self, dim:int, eps:float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# 频率缩放因子？
def precompute_freqs_cis(dim: int, end:int, theta: float=10000.0):
    freqs = 1.0/(theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin
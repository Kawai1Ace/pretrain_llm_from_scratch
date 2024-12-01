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



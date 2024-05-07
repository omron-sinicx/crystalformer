
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter, Module, Linear, Dropout, LayerNorm, ModuleList
from typing import List, Optional, Tuple, Union, Callable
from . import cuda_funcs

from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
import math
from . import global_config as config

def max_pool(x, batch, sizes):
    x = torch.split_with_sizes(x, sizes.tolist(), 0)
    x = torch.stack([torch.max(x,dim=0)[0] for x in x])
    return x

def avr_pool(x, batch, sizes):
    if config.REPRODUCIBLITY_STATE>=1 and cuda_funcs.CUPY_AVAILABLE:
        x = cuda_funcs.IrregularMeanCUDA.apply(x, batch, sizes)
    else:
        x = torch.split_with_sizes(x, sizes.tolist(), 0)
        x = torch.stack([torch.mean(x,dim=0) for x in x])
    return x


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


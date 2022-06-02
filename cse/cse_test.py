from re import A
from functorch import make_fx
import torch
from torch._decomp import decomposition_table, get_decompositions
import numpy
import torch
import torch.fx as fx
from functorch import make_fx
from functorch.compile import aot_function, print_compile
from torch.fx import symbolic_trace
import random 

from cse import modify


def f(x):
    vals = [x]
    ops = [torch.clone, torch.cos, torch.tanh, torch.nn.functional.gelu]
    for _ in range(100):
      new_val = random.choice(ops)(random.choice(vals))
      vals.append(new_val)
    return vals[-1]


f = fx.symbolic_trace(f)
f.graph.eliminate_dead_code()
f.recompile()
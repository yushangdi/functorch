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

from cse import modify



# class MyInt(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.x = torch.rand(3, 4)

#     def get(self):
#         return self.x
    
#     def forward(self, a):
#         return self.x + a

# module = MyInt()

# symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)

# # High-level intermediate representation (IR) - Graph representation
# fx_g = symbolic_traced.graph
# print(fx_g)
# print(symbolic_traced.code)

# new_graph = modify(symbolic_traced)
# print(new_graph)


# exit(0)



# class MyModule(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.param = torch.nn.Parameter(torch.rand(3, 4))
#         self.linear = torch.nn.Linear(4, 5)

#     def forward(self, x):
#         a = x.clamp()
#         return self.linear(a+x + self.param).clamp(min=0.0, max=1.0)


# module = MyModule()


# # Symbolic tracing frontend - captures the semantics of the module
# symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)

# # High-level intermediate representation (IR) - Graph representation
# fx_g = symbolic_traced.graph
# print(fx_g)
# print(symbolic_traced.code)

# new_graph = modify(symbolic_traced)
# print(new_graph)


# exit(0)

# def f(x):
#     a = x.cos()
#     b = x.cos()
#     return a+b

# def f(x):
#     a = x.cos()
#     b = x.cos()
#     c = a+b
#     d = a+b
#     return c+d

# def f(x):
#     g_cpu = torch.Generator()
#     g_cpu.manual_seed(2147483647)
#     a = torch.randn(4, generator = g_cpu)
#     b = torch.randn(4, generator = g_cpu)
#     print("a,b", a,b)
#     return a+b

# guard againt rand
def f(x):
    a = torch.ones_like(x)
    b = torch.ones_like(x)
    print(a,b)
    print()
    return a + b

    
t = torch.randn(2,2)
fx_g = make_fx(f)(t)
new_graph = modify(fx_g.graph)
new_g = fx.GraphModule(fx_g,new_graph)

print(fx_g.graph)
print(new_graph)

# print("true g:", fx_g)
# print("our g:", new_g)

print("true result:", fx_g(t))
print("our result:",new_g(t))



exit(0)

def f(x):
    return torch.mm(x, x).cos().cos()
t = torch.randn(3, 3, requires_grad=True)

fx_g = make_fx(f)(t)

back_g = None
def no_print(fx_g, _):
    return fx_g

def no_print_back(fx_g, _):
    global back_g
    back_g = fx_g
    return fx_g

aot_function(f, no_print, no_print_back)(t) 

print(back_g)
print(back_g.graph)
new_graph = modify(back_g)
print(new_graph)
print(fx_g(t))
print(fx.GraphModule({"x_1":torch.Tensor},new_graph)(t))


# def f(x):
#     vals = [x]
#     for _ in range(5):
#         vals.append(vals[-1].cos())
#     return vals[-1]

# print(make_fx(f)(torch.randn(5)).code)
# def f(x):
#     return torch.mm(x, x).cos().cos()

# aot_function(f, print_compile)(torch.randn(3, 3, requires_grad=True))
# exit(0)

# def f(x):
#     return x.cos().cos()

# fx_g = make_fx(f)(torch.randn(5))

# print(fx_g.graph)
# print(fx_g.code)
# for n in fx_g.graph.nodes:
#     if n.op == 'call_function':
#         if n.target == torch.ops.aten.cos:
#             n.target = torch.ops.aten.sin

# fx_g.recompile()
# print(fx_g.code)

# new_graph = fx.Graph()
# env = {}
# hash_env = {}
# for n in fx_g.graph.nodes:
#     if n.op != 'call_function':
#         new_node = new_graph.node_copy(n, lambda x: env[x])
#         env[n] = new_node
#     elif n.op == 'call_function':
#         hash_val = (n.target, tuple([env[n.args[0]]]))
#         if hash_val in hash_env:
#             env[n] = hash_env[hash_val]
#             continue

#         new_node = new_graph.call_function(torch.ops.aten.sin, tuple([env[n.args[0]]]))
#         hash_env[hash_val] = new_node
#         env[n] = new_node

# print(new_graph)

################ norm decomposition ################ ################ ################ ################ 

# def f(x, p):
#     return decomposition_table[torch.ops.aten.norm.Scalar](x, p)

# p=1
# # t = numpy.array([-4.5703, -1.1250, -1.4844, -4.4297,  6.2812])
# # a = torch.from_numpy(t)
# a = torch.tensor([-4.5703+1j, -1.1250, -1.4844, -4.4297,  6.2812])
# # a = a.to(torch.float16)

# print("a type:", a.dtype)
# true_answer = torch.ops.aten.norm(a, p)
# our_answer = f(a, p)
# # print(make_fx(f, decomposition_table=get_decompositions([torch.ops.aten.norm]))(a, p))
# print(true_answer)
# print(our_answer)

# print(true_answer.dtype)
# print(our_answer.dtype)



# from functorch import make_fx
# import torch
# from torch._decomp import decomposition_table, get_decompositions
# import numpy

# def f(x, p):
#     return decomposition_table[torch.ops.aten.norm.Scalar](x, p)

# p=2
# t = numpy.array([-4.5703, -1.1250, -1.4844, -4.4297,  6.2812])
# a = torch.from_numpy(t)
# a = a.to(torch.float16)
# print("a type:", a.dtype)
# true_answer = torch.ops.aten.norm(a, p)
# our_answer = f(a, p)
# print(make_fx(f, decomposition_table=get_decompositions([torch.ops.aten.norm]))(a, p))
# print(true_answer)
# print(our_answer)

# print(a.abs().type(torch.float64).pow(p).sum().pow(1/p))
# print(torch.linalg.norm(a, 2))
# S = torch.linalg.svdvals(a)
# print(max(S))

# print(torch.ops.aten.norm(a, 2, [0,1], False, int))

################ vmap ################ ################ ################ ################ 

# import torch
# from functorch import vmap
# import numpy as np

# def f(a, b, c):
#  return torch.addr(a, b, c)

# inps = [torch.randn(3, 4, 5), torch.randn(3, 4), torch.randn(3, 5)]

# for j in range(3):
#     result = torch.all(f(*[i[j] for i in inps]) == vmap(f)(*inps)[j]).numpy()
#     if(not result):
#         print("fail")
#         exit(1)
# print("pass")
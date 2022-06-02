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
import unittest

# check if the CSE modified graph of f has delta less nodes, and do not reduce the number of nodes further on a second pass.
# delta is an integer >= -1. If delta = -1, only check if the new graph
#   has less or equal number of nodes
#  
def check(f, t, delta):
    fx_g = make_fx(f)(t)
    new_graph = modify(fx_g.graph)
    new_g = fx.GraphModule({},new_graph)

    # the number of nodes decrease/ or stay the same
    old_num_nodes = len(fx_g.graph.nodes)
    new_num_nodes = len(new_graph.nodes)
    if delta == -1:
        assert  old_num_nodes >= new_num_nodes, f"number of nodes increased {old_num_nodes}, {new_num_nodes}"
    else:
        assert  old_num_nodes == new_num_nodes+delta, f"number of nodes not the same {old_num_nodes - delta}, {new_num_nodes}"
    
    # a second pass should not reduce more nodes
    pass_2_graph = modify(new_graph)
    pass_2_num_nodes = len(pass_2_graph.nodes)
    assert pass_2_num_nodes == new_num_nodes, f"second pass graph has less node {pass_2_num_nodes}, {new_num_nodes}"

    # check correctness
    true_result = fx_g(t)
    our_result = new_g(t)
    if true_result is None: # both return None
        assert our_result is None, f"true result is None, CSE result is {our_result}"
    else: # results returned are the same
        assert torch.all( true_result == our_result ), f"results are different {true_result}, {our_result}" #check results are the same


class NoChangeTestCase(unittest.TestCase):

    def test_nochange(self):
        def f(x):
            a = x+1
            b = x+a
            a = x
            d = x+a
            return b + d
        t = torch.randn(2,2)
        check(f,t, 0)

class ReduceTestCase(unittest.TestCase):

    def test_immutable_list_type(self):
        def f(x):
            a = x.sum(dim = 1)
            b = x.sum(dim = 1)
            c = x.sum()
            d = x.sum()
            return a+b+c+d
        t = torch.randn(2,2)
        check(f,t, 2)

    def test_simple(self):
        def f(x):
            a = x.cos()
            b = x.cos()
            c = a+a
            d = b+b
            return c+d
        t = torch.randn(2,2)
        check(f,t, 2)

    def test_simple_2(self):
        def f(x):
            a = x.cos().sin()
            b = x.cos().sin()
            c = a+a
            d = b+b
            return c+d
        t = torch.randn(1)
        check(f,t, 3)

    def test_simple_multiple_same_ops(self):
        def f(x):
            a = x.sum()
            b = x.sum()
            c = x.sum()
            d = x.sum()
            return a+b+c+d
        t = torch.randn(2,2)
        check(f,t, 3)

    # TODO: change to nested immutable list
    def test_nested_immutable_list_type(self):
        def f(x):
            a = x.sum(dim = 1)
            b = x.sum(dim = 1)
            c = x.sum()
            d = x.sum()
            return a+b+c+d
        t = torch.randn(2,2)
        check(f,t, 2)

    def test_empty(self):
        def f(x):
            pass
        t = torch.randn(2,2)
        check(f,t, 0)


    

if __name__ == '__main__':
    unittest.main()
import torch
import torch.fx as fx
from functorch import make_fx
from functorch.compile import aot_function, print_compile
from torch.fx.immutable_collections import immutable_list, immutable_dict
from torch.fx.node import Node

import hashlib
import json

def check_args(new_args, old_args, env):
    if (len(new_args)!=len(old_args)):
        return False
    for i in range(len(new_args)):
        if (new_args[i] != old_args[i]):
            if isinstance(new_args[i], list) or isinstance(new_args[i], tuple):
                if not check_args(new_args[i], old_args[i], env):
                    return False
            elif not isinstance(new_args[i], Node):
                return False
            elif (not old_args[i] in env):
                return False
            elif new_args[i] != env[old_args[i]]:
                return False
    return True

def check_kwargs(new_args, old_args, env):
    if (len(new_args)!=len(old_args)):
        return False
    for k,v in new_args.items():
        if not k in old_args: 
            return False
        old_v = old_args[k]
        if type(v) != type(old_v):
            return False
        if (v != old_v):
            if isinstance(v, list) or isinstance(v, tuple):
                if not check_args(v, old_v, env):
                    return False
            elif not isinstance(v, Node):
                return False
            elif (not old_v in env):
                return False
            elif v != env[old_v]:
                return False
    return True

# check if two nodes new_node and old_node are the same
# two nodes are the same if
# 1) they have the same target
# 2) their args and kwargs are the same. For node the elements of args and kwargs, they are the same
#    if the old element map to the new element in env. 
# essentially the following sementic with a more sophisticated implementtation of ==
#     node.target == n.target and node.args == n.args and node.kwargs == n.kwargs
# Note for args and kwargs with nested list or nested dict, we need to check each member
# recursively
def check_same(new_node: torch.fx.node.Node, old_node: torch.fx.node.Node, env: dict):
    if (new_node.target != old_node.target):
        return False
    if not check_args(new_node.args, old_node.args, env):
        return False
    if not check_kwargs(new_node.kwargs, old_node.kwargs, env):
        return False
    return True


# return a new graph with CSE applied to the input graph
# env stores a mapping from node in the old graph to node in the new graph
# The placehold, output, and get_attr nodes are copied to the new grpah without change
# The call nodes (call_function, call_module, call_method) are hashed to check if they
# have an equivalent node in the graph. If so, this node will not be copied, and a mapping
# to the duplicated node is stored in env
def modify(fx_g: torch.fx.graph.Graph):
    new_graph = fx.Graph()
    env = {} # map from node in the old graph to node in the new graph
    hash_env = {} # map from the computatio result to a node in the new graph
    for n in fx_g.nodes:
        if n.op == 'placeholder' or n.op == 'output' or n.op == 'get_attr': # != "call_function"
            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
        else: #n.op == 'call_function', we should never see n.op == 'call_module' or n.op == 'call_method'
            # print("======")
            # print(n.target)
            # print(n.args)
            # print(n.kwargs)

            # convert to mutable types for substitution
            args = list(n.args) 
            kwargs = dict(n.kwargs)

            # substitute members of a list to its mapping in env if exists
            # the subsitution is recursive for nested lists.
            # change unhashable types such as list and dictionary to tuples.
            def substitute_list(arg_list):
                for i in range(len(arg_list)):
                    v = arg_list[i]
                    # change the args to their mapping in env (if exist)
                    if isinstance(v, torch.fx.node.Node) and v in env:
                        arg_list[i] = env[v]
                    # recursively check each member of a list
                    # if the element is an immutable_list, we cast it to a mutable list, then cast back to tuple for hash
                    elif isinstance(v, list) or isinstance(v, tuple):
                        v = list(v)
                        substitute_list(v)
                        arg_list[i] = tuple(v)
                    elif isinstance(v, dict):
                        v = dict(v)
                        substitute_dict(v)
                        arg_list[i] = tuple(sorted(v))

            # substitute items of a dictionary to its mapping in env if exists
            # the subsitution is recursive for nested dictionary or list-type items
            def substitute_dict(kwarg_dict):
                for k,v in kwarg_dict.items():
                    # change the args to their mapping in env (if exist)
                    if isinstance(v, torch.fx.node.Node) and v in env:
                        kwarg_dict[k] = env[v]
                    elif isinstance(v, list) or isinstance(v, tuple):
                        v = list(v)
                        substitute_list(v)
                        kwarg_dict[k] = tuple(v)
                    elif isinstance(v, dict):
                        v = dict(v)
                        substitute_dict(v)
                        kwarg_dict[k] = tuple(sorted(v.items()))

            substitute_list(args)
            substitute_dict(kwargs)
            args = tuple(args)
            kwargs = tuple(sorted(kwargs))
            
            # hash substituted args to a number
            hash_arg = hash((args, kwargs))
            hash_val = (n.target, hash_arg)

            # check if a node can be eliminated. check both hash and node to avoid hash collision problem
            # if hash collision happens, only one set of equivalent nodes are eliminated
            # e.g. if hash(node1)=hash(node2) = hash(node3)=hash(node4), but node1=node2 != node3=node4, 
            # node 2 will be eliminated, but node 4 will not. 
            if hash_val in hash_env and check_same(hash_env[hash_val], n, env): 
                env[n] = hash_env[hash_val]
                continue
           
            new_node = new_graph.node_copy(n, lambda x: env[x])
            hash_env[hash_val] = new_node
            env[n] = new_node
            
    return new_graph






# hash arg and return a number
# if arg is a list, cast to tuple which is a hashable type
# for nested unhashable types, recursively hash each element and combine the hashcode of each element
# def fx_hash(arg):
#     if isinstance(arg, list): # torch.fx.immutable_collections.immutable_list is also a list
#         arg = tuple(arg)
#     elif isinstance(arg, dict): # torch.fx.immutable_collections.immutable_dict is a dict
#         arg = tuple(sorted(arg.items()))
#     try:
#         return hash(arg)
#     except TypeError:
#         if(isinstance(arg, tuple)): #TODO: remove parenthesis
#             # https://stackoverflow.com/questions/3054449/how-to-properly-define-hash-function-for-a-list-of-objects
#             hashCode = 1
#             for ele in arg:
#                 hashCode = 31*hashCode + (0 if ele is None else fx_hash(ele)) #TODO: but this works for set, but not for ordered list
#             return hashCode
#         else:
#             raise TypeError
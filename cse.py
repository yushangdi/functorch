import torch
import torch.fx as fx
from functorch import make_fx
from functorch.compile import aot_function, print_compile

import hashlib
import json


#TODO: another way: recursively change immutable_list to tuple? but how about dict nested in list?
# hash arg and return a number
# if arg is a list, cast to tuple which is a hashable type
# for nested unhashable types, recursively hash each element and combine the hashcode of each element
# for torch.fx.immutable_collections.immutable_dict, sort the disctionary keyss and dump to a json. Then
# use the hashlib.md5 to get a encoded version of the json. Finally, hash the code
def fx_hash(arg):
    # TODO: turn dict into key-value pairs
    if isinstance(arg, list): # torch.fx.immutable_collections.immutable_list is also a list
        arg = tuple(arg)
    elif isinstance(arg, dict): # torch.fx.immutable_collections.immutable_dict is a dict
        arg = tuple(sorted(arg.items()))
    try:
        return hash(arg) # cannot hash object?
    except TypeError:
        if(isinstance(arg, tuple)): #TODO: remove parenthesis
            # https://stackoverflow.com/questions/3054449/how-to-properly-define-hash-function-for-a-list-of-objects
            hashCode = 1
            for ele in arg:
                hashCode = 31*hashCode + (0 if ele is None else fx_hash(ele)) #TODO: but this works for set, but not for ordered list
            return hashCode
        # if(isinstance(arg, torch.fx.immutable_collections.immutable_dict)):
        #     dhash = hashlib.md5()
        #     encoded = json.dumps(arg, sort_keys=True).encode()
        #     dhash.update(encoded)
        #     return hash(dhash.hexdigest())
        else:
            raise TypeError



def check_args(new_args, old_args, env):
    if (len(new_args)!=len(old_args)):
        return False
    for i in range(len(new_args)):
        if (new_args[i] != old_args[i]):
            if not isinstance(new_args[i], torch.fx.node.Node):
                return False
            elif (not old_args[i] in env):
                return False
            elif new_args[i] != env[old_args[i]]:
                return False
    return True

# check if two nodes new_node and old_node are the same
# two nodes are the same if
# 1) they have the same target
# 2) their args and kwargs are the same. For node the elements of args and kwargs, they are the same
#    if the old element map to the new element in env. 
# essentially the following sementic with a more sophisticated implementtation of ==
#     node.target == n.target and node.args == n.args and node.kwargs == n.kwargs
def check_same(new_node: torch.fx.node.Node, old_node: torch.fx.node.Node, env: dict):
    if (new_node.target != old_node.target):
        return False
    if not check_args(new_node.args, old_node.args, env):
        return False
    if not check_args(new_node.kwargs, old_node.kwargs, env):
        return False
    return True


# return a new graph with CSE applied to the input graph
# env stores a mapping from node in the old graph to node in the new graph
# The placehold, output, and get_attr nodes are copied to the new grpah without change
# The call nodes (call_function, call_module, call_method) are hashed to check if they
# have an equivalent node in the graph. If so, this node will not be copied, and a mapping
# to the duplicated node is stored in env
def modify(fx_g: torch.fx.graph_module.GraphModule):
    new_graph = fx.Graph()
    env = {} # map from node in the old graph to node in the new graph
    hash_env = {} # map from the computatio result to a node in the new graph
    for n in fx_g.graph.nodes:
        if n.op == 'placeholder' or n.op == 'output' or n.op == 'get_attr': # != "call_function"
            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
        else: #n.op == 'call_function', we should never see n.op == 'call_module' or n.op == 'call_method'
            # print("======")
            # print(n.target)
            # print(n.args)
            # print(n.kwargs)
            # print(n.name) # e.g. cos_1
            # try: 

            # convert to list because tuple type is not mutable
            args = list(n.args)
            kwargs = list(n.kwargs)

            # change the args to their mapping in env (if exist)
            for i in range(len(args)):
                if isinstance(args[i], torch.fx.node.Node) and args[i] in env:
                    args[i] = env[args[i]]
            for i in range(len(kwargs)):
                if isinstance(kwargs[i], torch.fx.node.Node) and kwargs[i] in env:
                    kwargs[i] = env[kwargs[i]]
            
            # hash args to a number
            hash_arg = fx_hash([args, kwargs])
            hash_val = (n.target, hash_arg)

            # check if a node can be eliminated. check both hash and node to avoid hash collision problem
            # TODO: if node collision happens, only one set of nodes are eliminated
            if hash_val in hash_env and check_same(hash_env[hash_val], n, env): 
                env[n] = hash_env[hash_val]
                continue
            # except TypeError: # TODO: add try_catch for robustness? default to not copy when there's a type problem
            #     print("WARNING: TypeError: {}. Node not checked for CSE".format(n))
            #     new_node = new_graph.node_copy(n, lambda x: env[x])
            #     env[n] = new_node #maybe redundant?
            #     continue
           
            new_node = new_graph.node_copy(n, lambda x: env[x])
            hash_env[hash_val] = new_node
            env[n] = new_node
            
    return new_graph
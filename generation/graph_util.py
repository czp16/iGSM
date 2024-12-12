"""
define the node and graph classes
"""
from typing import List, Optional, Tuple, Dict, Union, Any, Literal, NewType
import random
from collections import deque, namedtuple

class Node:
    def __init__(
        self,
        parent_nodes: List["Node"] = [], 
        index: Tuple[int, int] = None,
        name: Optional[str] = None,
    ):
        """
        Node representing a node in a `StructureGraph`. Each node has an index, which is a tuple of
        (layer index, node index), e.g., (li, ni) means the ni-th node in the li-th layer.
        """
        self.parent_nodes = parent_nodes
        self.index = index
        self.name = name

    @property
    def in_degree(self):
        return len(self.parent_nodes)

class Parameter(Node):
    def __init__(
        self, 
        parent_nodes: List["Parameter"], 
        index: Union[Tuple[int, Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]],
    ):
        """
        Parameter representing a parameter derived from a `StructureGraph`. There are two types of parameters:
        - instance parameter: defined by each edge, e.g., the edge `((li, ni), (lj, nj))` defines a parameter
            which means "the number of `nodes[li][ni].name` in each `nodes[lj][nj].name`", represented as 
            `((li, ni), (lj, nj))`.
        - abstract parameter: not explicitly defined by the graph, but it is the parameter describing
            the total number of items in one category w.r.t. a node in a higher layer. E.g., one abstract
            parameter is "the total number of [Category of layer li] in each `nodes[lj][nj].name`",
            represented as `(li, (lj, nj))`.
        
        See the `StructureGraph` class for more details.

        Parameters:
        ------------
        parent_nodes: the parent parameters that it depends on.
        index: the index of the parameter, which is a tuple. For instance parameter, it is `((li, ni), (lj, nj))`,
            and for abstract parameter, it is `(li, (lj, nj))`.
        d_level: the difficulty level of abstract parameter, which is the difference of the layer index, 
            i.e., `lj - li`.
        """
        self.parent_nodes = parent_nodes
        self.index = index
        self.d_level = 0 if self.param_type == "instance" else index[1][0] - index[0]
    
    @property
    def param_type(self):
        return "instance" if isinstance(self.index[0], int) else "abstract"

class DependencyNode(Node):
    def __init__(
        self,
        parent_nodes: List["DependencyNode"],
        node_type: Literal["instance", "abstract", "rng"],
        op: List[str] = [], 
        name: Optional[str] = None,
        value: int = 0,
    ):
        """
        DependencyNode representing a node in a `DependencyGraph`. Each node corresponds to a parameter from
        the `StructureGraph`.

        Parameters:
        ------------
        parent_nodes: the parent nodes that it depends on.
        node_type: the type of the node, either "instance" or "abstract", corresponding to the parameter type.
        op: the operations that the node performs from parent nodes, e.g., "+", "*", "-".
        name: the name of the node.
        value: the correct value of the node by performing the operations from parent nodes.
        """
        self.parent_nodes = parent_nodes
        self.node_type = node_type
        self.op = op
        self.name = name
        self.value = value


class StructureGraph:
    def __init__(
        self, 
        nodes: List[List[int]],
        edges: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        names: List[List[str]],
        layer_categories: List[str],
    ):
        """
        Parameters:
        ------------
        nodes: list of nodes in each layer, index = (li,ni) where li is the layer index and ni is the node index
        edges: list of edges between nodes, each edge is represented as ((li, ni), (lj, nj)), 
            indicating that node (li, ni) is a parent of node (lj, nj)
        names: list of names for each layer
        """
        self.num_layers = len(nodes)
        assert len(nodes) == len(names) == len(layer_categories), "Number of layers and names do not match"
        self.nodes: List[List[Node]] = []
        self.edges = edges

        for i, layer in enumerate(nodes):
            self.nodes.append([Node(index=[i,j]) for j in layer])
        
        for (li, ni), (lj, nj) in self.edges:
            self.nodes[lj][nj].parent_nodes.append(self.nodes[li][ni])

        self.name_nodes(names)
        # self.name2node = {node.name: node for layer in self.nodes for node in layer}
        self.layer_categories = layer_categories

        self.construct_param_dependency_graph()
        

    def name_nodes(self, names: List[str]):
        # TODO: Implement this method
        # also define the class of each layer
        raise NotImplementedError
    

    def construct_param_dependency_graph(self):
        """
        Construct the dependency graph of all instance/abstract parameters from the structure graph.
        """
        self.params: List[Parameter] = []
        self.params.extend(self.get_all_instance_param())
        self.params.extend(self.get_all_abstract_param())
        
        index2param = {param.index: param for param in self.params}

        # add the dependency parameters to each abstract parameter
        for param in self.params:
            if param.param_type == "abstract":
                li, (lj, nj) = param.index
                if lj - li == 1:
                    # the parent parameters of the abstract parameter are the instance parameters
                    # defined by the edges between the layers li and lj
                    for parent in self.nodes[lj][nj].parent_nodes:
                        # TODO: remove the assert statement if the graph is correct
                        assert (parent.index, (lj, nj)) in self.edges, f"Invalid edge: {parent.index} -> {(lj,nj)}"
                        param.parent_nodes.append(index2param[(parent.index, (lj, nj))])
                elif lj - li > 1:
                    for parent in self.nodes[lj][nj].parent_nodes:
                        # TODO: remove the assert statement if the graph is correct
                        assert (parent.index, (lj, nj)) in self.edges, f"Invalid edge: {parent.index} -> {(lj,nj)}"
                        param.parent_nodes.append(index2param[(parent.index, (lj, nj))])
                        param.parent_nodes.append(index2param[(li, parent.index)])
                else:
                    raise ValueError(f"Invalid parameter index: {param.index}")

        
    
    def get_all_instance_param(self):
        """
        Instance parameters are defined by each edge, e.g., the edge `((li, ni), (lj, nj))` defines
        the instance parameter "the number of `nodes[li][ni].name` in each `nodes[lj][nj].name`".
        """
        all_instance_param = []
        for edge in self.edges:
            all_instance_param.append(Parameter([], edge))
        return all_instance_param
    

    def get_all_abstract_param(self):
        """
        Abstract parameters are not directly defined by the graph, they are the parameters
        describing the number of all items in the category in a node which is in a higher layer.
        E.g., suppose the category name of layer li is backpack and each node in layer li denotes
        a specific backpack (e.g., `nodes[li][ni].name` is school backpack), and the `nodes[lj][nj].name`
        is "art classroom". \
        Then there is an abstract parameter is 'the total number of backpack in each art classroom'.
        """
        all_abstract_param = []
        for lj, layer in enumerate(self.nodes):
            if lj == 0:
                continue
            for li in range(lj): # li: index of the lower layer
                for nj, _ in enumerate(layer):
                    param = Parameter([], (li, (lj, nj)))
                    all_abstract_param.append(param)
        return all_abstract_param
    

class DependencyGraph:
    def __init__(
        self, 
        Gs: StructureGraph, 
        max_op_stage1: int, 
        max_op_stage2: int,
        final_num_op: int,
        max_instance_in_degree: int = 4,
    ):
        self.Gs = Gs

        self.max_op_stage1 = max_op_stage1
        self.max_op_stage2 = max_op_stage2
        self.final_num_op = final_num_op
        self.max_instance_in_degree = max_instance_in_degree
        
        self.num_op = 0

        # TODO: add after constructing the graph
        self.leave_nodes = [node for layer in self.Gs.nodes for node in layer if node.in_degree == 0]
        self.root_nodes = None

    def construct_dependency_graph(self):
        """
        Construct the dependency graph of the structure graph.
        """
        # self.nodes = []
        # self.edges = []

        # This dictionary will map from index to its corresponding DependencyNode.
        self.idx2depnode: Dict[Tuple, DependencyNode] = {}
        # We will get all DependencyNodes in first 2 stages.
        self.construct_Gd1()
        self.construct_Gd2()
        self.depnodes = list(self.idx2depnode.values())

        self.construct_Gd3()


    def _add_ascendant_nodes_to_graph(self, root_param: Parameter):
        """
        Add all ascendant nodes of the root_param (including itself) to the dependency graph.
        Meanwhile keep the structure (parent-child relationship) of the nodes.

        Return:
        -------
        bool: whether the max_op_stage1 is reached
        """
        # temperary dictionary to map from index to DependencyNode
        # only store the newly added DependencyNodes, no overlap with self.idx2depnode
        tmp_idx2depnode = {}

        # First pass: 
        # Use BFS to create all DependencyNodes without linking their parents.
        open_list = deque([root_param])
        while open_list:
            param = open_list.popleft()

            # If we haven't created a node for this parameter yet, create it now.
            if param.index not in self.idx2depnode and param.index not in tmp_idx2depnode:
                dep_node = DependencyNode(
                    parent_nodes=[],  # will be linked later
                    node_type=param.param_type,
                    name=param.name,
                )
                
                tmp_idx2depnode[param.index] = dep_node

                for parent_param in param.parent_nodes:
                    if parent_param.index not in self.idx2depnode and parent_param.index not in tmp_idx2depnode:
                        open_list.append(parent_param)

        # check whether the max_op_stage1 is reached
        new_op = sum([max(1, self.Gs.index2param[idx].in_degree - 1) for idx in tmp_idx2depnode.keys()])
        if self.num_op + new_op > self.max_op_stage1:
            return False
        # if not reached, add the new DependencyNodes to the graph
        self.num_op += new_op
        self.idx2depnode.update(tmp_idx2depnode)

        # Second pass: 
        # Now that all DependencyNodes are created, link them together.
        for idx, dep_node in self.idx2depnode.items():
            param = self.Gs.index2param[idx]
            dep_node.parent_nodes = [self.idx2depnode[parent_param.idx] for parent_param in param.parent_nodes]
        return True
    
    def construct_Gd1(self):
        """
        Stage 1 of the dependency graph construction:
        Randomly select abstract parameters and its all dependent instance/abstract parameters
        recursively until the number of operations reaches max_op_stage1.
        """
        flag = True
        while flag:
            flag = False
            # select difficulty level from high to low to make the op_num as close to max_op_stage1 as possible
            for lvl in reversed(range(1, self.Gs.num_layers)):
                # TODO: maybe can optimize the valid_params selection, don't need to iterate all params every time
                valid_params = [p for p in self.Gs.params if p.d_level == lvl and p.index not in self.idx2depnode]
                if valid_params:
                    selected_param = random.choice(valid_params)
                    if self._add_ascendant_nodes_to_graph(selected_param):
                        flag = True
                        break
                    

    def construct_Gd2(self):
        """
        Stage 2 of the dependency graph construction:
        Randomly select instance parameters until the number of operations reaches max_op_stage2.
        """
        remaining_instance_params = set([p for p in self.Gs.params if p.param_type == "instance" and p.index not in self.idx2depnode])
        while self.num_op < self.max_op_stage2 and remaining_instance_params:
            # randomly pop a parameter from the remaining_instance_params
            selected_param = remaining_instance_params.pop()
            self.idx2depnode[selected_param.index] = DependencyNode(
                # though they should have no parents at this stage
                parent_nodes=[self.idx2depnode[p.index] for p in selected_param.parent_nodes],
                node_type="instance",
                name=selected_param.name,
            )
            self.num_op += max(1, selected_param.in_degree - 1)


    def construct_Gd3(self):
        """
        Stage 3 of the dependency graph construction:
        Get a random topological order of the nodes in the graph.
        """
        out_degree_map = {node: 0 for node in self.depnodes}
        for node in self.depnodes:
            for parent in node.parent_nodes:
                out_degree_map[parent] += 1

        remaining_nodes = set(self.depnodes)
        
        # start from the leave nodes
        # topo: the list of DependencyNodes in topological order 
        # (first it is from leaves to roots, will be reversed later)
        # group1: the parent nodes of nodes in topo, corresponding to Next1_Gd in paper
        # group2: the nodes having no children in remaining_nodes, corresponding to Next2_Gd in paper
        topo = []
        group1 = set()
        group2 = set([node for node in remaining_nodes if out_degree_map[node] == 0])
        
        while True:
            if not topo:
                node = group2.pop()
            else:
                node = (group1 & group2).pop()
            topo.append(node)
            remaining_nodes.remove(node)
            group1.remove(node)
            group2.remove(node)
            for p in node.parent_nodes:
                group1.add(p)
                out_degree_map[p] -= 1
                if out_degree_map[p] == 0:
                    group2.add(p)

            if not remaining_nodes:
                break
            if not (group1 & group2):
                if node.node_type == "abstract":
                    return False
                # non-uniformly random select a parent node
                # and create an edge p_node -> node
                p_node = random.choice(list(group1)) # TODO: non-uniformly random select
                node.parent_nodes.append(p_node)
                group1.add(p_node)
                
            elif node.node_type == "instance":
                if random.random() < 0.5:
                    # non-uniformly random select a parent node
                    # and create an edge p_node -> node
                    p_node = random.choice(list(remaining_nodes))
                    node.parent_nodes.append(p_node)
                    group1.add(p_node)

        topo.reverse()
        return topo


    def construct_Gd4(self, topo: List[DependencyNode]) -> bool:
        """
        Stage 4 of the dependency graph construction:
        Add additional dependency edges to make the graph more complex.
        Specifically, the in-degree of instance nodes should between 1 and 
        `self.max_instance_in_degree` and the final number of operations are
        exactly `self.final_num_op`.

        Return:
        -------
        bool: whether the graph is successfully constructed
        """
        curr_num_op = [max(1, node.in_degree - 1) for node in topo]
        max_num_op = [min(self.max_instance_in_degree - 1, max(1,i-1)) for i in range(len(topo))]
        while sum(curr_num_op) < self.final_num_op:
            # randomly select a node to increase its in-degree
            candidates = [i for i, node in enumerate(topo) if curr_num_op[i] < max_num_op[i] and node.node_type == "instance"]
            if not candidates:
                return False
            curr_num_op[random.choice(candidates)] += 1
        
        rng = DependencyNode([], "rng", name="RNG", value=0)
        self.depnodes.append(rng)

        for i, node in enumerate(topo):
            if node.node_type == "instance":
                pool = topo[:i] + [rng]
                
                dep_num = random.randint(1,2) if curr_num_op[i] == 1 else curr_num_op[i] + 1
                dep_num = min(dep_num, len(pool))
                if node.parent_nodes:
                    assert len(node.parent_nodes) == 1
                    pool.remove(node.parent_nodes[0])
                    dep_num -= 1
                if dep_num == len(pool):
                    node.parant_nodes.extend(pool)
                else:
                    if random.random() < 0.5:
                        node.parent_nodes.append(rng)
                        pool.remove(rng)
                        dep_num -= 1
                    node.parent_nodes.extend(random.sample(pool, dep_num))

        return True
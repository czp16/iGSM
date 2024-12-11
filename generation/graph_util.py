"""
define the node and graph classes
"""
from typing import List, Optional, Tuple, Dict, Union, Any, Literal, NewType
import random
from collections import deque, namedtuple

class Node:
    def __init__(
        self,
        child_nodes: List["Node"] = [], 
        index: Tuple[int, int] = None,
        name: Optional[str] = None,
    ):
        """
        Node representing a node in a `StructureGraph`. Each node has an index, which is a tuple of
        (layer index, node index), e.g., (li, ni) means the ni-th node in the li-th layer.
        """
        self.child_nodes = child_nodes
        self.index = index
        self.name = name

    @property
    def in_degree(self):
        return len(self.child_nodes)

class Parameter(Node):
    def __init__(
        self, 
        child_nodes: List["Parameter"], 
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
        child_nodes: the child parameters that it depends on.
        index: the index of the parameter, which is a tuple. For instance parameter, it is `((li, ni), (lj, nj))`,
            and for abstract parameter, it is `(li, (lj, nj))`.
        d_level: the difficulty level of abstract parameter, which is the difference of the layer index, 
            i.e., `lj - li`.
        """
        self.child_nodes = child_nodes
        self.index = index
        self.d_level = 0 if self.param_type == "instance" else index[1][0] - index[0]
    
    @property
    def param_type(self):
        return "instance" if isinstance(self.index[0], int) else "abstract"

class DependencyNode(Node):
    def __init__(
        self,
        child_nodes: List["DependencyNode"],
        node_type: Literal["instance", "abstract"],
        op: List[str] = [], 
        name: Optional[str] = None,
        value: int = 0,
    ):
        """
        DependencyNode representing a node in a `DependencyGraph`. Each node corresponds to a parameter from
        the `StructureGraph`.

        Parameters:
        ------------
        child_nodes: the child nodes that it depends on.
        node_type: the type of the node, either "instance" or "abstract", corresponding to the parameter type.
        op: the operations that the node performs from child nodes, e.g., "+", "*", "-".
        name: the name of the node.
        value: the correct value of the node by performing the operations from child nodes.
        """
        self.child_nodes = child_nodes
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
            indicating that node (li, ni) is a child of node (lj, nj)
        names: list of names for each layer
        """
        self.num_layers = len(nodes)
        assert len(nodes) == len(names) == len(layer_categories), "Number of layers and names do not match"
        self.nodes: List[List[Node]] = []
        self.edges = edges

        for i, layer in enumerate(nodes):
            self.nodes.append([Node(index=[i,j]) for j in layer])
        
        for (li, ni), (lj, nj) in self.edges:
            self.nodes[lj][nj].child_nodes.append(self.nodes[li][ni])

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
                    # the child parameters of the abstract parameter are the instance parameters
                    # defined by the edges between the layers li and lj
                    for child in self.nodes[lj][nj].child_nodes:
                        # TODO: remove the assert statement if the graph is correct
                        assert (child.index, (lj, nj)) in self.edges, f"Invalid edge: {child.index} -> {(lj,nj)}"
                        param.child_nodes.append(index2param[(child.index, (lj, nj))])
                elif lj - li > 1:
                    for child in self.nodes[lj][nj].child_nodes:
                        # TODO: remove the assert statement if the graph is correct
                        assert (child.index, (lj, nj)) in self.edges, f"Invalid edge: {child.index} -> {(lj,nj)}"
                        param.child_nodes.append(index2param[(child.index, (lj, nj))])
                        param.child_nodes.append(index2param[(li, child.index)])
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
    ):
        self.Gs = Gs
        # This dictionary will map from Parameter to its corresponding DependencyNode.
        self.param_idx2depnode = {}

        self.max_op_stage1 = max_op_stage1
        self.max_op_stage2 = max_op_stage2
        self.nodes = []
        self.edges = []
        self.num_op = 0

        # TODO: add after constructing the graph
        self.leave_nodes = [node for layer in self.Gs.nodes for node in layer if node.in_degree == 0]
        self.root_nodes = None


    def _add_descendant_nodes_to_graph(self, root_param: Parameter):
        """
        Add all descendant nodes of the root_param (including itself) to the dependency graph.
        Meanwhile keep the structure (parent-child relationship) of the nodes.

        Return:
        -------
        bool: whether the max_op_stage1 is reached
        """
        # temperary dictionary to map from index to DependencyNode
        # only store the newly added DependencyNodes, no overlap with self.param_idx2depnode
        tmp_idx2depnode = {}

        # First pass: 
        # Use BFS to create all DependencyNodes without linking their children.
        open_list = deque([root_param])
        while open_list:
            param = open_list.popleft()

            # If we haven't created a node for this parameter yet, create it now.
            if param.index not in self.param_idx2depnode and param.index not in tmp_idx2depnode:
                dep_node = DependencyNode(
                    child_nodes=[],  # will be linked later
                    node_type=param.param_type,
                    name=param.name,
                )
                
                tmp_idx2depnode[param.index] = dep_node

                for child_param in param.child_nodes:
                    if child_param.index not in self.param_idx2depnode and child_param.index not in tmp_idx2depnode:
                        open_list.append(child_param)

        # check whether the max_op_stage1 is reached
        new_op = sum([max(1, self.Gs.index2param[idx].in_degree - 1) for idx in tmp_idx2depnode.keys()])
        if self.num_op + new_op > self.max_op_stage1:
            return False
        # if not reached, add the new DependencyNodes to the graph
        self.num_op += new_op
        self.param_idx2depnode.update(tmp_idx2depnode)

        # Second pass: 
        # Now that all DependencyNodes are created, link them together.
        for idx, dep_node in self.param_idx2depnode.items():
            param = self.Gs.index2param[idx]
            dep_node.child_nodes = [self.param_idx2depnode[child_param.idx] for child_param in param.child_nodes]
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
                valid_params = [p for p in self.Gs.params if p.d_level == lvl and p.index not in self.param_idx2depnode]
                if valid_params:
                    selected_param = random.choice(valid_params)
                    if self._add_descendant_nodes_to_graph(selected_param):
                        flag = True
                        break
                    

    def construct_Gd2(self):
        """
        Stage 2 of the dependency graph construction:
        Randomly select instance parameters until the number of operations reaches max_op_stage2.
        """
        remaining_instance_params = [p for p in self.Gs.params if p.param_type == "instance" and p.index not in self.param_idx2depnode]
        while self.num_op < self.max_op_stage2:
            selected_param = random.choice(remaining_instance_params)
            remaining_instance_params.remove(selected_param)
            self.num_op += max(1, selected_param.in_degree - 1)

"""
define the node and graph classes
"""
from typing import List, Optional, Tuple, Dict, Union, Any

class Node:
    def __init__(
        self,
        child_nodes: List["Node"] = [], 
        value: Tuple[int, int] = None,
        name: Optional[str] = None,
    ):
        """
        Node representing a node in a `StructureGraph`. Each node has a value, which is a tuple of
        (layer index, node index), e.g., (li, ni) means the ni-th node in the li-th layer.
        """
        self.child_nodes = child_nodes
        self.value = value
        self.name = name

    @property
    def in_degree(self):
        return len(self.child_nodes)

class Parameter(Node):
    def __init__(
        self, 
        child_nodes: List["Parameter"], 
        value: Union[Tuple[int, Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]],
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

        The child nodes of a parameter are the parameters that it depends on.
        """
        self.child_nodes = child_nodes
        self.value = value
    
    @property
    def param_type(self):
        return "instance" if isinstance(self.value[0], int) else "abstract"

class DependencyNode(Node):
    def __init__(
        self,
        child_nodes: List["DependencyNode"], 
        value: Optional[int],
        op: str, 
        name: Optional[str] = None,
    ):
        super().__init__(child_nodes, name)
        self.value = value
        self.op = op


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
        nodes: list of nodes in each layer, value = (li,ni) where li is the layer index and ni is the node index
        edges: list of edges between nodes, each edge is represented as ((li, ni), (lj, nj)), 
            indicating that node (li, ni) is a child of node (lj, nj)
        names: list of names for each layer
        """
        self.num_layers = len(nodes)
        assert len(nodes) == len(names) == len(layer_categories), "Number of layers and names do not match"
        self.nodes: List[List[Node]] = []
        self.edges = edges

        for i, layer in enumerate(nodes):
            self.nodes.append([Node(value=[i,j]) for j in layer])
        
        for (li, ni), (lj, nj) in self.edges:
            self.nodes[lj][nj].child_nodes.append(self.nodes[li][ni])

        self.name_nodes(names)
        # self.name2node = {node.name: node for layer in self.nodes for node in layer}
        self.layer_categories = layer_categories
        

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
        
        value2param = {param.value: param for param in self.params}

        # add the dependency parameters to each abstract parameter
        for param in self.params:
            if param.param_type == "abstract":
                li, (lj, nj) = param.value
                if lj - li == 1:
                    # the child parameters of the abstract parameter are the instance parameters
                    # defined by the edges between the layers li and lj
                    for child in self.nodes[lj][nj].child_nodes:
                        # TODO: remove the assert statement if the graph is correct
                        assert (child.value, (lj, nj)) in self.edges, f"Invalid edge: {child.value} -> {(lj,nj)}"
                        param.child_nodes.append(value2param[(child.value, (lj, nj))])
                elif lj - li > 1:
                    for child in self.nodes[lj][nj].child_nodes:
                        # TODO: remove the assert statement if the graph is correct
                        assert (child.value, (lj, nj)) in self.edges, f"Invalid edge: {child.value} -> {(lj,nj)}"
                        param.child_nodes.append(value2param[(child.value, (lj, nj))])
                        param.child_nodes.append(value2param[(li, child.value)])
                else:
                    raise ValueError(f"Invalid parameter value: {param.value}")

        return self.params
        
    
    def get_all_instance_param(self):
        """
        instance parameters are defined by each edge, e.g., the edge `((li, ni), (lj, nj))` defines
        the instance parameter "the number of `nodes[li][ni].name` in each `nodes[lj][nj].name`".
        """
        all_instance_param = []
        for edge in self.edges:
            all_instance_param.append(Parameter([], edge))
        return all_instance_param
    

    def get_all_abstract_param(self):
        """
        abstract parameters are not directly defined by the graph, they are the parameters
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
        self.max_op_stage1 = max_op_stage1
        self.max_op_stage2 = max_op_stage2
        self.nodes = []
        self.edges = []
    
    def construct_Gd1(self):
        """
        Stage 1 of the dependency graph construction:
        Randomly select abstract parameters and its all dependent instance/abstract parameters
        until the number of operations reaches max_op_stage1.
        """

from typing import Dict, List, Tuple, Optional
import numpy as np
import random

from graph_util import Node, StructureGraph, DependencyNode, DependencyGraph

DEFAULT_CONFIG = {
    "structure_layers": 4,
    "min_items_per_layer": 2,
    "max_items_per_layer": 4,
    "arithmetic_mod": 23,
    "max_operations": 15,
    "max_attempts": 10,
}

# TODO: Implement load_categories
def load_categories() -> Dict[str, List[Dict]]:
    return {}

class ProblemGenerator:
    def __init__(self, config: Dict):
        self.config = config
        for k,v in DEFAULT_CONFIG:
            if k not in self.config:
                self.config[k] = v

        assert self.config["structure_layers"] <= DEFAULT_CONFIG["structure_layers"], \
            "Number of layers exceeds the default configuration"

    def _load_categories(self) -> Dict[str, List[Dict]]:
        return load_categories()
    
    def generate_structure(self) -> Dict[str, List[str]]:
        '''
        Randomly generates the structure of a graph for the problem.
        '''
        n_layer = self.config["structure_layers"]
        n_edge = self.config["desired_edges"]
        w0 = self.config["min_items_per_layer"]
        w1 = self.config["max_items_per_layer"]
        
        # 1. Randomly generate the number of items per layer
        flag = False
        _cnt = 0
        while not flag:
            _cnt += 1
            n_items_per_layer = np.random.randint(w0, w1+1, n_layer)
            e_minus = sum(n_items_per_layer[1:])
            e_plus = sum(n_items_per_layer[i]*n_items_per_layer[i+1] for i in range(n_layer-1))
            if e_minus <= n_edge <= e_plus or _cnt > self.config["max_attempts"]:
                flag = True
        if _cnt > self.config["max_attempts"]:
            return None
        
        # 2. Randomly generate the nodes and edges
        nodes = [] # list of nodes in each layer
        for i in range(n_layer):
            nodes.append(list(range(n_items_per_layer[i])))
        
        # 2.1 generate minimum number (e_minus) of edges
        # each edge is represented as ((li, ni), (lj, nj))
        # where li, lj are the layer index and ni, nj are the node index
        edges = []
        for i in range(1, n_layer):
            for j in range(n_items_per_layer[i]):
                k = np.random.randint(0, n_items_per_layer[i-1])
                edges.append(((i-1, k), (i, j)))
        
        # 2.2 generate the remaining edges
        n_current_edges = len(edges)
        while n_current_edges < n_edge:
            i = np.random.randint(1, n_layer)
            j = np.random.randint(0, n_items_per_layer[i])
            k = np.random.randint(0, n_items_per_layer[i-1])
            if ((i-1, k), (i, j)) not in edges:
                edges.append(((i-1, k), (i, j)))
                n_current_edges += 1

        # 3. add concept meaning to the nodes
        meaning_dict = {}
        # TODO: implement this

        return nodes, edges, meaning_dict
    

    def generate_dependency_graph(self, structure: Dict) -> Dict[str, Dict]:
        dependency_graph = {}
        parameters = self._flatten_structure(structure)
        operation_count = 0
        max_operations = self.config['max_operations']

        while parameters and operation_count < max_operations:
            param = random.choice(parameters)
            dependencies = self._generate_dependencies(parameters, param)
            operation = self._generate_operation(dependencies)
            dependency_graph[param] = {
                'dependencies': dependencies,
                'operation': operation
            }
            parameters.remove(param)
            operation_count += 1

        return dependency_graph
    
from typing import Dict, List, Tuple, Optional
import random
import numpy as np

from utils import softmax
from graph_util import Node, StructureGraph, DependencyNode, DependencyGraph

DEFAULT_CONFIG = {
    "structure_layers": 4,
    "min_items_per_layer": 2,
    "max_items_per_layer": 4,
    "max_instance_in_degree": 4,
    "arithmetic_mod": 23,
    "max_operations": 15,
    "max_attempts": 50,

    "max_instance_params": 20,
    "max_operations": 15,
    "force": False, # force the operation num to be exactly `max_operations``
}

# TODO: Implement load_categories
def load_categories() -> Dict[str, List[Dict]]:
    return {}

class ProblemGenerator:
    def __init__(self, config: Dict):
        self.config = config
        for k,v in DEFAULT_CONFIG.items():
            if k not in self.config:
                self.config[k] = v

        self._load_name_dictionary()

    def _load_name_dictionary(self):
        # TODO: modify it later
        self.name_dictionary = [
            [f"W{i}" for i in range(10)],
            [f"X{i}" for i in range(10)],
            [f"Y{i}" for i in range(10)],
            [f"Z{i}" for i in range(10)],
        ]
        self.categorey_name = ["WWW", "XXX", "YYY", "ZZZ"]
    
    
    def _get_structuregraph_hp(self, final_num_op: int) -> int:
        """
        Get the hyperparameter for the structure graph.

        Parameters:
        ----------
        final_num_op: int, the final number of operations of the question.
        
        Returns:
        ----------
        num_layers: int, the number of layers for the structure graph.
        w0: int, the minimum number of items per layer
        w1: int, the maximum number of items per layer
        num_edges: int, the number of edges in the structure graph.
        """
        assert self.config["structure_layers"] == 4
        max_ip = self.config["max_instance_params"]
        rel = (final_num_op - 1) / (max_ip - 1)
        _weights = np.array([-(rel-0.2)**2, -(rel-0.5)**2, -(rel-0.8)**2])

        num_layers = np.random.choice([2, 3, 4], p=softmax(_weights))
        _t0, _t1 = np.random.choice([2, 3, 4], size=2, replace=True, p=softmax(_weights))
        w0, w1 = min(_t0, _t1), max(_t0, _t1)
        _t0, _t1 = [random.randint((num_layers - 1) * w0, max_ip) for _ in range(2)]
        num_edges = min(_t0, _t1, (num_layers - 1) * w1 * w1)

        return w0, w1, num_layers, num_edges


    
    def draw_question(self):
        max_operations = self.config['max_operations']
        force = self.config['force']
        final_num_op = max_operations if force else min([random.randint(1, max_operations) for _ in range(2)])
        max_op_stage1 = max([random.randint(1, final_num_op) for _ in range(2)])
        max_op_stage2 = random.randint(max_op_stage1, final_num_op)

        w0, w1, num_layers, num_edges = self._get_structuregraph_hp(final_num_op)
        self.name_dictionary = self.name_dictionary[:num_layers]
        self.categorey_name = self.categorey_name[:num_layers]

        print(f"num_layers: {num_layers}, w0: {w0}, w1: {w1}, num_edges: {num_edges}")

        flag = False
        _cnt = 0
        while not flag:
            _cnt += 1
            if _cnt > self.config["max_attempts"]:
                raise RuntimeError("Cannot generate the structure graph.")

            # Generate the structure graph
            Gs = StructureGraph(w0, w1, num_layers, num_edges, self.name_dictionary, self.categorey_name)
            # Generate the dependency graph
            Gd = DependencyGraph(
                Gs, max_op_stage1, max_op_stage2, final_num_op,
                self.config["max_instance_in_degree"], self.config["arithmetic_mod"]
            )
            if Gd.construct_dependency_graph():
                flag = True
                self.Gs = Gs
                self.Gd = Gd
        

    def generate_question(self) -> str:
        question_desc = []
        for node in self.Gd.topo:
            if node.node_type == "instance":
                question_desc.append(self.Gd.gen_sentence(node))
        random.shuffle(question_desc)
        question_desc.append(self.Gd.gen_question(node))
        final_question = ". ".join(question_desc)
        return final_question
    
    def generate_answer(self) -> str:
        answer_desc = []
        all_variables = set(
            [chr(i) for i in range(ord('a'), ord('z') + 1)] + # lower case
            [chr(i) for i in range(ord('A'), ord('Z') + 1)] # upper case
        )
        for node in self.Gd.topo:
            _var_name = all_variables.pop()
            answer_desc.append(self.Gd.gen_answer(node, _var_name))
        final_answer = " ".join(answer_desc)
        return final_answer


if __name__ == "__main__":
    # seed = random.randint(0, 100000)
    seed = 57849
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed: {seed}")

    pg = ProblemGenerator(DEFAULT_CONFIG)
    pg.draw_question()
    print(pg.generate_question())
    print(pg.generate_answer())
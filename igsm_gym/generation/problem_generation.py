from typing import Dict, List, Tuple, Optional
import random
import json
import numpy as np

from igsm_gym.utils import softmax
from igsm_gym.generation.graph_util import Node, StructureGraph, DependencyNode, DependencyGraph

DEFAULT_CONFIG = {
    "english_path": "english/categorization.json",

    "max_structure_layers": 4,
    "min_items_per_layer": 2,
    "max_items_per_layer": 4,
    "max_instance_in_degree": 4,
    "arithmetic_mod": 23,
    "max_operations": 15,
    "max_attempts": 50,

    "max_instance_params": 20,
    "max_operations": 11,
    "force": True, # force the operation num to be exactly `max_operations`
}


class ProblemGenerator:
    def __init__(self, config: Dict = {}, debug: bool = False):
        self.config = config
        self.debug = debug
        for k,v in DEFAULT_CONFIG.items():
            if k not in self.config:
                self.config[k] = v

    def _load_name_dictionary(self):
        if self.debug:
            self.name_dictionary = [
                [f"W{i}" for i in range(10)],
                [f"X{i}" for i in range(10)],
                [f"Y{i}" for i in range(10)],
                [f"Z{i}" for i in range(10)],
            ]
            self.category_name = ["WWW", "XXX", "YYY", "ZZZ"]
        else:
            with open(self.config["english_path"], 'r') as f:
                all_data = json.load(f)
            system = all_data.get(random.choice(list(all_data.keys())))
            self.category_name = list(system.keys())
            
            self.name_dictionary = []
            for sub_category in system.values():
                self.name_dictionary.append(sub_category.get(random.choice(list(sub_category.keys()))))

            self.category_name.reverse()
            self.name_dictionary.reverse()
    
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
        assert self.config["max_structure_layers"] == 4
        max_ip = self.config["max_instance_params"]
        rel = (final_num_op - 1) / (max_ip - 1)
        _weights = np.array([-(rel-0.2)**2, -(rel-0.5)**2, -(rel-0.8)**2])

        num_layers = np.random.choice([2, 3, 4], p=softmax(_weights))
        _t0, _t1 = np.random.choice([2, 3, 4], size=2, replace=True, p=softmax(_weights))
        w0, w1 = min(_t0, _t1), max(_t0, _t1)
        _t0, _t1 = [random.randint((num_layers - 1) * w0, max_ip) for _ in range(2)]
        num_edges = min(_t0, _t1, (num_layers - 1) * w1 * w1)

        return w0, w1, num_layers, num_edges


    
    def draw_question(self) -> bool:
        """
        Randomly generate the structure graph and dependency graph to draw the question.

        Returns:
        ----------
        bool, whether the question is successfully generated
        """
        self._load_name_dictionary()

        max_operations = self.config['max_operations']
        force = self.config['force']
        final_num_op = max_operations if force else min([random.randint(1, max_operations) for _ in range(2)])
        max_op_stage1 = max([random.randint(1, final_num_op) for _ in range(2)])
        max_op_stage2 = random.randint(max_op_stage1, final_num_op)

        w0, w1, num_layers, num_edges = self._get_structuregraph_hp(final_num_op)
        _start_layer = random.randint(0, self.config["max_structure_layers"] - num_layers)
        self.name_dictionary = self.name_dictionary[_start_layer:_start_layer+num_layers]
        self.category_name = self.category_name[_start_layer:_start_layer+num_layers]

        # print(f"num_layers: {num_layers}, w0: {w0}, w1: {w1}, num_edges: {num_edges}")

        flag = False
        _cnt = 0
        while not flag:
            _cnt += 1
            if _cnt > self.config["max_attempts"]: 
                # will retry if stage 3 failed until max_attempts
                return False

            # Generate the structure graph
            Gs = StructureGraph(w0, w1, num_layers, num_edges, self.name_dictionary, self.category_name)
            # Generate the dependency graph
            Gd = DependencyGraph(
                Gs, max_op_stage1, max_op_stage2, final_num_op,
                self.config["max_instance_in_degree"], self.config["arithmetic_mod"]
            )
            result = Gd.construct_dependency_graph()
            if result == "success":
                flag = True
                self.Gs = Gs
                self.Gd = Gd
            elif result == "stage 4 failed": # will not retry if stage 4 failed
                return False
        return True
        

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
            if not all_variables:
                raise RuntimeError("No enough variable names for answer.")
            _var_name = all_variables.pop()
            answer_desc.append(self.Gd.gen_answer(node, _var_name))
        final_ans_statement = "Thus, the answer is {}.".format(node.value)
        answer_desc.append(final_ans_statement)
        
        final_answer = "\n".join(answer_desc)
        return final_answer


if __name__ == "__main__":
    seed = random.randint(0, 100000)
    # seed = 2927
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed: {seed}")

    pg = ProblemGenerator(DEFAULT_CONFIG)
    if pg.draw_question():
        print(pg.generate_question())
        print(pg.generate_answer())
    else:
        print("Failed to generate the question.")

    # _cnt = 0

    # for _ in range(100):
    #     pg = ProblemGenerator(DEFAULT_CONFIG)
    #     if pg.draw_question():
    #         _cnt += 1

    # print(f"Success rate: {_cnt} / {100}")
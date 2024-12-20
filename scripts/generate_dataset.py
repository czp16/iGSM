import os
import random
import numpy as np
import datasets

from igsm_gym.generation import ProblemGenerator




config = {
    "english_path": "igsm_gym/generation/english/categorization.json",
    "max_operations": 21,
    "force": False,
}

dataset_size = 50000
save_path = f"data/iGSM_train_maxop_{config['max_operations']}"
seed = 42
random.seed(seed)
np.random.seed(seed)

pg = ProblemGenerator(config)

questions = []
answers = []
num_ops = []
final_answers = []

while len(questions) < dataset_size:
    if pg.draw_question():
        questions.append(pg.generate_question())
        answers.append(pg.generate_answer())
        num_ops.append(pg.Gd.final_num_op)
        final_answers.append(pg.Gd.topo[-1].value)


dataset = datasets.Dataset.from_dict({
    "query": questions,
    "response": answers,
    "num_ops": num_ops,
    "final_answer": final_answers,
})

dataset.save_to_disk(save_path)
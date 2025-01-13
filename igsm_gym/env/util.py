import re

def extract_final_answer(text):
    match = re.search(r'\banswer is (\d+)\b', text)
    return int(match.group(1)) if match else None

DEFAULT_ENV_CONFIG = {
    "max_steps": 10,
    "reward_model": "outcome", # or "process"

    "start_word/intermediate_step": "Define",
    "start_word/final_step": "Thus, the answer is",

    "prompt_template": \
"""<|system|>
You're an expert at solving elementary math problems involving addition, subtraction, and multiplication. You solve all the problems in a uniform format. All calculations are done modulo 23.</s>
<|user|>
{query}</s>
<|assistant|>
""",
}

DEFAULT_REWARD_CONFIG = {
    "PRM/correct_step": 1.0,
    "PRM/wrong_calculation": 0.5,
    "PRM/wrong_step": -0.5,
    "PRM/redundant_step": -0.5,
    "PRM/mismatch_step": -1.0,
    "PRM": True,

    "ORM/correct_answer": 1.0,
    "ORM/wrong_answer": 0.0,
    "ORM/intermediate_step": 0.0,
}
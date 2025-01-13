from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from igsm_gym.generation import ProblemGenerator
from igsm_gym.env.reward_model import ProcessRewardModel, OutcomeRewardModel
from igsm_gym.env.util import DEFAULT_ENV_CONFIG, DEFAULT_REWARD_CONFIG

class ActionStep:
    """
    The action step in the environment
    """
    def __init__(self, content: str, misc: Any):
        self.content = content
        self.misc = misc

class IGSMEnv:
    """
    The iGSM environment
    """

    def __init__(
        self, 
        problem_generator: ProblemGenerator, 
        reward_model_type: str = DEFAULT_ENV_CONFIG["reward_model"],
        prompt_template: str = DEFAULT_ENV_CONFIG["prompt_template"],
        max_steps: int = DEFAULT_ENV_CONFIG["max_steps"]
    ):
        self.problem_generator = problem_generator
        self.prompt_template = prompt_template
        self.max_steps = max_steps

        if reward_model_type == "outcome":
            self.reward_model = OutcomeRewardModel(problem_generator.Gd)
        else:
            self.reward_model = ProcessRewardModel(problem_generator.Gd)

        self.query: str = ""
        self.answer_history: List[ActionStep] = []

    def seed(self, seed: int):
        self.problem_generator.seed(seed)


    def reset(self) -> Dict[str, np.ndarray]:
        """
        reset the environment

        Returns:
        -------
        str: the initial state
        Dict[str, np.ndarray]: info
        """
        question = self.problem_generator.draw_question()
        self.query = self.prompt_template.format(question)
        self.answer_history.clear()
        self._curr_step = 0 # current step 
        info = {}
        return self.get_state(), info
    
    def get_state(self) -> str:
        return self.query + "\n".join([step.content for step in self.answer_history])

    def get_reward(self, state: str, action: ActionStep) -> float:
        return self.reward_model(state, action.content)

    def step(
        self, action: ActionStep
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment
        """
        
        self._curr_step += 1
        if self._curr_step >= self.max_steps:
            truncated = True
        curr_state = self.get_state()
        reward = self.get_reward(curr_state, action)
        
        self.answer_history.append(action)
        next_state = self.get_state()
        terminated = action.content.startswith(DEFAULT_ENV_CONFIG["start_word/final_step"])
        info = {}

        return next_state, reward, terminated, truncated, info
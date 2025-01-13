from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

from igsm_gym.generation import ProblemGenerator
from igsm_gym.env.reward_model import ProcessRewardModel, OutcomeRewardModel
from igsm_gym.env.util import DEFAULT_ENV_CONFIG, DEFAULT_REWARD_CONFIG

class ActionStep:
    """
    The action step in the environment
    """
    def __init__(
        self,
        content: str,
        token_ids: np.ndarray,
        log_probs: np.ndarray,
    ):
        """
        Args:
            content (str): the content of the action step
            token_ids (np.ndarray): the token ids of the action step
            log_probs (np.ndarray): the log probabilities of each token in the action step,
                with the same shape as `token_ids`
        """
        self.content = content
        self.token_ids = token_ids
        self.log_probs = log_probs


class StepwiseLanguageEnv:
    """
    The step-wise language environment. Each step is a sub-sentence separated by a special token, e.g., "\n".
    """
    def __init__(
        self, 
        problem_generator: ProblemGenerator, 
        tokenizer: PreTrainedTokenizerBase,
        reward_model_type: str = DEFAULT_ENV_CONFIG["reward_model"],
        prompt_template: str = DEFAULT_ENV_CONFIG["prompt_template"],
        max_steps: int = DEFAULT_ENV_CONFIG["max_steps"]
    ):
        self.problem_generator = problem_generator
        self.tokenizer = tokenizer
        self.reward_model_type = reward_model_type
        self.prompt_template = prompt_template
        self.max_steps = max_steps

        self.query: str = ""
        self.answer_history: List[ActionStep] = []

    def seed(self, seed: int):
        self.problem_generator.seed(seed)


    def reset(self) -> str:
        """
        Reset the environment

        Returns:
            The initial state: str.
        """
        while True:
            if self.problem_generator.draw_question():
                break
        question = self.problem_generator.generate_question()
        # need to call `generate_answer` before creating the reward model
        # to compute the ground-truth answer
        example_answer = self.problem_generator.generate_answer()
        if self.reward_model_type == "outcome":
            self.reward_model = OutcomeRewardModel(self.problem_generator.Gd)
        else:
            self.reward_model = ProcessRewardModel(self.problem_generator.Gd)

        self.query = self.prompt_template.format(query=question)
        self.query_tokens = np.array(self.tokenizer(self.query)["input_ids"])
        self.answer_history.clear()
        self._curr_step = 0 # current step 
        return self.query
    
    # def get_state(self) -> str:
    #     return self.query + "".join([step.content for step in self.answer_history])

    def get_reward(self, state: str, action: str) -> float:
        return self.reward_model(state, action)

    def rollout(self, action_steps: List[ActionStep]) -> List[Dict[str, Any]]:
        """
        Do rollout in the environment with the given actions. Compute the obs, act, terminate, truncation in 
        each transition. Also assign a reward (by reward model) to each transition.

        Args:
            action_steps: the list of action steps
        
        Returns:
            A list of transitions, each transition is a dictionary with keys:
            - obs: the observation before the action
            - act: the action
            - obs_next: the observation after the action
            - rew: the reward of the action
            - terminated: whether the episode is terminated
            - truncated: whether the episode is truncated
            - log_probs: the log probabilities of each token in the action
            Note that the obs, act, obs_next are all token ids instead of text form.
        """
        # self.answer_history = action_steps
        next_obs, next_obs_ids = None, None

        transitions = []

        for i, step in enumerate(action_steps):
            # for string-form obs / action
            obs = self.query if next_obs is None else next_obs
            next_obs = obs + step.content
            act = step.content
            reward = self.get_reward(obs, act)
            
            # for token ids
            obs_ids = self.query_tokens if next_obs_ids is None else next_obs_ids
            next_obs_ids = np.concatenate([obs_ids, step.token_ids])
            act_ids = step.token_ids

            terminated = truncated = False
            if i == len(action_steps) - 1:
                if step.content.startswith(DEFAULT_ENV_CONFIG["start_word/final_step"]):
                    terminated = True
                else:
                    truncated = True

            # obs, act, obs_next, log_probs are with variable length
            # so we convert them to list and use np.array with dtype=object to store them
            transitions.append({
                "obs": obs_ids.tolist(),
                "act": act_ids.tolist(),
                "obs_next": next_obs_ids.tolist(), # concat of obs and act
                "log_probs": step.log_probs.tolist(),
                "rew": reward,
                "terminated": terminated,
                "truncated": truncated,
            })

        return transitions


    def batch_rollout(
        self, 
        action_steps_list: List[List[ActionStep]], 
        normalize_rewards: bool = True,
    ):
        """
        rollout in the environment with the given actions. Assign rewards to each action.

        Args:
            action_steps_list: the list of action steps for each episode
            normalize_rewards: whether to normalize the rewards

        Returns:
            A list of transitions for each episode.
        """
        transitions_list = []
        for action_steps in action_steps_list:
            transitions = self.rollout(action_steps)
            transitions_list.append(transitions)
        if normalize_rewards:
            if self.reward_model_type == "outcome":
                # apply reward normalization to the last transition of each episode
                rewards = [transitions[-1]["rew"] for transitions in transitions_list]
                r_mean, r_std = np.mean(rewards), np.std(rewards)
                for transitions in transitions_list:
                    # only normalize the last transition
                    for t in transitions[:-1]:
                        t["rew_normalized"] = t["rew"]
                    transitions[-1]["rew_normalized"] = (transitions[-1]["rew"] - r_mean) / r_std
            else:
                # apply reward normalization to all transitions
                rewards = [t["rew"] for transitions in transitions_list for t in transitions]
                r_mean, r_std = np.mean(rewards), np.std(rewards)
                for transitions in transitions_list:
                    for t in transitions:
                        t["rew_normalized"] = (t["rew"] - r_mean) / r_std
        return transitions_list
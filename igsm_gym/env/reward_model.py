from typing import Any, Optional, List, Tuple, Dict
import numpy as np
import re

from igsm_gym.generation import DependencyGraph
from igsm_gym.env.util import DEFAULT_REWARD_CONFIG, DEFAULT_ENV_CONFIG, extract_final_answer

# DEFAULT_REWARD_CONFIG = {
#     "correct_step_reward": 1.0,
#     "wrong_step_reward": -0.5,
#     "redundant_step_reward": 0.1,
#     "mismatch_step_reward": -0.1,
#     "mismatch_but_correct_reward": 0.5,
#     "format_error_reward": -1.0,
#     "PRM": True,

#     "start_word/intermediate_step": "Define",
#     "start_word/final_step": "Thus, the answer is",

#     "ORM/correct_answer": 1.0,
#     "ORM/wrong_answer": 0.0,
#     "ORM/intermediate_step": 0.0,
# }

class BaseRewardModel:
    def __init__(self, reward_config: Dict = {}) -> None:
        self.config = reward_config
        for k,v in DEFAULT_REWARD_CONFIG.items():
            if k not in self.config:
                self.config[k] = v

    def __call__(self, state: str, action: str) -> float:
        return 0.0
        
class ProcessRewardModel(BaseRewardModel):
    """
    Process Reward Model. We only check whether the intermediate step matches the ground truth answer.
    Here are several cases:
    1. The step is correct: get a "correct_step" reward
    2. The step calculates the correct variable but get the wrong value: get a "wrong_calculation" reward
    3. The step calculates the wrong variable: get a "wrong_step" reward
    4. The step is redundant: get a "redundant_step" reward
    5. The step is mismatched: get a "mismatch_step" reward

    Our checker will not strictly check the order of reasoning steps, i.e., it's possible that the agent
    calculates the correct answer but in a different order.
    """
    def __init__(self, Gd: DependencyGraph, reward_config = {}) -> None:
        super().__init__(reward_config)

        self.gt_answer = Gd.topo[-1].value
        self.is_final_step = lambda x: x.startswith(DEFAULT_ENV_CONFIG["start_word/final_step"])
        
        self.Gd = Gd
        self.Gd_topo_node = {node.name.lower(): node.value for node in self.Gd.topo}

        # the word "each" is actually a part of the node name,
        # so we need to add it to the node name after the extraction
        self.step_pattern = r"(?:Define each |Each )?(.+?) as (\w+)\. (?:So |so )?\2 = (?:.*= )?(\d+)\."

    def __call__(self, state: str, action: str):
        # Calculate reward based on llm_output trajectory (one turn)
        if self.is_final_step(action):
            final_answer = extract_final_answer(action)
            if self.gt_answer == final_answer:
                return self.config["PRM/correct_step"]
            else:
                return self.config["PRM/wrong_step"]
            
        else:
            matched = re.match(self.step_pattern, action)
            if matched:
                node_name = matched.group(1).lower()
                node_name = "each " + node_name
                var_name = matched.group(2)
                value = int(matched.group(3))
                if node_name in self.Gd_topo_node.keys():
                    if value == self.Gd_topo_node[node_name]:
                        return self.config["PRM/correct_step"]
                    else:
                        return self.config["PRM/wrong_calculation"]
                else:
                    return self.config["PRM/redundant_step"]
            else:
                return self.config["PRM/mismatch_step"]


class OutcomeRewardModel(BaseRewardModel):
    def __init__(self, Gd: DependencyGraph, reward_config = {}) -> None:
        super().__init__(reward_config)
        
        self.gt_answer = Gd.topo[-1].value
        self.is_final_step = lambda x: x.startswith(DEFAULT_ENV_CONFIG["start_word/final_step"])
        
    def __call__(self, state: str, action: str):
        if self.is_final_step(action):
            final_answer = extract_final_answer(action)
            if self.gt_answer == final_answer:
                return self.config["ORM/correct_answer"]
            else:
                return self.config["ORM/wrong_answer"]
        else:
            return self.config["ORM/intermediate_step"]


class RewardModel:
    def __init__(
        self,
        reward_config: Dict = {},
        Gd: Optional[DependencyGraph] = None,
    ) -> None:
        self.Gd = Gd
        self.config = reward_config
        for k,v in DEFAULT_REWARD_CONFIG.items():
            if k not in self.config:
                self.config[k] = v

        assert self.Gd is not None
        self.Gd_topo_node = {node.name : node.value for node in self.Gd.topo}
    
    def PRM(self, llm_output: str) -> List[float]: 
        # Process Reward Model
        # Calculate reward based on llm_output trajectory (one turn)
        
        sentence_list = llm_output.split('\n') # List["Define each Y9's X7 as b. So b = 9."]
        # print("sentence_list: ", sentence_list)
        
        final_answer = sentence_list.pop() # "Thus, the answer is ..."
        # print("final_answer: ", final_answer)
        
        try:
            final_answer_value = float(final_answer.split(' ')[-1]) 
        except ValueError:
            final_answer_value = -1 # invalid output
        
        
        
        sentence_info_list = []
        for sentence in sentence_list:
            # The template string
            template = sentence
            # Define the regular expression pattern
            
            pattern = r"(?:Define each |Each )?(.+?) as (\w+)\. (?:So |so )?\2 = (?:.*= )?(\d+)\." # r"Define (.+?) as (\w+)\. So \2 = .* = (\d+)\."
            # Match the pattern in the string
            match = re.match(pattern, template)
            # print("pattern:")
            # print(pattern)
            # print("template:")
            # print(template)
            # print("match:", match)
            # print()
            

            if match:
                node_name = match.group(1)  # node name
                node_name = "each " + node_name
                var_name = match.group(2)   # var name
                value = match.group(3)   # var value
                
                info = {
                    "node_name": node_name,
                    "var_name": var_name,
                    "value": int(value)
                }
                
            else:
                # print("@"*10)
                # print(sentence)
                # print("@"*10)
                info = {
                    "node_name": None,
                    "var_name": None,
                    "value": None,
                    "failure": "mismatch"
                }
            
            sentence_info_list.append(info)
        
        reward_list = [0] * len(sentence_info_list)
        for i in range(len(sentence_info_list)):
            info = sentence_info_list[i]
            
            if "mismatch" in info.values(): 
                # This step is redundant
                reward_list[i] = self.config["mismatch_step_reward"]
                continue
            
            elif not info["node_name"] in self.Gd_topo_node.keys():
                # This step is redundant
                reward_list[i] = self.config["redundant_step_reward"]
                continue
            
            elif info["node_name"] in self.Gd_topo_node.keys():
                # This is a correct and non-redundant step
                if info['value'] == self.Gd_topo_node[info["node_name"]]:
                    reward_list[i] = self.config["correct_step_reward"]
                else:
                    reward_list[i] = self.config["wrong_step_reward"]
                
                continue
            
            else:
                # This step is wrong
                reward_list[i] == self.config["wrong_step_reward"]
                continue
        
        # Final step 
        gt_ans = self.Gd.topo[-1].value
        # print("GT_final step:", gt_ans)
        if gt_ans == final_answer_value:
            reward_list.append(self.config["correct_step_reward"])
        elif final_answer_value == -1:
            reward_list.append(self.config["format_error_reward"])
        else:
            reward_list.append(self.config["wrong_step_reward"])
            
        # for i in range(len(sentence_info_list)):
        #     print("info {}".format(i))
        #     print(sentence_info_list[i])
        #     print("reward:", reward_list[i])
        #     print("="*10)

        return reward_list
    
    def ORM(self, llm_output: str) -> List[float]:
        # Outcome Reward Model
        # Calculate reward based on llm_output trajectory (one turn)
        
        sentence_list = llm_output.split('\n')
        final_answer = sentence_list[-1] # "Thus, the answer is ..."
        final_answer_value = int(final_answer.split(' ')[-1]) 
        gt_ans = self.Gd.topo[-1].value
        if gt_ans == final_answer_value:
            reward_list = [self.config["correct_step_reward"]] * len(sentence_list)
        else:
            reward_list = [self.config["wrong_step_reward"]] * len(sentence_list)
        
        return reward_list
    
    def recon_GD(self, sentence_list: List[str]):
        # reconstruct GD information based on the sentences in the answer output
        # TODO: if it is useful in reward caculation
        return
import numpy as np
from typing import Any, Optional, List, Tuple, Dict
from problem_generation import *
import re

"""_summary_

Reward model
Testing file is reward.ipynb

"""

REWARD_DEFAULT_CONFIG = {
    "correct_step_reward": 1.0,
    "wrong_step_reward": 0.0,
    "redundant_step_reward": 0.1,
    "PRM": True,
}

class RewardModel:
    def __init__(
        self,
        reward_config: Dict,
        Gd: DependencyGraph,
    ) -> None:
        self.Gd = Gd
        self.config = reward_config
        self.Gd_topo_node = {node.name : node.value for node in self.Gd.topo}
    
    def PRM(self, llm_output: str) -> List[float]: 
        # Process Reward Model
        # Calculate reward based on llm_output trajectory (one turn)
        
        sentence_list = llm_output.split('\n') # List["Define each Y9's X7 as b. So b = 9."]
        final_answer = sentence_list.pop() # "Thus, the answer is ..."
        final_answer_value = float(final_answer.split(' ')[-1]) 
        
        sentence_info_list = []
        for sentence in sentence_list:
            # The template string
            template = sentence
            # Define the regular expression pattern
            pattern = r"Define (.+?) as (\w+)\. So \2 = .* = (\d+)\."
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
                info = None
            
            sentence_info_list.append(info)
        
        reward_list = [0] * len(sentence_info_list)
        for i in range(len(sentence_info_list)):
            info = sentence_info_list[i]
            
            if info is None: 
                # This step is redundant
                reward_list[i] = self.config["redundant_step_reward"]
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
        if gt_ans == final_answer_value:
            reward_list.append(self.config["correct_step_reward"])
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
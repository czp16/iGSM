"""
implementation of the environment for the iGSM gym environment
the environment is a wrapper around the problem generation
The MDP includes
- state: query + all generated steps in answer
- action: next step of the answer
- reward: r(s,a) by the reward model
- transition: the next state is the current state + the next step
- terminal: TODO
"""

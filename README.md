# iGSM-gym: an RL env for LLM reasoning based on infinity GSM
(01/13) Update: The official repo of iGSM was released on 01/12, we will update our interface to fit the official repo.

We reproduce the generation of iGSM dataset and provide an RL interface for RL-based finetuning.

## Installation
```
cd iGSM
pip install -e .
```

You can generate a dataset by
```
python scripts/generate_dataset.py
```
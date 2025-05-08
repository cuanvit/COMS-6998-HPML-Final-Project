
# Training and Optimizing a Self-Created GPT and Fine-Tuned LLaMA for Domain Adaptation to Finance 

This project develops and benchmarks three large language models specialized for financial tasks:

1. A **custom GPT model** built from scratch using PyTorch and trained on transformed quantitative financial data.
2. A **custom optimzed GPT model** that optimizes the above model.
3. A **fine-tuned Meta LLaMA 3.2-3b model**, adapted for financial tasks using LoRA/qLoRA and other HPML (High-Performance Machine Learning) optimizations.

## Project Components

The **`own_gpt`** folder:

This contains the code for the custom gpt model.
The files in this folder have to be uploaded to google drive and then run the jupyter notebok to train the model. 
The path has to be changed to the google drive path.

**`model.py`:** contians the code for the gpt model.
**`utils.py`:** contains the code for to generate text.
**`data_utils.py`:** contains the code for the tokeizer and loading the dataset for the model.
**`finance_gpt_train.ipynb`:** for training the model on google collab.

The data is in the data folder


The **`own_gpt_optimized`** folder:

This contains the code for the custom optimized gpt model.
This model has to be trained in the same way as the previous model

**`model.py`:** contians the code for the gpt model.
**`utils.py`:** contains the code for to generate text.
**`data_utils.py`:** contains the code for the tokeizer and loading the dataset for the model.
**`finance_gpt_train.ipynb`:** for training the model on google collab.

The data is in the data folder

The llama_finetune:



We evaluate the impact of optimization techniques like LoRA, qLoRA, Torch.compile, mixed precision training, and KV cache improvements on model efficiency, latency, and performance.



# Training and Optimizing a Self-Created GPT and Fine-Tuned LLaMA for Domain Adaptation to Finance 

This project develops and benchmarks three large language models specialized for financial tasks:

1. A **custom GPT model** built from scratch using PyTorch and trained on transformed quantitative financial data.
2. A **custom optimzed GPT model** that optimizes the above model.
3. A **fine-tuned Meta LLaMA 3.2-3b model**, adapted for financial tasks using LoRA/qLoRA and other HPML (High-Performance Machine Learning) optimizations.

## Project Components

### From Scratch GPT

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

### LLaMA Fine-Tuning for Financial Domain Adaptation

This folder contains the code and output artifacts for **fine-tuning Meta's LLaMA 3.2-3B** model on financial news data using **LoRA/qLoRA** and high-performance ML techniques.

The fine-tuning leverages the [Unsloth](https://github.com/unslothai/unsloth) library to maximize throughput and minimize GPU memory usage.

---

#### Contents

- **`adapter_weights/`**  
  Contains LoRA adapter checkpoints saved after fine-tuning. These can be loaded for inference without modifying the base model.

- **`unsloth_a100_bs32_final.ipynb`**  
  The main notebook used for fine-tuning LLaMA 3.2-3B on an A100 GPU. This notebook includes:
  - Loading the 4-bit quantized base model
  - Applying LoRA adapters (rank = 16, alpha = 16)
  - Tokenizing and packing a finance news corpus
  - Running training for multiple epochs
  - Logging throughput, loss, and memory usage

---

#### Key Fine-Tuning Optimizations

- **LoRA (Low-Rank Adaptation):**  
  Trains only a small set of adapter parameters, reducing memory and compute cost.

- **qLoRA (4-bit Quantization):**  
  Loads the base model in 4-bit NF4 format, reducing VRAM usage by 75%.

- **Mixed Precision (bf16):**  
  All non-quantized layers use bf16, improving training speed without loss of accuracy.

- **Torch.compile:**  
  Uses PyTorch 2.0's graph compiler for faster training.

- **Packing:**  
  Packs multiple short sequences into fixed-length 128-token blocks for better GPU utilization.

- **KV Cache Enabled:**  
  Speeds up inference by reusing keys and values across tokens.

---

#### How to Use

1. **Upload the notebook to Colab** or run it in your local Jupyter environment with access to an A100 or compatible GPU.

2. Make sure `finance_corpus.txt` (cleaned financial news) is available in the dataset directory.

3. Run all cells in `unsloth_a100_bs32_final.ipynb` to:
   - Load model and tokenizer
   - Preprocess and tokenize data
   - Fine-tune using LoRA and optimizations
   - Save final adapter weights to `adapter_weights/`

4. For inference, use the saved LoRA weights like this:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-3B-bnb-4bit", 
    device_map="auto", 
    load_in_4bit=True, 
    torch_dtype="auto"
)

model = PeftModel.from_pretrained(
    base,
    "llama_finetune/adapter_weights/unsloth_a100_bs32_final"
)

tokenizer = AutoTokenizer.from_pretrained(
    "llama_finetune/adapter_weights/unsloth_a100_bs32_final"
)

model.eval()

def answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)



We evaluate the impact of optimization techniques like LoRA, qLoRA, Torch.compile, mixed precision training, and KV cache improvements on model efficiency, latency, and performance.


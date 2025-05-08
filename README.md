
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

### Dataset

**`llama_finetune/dataset/`** is the directory for preprocessing the dataset and dumping the corpus. We have scraped a list of all the S&P500 companies sorted by their market cap, and used the EODHD API to extract financial news on each of these companies. Eventually we ended up with **9285** articles.

Running 

```bash 
python preprocess.py
```
will extract all the news (you need an API KEY)

---

### LLaMA Fine-Tuning for Financial Domain Adaptation

**`llama_finetune/`** folder contains the code and output artifacts for **fine-tuning Meta's LLaMA 3.2-1B 4bit quantized** model on financial news data using **LoRA/qLoRA** and high-performance ML techniques.

The fine-tuning uses the [Unsloth](https://github.com/unslothai/unsloth) library to maximize throughput and minimize GPU memory usage.

---

#### Contents

- **`llama_finetune/adapter_weights/`**  
  Contains LoRA adapter checkpoints saved after fine-tuning. These can be loaded for inference without modifying the base model.

- **`llama_finetune/finetuning/llama_finance_finetune.ipynb`**  
  The main notebook used for fine-tuning LLaMA 3.2-1B on an A100 GPU. This notebook includes:
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
  Packs multiple short sequences into fixed-length 512-token blocks for better GPU utilization. Although currently Hugging face's packing is buggy so they have disabled it.

- **KV Cache Enabled:**  
  Speeds up inference by reusing keys and values across tokens. Unsloth uses this by default.

---

#### How to Use

1. **Upload the **`llama_finetune/finetuning/llama_finance_finetune.ipynb`** notebook to Colab** or run it in your local Jupyter environment with access to an A100 or compatible GPU. The colab file markdown has all the necessary instructions to run the file

2. Make sure **`finance_corpus.txt`** (cleaned financial news) is available in the dataset directory.

3. Run all cells in **`llama_finance_finetune.ipynb`** (using colab preferably unless you have access to NVIDIA GPUs locally) to:
   - Load model and tokenizer
   - Preprocess and tokenize data
   - Fine-tune using LoRA and optimizations
   - Save final adapter weights

4. If you are doing only inference first install unsloth.

```python
%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
    !pip install --no-deps unsloth

```


5. Then use the saved LoRA weights like this:

```python
import torch
from peft import prepare_model_for_kbit_training, PeftModel
from unsloth import FastLanguageModel


base, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Llama-3.2-1B-bnb-4bit",  
    max_seq_length = 128,
    dtype          = None,                   
    load_in_4bit   = True,
    device_map     = "auto",
)


tokenizer.pad_token = tokenizer.eos_token
base.config.pad_token_id = tokenizer.pad_token_id
base.config.use_cache      = True


base = prepare_model_for_kbit_training(base)


#-----THIS IS WHERE YOU PUT save_path AS THE FOLDER WHERE YOU SAVED THE WEIGHTS----#
model = PeftModel.from_pretrained(
    base,
    save_path,     # folder where we saved adapters + tokenizer
    device_map="auto",          
)


FastLanguageModel.for_inference(model)


def answer(prompt: str,
           max_new_tokens: int = 128,
           temperature: float    = 0.2,
           top_p: float          = 0.7,
           repetition_penalty: float = 1.2,
           no_repeat_ngram_size: int = 3):
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    input_ids = inputs["input_ids"]

    
    outputs = model.generate(
        **inputs,
        max_new_tokens       = max_new_tokens,
        temperature          = temperature,
        top_p                = top_p,
        do_sample            = True,
        repetition_penalty   = repetition_penalty,
        no_repeat_ngram_size = no_repeat_ngram_size,
        eos_token_id         = tokenizer.eos_token_id,
        pad_token_id         = tokenizer.pad_token_id,
        early_stopping       = True,
    )

    
    gen_ids = outputs[0][ input_ids.shape[-1] : ]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

```

We evaluate the impact of optimization techniques like LoRA, qLoRA, Torch.compile, mixed precision training, and KV cache improvements on model efficiency, latency, and performance.


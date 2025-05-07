# data_utils.py
import torch
from torch.utils.data import Dataset, random_split
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


class FinancialNewsDataset(Dataset):
    """
    Dataset for financial news text. Each item is (input_ids, target_ids) for next-token LM.
    """
    def __init__(self, file_paths, tokenizer, block_size=256):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.token_to_id("[PAD]")

        # Read and concatenate all input text
        all_text = ""
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                all_text += f.read() + "\n"
        # Tokenize entire corpus to IDs
        enc = tokenizer.encode(all_text)
        ids = enc.ids  # list of token IDs

        # Pad so that total length is a multiple of block_size
        n_pad = (block_size - (len(ids) % block_size)) % block_size
        ids += [self.pad_id] * n_pad

        # Convert to tensor of shape (num_seqs, block_size)
        ids = torch.tensor(ids, dtype=torch.long)
        self.num_sequences = len(ids) // block_size
        self.data = ids.view(self.num_sequences, block_size)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Returns (input_ids, target_ids) for one sequence.
        Target is input shifted by 1. Last token target is PAD (ignored in loss).
        """
        seq = self.data[idx]
        x = seq.clone()
        y = seq.clone()
        y[:-1] = x[1:]          # shift left
        y[-1] = self.pad_id     # no prediction for last token
        return x, y

def load_data(file_paths, tokenizer, block_size=256, train_frac=0.9):
    """
    Create train/validation datasets from text files.
    """
    dataset = FinancialNewsDataset(file_paths, tokenizer, block_size=block_size)
    # Randomly split into train/val
    n = len(dataset)
    n_train = int(train_frac * n)
    n_val = n - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    return train_dataset, val_dataset

# data_utils.py (excerpt)

def train_tokenizer(file_paths, vocab_size=10000, save_path="tokenizer.json"):
    """
    Train a BPE tokenizer on given text files and save to disk.
    """
    # Initialize BPE tokenizer with [UNK] token
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # Use whitespace pre-tokenization (split on spaces):contentReference[oaicite:4]{index=4}
    tokenizer.pre_tokenizer = Whitespace()
    # Set up trainer with special tokens [PAD] and [UNK]
    special_tokens = ["[PAD]", "[UNK]"]
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    # Train on the provided text files
    tokenizer.train(files=file_paths, trainer=trainer)  # trains BPE merges:contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}
    # Save tokenizer to JSON
    tokenizer.save(save_path)
    print(f"Tokenizer trained and saved to {save_path}")
    return tokenizer

def load_tokenizer(tokenizer_path):
    """
    Load a trained tokenizer from file.
    """
    return Tokenizer.from_file(tokenizer_path)


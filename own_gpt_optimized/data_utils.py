import os
import torch
from torch.utils.data import Dataset, random_split
import tiktoken


class FinancialNewsDataset(Dataset):
    """
    Dataset for financial news text. Each item is (input_ids, target_ids) for next-token LM.
    """
    def __init__(self, file_paths, tokenizer, block_size=256):
        self.block_size = block_size
        self.tokenizer = tokenizer

        # Read and concatenate all input text
        all_text = ""
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                all_text += f.read() + "\n"
        # Tokenize entire corpus to IDs
        ids = tokenizer.encode(all_text)
        self.pad_id = tokenizer.encode("<|pad|>")[0]

        # Pad so that total length is a multiple of block_size
        n_pad = (block_size - (len(ids) % block_size)) % block_size
        ids += [self.pad_id] * n_pad

        self.data = torch.tensor(ids, dtype=torch.long).view(-1, block_size)
        self.num_sequences = self.data.size(0)


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
        return x,y

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

def load_tokenizer(encoding_name):
    """
    Load a trained tokenizer from file.
    """
    return tiktoken.get_encoding(encoding_name)



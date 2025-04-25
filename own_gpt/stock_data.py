import yfinance as yf
import numpy as np
import numpy as np
import torch
from torch.utils.data import Dataset
from .quantizer import Quantizer

# minGPT/data/stock_dataset.py
class StockSequenceDataset(Dataset):
    def __init__(self, tickers, seq_len, split='train', clip=0.05, bins=256):
        """
        tickers: list of symbols, last one is held-out for zero-shot if split='test'
       seq_len: context window
        """
        quant = Quantizer(num_bins=bins, clip=clip)
        all_tokens = []
        for tk in tickers:
            data = yf.download(tk, period='2y', interval='1d')['Close'].values
            rets = np.diff(data) / data[:-1]
            all_tokens.append(quant.encode(rets))

        # split out the last ticker as zero-shot test
        if split == 'train':
            tokens = np.concatenate(all_tokens[:-1])
        else:
            tokens = all_tokens[-1]

        self.x, self.y = [], []
        for i in range(len(tokens)-seq_len):
            window = tokens[i:i+seq_len+1]
            self.x.append(window[:-1])
            self.y.append(window[1:])

        self.x = torch.tensor(self.x, dtype=torch.long)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

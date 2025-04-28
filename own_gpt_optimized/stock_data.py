import yfinance as yf
import numpy as np
import torch
from torch.utils.data import Dataset
from .tokenizer import TikTokenTokenizer

class StockDataset(Dataset):
    def __init__(self, tickers, seq_len, split='train'):
        quant = TikTokenTokenizer()
        all_tokens = []
        for tk in tickers:
            df = yf.download(tk, period='2y', interval='1d', auto_adjust=False, progress=False)
            rets = df['Close'].pct_change().dropna().to_numpy()
            all_tokens.append(quant.encode(rets))

        tokens = np.concatenate(all_tokens[:-1]) if split=='train' else all_tokens[-1]

        xs, ys = [], []
        for i in range(len(tokens) - seq_len):
            window = tokens[i : i + seq_len + 1]
            xs.append(window[:-1])
            ys.append(window[1:])

        xs = np.stack(xs, axis=0).astype(np.int64)
        ys = np.stack(ys, axis=0).astype(np.int64)

        x_tensor = torch.from_numpy(xs)
        y_tensor = torch.from_numpy(ys)

        # **KEY FIX**: collapse any stray singleton dims
        self.x = x_tensor.view(x_tensor.size(0), -1)
        self.y = y_tensor.view(y_tensor.size(0), -1)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# train_stock.py

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# adjust this import if your package folder is named min_gpt
from own_gpt.model       import GPT
from own_gpt.trainer     import Trainer, TrainerConfig
from own_gpt.quantizer   import Quantizer
from own_gpt.stock_data  import StockDataset

def main():
    # Read arguments
    parser = argparse.ArgumentParser(description="Zero-shot stock forecasting with minGPT")
    parser.add_argument('--tickers',   type=str,   default='AAPL,MSFT,GOOG,TSLA',
                        help='comma-separated list; last ticker held out for zero-shot eval')
    parser.add_argument('--seq_len',   type=int,   default=64,
                        help='context window length')
    parser.add_argument('--bins',      type=int,   default=256,
                        help='number of quantization bins')
    parser.add_argument('--clip',      type=float, default=0.05,
                        help='max |return| before clipping')
    parser.add_argument('--batch_size',type=int,   default=32)
    parser.add_argument('--max_epochs',type=int,   default=5)
    parser.add_argument('--lr',        type=float, default=5e-4)
    args = parser.parse_args()

    
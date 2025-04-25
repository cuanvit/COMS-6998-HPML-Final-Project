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

    # set up quantizer and load the data
    quant    = Quantizer(num_bins=args.bins, clip=args.clip)
    symbols  = args.tickers.split(',')
    train_ds = StockDataset(symbols, seq_len=args.seq_len,
                                    split='train', quantizer=quant)
    val_ds   = StockDataset(symbols, seq_len=args.seq_len,
                                    split='test',  quantizer=quant)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size)

    # build a small GPT
    model = GPT(
        vocab_size = quant.num_bins,
        block_size = args.seq_len,
        n_layer     = 4,
        n_head      = 4,
        n_embd      = 128
    )

    # config for trainer
    tconf = TrainerConfig(
        max_epochs   = args.max_epochs,
        batch_size   = args.batch_size,
        learning_rate= args.lr,
        lr_decay     = True,
        warmup_tokens= 10 * len(train_ds) * args.seq_len,
        final_tokens = len(train_ds) * args.seq_len * args.max_epochs
    )

    trainer = Trainer(model, train_dl, val_dl, tconf)
    print("Starting training")
    trainer.train()

    ckpt_path = "ckpt_stock.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")

if __name__ == '__main__':
    main()

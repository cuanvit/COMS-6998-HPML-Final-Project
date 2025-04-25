# train_stock.py

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np

from own_gpt.utils      import set_seed, CfgNode
from own_gpt.quantizer  import Quantizer
from own_gpt.stock_data import StockDataset
from own_gpt.model      import GPT
from own_gpt.trainer    import Trainer


def main():
    # # Read arguments
    # parser = argparse.ArgumentParser(description="Zero-shot stock forecasting with minGPT")
    # parser.add_argument('--tickers',   type=str,   default='AAPL,MSFT,GOOG,TSLA',
    #                     help='comma-separated list; last ticker held out for zero-shot eval')
    # parser.add_argument('--seq_len',   type=int,   default=64,
    #                     help='context window length')
    # parser.add_argument('--bins',      type=int,   default=256,
    #                     help='number of quantization bins')
    # parser.add_argument('--clip',      type=float, default=0.05,
    #                     help='max |return| before clipping')
    # parser.add_argument('--batch_size',type=int,   default=32)
    # parser.add_argument('--max_epochs',type=int,   default=5)
    # parser.add_argument('--lr',        type=float, default=5e-4)
    # args = parser.parse_args()
    set_seed(1234)
    tickers    = ['AAPL','MSFT','GOOG','TSLA']   # TSLA held-out
    seq_len    = 64
    batch_size = 32
    bins       = 256
    clip       = 0.05
    epochs     = 5
    lr         = 5e-4

    # set up quantizer and load the data
    quant    = Quantizer(num_bins=bins, clip=clip)
    # symbols  = args.tickers.split(',')
    train_ds = StockDataset(symbols = tickers, seq_len=seq_len,
                                    split='train', quantizer=quant)
    # val_ds   = StockDataset(symbols=tickers, seq_len=seq_len,
    #                                 split='test',  quantizer=quant)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # val_dl   = DataLoader(val_ds,   batch_size=batch_size)
    
    iters_per_epoch = len(train_ds) // batch_size
    ma_iters = epochs * iters_per_epoch

    # Make the GPT
    model_cfg = CfgNode()
    model_cfg.model_type = 'gpt-micro'   # 4-layer, 4-head, 128-dim
    model_cfg.vocab_size = bins
    model_cfg.block_size = seq_len
    model = GPT(model_cfg)

    # config for trainer
    config = Trainer.get_default_config()
    config.device = "auto"
    config.batch_size = batch_size
    config.learning_rate = lr
    config.max_iters = ma_iters
    
    trainer = Trainer(config, model, train_ds)
    
    print("Starting training")
    trainer.run()

    ckpt_path = "ckpt_stock.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")

if __name__ == '__main__':
    main()

# zero_shot_inference.py

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from own_gpt.utils      import set_seed
from own_gpt.stock_data import StockDataset
from own_gpt.model      import GPT
from own_gpt.quantizer  import Quantizer

def fetch_sp500_tickers():
    tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', header=0)
    for df in tables:
        if 'Symbol' in df.columns:
            return df['Symbol'].str.replace('.', '-', regex=False).tolist()
    raise RuntimeError("Could not find S&P 500 table on Wikipedia")

def main():
    # Setting up information
    set_seed(42)
    DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEQ_LEN, BINS  = 64, 64
    CLIP           = 0.05
    BATCH_SIZE     = 32
    CHECKPOINT_PATH = 'ckpt_stock_weighted.pth'

    # Fetch symbols & build test set
    symbols = fetch_sp500_tickers()
    print(f"Loaded {len(symbols)} tickers, zero-shot on: {symbols[-1]}")
    test_ds = StockDataset(
        tickers=symbols,
        seq_len=SEQ_LEN,
        split='test',
        clip=CLIP,
        bins=BINS
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Load model being used
    cfg = GPT.get_default_config()
    cfg.model_type  = 'gpt-mini'  # same as training
    cfg.vocab_size  = BINS
    cfg.block_size  = SEQ_LEN
    model = GPT(cfg).to(DEVICE)
    print("loading model")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print("evaluating model")
    model.eval()

    # load quantizer for decoding
    quant = Quantizer(num_bins=BINS, clip=CLIP)

    # zero-shot evaluation
    total_ce, total_mae, total_tokens = 0.0, 0.0, 0
    correct_dirs = 0
    
    print("Starting predictions")
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x, y)
            B, T, V = logits.shape

            # 1) cross-entropy
            ce = F.cross_entropy(logits.view(B*T, V), y.view(B*T), reduction='sum')
            total_ce += ce.item()

            # 2) MAE + directional accuracy in real-return space
            # Shape for prediction tensors is (B, T)
            preds = logits.argmax(-1).cpu().numpy()
            trues = y.cpu().numpy()
            pred_rets = quant.decode(preds)
            true_rets = quant.decode(trues)
            total_mae += np.sum(np.abs(pred_rets - true_rets))
            correct_dirs += np.sum(np.sign(pred_rets) == np.sign(true_rets))

            total_tokens += B * T
    
    avg_ce  = total_ce  / total_tokens
    avg_mae = total_mae / total_tokens
    dir_acc = correct_dirs / total_tokens

    print(f"\nZero-shot results on {symbols[-1]}:")
    print(f"Cross-Entropy:       {avg_ce:.4f}")
    print(f" MAE (in return %):  {avg_mae:.4f}")
    print(f"Directional Acc.:   {dir_acc*100:.2f}%\n")

    # ── zero-shot forecasting ──────────────────────────────
    # take the *last* window of real test data, generate next 10 returns
    last_window = test_ds.x[-1].unsqueeze(0).to(DEVICE)  # shape (1, SEQ_LEN)
    generated = model.generate(
        last_window,
        max_new_tokens=10,
        do_sample=True,
        top_k=5
    )  # shape (1, SEQ_LEN + 10)
    new_ids = generated[0, -10:].cpu().numpy()
    new_rets = quant.decode(new_ids)

    print("Zero-shot forecast — next 10 returns (%):")
    print(np.round(new_rets * 100, 2))

if __name__ == '__main__':
    main()

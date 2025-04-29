import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import pandas as pd

from own_gpt.utils      import set_seed
from own_gpt.stock_data import StockDataset
from own_gpt.model      import GPT

def fetch_sp500_tickers():
    tables = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        header=0
    )
    for df in tables:
        if 'Symbol' in df.columns:
            return df['Symbol'].str.replace('.', '-', regex=False).tolist()
    raise RuntimeError("Couldn't find S&P 500 table")

def train_model():
    set_seed(42)

    # Load tickers
    symbols = fetch_sp500_tickers()
    print(f"Loaded {len(symbols)} tickers")

    # Hyperparameters: Change to args later
    seq_len    = 64
    bins       = 64
    clip       = 0.05
    batch_size = 32
    epochs     = 5
    lr         = 1e-3

    # Dataset & class weights
    train_ds = StockDataset(
        tickers=symbols,
        seq_len=seq_len,
        split='train',
        clip=clip,
        bins=bins
    )

    # compute per-class inverse-frequency weights
    all_labels   = train_ds.y.view(-1).numpy()
    counts       = np.bincount(all_labels, minlength=bins).astype(float)
    class_weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32)
    class_weights = (class_weights / class_weights.sum() * bins)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights = class_weights.to(device)

    # Weighted sampling
    inv_freq = class_weights.cpu().numpy()
    window_weights = []
    for window in train_ds.y.numpy():
        window_weights.append(inv_freq[window].mean())
    window_weights = torch.DoubleTensor(window_weights)

    sampler = WeightedRandomSampler(
        weights=window_weights,
        num_samples=len(window_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # Model setup (no dropout as our loss wasn't good with dropout) 
    cfg = GPT.get_default_config()
    cfg.model_type   = 'gpt-mini'
    cfg.vocab_size   = bins
    cfg.block_size   = seq_len
    cfg.embd_pdrop   = 0.0
    cfg.resid_pdrop  = 0.0
    cfg.attn_pdrop   = 0.0

    model = GPT(cfg).to(device)

    # Optimizer + LR schedule
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    total_steps = epochs * len(train_loader)
    warmup      = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        return max(0.0, (total_steps - step) / (total_steps - warmup))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Training loop
    model.train()
    step, running_loss = 0, 0.0
    for epoch in range(1, epochs+1):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # forward + weighted cross-entropy for better loss calculation
            logits, _ = model(x)      
            B, T, V    = logits.shape
            loss = F.cross_entropy(
                logits.view(B*T, V),
                y.view(B*T),
                weight=class_weights
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Logging to check. Can comment out if loss looks ok
            step += 1
            running_loss += loss.item()
            if step % 100 == 0:
                avg = running_loss / 100
                print(f"[{step:5d}/{total_steps:5d}] "
                      f"Epoch {epoch}/{epochs}  avg loss: {avg:.4f}")
                running_loss = 0.0

    # Save checkpoint so we can use it
    torch.save(model.state_dict(), 'ckpt_stock_weighted.pth')
    print("Training complete, checkpoint saved to ckpt_stock_weighted.pth")

if __name__ == '__main__':
    train_model()





    # # Read arguments. Add this stuff in later
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
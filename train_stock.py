import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from own_gpt.utils      import set_seed
from own_gpt.stock_data import StockDataset
from own_gpt.model      import GPT
import pandas as pd

def main():

    # 1) Grab all the tables on the page…
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url, header=0)

    # 2) Select the one that actually has the “Symbol” column  
    for df in tables:
        if 'Symbol' in df.columns:
            sp500 = df
            break

    symbols = sp500['Symbol'].str.replace('.', '-', regex=False).tolist()

    set_seed(42)

    # hyperparameters. replace with args later
    seq_len    = 64
    bins       = 256
    clip       = 0.05
    batch_size = 32
    epochs     = 10
    lr         = 1e-3

    # data loader setup
    train_ds = StockDataset(
        tickers=symbols,
        seq_len=seq_len,
        split='train',
        clip=clip,
        bins=bins
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # # Checking shapes to make sure everything works
    # x0, y0 = train_ds[0]
    # print(f"▶ Sample shapes: x0 {tuple(x0.shape)}, y0 {tuple(y0.shape)}")


    # Setting up the model
    cfg = GPT.get_default_config()
    cfg.model_type = 'gpt-mini'
    cfg.vocab_size = bins
    cfg.block_size = seq_len
    model = GPT(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # training loop 
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            # batch shapes
            # x: (batch_size, seq_len)
            # y: (batch_size, seq_len)
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                avg100 = running_loss / 100
                print(f"Epoch {epoch+1}/{epochs}, batch {batch_idx}, avg loss over last 100 batches: {avg100:.4f}")
                running_loss = 0.0

    # ── save checkpoint ───────────────────────────────────
    torch.save(model.state_dict(), 'ckpt_stock.pth')
    print("Training complete, checkpoint saved to chpt_stock.pth")

if __name__ == '__main__':
    main()




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
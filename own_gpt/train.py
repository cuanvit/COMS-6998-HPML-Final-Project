# train.py
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import MiniGPT
from data_utils import train_tokenizer, load_tokenizer, load_data

def train(model, train_loader, val_loader, device, pad_id, epochs=5, lr=3e-4):
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        # Training
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)  # (B, T, vocab)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} - Training loss: {avg_loss:.4f}, Perplexity: {torch.exp(torch.tensor(avg_loss)):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} - Validation loss: {avg_val_loss:.4f}, Perplexity: {torch.exp(torch.tensor(avg_val_loss)):.4f}")

    print("Training complete.")

if __name__ == "__main__":
    # Paths and hyperparameters
    data_files = ["data/financial_news.txt"]    # list of training text files
    tokenizer_path = "tokenizer_finance.json"
    vocab_size = 10000
    block_size = 256

    # Train or load tokenizer
    if not os.path.exists(tokenizer_path):
        tokenizer = train_tokenizer(data_files, vocab_size=vocab_size, save_path=tokenizer_path)
    else:
        tokenizer = load_tokenizer(tokenizer_path)
    pad_id = tokenizer.token_to_id("[PAD]")

    # Load datasets
    train_dataset, val_dataset = load_data(data_files, tokenizer, block_size=block_size)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Initialize model and start training
    model = MiniGPT(vocab_size=vocab_size, block_size=block_size, n_layer=4, n_head=4, n_embd=256, dropout=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train(model, train_loader, val_loader, device, pad_id, epochs=10, lr=3e-4)

    # Save the trained model
    torch.save(model.state_dict(), "finance_gpt.pth")
    print("Model saved to finance_gpt.pth")

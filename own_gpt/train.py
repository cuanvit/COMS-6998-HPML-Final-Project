import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from model import GPTModel
from gpt_config import GPTConfig
from data_utils import train_tokenizer, load_tokenizer, load_data
from utils import generate_text, calculate_perplexity

# Set up paths

def train(epochs = 10, lr = 3e-4):
    data_file = "data/finance_corpus.txt"
    tokenizer_path = "tokenizer_finance.json"

    # Load tokenizer or train a new one
    if not os.path.exists(tokenizer_path):
        tokenizer = train_tokenizer([data_file], vocab_size=10000, save_path=tokenizer_path)
    else:
        tokenizer = load_tokenizer(tokenizer_path)

    pad_id = tokenizer.token_to_id("[PAD]")

    # Load data
    block_size = 256
    vocab_size = 10000
    train_dataset, val_dataset = load_data(data_file, tokenizer, block_size)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Model setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = GPTConfig(vocab_size=vocab_size, block_size=block_size)
    model = GPTModel(config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # Training loop
    total_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_start = time.time()

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_duration = time.time() - epoch_start
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Epoch Time = {epoch_duration:.2f} seconds")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/gpt_finance.pt")

    # Total training time
    total_time = time.time() - total_start_time
    print(f"Total training time: {total_time:.2f} seconds")


if __name__ == "__main__":
    train()


# utils.py
import torch
import torch.nn.functional as F

def generate_text(model, tokenizer, prompt, max_length=50, top_k=0, device=None):
    """
    Generate text given a prompt. Supports greedy (top_k=0) or top-k sampling.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # Encode prompt to IDs
    input_ids = tokenizer.encode(prompt).ids.copy()  # get list of token IDs

    for _ in range(max_length):
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_tensor)  # (1, seq_len, vocab)
        logits = logits[0, -1, :]        # last token's logits
        probs = F.softmax(logits, dim=-1)

        if top_k > 0:
            # Top-k filtering
            top_probs, top_idx = torch.topk(probs, top_k)
            top_probs = top_probs / top_probs.sum()  # renormalize
            next_id = top_idx[torch.multinomial(top_probs, 1)].item()
        else:
            # Greedy
            next_id = torch.argmax(probs).item()
        input_ids.append(next_id)

    text = tokenizer.decode(input_ids)
    return text

def calculate_perplexity(model, data_loader, pad_id, device=None):
    """
    Compute model perplexity on a DataLoader.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
            total_tokens += (targets != pad_id).sum().item()
    # Perplexity = exp(total_loss / total_tokens)
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

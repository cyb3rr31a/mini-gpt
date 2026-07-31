import torch
import torch.nn as nn
import urllib.request
import os
import tiktoken # type:ignore

from app.model.architecture import MiniGPT
from app.model.generator import generative_naive

# Download Tiny Shakespeare
data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_path = "input.txt"

if not os.path.exists(data_path):
    print("Downloading Tiny Shakespeare...")
    urllib.request.urlretrieve(data_url, data_path)

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

# BPE Tokenizer
print("Encoding dataset with BPE...")
enc = tiktoken.get_encoding("gpt2")

# Convert entire dataset to a PyTorch tensor
data = torch.tensor(enc.encode(text), dtype=torch.long)
vocab_size = enc.n_vocab

print(f"Dataset has {len(text)} characters. Vocabulary size: {vocab_size}")

# Split into train and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Hyperparameters
BATCH_SIZE = 16
SEQ_LENGTH = 128
MAX_ITERS = 5000
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Training on {DEVICE}...")

# Initialize model
model = MiniGPT(vocab_size=vocab_size, embed_size=384, max_seq_length=SEQ_LENGTH, num_layers=6, heads=6)
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

def get_batch(split):
    dataset = train_data if split == 'train' else val_data
    ix = torch.randint(len(dataset) - SEQ_LENGTH, (BATCH_SIZE,))
    x = torch.stack([dataset[i:i+SEQ_LENGTH] for i in ix])
    y = torch.stack([dataset[i+1:i+SEQ_LENGTH+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# Training Loop
model.train()
for iter in range(MAX_ITERS):
    xb, yb = get_batch('train')

    # Forward pass
    logits, _ = model(xb)

    # Calculate loss
    B, T, C = logits.shape
    logits_flat = logits.view(B * T, C)
    yb_flat = yb.view(B * T)
    loss = nn.functional.cross_entropy(logits_flat, yb_flat)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print(f"Step {iter} | Loss: {loss.item():.4f}")

# Save trained weights
torch.save(model.state_dict(), "minigpt_bpe.pth")
print("Training complete! Weights saved to minigpt_weights.pth")

"""
# Quick Test Generation
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
context = generative_naive(model=model, context=context, max_new_tokens=200)

print("\n--- Generate Text ---")
print(decode(context[0].tolist()))
"""
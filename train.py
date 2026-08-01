import os
import math
import urllib.request

import torch
import torch.nn as nn
import tiktoken # type:ignore

from app.model.architecture import MiniGPT
from app.model.generator import generative_naive

torch.manual_seed(1337)

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
MIN_LR = LEARNING_RATE * 0.1
WARMUP_ITERS = 100
EVAL_INTERVAL = 250
EVAL_ITERS = 50
GRAD_CLIP = 1.0
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

def compute_loss(logits, targets):
    B, T, C = logits.shape # batch size, sequence length, vocabulary size
    return nn.CrossEntropyLoss()(logits.view(B * T, C), targets.view(B * T))

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            logits, _ = model(xb)
            losses[k] = compute_loss(logits, yb).item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it):
    # Linear warmup then cosine decay down to MIN_LR
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / WARMUP_ITERS
    progress = (it - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# Training Loop
model.train()
best_val_loss = float('inf')

for iter in range(MAX_ITERS):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    xb, yb = get_batch('train')

    # Forward pass
    logits, _ = model(xb)

    # Calculate loss
    loss = compute_loss(logits, yb)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    if iter % 100 == 0:
        print(f"Step {iter} | Loss: {loss.item():.4f}")

    # Periodic evaluation
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"Step {iter} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'vocab_size': vocab_size,
                    'embed_size': 384,
                    'max_seq_length': SEQ_LENGTH,
                    'num_layers': 6,
                    'heads': 6,
                    'iter': iter,
                    'val_loss': best_val_loss
                },
                "bpe_weights.pth",
            )
            print(f"New best model saved at step {iter} with val loss {best_val_loss:.4f}")

# Save final weights
torch.save(
    {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'embed_size': 384,
        'max_seq_length': SEQ_LENGTH,
        'num_layers': 6,
        'heads': 6
    },
    "minigpt_weights.pth"
)
print("Training complete! Final weights saved to minigpt_weights.pth")
print(f"Best validation loss achieved: {best_val_loss:.4f}")

# Quick Test Generation
model.eval()
prompt = "ROMEO:"
prompt_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long).to(DEVICE)

genrated_ids = generative_naive(model=model, context=prompt_ids, max_new_tokens=200)
generated_text = enc.decode(genrated_ids[0].tolist())

print("\n--- Generated Text ---")
print(generated_text)
import torch
import torch.nn as nn
from torch.nn import functional as F

# 1: Token and Positional Embedding
class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_size, max_seq_length, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_seq_length, embed_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, start_pos=0):
        seq_length = x.size(1)
        positions = torch.arange(start_pos, start_pos + seq_length, device=x.device)
        return self.drop(self.token_embedding(x) + self.position_embedding(positions))

# 2: Causal Self-Attention
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # The Q,K,V linear transformation
        self.qkv = nn.Linear(embed_size, embed_size * 3)
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, layer_cache=None):
        # Batch, Time(sequence length), channels(embed_size)
        B, T, C = x.size() 

        # Generate Q, K, V for all tokens
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.embed_size, dim=2)

        # KV Cache Implementation
        if layer_cache is not None:
            past_k, past_v = layer_cache

            # Append the current K and V to the previous ones
            k = torch.cat((past_k, k), dim=1)
            v = torch.cat((past_v, v), dim=1)

        new_layer_cache = (k, v)

        # Calculate attention scores (Q dot K)
        # Transpose K to align dimensions for matrix multiplication
        T_total = k.size(1)
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)

        # The Causal Mask
        if T > 1:
            mask = torch.tril(torch.ones(T, T)).to(x.device)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Convert to probabilities (softmax) and multiply by Values
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attn_drop(attention_weights)
        out = attention_weights @ v

        return self.resid_drop(self.fc_out(out)), new_layer_cache

# 3 : The Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super().__init__()
        # Expand hidden layer by 4x then project back
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# 4. Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super().__init__()
        self.attention = CausalSelfAttention(embed_size, heads, dropout)
        self.ffn = FeedForward(embed_size, dropout)

        # Normalized layers
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x, layer_cache=None):
        attn_out, updated_cache = self.attention(self.ln1(x), layer_cache)
        # Apply attention, add residual, apply norm
        x = x + attn_out
        # Apply FFN, add residual, apply norm
        x = x + self.ffn(self.ln2(x))
        return x, updated_cache

# Full mini-gpt
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, max_seq_length, num_layers, heads, dropout=0.1):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, embed_size, max_seq_length, dropout)

        # stack multiple blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_size, heads, dropout) for _ in range(num_layers)]
        )

        self.ln_f = nn.LayerNorm(embed_size) #final normalization
        self.lm_head = nn.Linear(embed_size, vocab_size) #predict next token

    def forward(self, x, past_kv_cache=None):
        if past_kv_cache is not None:
            start_pos = past_kv_cache[0][0].size(1)
        else:
            start_pos = 0
 
        x = self.embeddings(x, start_pos)
 
        new_kv_cache = []
 
        for i, block in enumerate(self.blocks):
            layer_cache = past_kv_cache[i] if past_kv_cache is not None else None
 
            x, updated_cache = block(x, layer_cache)
            new_kv_cache.append(updated_cache)
 
        x = self.ln_f(x)
        logits = self.lm_head(x) 
 
        return logits, new_kv_cache
import torch
import torch.nn.functional as F

# Naive Generator
@torch.no_grad()
def generative_naive(model, context, max_new_tokens):
    # Dynamically find out what the model's max sequence length is
    max_seq_length = model.embeddings.position_embedding.num_embeddings

    # context in a 2D tensor of token IDs
    for _ in range(max_new_tokens):
        context_cropped = context[:, -max_seq_length:]
        # Pass entire sequence through model each time
        logits, _ = model(context_cropped)

        # Pluck the logits
        next_token = logits[:, -1, :]

        # Greedily pick the most likely next token
        next_token = torch.argmax(next_token, dim=-1, keepdim=True)

        # Append to the context
        context = torch.cat((context, next_token), dim=1)

    return context

@torch.no_grad()
def generative_cached(model, context, max_new_tokens):
    max_seq_length = model.embeddings.position_embedding.num_embeddings
 
    def prime(ctx):
        # Fresh forward pass with no cache, over up to max_seq_length tokens.
        ctx_cropped = ctx[:, -max_seq_length:]
        logits, kv_cache = model(ctx_cropped)
        cached_length = ctx_cropped.size(1)
        return logits, kv_cache, cached_length
 
    # Initial prefill over the prompt
    logits, kv_cache, cached_length = prime(context)
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    context = torch.cat((context, next_token), dim=1)
 
    for _ in range(max_new_tokens - 1):
        if cached_length >= max_seq_length:
            logits, kv_cache, cached_length = prime(context)
        else:
            logits, kv_cache = model(next_token, past_kv_cache=kv_cache)
            cached_length += 1
 
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        context = torch.cat((context, next_token), dim=1)
 
    return context
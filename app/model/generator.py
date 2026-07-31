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
def generate_cached(model, context, max_new_tokens, temperature=0.8):
    kv_cache = None
    generated_sequence = context

    max_seq_length = model.embeddings.position_embedding.num_embeddings

    for _ in range(max_new_tokens):
        # Forward pass: returns logits and the updated cache
        context_cropped = context[:, -max_seq_length:]
        logits, kv_cache = model(context_cropped, kv_cache)
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / temperature

        # Convert to probabilities
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1) # roll weighted die

        # Append to final output list
        generated_sequence = torch.cat((generated_sequence, next_token), dim=1)
        context = next_token

    return generated_sequence
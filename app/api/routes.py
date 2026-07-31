import asyncio
import torch
import tiktoken # type:ignore
from fastapi import APIRouter # type:ignore
from pydantic import BaseModel # type:ignore
from app.model.generator import generate_cached
from app.model.architecture import MiniGPT

router = APIRouter()

# Server Configuration
MAX_BATCH_SIZE = 8
MAX_WAIT_TIME = 0.05
request_queue = asyncio.Queue()

# BPE Tokenizer
enc = tiktoken.get_encoding("gpt2")

def encode(s: str) -> list[int]:
    return enc.encode(s)

def decode(l: list[int]) -> str:
    return enc.decode(l)

model = MiniGPT(vocab_size=50257, embed_size=384, max_seq_length=128, num_layers=6, heads=6)
model.load_state_dict(torch.load("minigpt_bpe.pth", weights_only=True))
model.eval()

class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

class QueueItem:
    def __init__(self, prompt_tokens, max_new_tokens):
        self.prompt_tokens = prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.result = None
        self.event = asyncio.Event()

async def batching_worker():
    while True:
        batch = []
        item = await request_queue.get()
        batch.append(item)

        end_time = asyncio.get_event_loop().time() + MAX_WAIT_TIME

        while len(batch) < MAX_BATCH_SIZE:
            remaining_time = end_time - asyncio.get_event_loop().time()
            if remaining_time <= 0:
                break
            try:
                next_item = await asyncio.wait_for(request_queue.get(), timeout=remaining_time)
                batch.append(next_item)
            except asyncio.TimeoutError:
                break

        print(f"Processing batch of size {len(batch)}")

        try:
            # Enforce all prompts to be same length
            input_tensor = torch.tensor([req.prompt_tokens for req in batch])

            # Call the generator
            batched_output = await asyncio.to_thread(generate_cached, model, input_tensor, 20)

            for i, queued_item in enumerate(batch):
                queued_item.result = batched_output[i].tolist()
                queued_item.event.set()

        except Exception as e:
            print(f"Worker Error: {e}")
            # If it crashes, tell every waiting request about the error!
            for queued_item in batch:
                queued_item.result = {"error": str(e)}
                queued_item.event.set()

@router.post("/generate")
async def generate_text(req: InferenceRequest):
    # Translate text string into integers
    prompt_tokens = encode(req.prompt)
    item = QueueItem(prompt_tokens, req.max_new_tokens)
    await request_queue.put(item)
    await item.event.wait()

    # Error handling
    if isinstance(item.result, dict) and "error" in item.result:
        return item.result

    if item.result is None:
        return {"error": "No result available"}

    generated_text = decode(item.result)
    return {"generated_tokens": generated_text}
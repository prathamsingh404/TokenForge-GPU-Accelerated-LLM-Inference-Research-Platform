"""
FastAPI Dashboard for the LLM Inference Optimization Laboratory.
Serves the web UI and provides streaming inference and telemetry endpoints.
"""

import asyncio
import json
import time
from typing import AsyncGenerator

import torch
import pynvml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from threading import Lock

from core.config import get_config

app = FastAPI(title="LLM Inference Lab Dashboard")

# Ensure templates directory exists relative to this file
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Global State
class EngineState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.lock = Lock()
        
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
        except Exception:
            self.handle = None
            self.gpu_name = "Unknown CPU/GPU"

engine = EngineState()

def load_model():
    """Load model in background if not already loaded."""
    if engine.is_loaded:
        return
    with engine.lock:
        if engine.is_loaded:
            return
        
        cfg = get_config()
        model_name = cfg.models.small_model
        print(f"Loading {model_name}...")
        
        # Fast load on CUDA
        engine.tokenizer = AutoTokenizer.from_pretrained(model_name)
        engine.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        engine.is_loaded = True
        print("Model loaded successfully.")


@app.on_event("startup")
async def startup_event():
    # Trigger async model load to not block server startup
    asyncio.create_task(asyncio.to_thread(load_model))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/stats")
async def get_stats():
    """Hardware and environment stats for the dashboard."""
    stats = {
        "env": {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
        },
        "gpu": None
    }
    
    if engine.handle:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(engine.handle)
            stats["gpu"] = {
                "name": engine.gpu_name,
                "vram_used_mb": info.used / 1024**2,
                "vram_total_mb": info.total / 1024**2
            }
        except Exception:
            pass
            
    return stats


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7


async def generate_stream(prompt: str, max_new_tokens: int) -> AsyncGenerator[str, None]:
    """Streaming generator for the chat interface."""
    # Wait for model to load if requested early
    while not engine.is_loaded:
        yield f"data: {json.dumps({'token': '...loading model...'})}\n\n"
        await asyncio.sleep(1)
        
    input_ids = engine.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    
    # We use a simple loop with torch.no_grad for streaming 
    # (HuggingFace's TextIteratorStreamer is better for production, but this demonstrates the concept clearly)
    past_key_values = None
    generated_ids = input_ids.clone()
    
    for _ in range(max_new_tokens):
        # Must yield control to event loop so we don't block other requests
        await asyncio.sleep(0.001)
        
        with torch.no_grad():
            if past_key_values is None:
                outputs = engine.model(generated_ids, use_cache=True)
            else:
                outputs = engine.model(
                    generated_ids[:, -1:], 
                    past_key_values=past_key_values, 
                    use_cache=True
                )
                
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            # Decode just the new token
            token_str = engine.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            
            if next_token_id.item() == engine.tokenizer.eos_token_id:
                break
                
            if token_str:
                yield f"data: {json.dumps({'token': token_str})}\n\n"


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """Endpoint for streaming text generation."""
    return StreamingResponse(
        generate_stream(req.prompt, req.max_new_tokens), 
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

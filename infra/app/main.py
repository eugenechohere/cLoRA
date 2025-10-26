from __future__ import annotations

import json
import random
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests
from fastapi import Body, FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RECENT_EXAMPLES_PATH = DATA_DIR / "recent_examples.jsonl"
WORKFLOW_LOG_PATH = Path("/home/ubuntu/calhacks-continual-learning/infra/workflow.log")

# Single source of truth for batch size
BATCH_SIZE = 512
TRAIN_TRIGGER_THRESHOLD = BATCH_SIZE
TRAINING_SERVICE_URL = "http://0.0.0.0:8001/train-and-update"
VLLM_URL = "http://0.0.0.0:8000/v1/completions"

# Global variable to store latest loss (updated by background thread)
# No lock needed - reading/writing a float in Python is atomic
latest_loss_value: Optional[float] = None

app = FastAPI()


def tail_log_file():
    """Background thread that tails the log file and updates latest_loss_value."""
    global latest_loss_value
    
    if not WORKFLOW_LOG_PATH.exists():
        print(f"Warning: Log file {WORKFLOW_LOG_PATH} does not exist. Loss monitoring disabled.")
        return
    
    pattern = r"\{'loss':\s*([\d.]+)"
    
    try:
        # Start tail -f process
        process = subprocess.Popen(
            ["tail", "-f", "-n", "100", str(WORKFLOW_LOG_PATH)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        print(f"Started tailing log file: {WORKFLOW_LOG_PATH}")
        
        # Read lines as they come
        while True:
            line = process.stdout.readline()
            if not line:
                break
            match = re.search(pattern, line)
            if match:
                loss_val = float(match.group(1))
                latest_loss_value = loss_val  # Atomic write in Python
                print(f"Updated loss: {loss_val}", flush=True)
                
    except Exception as e:
        print(f"Error in tail_log_file: {e}")


@app.on_event("startup")
async def startup_event():
    """Start the background thread when FastAPI starts."""
    thread = threading.Thread(target=tail_log_file, daemon=True)
    thread.start()
    print("Background log tailer started")

class InferenceRequest(BaseModel):
    prompt: str

@app.post("/upload")
async def upload_example(
    payloads: List[Dict[str, str]] = Body(...),
    background_tasks: BackgroundTasks = None,
) -> Dict[str, Any]:
    """
    Append prompt/completion pairs and, whenever possible, emit as many full
    BATCH_SIZE training files as can be formed (each exactly BATCH_SIZE examples),
    triggering one training request per file. Any remainder (< BATCH_SIZE)
    stays in RECENT_EXAMPLES_PATH.
    """
    total_after_append, appended = _append_examples(RECENT_EXAMPLES_PATH, payloads)
    print(total_after_append, appended)

    train_files: List[Path] = []

    # If we have at least one full batch, cut all possible batches and trigger training per batch
    if total_after_append >= TRAIN_TRIGGER_THRESHOLD:
        train_files, remaining = _prepare_training_batches(
            RECENT_EXAMPLES_PATH, TRAIN_TRIGGER_THRESHOLD
        )
        print(train_files, remaining)
        

        # Fire-and-forget training requests in background to avoid blocking the server
        for batch_path in train_files:
            if background_tasks is not None:
                background_tasks.add_task(_trigger_training, batch_path)
            else:
                # Fallback (shouldn't happen in FastAPI) – run synchronously
                _trigger_training(batch_path)

        total_after_append = remaining

    return {
        "status": "ok",
        "count": len(payloads),
        "appended": appended,
        "pending_examples": total_after_append,   # remainder kept in persistent JSONL
        "training_triggered": bool(train_files),
        "training_requests": len(train_files),
        "train_files": [str(p) for p in train_files],
    }



def _trigger_training(train_file: Path) -> bool:
    """
    Notify the training service about a ready-to-train file.
    Requests are independent; success for one does not depend on others.
    """
    print(f"Triggering training for {train_file}")
    response = requests.post(
        TRAINING_SERVICE_URL,
        json={
            "data_path": str(train_file),
        },
        timeout=300,  # 10 minutes - training can take a while
    )
    response.raise_for_status()
    data = response.json()
    print(data)
    return data



def _append_examples(path: Path, payloads: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Append valid {prompt, completion} pairs as JSONL to `path`.

    Returns:
        total_count_after_append, appended_count
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Count current lines then append
    current_total = 0
    with path.open("r", encoding="utf-8") as h:
        current_total = sum(1 for line in h if line.strip())

    with path.open("a", encoding="utf-8") as h:
        for payload in payloads:
            h.write(json.dumps(payload, ensure_ascii=False))
            h.write("\n")

    return current_total + len(payloads), len(payloads)


def _prepare_training_batches(path: Path, batch_size: int) -> Tuple[List[Path], int]:
    """
    Split `path` into as many full batches (each exactly `batch_size` lines) as possible.
    Create a sandboxed file per batch. Remaining (< batch_size) lines are written back
    to `path`.

    Strategy: take batches from the end (most-recent-first) to form files; the remainder
    stays in `path`. Requests are independent of order.
    """
    
    with path.open("r", encoding="utf-8") as h:
        lines = [ln for ln in h if ln.strip()]

    total = len(lines)
    assert total >= batch_size, "Not enough examples to form a batch"

    num_batches = total // batch_size
    remainder_count = total % batch_size

    # indices for the chunked region we’ll export into batch files
    export_start = total - (num_batches * batch_size)
    export_region = lines[export_start:]  # length == num_batches * batch_size

    batch_files: List[Path] = []
    timestamp_base = datetime.utcnow().strftime("%Y%m%d_%H%M%S%f")

    # Create files in order, each with exactly batch_size lines
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_lines = export_region[start:end]

        batch_path = path.with_name(f"train_batch_{timestamp_base}_{i+1:03d}.jsonl")
        with batch_path.open("w", encoding="utf-8") as bh:
            bh.writelines(batch_lines)
        batch_files.append(batch_path)

    # Write remainder back to persistent jsonl (the older lines)
    remainder_lines = lines[:export_start]
    with path.open("w", encoding="utf-8") as h:
        h.writelines(remainder_lines)

    return batch_files, remainder_count


@app.get("/get_data")
async def get_data(samples_per_batch: int = 10) -> List[List[Dict[str, Any]]]:
    """
    Returns randomly sampled data from finalized training batches only.
    
    Args:
        samples_per_batch: Number of samples to randomly select from each batch (default: 10)
    
    Returns:
        List of lists, where each inner list contains N random samples from a batch.
        Example: [[sample1, sample2, ...], [sample1, sample2, ...], ...]
    """
    result = []
    
    # Get all train_batch files sorted by timestamp (latest first)
    batch_files = sorted(
        DATA_DIR.glob("train_batch_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True  # Latest first
    )
 
    # For each batch file, read all data and randomly sample N examples
    for batch_file in batch_files:
        batch_data = []
        with batch_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        batch_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        # Randomly sample examples from this batch (or all if less than requested)
        sample_size = min(samples_per_batch, len(batch_data))
        if sample_size > 0:
            sampled = random.sample(batch_data, sample_size)
            result.append(sampled)
    
    return result


@app.get("/latest_loss")
async def get_latest_loss() -> Dict[str, Any]:
    """
    Returns the latest loss value from memory (updated by background tail process).
    This is instant - no file I/O on each request!
    
    Returns:
        {"loss": float} - The most recent loss value from training
    """
    if latest_loss_value is None:
        raise HTTPException(status_code=404, detail="No loss value available yet")
    
    return {"loss": latest_loss_value}


@app.post("/infer")
async def infer(request: InferenceRequest):
    payload = {
        "model": "Qwen/Qwen3-8B",
        "prompt": request.prompt,
        "max_tokens": 512,
        "temperature": 0.7,
    }

    try:
        resp = requests.post(VLLM_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return {"output": data["choices"][0]["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

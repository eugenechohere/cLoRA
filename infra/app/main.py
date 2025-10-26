from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from fastapi import Body, FastAPI


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RECENT_EXAMPLES_PATH = DATA_DIR / "recent_examples.jsonl"

# Single source of truth for batch size
BATCH_SIZE = 64
TRAIN_TRIGGER_THRESHOLD = BATCH_SIZE
TRAINING_SERVICE_URL = "http://127.0.0.1:9000/train"

app = FastAPI()

@app.post("/upload")
async def upload_example(payloads: List[Dict[str, Any]] = Body(...)) -> Dict[str, Any]:
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

        # Fire independent requests (no interactions among them)
        for batch_path in train_files:
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
 
    response = requests.post(
        TRAINING_SERVICE_URL,
        json={
            "train_file": str(train_file),
        },
        timeout=5,
    )
    response.raise_for_status()
    data = response.json()
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
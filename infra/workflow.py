import os
import subprocess
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lora import train_lora
from uuid import uuid4
from time import time

app = FastAPI()

VLLM_URL = "http://localhost:8000"
LORA_OUTPUT_DIR = "lora_adapters"
os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)

current_adapter_path = None

class TrainRequest(BaseModel):
    data_path: str

@app.post("/train-and-update")
async def train_and_update(request: TrainRequest):
    global current_adapter_path
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_name = str(uuid4())[:10]
    new_adapter_path = os.path.abspath(f"{LORA_OUTPUT_DIR}/{adapter_name}_{timestamp}")
    
    try:
        train_start_time = time()
        train_lora(
            data_path=request.data_path,
            output_path=new_adapter_path,
            gpu_id=0,
            base_adapter_path=current_adapter_path,
        )
        print(f'total train time: {time() - train_start_time}')
    except Exception as e:
        print(f'Error in train_lora: {e}')
        raise RuntimeError(e)
    try:
        lora_reload_time = time()
        response = requests.post(
            f"{VLLM_URL}/v1/load_lora_adapter",
            json={"lora_name": adapter_name, "lora_path": new_adapter_path},
            timeout=60
        )
        print(f'lora reload time: {time() - lora_reload_time}')
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load adapter: {str(e)}")
    
    old_adapter = current_adapter_path
    current_adapter_path = new_adapter_path

    # TODO: delete old_adapter path.
    # TODO: sample by using the adapter_name
    return {
        "status": "success",
        "adapter_name": adapter_name,
        "new_adapter_path": new_adapter_path,
        "previous_adapter_path": old_adapter
    }

@app.get("/current-adapter")
async def get_current_adapter():
    return {"adapter_path": current_adapter_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
import os
import subprocess
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

VLLM_URL = "http://localhost:8000"
LORA_OUTPUT_DIR = "lora_adapters"
os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)

current_adapter_path = None

class TrainRequest(BaseModel):
    data_path: str
    adapter_name: str = "latest"

@app.post("/train-and-update")
async def train_and_update(request: TrainRequest):
    global current_adapter_path
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_adapter_path = os.path.abspath(f"{LORA_OUTPUT_DIR}/{request.adapter_name}_{timestamp}")
    
    train_cmd = ["python", "lora.py", request.data_path, new_adapter_path, "1"]
    if current_adapter_path:
        train_cmd.append(current_adapter_path)
    
    try:
        subprocess.run(train_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e.stderr}")
    
    try:
        response = requests.post(
            f"{VLLM_URL}/v1/load_lora_adapter",
            json={"lora_name": request.adapter_name, "lora_path": new_adapter_path},
            timeout=60
        )
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load adapter: {str(e)}")
    
    old_adapter = current_adapter_path
    current_adapter_path = new_adapter_path

    # TODO: delete old_adapter path.
    
    return {
        "status": "success",
        "adapter_name": request.adapter_name,
        "new_adapter_path": new_adapter_path,
        "previous_adapter_path": old_adapter
    }

@app.get("/current-adapter")
async def get_current_adapter():
    return {"adapter_path": current_adapter_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
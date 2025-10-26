#!/bin/bash

# for vllm, we use gpu=0
export CUDA_VISIBLE_DEVICES=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# by default, serve the og model. But on a per-request basis, we pass in the lora adapters path for inference.

# sj: max-loras refers to the # of concurrent lora adaptes we can load.
# SJ: vllm server OOMs without gpu-util param
vllm serve Qwen/Qwen3-8B \
    --enable-lora \
    --max-lora-rank 16 \
    --max-loras 4 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
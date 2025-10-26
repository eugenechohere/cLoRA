from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import requests


DEFAULT_UPLOAD_URL = "https://calhacks-monitor-backend.ngrok.pizza/upload"
DEFAULT_INFER_URL = "http://0.0.0.0:8002/infer"
DEFAULT_GET_DATA_URL = "https://calhacks-monitor-backend.ngrok.pizza/get_data"
DEFAULT_LATEST_LOSS_URL = "https://calhacks-monitor-backend.ngrok.pizza/latest_loss"
TRAIN_TRIGGER_THRESHOLD = 64


def make_payload(start: int, count: int) -> List[Dict[str, str]]:
    return [
        {"question": f"prompt-{i}", "answer": f"completion-{i}"}
        for i in range(start, start + count)
    ]


def post_payload(url: str, payload: List[Dict[str, str]]) -> Dict[str, any]:
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def test_inference(url: str, prompt: str) -> Dict[str, any]:
    """Test the inference endpoint with a sample prompt."""
    payload = {"prompt": prompt}
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def test_get_data(url: str, samples_per_batch) -> List[List[Dict[str, any]]]:
    """Test the get_data endpoint to retrieve sampled training data."""
    params = {"samples_per_batch": samples_per_batch}
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def test_latest_loss(url: str) -> Dict[str, any]:
    """Test the latest_loss endpoint to get the most recent training loss."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:

    # print(f"[test] Sending first batch of {TRAIN_TRIGGER_THRESHOLD - 1} examples...")
    # first_payload = make_payload(0, 256)
    # first_response = post_payload(DEFAULT_UPLOAD_URL, first_payload)
    # print(f"[test] First response: {first_response}")
    
    print(f"\n[test] Testing get_data endpoint...")
    get_data_response = test_get_data(DEFAULT_GET_DATA_URL, samples_per_batch=5)
    for i in get_data_response:
        print(i[:3], '\n\n\n')
        print(len(i))
    
    print(f"\n[test] Testing latest_loss endpoint...")
    loss_response = test_latest_loss(DEFAULT_LATEST_LOSS_URL)
    print(f"[test] Latest training loss: {loss_response['loss']}")
    
    # print(f"[test] Sending second batch to cross threshold...")
    # second_payload = make_payload(TRAIN_TRIGGER_THRESHOLD - 1, 2)
    # second_response = post_payload(DEFAULT_UPLOAD_URL, second_payload)

    # print(f"[test] Testing inference endpoint...")
    # test_prompt = "What is 10 + 10?"
    # inference_response = test_inference(DEFAULT_INFER_URL, test_prompt)
    # print(f"[test] Inference response: {inference_response}")





if __name__ == "__main__":
    main()

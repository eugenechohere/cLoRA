curl -X POST http://localhost:8001/train-and-update \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "data/batch.jsonl"
  }'
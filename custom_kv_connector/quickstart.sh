#!/bin/bash

# Quick start script for CustomFileKVConnector
# This script runs a complete end-to-end test

set -e  # Exit on error

echo "======================================================================"
echo "CustomFileKVConnector Quick Start"
echo "======================================================================"
echo ""

# Configuration
MODEL="${MODEL:-facebook/opt-125m}"
PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog. This is a test sentence to demonstrate KV caching.}"
CACHE_TOKENS="${CACHE_TOKENS:-20}"
CACHE_DIR="kv_cache_dir"
CACHE_FILE="${CACHE_DIR}/quickstart_cache.pt"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Cache tokens: $CACHE_TOKENS"
echo "  Cache directory: $CACHE_DIR"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# Step 1: Simple test (no model)
echo "Step 1: Running simple test (no model download)..."
echo "----------------------------------------------------------------------"
python simple_test.py
if [ $? -ne 0 ]; then
    echo "Simple test failed!"
    exit 1
fi
echo ""

# Step 2: Prepare KV cache
echo "Step 2: Preparing KV cache..."
echo "----------------------------------------------------------------------"
python prepare_kv_cache.py \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --output "$CACHE_FILE" \
    --num-tokens "$CACHE_TOKENS" \
    --method transformers

if [ $? -ne 0 ]; then
    echo "KV cache preparation failed!"
    exit 1
fi
echo ""

# Step 3: Verify cache file
echo "Step 3: Verifying KV cache file..."
echo "----------------------------------------------------------------------"
python -c "
import torch
loaded = torch.load('$CACHE_FILE')
print(f'✓ KV cache verified:')
print(f'  Keys: {loaded[\"keys\"].shape}')
print(f'  Values: {loaded[\"values\"].shape}')
print(f'  Tokens: {len(loaded[\"prompt_token_ids\"])}')
"
echo ""

# Step 4: Run full test
echo "Step 4: Running full test with vLLM..."
echo "----------------------------------------------------------------------"
echo "Note: This will download the model if not already cached"
echo ""

python test_connector.py \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --cache-tokens "$CACHE_TOKENS" \
    --cache-file "$CACHE_FILE"

if [ $? -ne 0 ]; then
    echo "Full test failed!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✓ Quick start completed successfully!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Check the README.md for detailed usage"
echo "  2. Try with your own model and prompts"
echo "  3. Integrate into your application"
echo ""
echo "Example usage in your code:"
echo "  python -c '"
echo "  import custom_kv_connector.register_connector"
echo "  from vllm import LLM, SamplingParams"
echo "  llm = LLM("
echo "      model=\"$MODEL\","
echo "      kv_transfer_config={"
echo "          \"kv_connector\": \"CustomFileKVConnector\","
echo "          \"kv_role\": \"kv_both\","
echo "          \"kv_connector_extra_config\": {\"kv_cache_dir\": \"$CACHE_DIR\"}"
echo "      }"
echo "  )"
echo "  outputs = llm.generate(\"$PROMPT\", SamplingParams(max_tokens=20))"
echo "  print(outputs[0].outputs[0].text)"
echo "  '"
echo ""


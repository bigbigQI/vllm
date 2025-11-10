# Custom File KV Connector for vLLM

A custom KV connector that enables loading pre-computed KV cache from local files to accelerate vLLM inference by skipping computation for cached tokens.

## Features

- ✅ Load pre-computed KV cache from disk
- ✅ Skip computation for cached prompt tokens
- ✅ Support partial prompt caching
- ✅ Automatic prefix matching
- ✅ Compatible with vLLM v1 architecture
- ✅ Support multiple KV cache layouts (Flash Attention, FlashInfer)

## Quick Start

### 1. Installation

No additional installation required. Just ensure vLLM is properly installed:

```bash
cd /lustre/fsw/portfolios/coreai/users/larkz/vllm
pip install -e .
```

### 2. Simple Test (No Model Required)

Run a simple test to verify the connector works:

```bash
cd custom_kv_connector
python simple_test.py
```

This creates a dummy KV cache and verifies the connector can load it.

### 3. Prepare KV Cache

Extract KV cache from your model for a specific prompt:

```bash
python prepare_kv_cache.py \
    --model facebook/opt-125m \
    --prompt "The quick brown fox jumps over the lazy dog." \
    --output kv_cache_dir/kv_cache.pt \
    --num-tokens 20
```

This will:
- Load the model
- Compute KV cache for the first 20 tokens
- Save to `kv_cache_dir/kv_cache.pt`

**Output format:**
```python
{
    'prompt_token_ids': List[int],  # Token IDs of cached prompt
    'keys': Tensor(num_layers, 1, num_heads, seq_len, head_dim),
    'values': Tensor(num_layers, 1, num_heads, seq_len, head_dim)
}
```

### 4. Run Full Test

Test the connector end-to-end with vLLM:

```bash
python test_connector.py \
    --model facebook/opt-125m \
    --prompt "The quick brown fox jumps over the lazy dog. This is a longer sentence." \
    --cache-tokens 15 \
    --cache-file kv_cache_dir/test_kv.pt
```

This will:
1. Create KV cache for the first 15 tokens
2. Run baseline inference (without cache)
3. Run inference with cached KV
4. Compare outputs and performance

Expected output:
```
✓ PASS: Outputs match perfectly!

Performance:
  Baseline time: 1.23s
  Cached time: 0.89s
  Time saved: 0.34s
  Speedup: 1.38x

✓ ALL TESTS PASSED!
```

## Usage in Your Application

### Method 1: Specify KV Cache File in Request

```python
import sys
sys.path.insert(0, "/lustre/fsw/portfolios/coreai/users/larkz/vllm")

# Register the connector
import custom_kv_connector.register_connector

from vllm import LLM, SamplingParams

# Initialize vLLM with the connector
llm = LLM(
    model="your-model",
    kv_transfer_config={
        "kv_connector": "CustomFileKVConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "kv_cache_dir": "./kv_cache"
        }
    }
)

# Generate with cached KV
sampling_params = SamplingParams(
    max_tokens=50,
    extra_args={
        "kv_transfer_params": {
            "kv_cache_file": "./kv_cache/my_cache.pt"
        }
    }
)

outputs = llm.generate("Your prompt here...", sampling_params)
print(outputs[0].outputs[0].text)
```

### Method 2: Automatic Cache Matching

The connector can automatically find matching KV cache files based on prompt hash:

```python
# 1. Prepare cache with consistent naming
python prepare_kv_cache.py \
    --model your-model \
    --prompt "Common prefix that appears in many requests" \
    --output kv_cache_dir/kv_cache.pt

# 2. Use in vLLM - connector will auto-detect matching cache
llm = LLM(
    model="your-model",
    kv_transfer_config={
        "kv_connector": "CustomFileKVConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "kv_cache_dir": "./kv_cache"
        }
    }
)

# Any request with matching prefix will use the cache automatically
outputs = llm.generate(
    "Common prefix that appears in many requests and then continues...",
    SamplingParams(max_tokens=50)
)
```

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Scheduler Process                     │
│  1. Check if request has matching KV cache              │
│  2. Allocate blocks for external KV tokens              │
│  3. Build metadata for workers                          │
└────────────────────┬────────────────────────────────────┘
                     │ Metadata
                     ↓
┌─────────────────────────────────────────────────────────┐
│                    Worker Process                        │
│  1. Load KV cache file from disk                        │
│  2. Convert continuous KV → paged format                │
│  3. Inject into vLLM's paged KV buffer                  │
│  4. Skip computation for cached tokens                  │
└─────────────────────────────────────────────────────────┘
```

### KV Cache Format Conversion

```python
# Input: Your format (continuous)
keys:   (num_layers, 1, num_heads, seq_len, head_dim)
values: (num_layers, 1, num_heads, seq_len, head_dim)

# Output: vLLM's format (paged)
kv_cache: [2, num_blocks, block_size, num_heads, head_dim]
          ↑                ↑
          |                └─ Fixed block size (e.g., 16 tokens)
          └─ [0]=keys, [1]=values

# Mapping example (block_size=16, seq_len=35):
# Token 0-15  → Block 0 [fully filled]
# Token 16-31 → Block 1 [fully filled]
# Token 32-35 → Block 2 [partially filled, 4/16 tokens]
```

### Workflow

1. **Scheduler Checks Cache**:
   - When new request arrives, check for matching KV cache file
   - Count matching prefix tokens (aligned to block boundary)
   - Allocate blocks for external KV

2. **Worker Loads KV**:
   - Load KV cache file into GPU memory
   - Convert from continuous format to paged format
   - Inject into allocated blocks

3. **Model Forward**:
   - Attention uses cached KV for first N tokens
   - Only computes KV for remaining tokens
   - Combines cached and new KV seamlessly

## Configuration Options

### KV Transfer Config

```python
kv_transfer_config = {
    "kv_connector": "CustomFileKVConnector",
    "kv_role": "kv_both",  # or "kv_consumer" for decode-only
    "kv_connector_extra_config": {
        "kv_cache_dir": "./kv_cache"  # Directory containing KV cache files
    }
}
```

### Request-Specific Config

```python
sampling_params = SamplingParams(
    max_tokens=100,
    extra_args={
        "kv_transfer_params": {
            "kv_cache_file": "/path/to/specific/cache.pt"
        }
    }
)
```

## Performance Tips

### 1. Cache Alignment

Always align cached tokens to block boundaries for maximum efficiency:

```python
# Good: Cache 32 tokens (2 blocks @ block_size=16)
num_tokens_to_cache = 32

# Bad: Cache 30 tokens (only 1 full block can be used)
num_tokens_to_cache = 30
```

### 2. Caching Strategy

Cache the most expensive parts of your prompts:

```python
# System prompt (expensive, rarely changes)
system_prompt = "You are a helpful AI assistant. [lots of instructions...]"
cache_system_prompt(system_prompt)  # Cache once

# User queries (cheap, vary for each request)
# Don't cache these - compute on the fly
```

### 3. File Management

For production use, organize cache files efficiently:

```python
kv_cache_dir/
├── system_prompts/
│   ├── assistant_v1.pt      # Common system prompt
│   └── coder_v1.pt          # Code-specific system prompt
├── common_prefixes/
│   └── product_catalog.pt   # Product descriptions
└── user_specific/
    └── user_123_history.pt  # User conversation history
```

## Troubleshooting

### Issue: KV cache not being used

**Check:**
1. Prompt prefix matches cached tokens exactly
2. Block alignment is correct
3. Connector is properly registered
4. File path is correct

**Debug:**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check connector logs
# You should see:
# "Request {id}: KV cache hit! X/Y tokens cached"
```

### Issue: Output differs from baseline

**Possible causes:**
1. KV cache was generated with different model version
2. Numerical precision issues (fp16 vs fp32)
3. Different attention backend

**Solution:**
```python
# Regenerate KV cache with exact same model and dtype
prepare_kv_cache.py --model exact-model-path --dtype float16
```

### Issue: Out of memory

**Solutions:**
1. Reduce number of cached tokens
2. Use fp16 instead of fp32 for KV cache
3. Clear cache after use:

```python
connector._loaded_kv_cache.clear()
```

## Advanced Usage

### Multiple KV Caches

Handle requests with different prefixes:

```python
# Prepare multiple caches
prepare_kv_cache(prompt_a, output="cache_a.pt")
prepare_kv_cache(prompt_b, output="cache_b.pt")

# Specify cache per request
for prompt, cache_file in requests:
    sampling_params = SamplingParams(
        extra_args={
            "kv_transfer_params": {"kv_cache_file": cache_file}
        }
    )
    llm.generate(prompt, sampling_params)
```

### Dynamic Cache Generation

Generate cache on-demand:

```python
def get_or_create_cache(prompt):
    cache_file = f"kv_cache_dir/{hash(prompt)}.pt"
    if not os.path.exists(cache_file):
        prepare_kv_cache(prompt, cache_file)
    return cache_file

# Use in production
cache_file = get_or_create_cache(common_prefix)
outputs = llm.generate(full_prompt, SamplingParams(
    extra_args={"kv_transfer_params": {"kv_cache_file": cache_file}}
))
```

## Limitations

1. **Block Alignment**: Only full blocks can be cached efficiently
2. **Exact Matching**: Prompt prefix must match exactly (no fuzzy matching)
3. **Single File**: Each request uses one cache file (no merging)
4. **V1 Only**: Currently only works with vLLM v1 architecture
5. **Synchronous Loading**: KV loading is synchronous (future: async)

## Future Enhancements

- [ ] Async KV loading during model execution
- [ ] Fuzzy prefix matching with similarity threshold
- [ ] Automatic cache warming on startup
- [ ] Cache compression for storage efficiency
- [ ] Multi-file cache merging
- [ ] Distributed cache storage (Redis, S3)

## Contributing

Found a bug or have a feature request? Please open an issue!

## License

Apache 2.0 - Same as vLLM

## References

- [vLLM Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)
- [vLLM KV Transfer](https://github.com/vllm-project/vllm/tree/main/vllm/distributed/kv_transfer)
- [Paged Attention Paper](https://arxiv.org/abs/2309.06180)


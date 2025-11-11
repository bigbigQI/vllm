#!/usr/bin/env python3
"""
Script to perform inference with KV cache and save token/logprob information.

This script:
1. Loads a model with KV cache connector
2. Performs inference using cached KV
3. Saves prompt token IDs and response token IDs with logprobs to JSON
"""

import argparse
import sys
import json
import time
import torch
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Register the connector
import custom_kv_connector.register_connector  # noqa: F401

from vllm import LLM, SamplingParams


def inference_with_cache(
    model_path: str,
    prompt: str,
    kv_cache_file: str,
    max_tokens: int = 20
) -> dict:
    """Run inference with KV cache connector and return token/logprob information
    
    Args:
        model_path: Path to the model
        prompt: Input prompt text
        kv_cache_file: Path to the KV cache file
        max_tokens: Maximum tokens to generate (default: 20)
    
    Returns:
        Dictionary containing:
        - prompt_token_ids: List of token IDs in the prompt
        - response_tokens: List of dicts with token_id and logprob for response
        - num_cached_tokens: Number of tokens loaded from cache
        - elapsed_time: Inference time in seconds
    """
    print("\n" + "="*60)
    print("Inference with Cached KV")
    print("="*60 + "\n")
    
    # Configure vLLM to use our connector
    llm = LLM(
        model=model_path,
        enforce_eager=True,
        max_model_len=2048,
        kv_transfer_config={
            "kv_connector": "CustomFileKVConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "kv_cache_dir": str(Path(kv_cache_file).parent)
            }
        },
        hf_overrides={
            "head_dtype": "float32"
        }
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=1,  # Get logprobs for generated tokens
        # NOTE: prompt_logprobs is disabled for KV connector because vLLM cannot
        # compute logprobs for externally loaded cached tokens
        prompt_logprobs=None,
        # Pass the KV cache file path via extra_args
        extra_args={
            "kv_transfer_params": {
                "kv_cache_file": kv_cache_file
            }
        }
    )
    
    print(f"Prompt: {prompt[:100]}...")
    print(f"KV cache file: {kv_cache_file}")
    print("Running inference...")
    
    start_time = time.time()
    outputs = llm.generate(prompt, sampling_params)
    elapsed = time.time() - start_time
    
    output = outputs[0]
    generated_text = output.outputs[0].text
    
    # Get cache hit statistics
    num_cached = getattr(output, 'num_cached_tokens', 0)
    
    # Extract prompt token IDs
    prompt_token_ids = []
    if hasattr(output, 'prompt_token_ids'):
        raw_tokens = output.prompt_token_ids
        if isinstance(raw_tokens, torch.Tensor):
            prompt_token_ids = raw_tokens.tolist()
        elif isinstance(raw_tokens, list):
            prompt_token_ids = [int(t) if hasattr(t, '__int__') else t for t in raw_tokens]
        else:
            prompt_token_ids = list(raw_tokens)
    
    # Extract response token information (token_id and logprob)
    response_tokens = []
    
    for output_token in output.outputs[0].logprobs:
        if output_token is not None:
            # Get the token with highest probability (the selected token)
            for token_id, logprob_obj in output_token.items():
                # Ensure token_id is a Python int
                token_id_int = int(token_id) if hasattr(token_id, '__int__') else token_id
                response_tokens.append({
                    "token_id": token_id_int,
                    "logprob": float(logprob_obj.logprob)
                })
                break  # Only take the first (selected) token
    
    print(f"✓ Generated in {elapsed:.2f}s")
    print(f"Output: {generated_text}")
    print(f"Number of cached tokens: {num_cached}")
    print(f"Prompt tokens: {len(prompt_token_ids)}")
    print(f"Response tokens: {len(response_tokens)}")
    
    return {
        "prompt_token_ids": prompt_token_ids,
        "response_tokens": response_tokens,
        "num_cached_tokens": num_cached,
        "elapsed_time": elapsed,
        "generated_text": generated_text
    }


def save_to_json(
    result_data: dict,
    output_file: str,
    model_path: str,
    prompt: str,
    kv_cache_file: str
):
    """Save inference results to JSON file
    
    Args:
        result_data: Dictionary with inference results
        output_file: Path to output JSON file
        model_path: Path to the model
        prompt: Input prompt text
        kv_cache_file: Path to the KV cache file
    """
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model_path,
            "prompt": prompt,
            "kv_cache_file": kv_cache_file,
            "num_cached_tokens": result_data["num_cached_tokens"],
            "elapsed_time": result_data["elapsed_time"],
            "generated_text": result_data["generated_text"]
        },
        "prompt": {
            "token_ids": result_data["prompt_token_ids"]
        },
        "response": {
            "tokens": result_data["response_tokens"]
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print(f"✓ Results saved to: {output_file}")
    print(f"  Prompt tokens: {len(result_data['prompt_token_ids'])}")
    print(f"  Response tokens: {len(result_data['response_tokens'])}")
    print(f"  Cached tokens used: {result_data['num_cached_tokens']}")
    print(f"  Inference time: {result_data['elapsed_time']:.2f}s")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Perform inference with KV cache and save token/logprob information"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/apps/models/Qwen3-8B-Base",
        help="Model path (default: /apps/models/Qwen3-8B-Base)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. ",
        help="Input prompt text"
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="/apps/sharon_verl_new/verl/kv_cache.pt",
        help="Path to KV cache file"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cache_vllm_inference_output.json",
        help="Output JSON file path (default: cache_inference_output.json)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("KV Cache Inference - Save Token/Logprob Script")
    print("="*60)
    
    try:
        # Verify cache file exists
        if not Path(args.cache_file).exists():
            print(f"✗ ERROR: KV cache file not found: {args.cache_file}")
            return 1
        
        # Perform inference with cache
        result_data = inference_with_cache(
            model_path=args.model,
            prompt=args.prompt,
            kv_cache_file=args.cache_file,
            max_tokens=args.max_tokens
        )
        
        # Save results to JSON
        save_to_json(
            result_data=result_data,
            output_file=args.output,
            model_path=args.model,
            prompt=args.prompt,
            kv_cache_file=args.cache_file
        )
        
        print("✓ Successfully completed!")
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

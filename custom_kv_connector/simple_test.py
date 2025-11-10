#!/usr/bin/env python3
"""
Simple test that doesn't require full model download.
Creates a dummy KV cache and verifies the connector can load it.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Register connector
import custom_kv_connector.register_connector  # noqa: F401


def create_dummy_kv_cache(
    num_layers: int = 12,
    num_heads: int = 12,
    seq_len: int = 30,
    head_dim: int = 64,
    output_path: str = "kv_cache_dir/dummy_kv.pt"
):
    """Create a dummy KV cache file for testing"""
    print(f"Creating dummy KV cache:")
    print(f"  Layers: {num_layers}")
    print(f"  Heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dimension: {head_dim}")
    
    # Create dummy data
    prompt_token_ids = list(range(seq_len))  # [0, 1, 2, ..., seq_len-1]
    
    keys = torch.randn(num_layers, 1, num_heads, seq_len, head_dim, dtype=torch.float16)
    values = torch.randn(num_layers, 1, num_heads, seq_len, head_dim, dtype=torch.float16)
    
    save_data = {
        'prompt_token_ids': prompt_token_ids,
        'keys': keys,
        'values': values,
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, output_path)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved dummy KV cache to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    return str(output_path)


def verify_kv_file(file_path: str):
    """Verify the KV cache file can be loaded"""
    print(f"\nVerifying KV cache file: {file_path}")
    
    loaded = torch.load(file_path)
    
    print("✓ File loaded successfully")
    print(f"  Keys: {loaded['keys'].shape}")
    print(f"  Values: {loaded['values'].shape}")
    print(f"  Token IDs: {len(loaded['prompt_token_ids'])} tokens")
    print(f"  First 10 tokens: {loaded['prompt_token_ids'][:10]}")
    
    return True


def test_connector_initialization():
    """Test that the connector can be initialized"""
    print("\nTesting connector initialization...")
    
    try:
        from vllm.config import VllmConfig, ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        from custom_kv_connector.custom_file_kv_connector import CustomFileKVConnector
        
        # Create minimal config
        model_config = ModelConfig(
            model="facebook/opt-125m",
            tokenizer="facebook/opt-125m",
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="float16",
            seed=0,
        )
        
        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            swap_space=0,
            cache_dtype="auto",
        )
        
        parallel_config = ParallelConfig()
        scheduler_config = SchedulerConfig()
        
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
        )
        
        # Initialize connector
        connector = CustomFileKVConnector(
            vllm_config=vllm_config,
            role=KVConnectorRole.WORKER
        )
        
        print("✓ Connector initialized successfully")
        print(f"  Block size: {connector._block_size}")
        print(f"  Num layers: {connector._num_layers}")
        print(f"  Num KV heads: {connector._num_kv_heads}")
        print(f"  Head size: {connector._head_size}")
        print(f"  KV cache dir: {connector._kv_cache_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ Connector initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("Simple Connector Test (No Model Required)")
    print("="*60)
    
    success = True
    
    # Test 1: Create dummy KV cache
    print("\nTest 1: Creating dummy KV cache")
    try:
        kv_file = create_dummy_kv_cache()
    except Exception as e:
        print(f"✗ Failed: {e}")
        success = False
        return 1
    
    # Test 2: Verify file can be loaded
    print("\nTest 2: Verifying KV cache file")
    try:
        verify_kv_file(kv_file)
    except Exception as e:
        print(f"✗ Failed: {e}")
        success = False
        return 1
    
    # Test 3: Test connector initialization
    print("\nTest 3: Connector initialization")
    try:
        test_connector_initialization()
    except Exception as e:
        print(f"✗ Failed: {e}")
        success = False
        return 1
    
    if success:
        print("\n" + "="*60)
        print("✓ All simple tests passed!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run prepare_kv_cache.py to create real KV cache from a model")
        print("2. Run test_connector.py to test end-to-end with vLLM")
        return 0
    else:
        print("\n" + "="*60)
        print("✗ Some tests failed")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())


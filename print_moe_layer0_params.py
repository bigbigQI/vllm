#!/usr/bin/env python3
"""
Script to load an MoE model with vLLM and print all parameter names from layer 0.

This script demonstrates how to:
1. Load an MoE model using vLLM's LLM class
2. Access the underlying model from the engine
3. Print all parameter names from the first layer (layer 0)

Usage:
    python print_moe_layer0_params.py
"""

import os
from vllm import LLM

def print_layer0_parameters(model_path: str):
    """
    Load an MoE model and print all parameter names from layer 0.
    
    Args:
        model_path: Path to the MoE model
    """
    print(f"Loading MoE model: {model_path}")
    print("=" * 80)
    
    # Initialize vLLM with the MoE model
    # Using smaller max_model_len to reduce memory usage
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,  # Adjust based on your hardware
        max_model_len=512,       # Smaller length for initialization
        enforce_eager=True,      # Use eager mode for easier debugging
        trust_remote_code=True,  # Some MoE models may need this
    )
    
    print("\nModel loaded successfully!")
    print("=" * 80)
    
    # Access the underlying model from the engine
    # The path depends on vLLM version and executor type
    try:
        # For V1 engine (if VLLM_USE_V1=1)
        if hasattr(llm.llm_engine, 'engine_core'):
            engine_core = llm.llm_engine.engine_core
            if hasattr(engine_core, 'engine_core'):
                # Multiprocess mode
                print("Note: Cannot access model in multiprocess mode.")
                print("Please set VLLM_ENABLE_V1_MULTIPROCESSING=0 to access the model.")
                return
            # Access the model executor
            executor = engine_core.model_executor
            # Get the model from the driver worker
            model = executor.driver_worker.model_runner.model
        # For V0 engine (if VLLM_USE_V1=0 or not set)
        elif hasattr(llm.llm_engine, 'model_executor'):
            executor = llm.llm_engine.model_executor
            # Try to get model from driver_worker
            if hasattr(executor, 'driver_worker'):
                model = executor.driver_worker.model_runner.model
            else:
                print("Cannot access model from executor")
                return
        else:
            print("Cannot determine engine type")
            return
            
    except AttributeError as e:
        print(f"Error accessing model: {e}")
        print("\nTrying alternative access method...")
        # Alternative: try to access through model_executor
        if hasattr(llm.llm_engine, 'model_executor'):
            model_executor = llm.llm_engine.model_executor
            print(f"Model executor type: {type(model_executor)}")
            print(f"Available attributes: {dir(model_executor)}")
        return
    
    print("\nSuccessfully accessed the model!")
    print(f"Model type: {type(model).__name__}")
    print("=" * 80)
    
    # Print all parameter names from layer 0
    print("\n" + "=" * 80)
    print("Parameter names from Layer 0:")
    print("=" * 80)
    
    layer0_params = []
    
    # Iterate through all parameters and find those belonging to layer 0
    for name, param in model.named_parameters():
        # Check if parameter belongs to layer 0
        # Common patterns: "layers.0.", "model.layers.0.", "transformer.layers.0."
        if ".layers.0." in name or ".layers.0/" in name:
            layer0_params.append((name, param.shape, param.dtype))
    
    if not layer0_params:
        print("No parameters found for layer 0.")
        print("\nAll parameter names (for debugging):")
        print("-" * 80)
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape} ({param.dtype})")
    else:
        print(f"\nFound {len(layer0_params)} parameters in layer 0:\n")
        for name, shape, dtype in layer0_params:
            print(f"  {name}")
            print(f"    Shape: {shape}")
            print(f"    Dtype: {dtype}")
            print()
    
    print("=" * 80)
    print("Done!")


def main():
    # Example MoE model path - replace with your actual model path
    # Some popular MoE models:
    # - "mistralai/Mixtral-8x7B-v0.1"
    # - "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # - "Qwen/Qwen1.5-MoE-A2.7B"
    # - "/apps/models/Qwen3-30B-A3B-Instruct-2507"
    
    model_path = "/apps/models/Qwen3-30B-A3B-Instruct-2507"  # Default from your moe.py
    
    # You can also specify a different model path via environment variable
    model_path = os.environ.get("MODEL_PATH", model_path)
    
    print("=" * 80)
    print("vLLM MoE Model - Layer 0 Parameter Printer")
    print("=" * 80)
    print(f"\nModel path: {model_path}\n")
    
    # Check if we should disable multiprocessing for model access
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "1":
        print("WARNING: VLLM_ENABLE_V1_MULTIPROCESSING is enabled.")
        print("Model access may not be available in multiprocess mode.")
        print("Consider setting VLLM_ENABLE_V1_MULTIPROCESSING=0\n")
    
    try:
        print_layer0_parameters(model_path)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Make sure the model path is correct")
        print("2. Ensure you have enough GPU memory")
        print("3. Try setting VLLM_ENABLE_V1_MULTIPROCESSING=0")
        print("4. Check if the model is actually an MoE model")


if __name__ == "__main__":
    main()








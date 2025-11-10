#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
演示如何查看 vLLM 推理时 KV Cache 重用的信息

这个脚本展示了：
1. 如何启用 Prefix Caching（自动 KV Cache 重用）
2. 如何从输出中读取 num_cached_tokens（重用的 KV cache token 数量）
3. 如何在日志中查看 prefix cache hit rate（前缀缓存命中率）

Run:
    python examples/kv_cache_reuse_demo.py
"""

import time
from vllm import LLM, SamplingParams

# 一个长提示词作为共享前缀
LONG_PROMPT = """You are a helpful AI assistant. Please answer the following question based on the context below.

Context:
Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on 
the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.

IBM has a rich history with machine learning. One of its own, Arthur Samuel, is credited for coining 
the term, "machine learning" with his research (PDF, 481 KB) (link resides outside IBM) around the 
game of checkers. Robert Nealey, the self-proclaimed checkers master, played the game on an IBM 7094 
computer in 1962, and he lost to the computer. Compared to what can be done today, this feat seems 
trivial, but it's considered a major milestone in the field of artificial intelligence.

Over the last couple of decades, the technological advances in storage and processing power have 
enabled some innovative products based on machine learning, such as Netflix's recommendation engine 
and self-driving cars.

Deep learning is a subset of machine learning, which is essentially a neural network with three or 
more layers. These neural networks attempt to simulate the behavior of the human brain—albeit far 
from matching its ability—allowing it to "learn" from large amounts of data. While a neural network 
with a single layer can still make approximate predictions, additional hidden layers can help to 
optimize and refine for accuracy.

"""


def print_kv_cache_stats(output, query_desc):
    """打印 KV Cache 重用统计信息"""
    print(f"\n{'='*80}")
    print(f"查询: {query_desc}")
    print(f"{'='*80}")
    
    # 获取第一个输出
    if len(output) > 0:
        request_output = output[0]
        
        # 打印基本信息
        print(f"Request ID: {request_output.request_id}")
        print(f"Prompt tokens: {len(request_output.prompt_token_ids) if request_output.prompt_token_ids else 'N/A'}")
        
        # 关键信息：打印重用的 KV cache token 数量
        if request_output.num_cached_tokens is not None:
            print(f"✅ 重用的 KV cache tokens: {request_output.num_cached_tokens}")
            if request_output.prompt_token_ids:
                cache_hit_rate = (request_output.num_cached_tokens / 
                                 len(request_output.prompt_token_ids) * 100)
                print(f"✅ KV cache 命中率: {cache_hit_rate:.1f}%")
        else:
            print("❌ num_cached_tokens: None (可能未启用 prefix caching)")
        
        # 打印生成的文本
        if len(request_output.outputs) > 0:
            generated_text = request_output.outputs[0].text[:100]  # 只显示前100个字符
            print(f"\n生成的文本（前100字符）: {generated_text}...")
        
        # 打印指标信息
        if request_output.metrics:
            print(f"\n其他指标:")
            for key, value in vars(request_output.metrics).items():
                if value is not None:
                    print(f"  {key}: {value}")
    
    print(f"{'='*80}\n")


def main():
    print("=== vLLM KV Cache 重用演示 ===\n")
    
    # 初始化 LLM，启用 prefix caching
    print("1. 初始化 vLLM 引擎（启用 enable_prefix_caching=True）...")
    llm = LLM(
        model="/apps/models/Qwen3-8B",  # 使用一个小模型进行演示
        enable_prefix_caching=True,  # 启用 prefix caching
        gpu_memory_utilization=0.3,
        max_model_len=512,
    )
    print("   ✅ 引擎初始化完成\n")
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.0,  # 确定性输出
        max_tokens=50,
    )
    
    # 第一个查询 - 没有缓存
    print("2. 第一次查询（没有缓存，需要计算完整的 KV cache）...")
    prompt1 = LONG_PROMPT + "\n\nQuestion: What is machine learning?\nAnswer:"
    start_time = time.time()
    output1 = llm.generate([prompt1], sampling_params=sampling_params)
    time1 = time.time() - start_time
    
    print_kv_cache_stats(output1, "第一次查询 - What is machine learning?")
    print(f"   ⏱️  推理时间: {time1:.3f} 秒\n")
    
    # 第二个查询 - 应该重用大部分 KV cache
    print("3. 第二次查询（共享相同前缀，应该重用大量 KV cache）...")
    prompt2 = LONG_PROMPT + "\n\nQuestion: What is deep learning?\nAnswer:"
    start_time = time.time()
    output2 = llm.generate([prompt2], sampling_params=sampling_params)
    time2 = time.time() - start_time
    
    print_kv_cache_stats(output2, "第二次查询 - What is deep learning?")
    print(f"   ⏱️  推理时间: {time2:.3f} 秒")
    
    # 计算加速比
    if time2 > 0:
        speedup = time1 / time2
        print(f"\n   🚀 加速比: {speedup:.2f}x (由于 KV cache 重用)\n")
    
    # 第三个查询 - 完全不同的提示词
    print("4. 第三次查询（完全不同的提示词，无法重用 KV cache）...")
    prompt3 = "What is the capital of France? Answer:"
    start_time = time.time()
    output3 = llm.generate([prompt3], sampling_params=sampling_params)
    time3 = time.time() - start_time
    
    print_kv_cache_stats(output3, "第三次查询 - 完全不同的提示词")
    print(f"   ⏱️  推理时间: {time3:.3f} 秒\n")
    
    print("\n=== 总结 ===")
    print("✅ vLLM 的 RequestOutput 对象包含 num_cached_tokens 字段")
    print("✅ 该字段表示本次请求重用了多少个 KV cache tokens")
    print("✅ 通过启用 enable_prefix_caching=True 可以自动重用共享前缀的 KV cache")
    print("✅ 在日志中也可以看到全局的 Prefix cache hit rate 统计信息")
    print("\n如何在代码中访问:")
    print("  output = llm.generate(prompts, ...)")
    print("  num_cached = output[0].num_cached_tokens  # 重用的 token 数量")
    print("  total_tokens = len(output[0].prompt_token_ids)  # 总 token 数量")
    print("  hit_rate = num_cached / total_tokens * 100  # 命中率 (%)")


if __name__ == "__main__":
    main()


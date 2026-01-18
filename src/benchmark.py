#!/usr/bin/env python3
"""
LLM Inference Optimization Benchmark Script

Usage:
    python benchmark.py --model gpt2 --config fp16
    python benchmark.py --model mistralai/Mistral-7B-v0.1 --config flash_attn
"""

import argparse
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

def benchmark_model(model_name, config, num_iterations=8):
    """Benchmark a model with specified configuration."""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"Configuration: {config}")
    print(f"{'='*60}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model based on configuration
    if config == "baseline":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto"
        )
    elif config == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    elif config == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto"
        )
    elif config == "4bit":
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
    elif config == "flash_attn":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
    else:
        raise ValueError(f"Unknown config: {config}")
    
    # Benchmark
    prompt = "Explain quantum computing:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    times = []
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=100)
    
    # Timed runs
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        
        torch.cuda.synchronize()
        end = time.time()
        
        elapsed = end - start
        times.append(elapsed)
        print(f"Iteration {i+1}/{num_iterations}: {elapsed:.3f}s")
    
    # Calculate metrics
    avg_time = sum(times) / len(times)
    throughput = 100 / avg_time  # tokens per second
    memory_gb = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"Average Time: {avg_time:.3f}s")
    print(f"Throughput: {throughput:.2f} tokens/sec")
    print(f"GPU Memory: {memory_gb:.2f} GB")
    print(f"{'='*60}\n")
    
    return {
        'model': model_name,
        'config': config,
        'avg_time_s': avg_time,
        'throughput_tok_s': throughput,
        'memory_gb': memory_gb
    }

def main():
    parser = argparse.ArgumentParser(description='Benchmark LLM inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., gpt2, mistralai/Mistral-7B-v0.1)')
    parser.add_argument('--config', type=str, required=True,
                       choices=['baseline', 'fp16', 'int8', '4bit', 'flash_attn'],
                       help='Optimization configuration')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--iterations', type=int, default=8,
                       help='Number of benchmark iterations')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = benchmark_model(args.model, args.config, args.iterations)
    
    # Save results
    df = pd.DataFrame([results])
    output_path = f"{args.output_dir}/benchmark_{args.model.replace('/', '_')}_{args.config}.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()

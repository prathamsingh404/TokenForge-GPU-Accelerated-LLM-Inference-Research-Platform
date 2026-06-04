# TokenForge: GPU-Accelerated LLM Inference Research Platform

## Overview

TokenForge is a comprehensive, ground-up research platform built in PyTorch and CUDA. It is designed to evaluate, benchmark, and optimize the inference performance of Large Language Models (LLMs) on consumer and enterprise GPU hardware.

This project is not an AI chatbot or a wrapper around existing APIs. Instead, it serves as a miniature version of production-grade serving infrastructure (such as vLLM, TensorRT-LLM, or Orca). It exists to answer fundamental engineering questions regarding LLM deployment: How fast can a model be served, and exactly which mathematical and structural techniques yield the highest performance gains?

By implementing advanced memory management, scheduling algorithms, and custom GPU kernels from scratch, TokenForge exposes the internal mechanics of modern LLM inference.

## Core Features and Architecture

The platform is divided into distinct optimization modules, each addressing a specific bottleneck in the LLM inference pipeline.

### 1. Benchmark Engine (Telemetry and Profiling)
The foundation of the platform is an automated benchmarking orchestrator. It manages the lifecycle of inference experiments by executing warmup cycles, running timed generations, and persisting results to a local SQLite database. It captures critical metrics including Time To First Token (TTFT), tokens per second (throughput), End-to-End latency, and hardware telemetry (VRAM utilization, power draw, and temperature) via PyNVML. 

### 2. Quantization Engine
LLMs are heavily bound by memory bandwidth. Moving weight matrices from High Bandwidth Memory (HBM) to the Streaming Multiprocessors (SM) takes more time than the actual matrix multiplication. The quantization engine tests multiple precision formats (FP16, INT8, and 4-bit NormalFloat) to demonstrate the exact VRAM savings and throughput gains achieved by reducing precision, as well as the mechanisms required to dequantize weights on the fly during the forward pass.

### 3. Continuous Batching
Traditional static batching forces the GPU to wait until all sequences in a batch have finished generating before a new batch can begin. TokenForge implements iteration-level scheduling (Continuous Batching). The engine evaluates the request queue at every single decode step. It evicts completed requests and immediately injects new requests into the empty slots of the active batch. This ensures the GPU remains fully saturated, drastically increasing total throughput.

### 4. KV Cache Memory Management
During autoregressive generation, a model must attend to all previous tokens. Computing this from scratch at every step results in quadratic time complexity. The standard solution is to cache the Key (K) and Value (V) states. However, dynamic memory allocation leads to severe VRAM fragmentation. TokenForge implements a custom PagedAttention-style memory manager that pre-allocates a continuous block of GPU memory upon initialization. Logical tokens are mapped to physical memory pages via a block table, completely eliminating external fragmentation and allowing for significantly larger batch sizes.

### 5. Speculative Decoding
Speculative decoding accelerates generation by guessing future tokens. The platform pairs a small, extremely fast draft model with a large target model. The draft model generates a sequence of tokens autoregressively. The target model then verifies this entire sequence in a single parallel forward pass. Using rejection sampling, matching tokens are accepted, allowing the system to output multiple tokens per step without altering the mathematical distribution of the target model's output.

### 6. Prefix Caching
For systems that frequently process the same system prompts or document contexts, recomputing the initial KV states is inefficient. The Prefix Caching module implements a Radix/Trie-based LRU cache. It stores the KV states of common prompt prefixes, allowing subsequent requests sharing the same prefix to instantly retrieve the precomputed states, effectively bypassing the prefill compute phase.

### 7. Custom CUDA and Triton Kernels
To achieve maximum hardware utilization, standard PyTorch operations are often insufficient. The platform includes custom kernels written in both raw C++/CUDA and OpenAI Triton.
- **CUDA Kernels**: Implementations of fundamental operations including Matrix Multiplication (utilizing tiled shared memory), LayerNorm (using Welford's online variance algorithm), and Softmax (online reduction).
- **Triton Kernels**: High-level Python-based GPU kernels that rival raw CUDA performance, including a fused Flash-Attention implementation that avoids materializing the massive attention matrix in global memory.

### 8. Real-time Telemetry Dashboard
All data and live inference capabilities are surfaced through a FastAPI backend and a custom frontend dashboard. The dashboard provides a real-time Inference Playground utilizing Server-Sent Events (SSE) to stream text generation while simultaneously graphing throughput and monitoring GPU hardware states.

## Project Structure

- `core/`: System configuration, hardware detection, SQLite database management, and metric definitions.
- `benchmark_engine/`: Orchestration scripts for running automated sweeps and generating reports.
- `quantization/`: Implementations of FP16, INT8, and INT4 runners and precision degradation comparators.
- `batching/`: Static batching scripts and batch size saturation sweepers.
- `continuous_batching/`: The core iteration-level scheduler and request queue.
- `kv_cache/`: Memory block allocators and fragmentation analyzers.
- `speculative_decoding/`: Draft models, target verifiers, and rejection sampling logic.
- `prefix_caching/`: Radix trie data structures and cache engines.
- `profiling/`: PyTorch profiler wrappers and Roofline model plotting.
- `cuda_kernels/`: Raw C++ and CUDA source files with JIT compilation hooks.
- `triton_kernels/`: OpenAI Triton Python scripts for fused GPU operations.
- `dashboard/`: FastAPI web server, WebSocket streams, and HTML/CSS/JS frontend.
- `reports/`: Detailed technical documentation explaining the theory behind each module.

## Installation and Usage

The platform requires a Linux/Windows environment with a CUDA-capable NVIDIA GPU and Python 3.10+.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/prathamsingh404/TokenForge-GPU-Accelerated-LLM-Inference-Research-Platform.git
   cd TokenForge-GPU-Accelerated-LLM-Inference-Research-Platform
   ```

2. **Initialize the environment:**
   ```bash
   python setup_env.py
   python -m venv .venv
   source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard:**
   ```bash
   python -m dashboard.app
   ```
   Navigate to `http://localhost:8000` to access the real-time inference console and telemetry viewer.

4. **Run specific experiments:**
   Individual modules can be executed directly from the root directory to view terminal-based reports:
   ```bash
   python -m triton_kernels.benchmark
   python -m continuous_batching.benchmark
   ```

<div align="center">
  <img src="https://via.placeholder.com/150/0f172a/10b981?text=TokenForge" alt="TokenForge Logo" width="150"/>
  <h1>TokenForge</h1>
  <p><b>The open-source playground for studying, benchmarking, profiling, and optimizing LLM inference.</b></p>
  
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
  [![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

<br/>

**TokenForge** is an inference infrastructure ecosystem built in PyTorch and CUDA. It is designed to evaluate, benchmark, and optimize the inference performance of Large Language Models (LLMs) on consumer and enterprise GPU hardware.

Unlike a generic chatbot wrapper, TokenForge is a highly modular research platform that exposes the internal mechanics of modern LLM inference. It answers fundamental engineering questions: How fast can a model be served? Which mathematical techniques yield the highest performance gains? What is the impact of different scheduling strategies on P99 latency?

---

##  Performance Showcase

TokenForge implements iteration-level continuous batching, custom CUDA/Triton kernels, and advanced memory management to dramatically outperform naive HuggingFace inference.

### Throughput (Tokens/sec) vs. HuggingFace Transformers
*Measured on Llama-3-8B, RTX 4090, FP16, Batch Size 16, Sequence Length 1024*

| Engine | Throughput (Tokens/s) | Speedup | Time to First Token (TTFT) | Memory Fragmentation |
|--------|----------------------:|--------:|---------------------------:|---------------------:|
| HF `generate()` | ~450 tok/s | Baseline | 85 ms | High |
| HF + `compile` | ~620 tok/s | 1.37x | 65 ms | High |
| **TokenForge** | **~2,850 tok/s** | **6.33x** | **22 ms** | **Zero (Paged)** |

### KV Cache Memory Savings
*Measured on 32k context length*

| Feature | VRAM Footprint | Compression Ratio | Quality Loss |
|---------|---------------:|------------------:|-------------:|
| Standard FP16 | 16.0 GB | 1.0x | None |
| FP16 + INT8 (Medium Age) | 10.5 GB | 1.52x | Negligible |
| **TokenForge FP16+INT8+INT4** | **6.2 GB** | **2.58x** | **Minimal** |

---

## 🧩 Feature Matrix

How TokenForge compares to production inference engines:

| Feature | TokenForge | vLLM | TensorRT-LLM | HuggingFace |
|---------|:---:|:---:|:---:|:---:|
| **Continuous Batching** | ✅ | ✅ | ✅ | ❌ |
| **Paged KV Cache** | ✅ | ✅ | ✅ | ❌ |
| **Flash Attention** | ✅ (Triton) | ✅ | ✅ | ✅ (Opt) |
| **Pluggable Schedulers** | ✅ (FIFO, SRPT, Fair, Deadline) | ❌ (FIFO/Priority only) | ❌ | ❌ |
| **KV Cache Compression** | ✅ (Age-based FP16/INT8/INT4) | ❌ (Uniform only) | ✅ | ❌ |
| **Adaptive Eviction (H2O)** | ✅ | ❌ | ❌ | ❌ |
| **Energy/Power Metrics** | ✅ (Joules/token) | ❌ | ❌ | ❌ |
| **Hardware Simulator** | ✅ (TP/PP on single GPU) | ❌ | ❌ | ❌ |
| **Ease of Hacking/Extending** | ⭐⭐⭐⭐⭐ (Pure Python/PyTorch) | ⭐⭐⭐ (Complex C++) | ⭐ (Closed/Complex) | ⭐⭐⭐⭐ 

---

##  Quick Start

TokenForge requires a Linux/Windows environment with a CUDA-capable NVIDIA GPU and Python 3.10+.

```bash
# 1. Clone the repository
git clone https://github.com/prathamsingh404/TokenForge.git
cd TokenForge

# 2. Setup environment (creates .venv and installs dependencies)
python setup_env.py
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Run an automated benchmark simulation
tokenforge benchmark --model Qwen/Qwen2.5-1.5B --workload chatgpt --scheduler token_fair

# 4. Analyze results for automated optimization recommendations
tokenforge analyze
```

---

## 📊 Comprehensive Infrastructure Ecosystem

TokenForge provides a suite of tools for the full lifecycle of inference optimization.

### 1. Workload Simulation
Instead of basic "1 prompt, 1 user" testing, simulate real-world traffic patterns:
* **ChatGPT Traffic:** Bursty arrivals, Poisson distribution, varied prompt/output lengths.
* **Coding Assistant:** Extremely low latency constraints, short outputs.
* **RAG Workloads:** Massive input contexts, short deterministic outputs.

### 2. Pluggable Scheduler Framework
Easily swap scheduling strategies and visualize the results:
* **FIFO:** First-Come, First-Served continuous batching.
* **Shortest Remaining Processing Time (SRPT):** Minimizes average completion time.
* **Deadline-Aware:** Guarantees SLAs for real-time applications.
* **Token-Fair:** Prevents starvation, inspired by Linux CFS.

### 3. Advanced Memory Engineering
Push the limits of context length on constrained hardware:
* **Age-based Compression:** Recent tokens in FP16, medium in INT8, old in INT4.
* **Adaptive Eviction:** Drops least-attended tokens (H2O/StreamingLLM styles) rather than naive LRU.
* **Hierarchical Cache:** Seamlessly tier KV states across GPU HBM → CPU RAM → NVMe.

### 4. Distributed Inference Simulation
Study multi-GPU scaling without the hardware:
* **Tensor Parallelism Simulator:** Analyzes communication volume vs. compute time for column/row splits.
* **Pipeline Parallelism Simulator:** Models micro-batching and bubble overhead.
* **MoE Expert Parallelism:** Tracks load imbalance and all-to-all networking costs.

---

## 🛠️ CLI Reference

The `tokenforge` CLI is your gateway to the platform:

```bash
# Run a specific workload with a custom scheduler
tokenforge benchmark --model meta-llama/Llama-3.2-1B --workload coding --scheduler deadline_aware

# Profile a model at the kernel level
tokenforge profile --model Qwen/Qwen2.5-1.5B --depth kernel

# Compare two experiments
tokenforge compare --experiments exp_fifo,exp_srpt

# Start the interactive UI dashboard
tokenforge dashboard
```

---

## 🔭 Jupyter Notebook API

TokenForge acts as a Python library for researchers building custom pipelines:

```python
from tokenforge import Benchmark, WorkloadConfig

# Initialize benchmark engine
bench = Benchmark(model="Qwen/Qwen2.5-1.5B-Instruct")

# Define traffic pattern
traffic = WorkloadConfig.chatgpt_traffic()
traffic.num_users = 500

# Run with custom scheduler
result = bench.run(workload=traffic, scheduler="priority")

# Export interactive report
result.export_report("qwen_priority_analysis.html")
```

---

##  Architecture Overview

TokenForge is designed around strict decoupling. You can extend any component without touching the core engine.

- `tokenforge/cli.py`: Unified command-line interface.
- `tokenforge/model.py`: Unified architecture-aware model loader.
- `tokenforge/workloads/`: Poisson generators, trace replays, and workload definitions.
- `tokenforge/schedulers/`: ABCs and implementations for iteration-level scheduling.
- `tokenforge/cache/`: Advanced PagedAttention, mixed-precision, and hierarchical offloading.
- `tokenforge/distributed/`: Single-GPU simulators for Tensor, Pipeline, and Expert parallelism.
- `tokenforge/plugins/`: Extension registry for custom researchers.
- `tokenforge/visualization/`: Heatmaps, rooflines, and Gantt charts.
- `tokenforge/analyze.py`: Automated optimization recommendation engine.
- `core/`: Base PyTorch engine, SQLite telemetry database, and hardware metrics.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.


<!-- TokenForge Platform -->

"""
Model-agnostic loader with architecture detection.

Wraps HuggingFace AutoModel to provide a unified interface that
automatically detects model families, applies quantization, and
returns generation results with built-in timing metadata.

Usage:
    model = TokenForgeModel.load("Qwen/Qwen3-8B")
    model = TokenForgeModel.load("meta-llama/Llama-3.2-1B", quantization="int4")
    result = model.generate("Explain attention mechanisms.")
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from tokenforge.model_registry import detect_architecture, ArchitectureInfo


@dataclass
class GenerationResult:
    """Output of a single generation call with built-in timing."""
    text: str
    tokens_generated: int
    input_tokens: int
    total_time_s: float
    time_to_first_token_s: float
    tokens_per_second: float
    token_ids: list[int] = field(default_factory=list)

    @property
    def time_per_output_token_ms(self) -> float:
        if self.tokens_generated <= 1:
            return 0.0
        decode_time = self.total_time_s - self.time_to_first_token_s
        return (decode_time / max(self.tokens_generated - 1, 1)) * 1000


class TokenForgeModel:
    """
    Model-agnostic inference wrapper.

    Provides a single interface for loading any supported model
    family (Llama, Qwen, Gemma, Mistral, DeepSeek, Phi) with
    automatic architecture detection and quantization support.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config,
        arch_info: ArchitectureInfo,
        model_id: str,
        quantization: str,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.arch_info = arch_info
        self.model_id = model_id
        self.quantization = quantization
        self.device = device

    @classmethod
    def load(
        cls,
        model_id: str,
        quantization: str = "fp16",
        device: str = "auto",
        trust_remote_code: bool = False,
        max_memory: Optional[dict] = None,
    ) -> "TokenForgeModel":
        """
        Load any supported model with automatic optimization.

        Args:
            model_id: HuggingFace model ID or local path.
            quantization: Precision format — fp32, fp16, bf16, int8, int4, auto.
            device: Device placement — auto, cuda, cpu.
            trust_remote_code: Whether to trust remote code in model configs.
            max_memory: Per-device memory limits for device_map="auto".

        Returns:
            TokenForgeModel ready for generation.

        Example:
            model = TokenForgeModel.load("Qwen/Qwen3-8B")
            model = TokenForgeModel.load("meta-llama/Llama-3.2-1B", quantization="int4")
        """
        from rich.console import Console
        console = Console()

        # Detect architecture before loading weights
        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code,
        )
        arch_info = detect_architecture(model_id, config)

        console.print(f"[bold cyan]TokenForge[/] Loading [bold]{model_id}[/]")
        console.print(f"  Family: {arch_info.family}")
        console.print(f"  Layers: {arch_info.num_layers}")
        console.print(f"  Attention heads: {arch_info.num_attention_heads} "
                       f"(KV heads: {arch_info.num_kv_heads})")
        console.print(f"  Hidden size: {arch_info.hidden_size}")
        console.print(f"  Context length: {arch_info.max_position_embeddings}")
        if arch_info.is_moe:
            console.print(f"  MoE experts: {arch_info.num_experts} "
                           f"(active: {arch_info.num_experts_per_token})")
        console.print(f"  Quantization: {quantization}")

        # Build loading kwargs based on quantization
        load_kwargs = {"trust_remote_code": trust_remote_code}

        if max_memory:
            load_kwargs["max_memory"] = max_memory

        if device == "auto" and torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        elif device == "cuda":
            load_kwargs["device_map"] = "cuda"
        else:
            load_kwargs["device_map"] = None

        # Auto quantization: pick the best precision for available VRAM
        if quantization == "auto":
            quantization = cls._auto_quantization(arch_info)
            console.print(f"  [dim]Auto-selected: {quantization}[/]")

        if quantization == "fp32":
            load_kwargs["torch_dtype"] = torch.float32
        elif quantization == "fp16":
            load_kwargs["torch_dtype"] = torch.float16
        elif quantization == "bf16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif quantization == "int8":
            load_kwargs["load_in_8bit"] = True
        elif quantization == "int4":
            load_kwargs["load_in_4bit"] = True

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.eval()

        resolved_device = next(model.parameters()).device

        # Report VRAM usage
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            console.print(f"  VRAM used: {vram_mb:.0f} MB")

        console.print(f"  [bold green]✓ Model loaded[/]\n")

        return cls(
            model=model,
            tokenizer=tokenizer,
            config=config,
            arch_info=arch_info,
            model_id=model_id,
            quantization=quantization,
            device=resolved_device,
        )

    @staticmethod
    def _auto_quantization(arch_info: ArchitectureInfo) -> str:
        """Pick the best quantization for available VRAM."""
        if not torch.cuda.is_available():
            return "fp32"

        vram_mb = torch.cuda.get_device_properties(0).total_mem // (1024 ** 2)

        # Rough model size estimation (parameters × bytes_per_param)
        param_count = arch_info.estimated_param_count
        fp16_size_mb = (param_count * 2) / (1024 ** 2)

        if fp16_size_mb < vram_mb * 0.7:
            return "fp16"
        elif fp16_size_mb * 0.5 < vram_mb * 0.7:
            return "int8"
        else:
            return "int4"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        do_sample: bool = False,
        use_cache: bool = True,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate text with built-in timing instrumentation.

        Returns a GenerationResult with tokens, timing, and throughput.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        # Measure TTFT (first token)
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        start_time = time.perf_counter()

        with torch.no_grad():
            # Generate first token for TTFT measurement
            first_out = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                use_cache=use_cache,
            )

        torch.cuda.synchronize() if self.device.type == "cuda" else None
        ttft = time.perf_counter() - start_time

        # Full generation
        gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                use_cache=use_cache,
                **kwargs,
            )

        torch.cuda.synchronize() if self.device.type == "cuda" else None
        total_time = time.perf_counter() - gen_start

        # Decode
        generated_ids = outputs[0][input_length:].tolist()
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens_generated = len(generated_ids)

        tps = tokens_generated / total_time if total_time > 0 else 0

        return GenerationResult(
            text=text,
            tokens_generated=tokens_generated,
            input_tokens=input_length,
            total_time_s=total_time,
            time_to_first_token_s=ttft,
            tokens_per_second=tps,
            token_ids=generated_ids,
        )

    @property
    def architecture_info(self) -> dict:
        """Return architecture details as a dictionary."""
        return {
            "family": self.arch_info.family,
            "num_layers": self.arch_info.num_layers,
            "num_attention_heads": self.arch_info.num_attention_heads,
            "num_kv_heads": self.arch_info.num_kv_heads,
            "hidden_size": self.arch_info.hidden_size,
            "intermediate_size": self.arch_info.intermediate_size,
            "head_dim": self.arch_info.head_dim,
            "max_position_embeddings": self.arch_info.max_position_embeddings,
            "vocab_size": self.arch_info.vocab_size,
            "is_gqa": self.arch_info.is_gqa,
            "is_moe": self.arch_info.is_moe,
            "num_experts": self.arch_info.num_experts,
            "estimated_param_count": self.arch_info.estimated_param_count,
        }

    def estimate_kv_cache_mb(
        self,
        seq_len: int = 2048,
        batch_size: int = 1,
    ) -> float:
        """Estimate KV cache memory for a given sequence length."""
        ai = self.arch_info
        bytes_per = 2  # FP16
        total = (2 * ai.num_layers * batch_size * ai.num_kv_heads
                 * seq_len * ai.head_dim * bytes_per)
        return total / (1024 ** 2)

    def cleanup(self):
        """Release model memory."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return (
            f"TokenForgeModel('{self.model_id}', "
            f"quantization='{self.quantization}', "
            f"family='{self.arch_info.family}')"
        )

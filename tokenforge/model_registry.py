"""
Model architecture detection and registry.

Maps HuggingFace model configs to their architecture characteristics
(GQA, MoE, head counts, context length, etc.) without loading weights.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ArchitectureInfo:
    """Detected architecture characteristics of a model."""
    family: str  # llama, qwen, gemma, mistral, phi, deepseek, gpt2, etc.
    model_type: str  # Raw model_type from HF config

    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    hidden_size: int
    intermediate_size: int
    head_dim: int
    max_position_embeddings: int
    vocab_size: int

    # Grouped Query Attention
    is_gqa: bool

    # Mixture of Experts
    is_moe: bool
    num_experts: int = 0
    num_experts_per_token: int = 0

    # Rope / positional encoding
    rope_type: Optional[str] = None

    @property
    def estimated_param_count(self) -> int:
        """Rough parameter count estimate (no weights loaded)."""
        # Embedding + LM Head
        embed_params = self.vocab_size * self.hidden_size * 2

        # Per transformer layer:
        # Q, K, V projections + output projection
        attn_params = (
            self.hidden_size * self.num_attention_heads * self.head_dim  # Q
            + self.hidden_size * self.num_kv_heads * self.head_dim  # K
            + self.hidden_size * self.num_kv_heads * self.head_dim  # V
            + self.num_attention_heads * self.head_dim * self.hidden_size  # O
        )

        # FFN (gate + up + down for SwiGLU, or just up + down)
        ffn_params = self.hidden_size * self.intermediate_size * 3  # SwiGLU

        # LayerNorm (2 per layer)
        norm_params = self.hidden_size * 4

        if self.is_moe:
            # MoE: multiply FFN by num_experts
            layer_params = attn_params + (ffn_params * self.num_experts) + norm_params
        else:
            layer_params = attn_params + ffn_params + norm_params

        total = embed_params + (layer_params * self.num_layers)
        return total

    @property
    def param_count_str(self) -> str:
        count = self.estimated_param_count
        if count >= 1e9:
            return f"{count / 1e9:.1f}B"
        elif count >= 1e6:
            return f"{count / 1e6:.0f}M"
        return str(count)


# ─── Family Detection ─────────────────────────────────────────────────

# Maps HuggingFace model_type strings to TokenForge family names
_FAMILY_MAP = {
    "llama": "llama",
    "mistral": "mistral",
    "mixtral": "mistral",
    "qwen2": "qwen",
    "qwen2_moe": "qwen",
    "qwen3": "qwen",
    "qwen3_moe": "qwen",
    "gemma": "gemma",
    "gemma2": "gemma",
    "gemma3": "gemma",
    "phi": "phi",
    "phi3": "phi",
    "phimoe": "phi",
    "deepseek_v2": "deepseek",
    "deepseek_v3": "deepseek",
    "gpt2": "gpt2",
    "gpt_neox": "gpt_neox",
    "falcon": "falcon",
    "starcoder2": "starcoder",
    "cohere": "command",
    "cohere2": "command",
    "internlm2": "internlm",
}


def _detect_family(model_type: str) -> str:
    """Map model_type to a high-level family name."""
    model_type_lower = model_type.lower()
    return _FAMILY_MAP.get(model_type_lower, model_type_lower)


def detect_architecture(
    model_id: str,
    config,
) -> ArchitectureInfo:
    """
    Extract architecture information from a HuggingFace config.

    Works with any model that has a standard config (no weights loaded).
    """
    model_type = getattr(config, "model_type", "unknown")
    family = _detect_family(model_type)

    # Standard attribute extraction with fallbacks
    num_layers = getattr(config, "num_hidden_layers", 12)

    num_attention_heads = getattr(config, "num_attention_heads", 12)

    # GQA detection: num_key_value_heads < num_attention_heads
    num_kv_heads = getattr(
        config, "num_key_value_heads",
        num_attention_heads,  # Default: MHA (no GQA)
    )

    hidden_size = getattr(config, "hidden_size", 768)
    intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
    head_dim = hidden_size // num_attention_heads if num_attention_heads > 0 else 64

    max_position_embeddings = getattr(config, "max_position_embeddings", 2048)
    vocab_size = getattr(config, "vocab_size", 32000)

    is_gqa = num_kv_heads < num_attention_heads

    # MoE detection
    num_experts = getattr(config, "num_local_experts",
                          getattr(config, "num_experts", 0))
    num_experts_per_token = getattr(config, "num_experts_per_tok",
                                    getattr(config, "num_selected_experts", 0))
    is_moe = num_experts > 0

    # RoPE type
    rope_config = getattr(config, "rope_scaling", None)
    rope_type = None
    if rope_config and isinstance(rope_config, dict):
        rope_type = rope_config.get("type", rope_config.get("rope_type"))

    return ArchitectureInfo(
        family=family,
        model_type=model_type,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        vocab_size=vocab_size,
        is_gqa=is_gqa,
        is_moe=is_moe,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        rope_type=rope_type,
    )

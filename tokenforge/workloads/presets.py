"""
Pre-built workload presets for common inference traffic patterns.

Each preset represents a real-world deployment scenario with tuned
parameters for arrival patterns, prompt distributions, and output
lengths.

Usage:
    from tokenforge.workloads.presets import WORKLOAD_PRESETS

    config = WORKLOAD_PRESETS["chatgpt"]()
    requests = config.generate_requests()
"""

from tokenforge.workloads.generator import WorkloadConfig


# Registry of available workload presets
WORKLOAD_PRESETS = {
    "chatgpt": WorkloadConfig.chatgpt_traffic,
    "rag": WorkloadConfig.rag_traffic,
    "coding": WorkloadConfig.coding_assistant,
    "customer_support": WorkloadConfig.customer_support,
}


def list_presets() -> list[str]:
    """Return names of all available workload presets."""
    return list(WORKLOAD_PRESETS.keys())


def get_preset(name: str) -> WorkloadConfig:
    """Get a workload config by preset name."""
    if name not in WORKLOAD_PRESETS:
        available = ", ".join(WORKLOAD_PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return WORKLOAD_PRESETS[name]()


# ─── Preset Descriptions ──────────────────────────────────────────────

PRESET_DESCRIPTIONS = {
    "chatgpt": {
        "name": "ChatGPT Traffic",
        "description": "Conversational AI — many short prompts, Poisson arrivals, moderate output lengths.",
        "typical_use": "Consumer chatbot serving",
        "arrival": "poisson",
        "rate": "20 req/s",
        "prompt_range": "20–2000 chars",
        "output_range": "30–300 tokens",
    },
    "rag": {
        "name": "Enterprise RAG",
        "description": "Retrieved context + question — long prompts, bursty arrivals, short outputs.",
        "typical_use": "Enterprise knowledge bases, document Q&A",
        "arrival": "burst",
        "rate": "5 req/s (30 during bursts)",
        "prompt_range": "1000–8000 chars",
        "output_range": "20–200 tokens",
    },
    "coding": {
        "name": "Coding Assistant",
        "description": "Code context + completion — medium prompts, uniform arrivals, very short outputs.",
        "typical_use": "IDE autocomplete (Copilot-style)",
        "arrival": "uniform",
        "rate": "50 req/s",
        "prompt_range": "100–3000 chars",
        "output_range": "10–150 tokens",
    },
    "customer_support": {
        "name": "Customer Support",
        "description": "Short questions, detailed answers — low rate, steady, priority-based.",
        "typical_use": "Customer service chatbots with VIP tiers",
        "arrival": "poisson",
        "rate": "5 req/s",
        "prompt_range": "30–500 chars",
        "output_range": "50–400 tokens",
    },
}

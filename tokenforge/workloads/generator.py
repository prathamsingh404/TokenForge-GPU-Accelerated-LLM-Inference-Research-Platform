"""
Workload generator for realistic inference benchmarking.

Instead of "1 prompt, 1 user, measure speed," this generates
realistic request streams that model production traffic patterns.

Usage:
    # Pre-built profile
    config = WorkloadConfig.chatgpt_traffic()
    requests = config.generate_requests()

    # Custom configuration
    config = WorkloadConfig(
        num_users=500,
        arrival_pattern="poisson",
        arrival_rate=50.0,
        prompt_length_min=50,
        prompt_length_max=4000,
        output_length_min=20,
        output_length_max=500,
        duration_seconds=3600,
    )
    requests = config.generate_requests()
"""

import random
from dataclasses import dataclass, field
from typing import Optional

from tokenforge.workloads.arrival import (
    ArrivalEvent,
    poisson_arrivals,
    uniform_arrivals,
    burst_arrivals,
    ARRIVAL_GENERATORS,
)


# ─── Prompt Templates ─────────────────────────────────────────────────

# Realistic prompt snippets used to generate synthetic requests
# with varied content and lengths. These are combined/extended to
# reach the target prompt length.

_PROMPT_TEMPLATES = [
    "Explain the concept of {topic} in detail.",
    "Write a comprehensive analysis of {topic}.",
    "Compare and contrast {topic_a} with {topic_b}.",
    "Summarize the following document about {topic}:\n\n{context}",
    "Given the following context:\n\n{context}\n\nAnswer: {question}",
    "You are a helpful assistant. The user asks about {topic}. Respond thoroughly.",
    "Debug the following code:\n\n```python\n{code}\n```",
    "Translate the following to {language}:\n\n{text}",
    "Create a step-by-step tutorial for {topic}.",
    "What are the key differences between {topic_a} and {topic_b}?",
]

_TOPICS = [
    "transformer architectures", "GPU memory management", "CUDA kernels",
    "attention mechanisms", "KV cache optimization", "quantization",
    "continuous batching", "speculative decoding", "tensor parallelism",
    "flash attention", "roofline analysis", "memory bandwidth",
    "neural network training", "distributed systems", "operating systems",
    "compiler optimization", "database indexing", "network protocols",
    "machine learning pipelines", "cloud infrastructure",
]

_FILLER_TEXT = (
    "The fundamental theorem states that for any sufficiently complex system, "
    "there exists a set of optimizations that can reduce computational overhead "
    "while maintaining output quality within acceptable bounds. This principle "
    "applies broadly across domains including language modeling, scientific "
    "computing, and real-time inference systems. "
)


@dataclass
class SimulatedRequest:
    """A single request in the workload simulation."""
    request_id: str
    prompt: str
    max_new_tokens: int
    priority: int = 0  # Higher = more urgent
    deadline_ms: Optional[float] = None  # For deadline-aware scheduling
    arrival_time: float = 0.0  # Seconds from simulation start
    user_id: int = 0

    @property
    def prompt_tokens_estimate(self) -> int:
        """Rough token count estimate (4 chars per token heuristic)."""
        return max(1, len(self.prompt) // 4)


@dataclass
class WorkloadConfig:
    """
    Configurable traffic simulation for realistic benchmarking.

    Defines arrival patterns, prompt/output length distributions,
    and duration for simulating production inference traffic.
    """
    num_users: int = 100
    arrival_pattern: str = "poisson"  # poisson | uniform | burst
    arrival_rate: float = 10.0  # requests/sec

    prompt_length_min: int = 50  # characters
    prompt_length_max: int = 4000
    output_length_min: int = 20  # tokens
    output_length_max: int = 500

    duration_seconds: float = 60.0

    # Priority distribution (for priority scheduling)
    priority_levels: int = 3  # 0=low, 1=medium, 2=high
    priority_weights: list[float] = field(
        default_factory=lambda: [0.6, 0.3, 0.1],  # 60% low, 30% med, 10% high
    )

    # Deadline distribution (for deadline-aware scheduling)
    deadline_min_ms: float = 500.0
    deadline_max_ms: float = 10000.0

    # Burst-specific parameters
    burst_rate: float = 100.0  # req/s during bursts
    burst_duration: float = 5.0  # seconds
    burst_interval: float = 30.0  # seconds between burst starts

    # Reproducibility
    seed: Optional[int] = 42

    @classmethod
    def chatgpt_traffic(cls) -> "WorkloadConfig":
        """ChatGPT-style conversational traffic.

        Characteristics:
        - Many short prompts, occasional long ones
        - Medium output lengths
        - Poisson arrivals (independent users)
        - Moderate request rate
        """
        return cls(
            num_users=200,
            arrival_pattern="poisson",
            arrival_rate=20.0,
            prompt_length_min=20,
            prompt_length_max=2000,
            output_length_min=30,
            output_length_max=300,
            duration_seconds=120,
            seed=42,
        )

    @classmethod
    def rag_traffic(cls) -> "WorkloadConfig":
        """Enterprise RAG (Retrieval-Augmented Generation) traffic.

        Characteristics:
        - Long prompts (retrieved documents included)
        - Short to medium outputs
        - Bursty (batch queries)
        - Lower overall rate
        """
        return cls(
            num_users=50,
            arrival_pattern="burst",
            arrival_rate=5.0,
            prompt_length_min=1000,
            prompt_length_max=8000,
            output_length_min=20,
            output_length_max=200,
            duration_seconds=120,
            burst_rate=30.0,
            burst_duration=3.0,
            burst_interval=20.0,
            seed=42,
        )

    @classmethod
    def coding_assistant(cls) -> "WorkloadConfig":
        """Coding assistant traffic (Copilot-style).

        Characteristics:
        - Medium prompts (code context)
        - Short outputs (completions)
        - High rate, latency-sensitive
        - Uniform arrivals (IDE keystrokes)
        """
        return cls(
            num_users=100,
            arrival_pattern="uniform",
            arrival_rate=50.0,
            prompt_length_min=100,
            prompt_length_max=3000,
            output_length_min=10,
            output_length_max=150,
            duration_seconds=60,
            deadline_min_ms=200.0,
            deadline_max_ms=2000.0,
            seed=42,
        )

    @classmethod
    def customer_support(cls) -> "WorkloadConfig":
        """Customer support chatbot traffic.

        Characteristics:
        - Short prompts (customer questions)
        - Medium outputs (detailed answers)
        - Lower rate, steady
        - Priority-based (VIP customers)
        """
        return cls(
            num_users=30,
            arrival_pattern="poisson",
            arrival_rate=5.0,
            prompt_length_min=30,
            prompt_length_max=500,
            output_length_min=50,
            output_length_max=400,
            duration_seconds=300,
            priority_levels=3,
            priority_weights=[0.5, 0.35, 0.15],
            seed=42,
        )

    def generate_requests(self) -> list[SimulatedRequest]:
        """
        Generate the full set of simulated requests.

        Returns a list of SimulatedRequest objects sorted by arrival time,
        each with a synthetic prompt, output length target, priority,
        and optional deadline.
        """
        rng = random.Random(self.seed)

        # Generate arrival times
        if self.arrival_pattern == "poisson":
            arrivals = poisson_arrivals(
                rate=self.arrival_rate,
                duration=self.duration_seconds,
                seed=self.seed,
            )
        elif self.arrival_pattern == "uniform":
            arrivals = uniform_arrivals(
                rate=self.arrival_rate,
                duration=self.duration_seconds,
                jitter=0.1,
                seed=self.seed,
            )
        elif self.arrival_pattern == "burst":
            arrivals = burst_arrivals(
                base_rate=self.arrival_rate,
                burst_rate=self.burst_rate,
                duration=self.duration_seconds,
                burst_duration=self.burst_duration,
                burst_interval=self.burst_interval,
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unknown arrival pattern: {self.arrival_pattern}")

        # Build requests
        requests = []
        for event in arrivals:
            prompt = self._generate_prompt(rng)
            output_len = rng.randint(self.output_length_min, self.output_length_max)
            priority = self._sample_priority(rng)
            deadline = rng.uniform(self.deadline_min_ms, self.deadline_max_ms)
            user_id = rng.randint(0, self.num_users - 1)

            requests.append(SimulatedRequest(
                request_id=f"req_{event.request_index:06d}",
                prompt=prompt,
                max_new_tokens=output_len,
                priority=priority,
                deadline_ms=deadline,
                arrival_time=event.timestamp,
                user_id=user_id,
            ))

        return requests

    def _generate_prompt(self, rng: random.Random) -> str:
        """Generate a synthetic prompt of the target length."""
        target_len = rng.randint(self.prompt_length_min, self.prompt_length_max)

        # Start with a template
        template = rng.choice(_PROMPT_TEMPLATES)
        topic = rng.choice(_TOPICS)
        topic_b = rng.choice(_TOPICS)

        prompt = template.format(
            topic=topic,
            topic_a=topic,
            topic_b=topic_b,
            context=_FILLER_TEXT[:min(target_len // 2, len(_FILLER_TEXT))],
            question=f"What is the impact of {topic}?",
            code="def optimize(x): return x * 2  # TODO: implement",
            language="Spanish",
            text="The inference system processes requests efficiently.",
        )

        # Extend to target length with filler text
        while len(prompt) < target_len:
            prompt += " " + _FILLER_TEXT

        return prompt[:target_len]

    def _sample_priority(self, rng: random.Random) -> int:
        """Sample a priority level based on configured weights."""
        r = rng.random()
        cumulative = 0.0
        for level, weight in enumerate(self.priority_weights):
            cumulative += weight
            if r <= cumulative:
                return level
        return 0

    def summary(self) -> str:
        """Human-readable summary of the workload configuration."""
        expected_requests = int(self.arrival_rate * self.duration_seconds)
        return (
            f"WorkloadConfig(\n"
            f"  users={self.num_users}, "
            f"arrival={self.arrival_pattern} @ {self.arrival_rate} req/s,\n"
            f"  prompts={self.prompt_length_min}-{self.prompt_length_max} chars, "
            f"outputs={self.output_length_min}-{self.output_length_max} tokens,\n"
            f"  duration={self.duration_seconds}s, "
            f"~{expected_requests} total requests\n"
            f")"
        )

    def __repr__(self) -> str:
        return self.summary()

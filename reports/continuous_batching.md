# Continuous Batching (Iteration-Level Scheduling)

Traditional "Static Batching" waits for all sequences in a batch to complete before launching a new batch. If one sequence generates 100 tokens and another generates 5, the GPU sits idle for 95 steps waiting for the longer sequence.

## The Continuous Batching Algorithm
Continuous Batching operates at the *iteration level* (token-by-token):
1. Compute a forward pass for all active sequences.
2. Identify completed sequences and immediately evict them.
3. Immediately inject new sequences from the queue into the empty slots of the active batch.
4. The new sequences undergo a "prefill" phase while the existing sequences undergo a "decode" phase.

This drastically improves overall hardware utilization, often yielding 10x-20x throughput increases in production serving environments compared to static batching.


<!-- TokenForge Platform -->

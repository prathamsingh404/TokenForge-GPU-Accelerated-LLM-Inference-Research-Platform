# Speculative Decoding

Autoregressive decoding is heavily memory-bandwidth bound. For an LLM to generate one token, it must load its entire multi-gigabyte weight matrix from HBM into the SM registers. 

Speculative Decoding breaks this bottleneck by guessing tokens in advance.

## The Draft & Verify Loop
1. **Draft Phase**: A small, lightning-fast model (the "Draft Model") autoregressively generates $K$ tokens. Because the model is tiny, this is very fast.
2. **Verify Phase**: The large, accurate model (the "Target Model") processes all $K$ drafted tokens in a *single forward pass*. Since the Target model processes them in parallel, it only loads its weights once.
3. **Acceptance**: We compare the Target Model's output distribution against the Draft Model's distribution using Rejection Sampling. If the draft matches the target's prediction, the token is accepted. 
4. Upon the first rejection, the sequence is corrected by sampling from the modified target distribution, and the draft loop restarts.

This provides mathematically identical outputs to the Target model, but at 2x-3x the speed.

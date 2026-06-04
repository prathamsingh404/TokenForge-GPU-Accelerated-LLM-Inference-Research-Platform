"""
Simplified Fused Attention implementation using OpenAI Triton.
Mimics FlashAttention by fusing the matmuls and softmax.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    q = tl.load(Q_block_ptr)
    q = (q * sm_scale).to(tl.float16)
    
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        acc_scale = l_i * 0 + alpha
        acc = acc * acc_scale[:, None]
        
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        
    acc = acc / l_i[:, None]
    
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(tl.float16))

def attention_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    
    sm_scale = 1.0 / (Lq ** 0.5)
    batch, heads, seq_len, d_head = q.shape
    
    o = torch.empty_like(q)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)
    
    _fwd_kernel[grid](
        q, k, v, sm_scale,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch, heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=d_head
    )
    
    return o

def benchmark_triton_attention():
    from rich.console import Console
    from rich.table import Table
    console = Console()
    sizes = [(1, 8, 256, 64), (1, 8, 512, 64), (1, 8, 1024, 64)]

    table = Table(title="Triton Fused Attention Benchmark")
    table.add_column("Shape (B, H, Seq, D)", justify="right")
    table.add_column("Triton (ms)", justify="right", style="green")
    table.add_column("PyTorch (ms)", justify="right", style="yellow")
    
    for b, h, s, d in sizes:
        q = torch.randn((b, h, s, d), device='cuda', dtype=torch.float16)
        k = torch.randn((b, h, s, d), device='cuda', dtype=torch.float16)
        v = torch.randn((b, h, s, d), device='cuda', dtype=torch.float16)
        
        # Warmup
        for _ in range(5):
            attention_triton(q, k, v)
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
            
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(20):
            attention_triton(q, k, v)
        end.record()
        torch.cuda.synchronize()
        triton_ms = start.elapsed_time(end) / 20
        
        start.record()
        for _ in range(20):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
        torch_ms = start.elapsed_time(end) / 20
        
        table.add_row(f"{b}x{h}x{s}x{d}", f"{triton_ms:.4f}", f"{torch_ms:.4f}")
        
    console.print(table)

if __name__ == "__main__":
    q = torch.randn((1, 4, 128, 64), device='cuda', dtype=torch.float16)
    k = torch.randn((1, 4, 128, 64), device='cuda', dtype=torch.float16)
    v = torch.randn((1, 4, 128, 64), device='cuda', dtype=torch.float16)
    
    out_triton = attention_triton(q, k, v)
    out_torch = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    assert torch.allclose(out_triton, out_torch, atol=1e-2), "Triton attention mismatch!"
    print("Triton Fused Attention: PASS")
    benchmark_triton_attention()

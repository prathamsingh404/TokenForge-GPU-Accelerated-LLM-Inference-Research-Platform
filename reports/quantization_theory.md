# Quantization Theory & Practice

Quantization aims to reduce the memory footprint and bandwidth requirements of LLM weights and activations by mapping high-precision floats to lower-precision integers.

## Floating Point 16 (FP16/BF16)
The baseline for modern inference. Preserves dynamic range while halving VRAM requirements compared to FP32. 

## 8-Bit Integer (INT8)
Vector-wise quantization requires dynamic scales.
$$ X_{int8} = \text{round}\left( \frac{X_{fp16}}{S} \right) $$
Where $S = \frac{\max(|X|)}{127}$. 
While weights are quantized, activations often contain outliers. SmoothQuant addresses this by migrating the quantization difficulty from activations to weights.

## 4-Bit NormalFloat (NF4)
Information-theoretically optimal data type for normally distributed weights. BitsAndBytes implements this by estimating the quantiles of a standard normal distribution and mapping weight tensors into those 16 buckets. Perfect for running large models on consumer GPUs (e.g., RTX 5050 8GB).


<!-- TokenForge Platform -->

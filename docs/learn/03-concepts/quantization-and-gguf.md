# Quantization and GGUF

## Why a 7B model does not need 28 GB

A model's parameters are numbers. Trained in 16-bit floats (FP16 or BF16), a 7 billion
parameter model is 14 GB of weights; in 32-bit it is 28 GB. Consumer hardware has 8 to 24 GB
of GPU memory and free-tier VMs have 1 to 24 GB of RAM. Quantization stores each weight in
fewer bits, typically 4 to 8, so the same 7B model fits in 4 to 7 GB and, just as important,
moves through memory faster. Generation is memory-bandwidth bound: each new token requires
reading every weight once, so halving the bytes roughly doubles tokens per second on CPU.

The cost is precision. Weights are grouped into blocks (often 32), each block stores a scale
(and sometimes a minimum), and the individual weights become small integers relative to that
scale. Modern schemes keep the sensitive tensors (attention output, embeddings) at higher
precision and squeeze the rest. The quality loss at 4 bits is usually small for 7B+ models and
noticeable for tiny models, which is why the default here is Q4_K_M on a 0.5B model only
because it must fit CI and free CPUs; on the Oracle VM you would use a 3B or 7B.

## Reading a quant name

GGUF files from the llama.cpp ecosystem use names like `qwen2.5-0.5b-instruct-q4_k_m.gguf`.

| Token | Meaning |
|---|---|
| `Q4` | about 4 bits per weight |
| `_K` | "k-quants": blocks with per-block scales and mixed precision for important tensors |
| `_S` / `_M` / `_L` | small, medium, large: how many tensors are kept at higher precision |
| `Q8_0` | 8 bits, legacy scheme, nearly lossless, twice the size of Q4 |
| `Q2_K`, `IQ2_XS` | 2 bit, aggressive, visible quality loss, for fitting 70B into 24 GB |
| `F16` / `BF16` | unquantised |

Rules of thumb: Q4_K_M is the default trade-off; Q5_K_M or Q6_K when you have the memory and
want quality; Q8_0 for small models where every bit of quality matters and the file is small
anyway. A model's quality floor drops faster as the model gets smaller, so quantise a 70B hard
and a 1B gently.

Size estimate: `parameters * bits / 8`. Qwen2.5-0.5B at Q4_K_M is about 400 MB; Llama 3.1 8B at
Q4_K_M is about 4.9 GB; at Q8_0 about 8.5 GB. Add 10 to 30 percent for the KV cache depending on
context length.

## GGUF, the file format

GGUF ("GPT-Generated Unified Format") is a single binary file holding both metadata and tensors:
the architecture, vocabulary and tokenizer, chat template, context length, quantisation type per
tensor, and the weights laid out for memory mapping. One file is the whole model. Because it
mmaps, loading is fast and the OS page cache does the work; two processes loading the same file
share pages.

This is what makes "the same model on any hardware" literal. The GGUF does not know whether it
will run on AVX2, Metal or CUDA; the ggml backend chosen at build time decides, and
`n_gpu_layers` decides how many layers go to the accelerator. The hardware module in this repo
(`openinfer/hardware/detect.py`) recommends that number.

Hugging Face hosts GGUF conversions for most open models, often in repositories named `*-GGUF`
by the model vendor (Qwen publishes its own) or by community quantisers. `openinfer
download-model` uses `huggingface_hub.hf_hub_download` to fetch one file from such a repo
(`openinfer/cli.py`).

## Other quantisation worlds

GPU-serving engines use different formats: GPTQ and AWQ (4-bit, calibrated on sample data,
optimised for CUDA kernels), FP8 (on Hopper and newer), and bitsandbytes (on-the-fly 8/4-bit for
PyTorch). vLLM consumes those, not GGUF (support is partial). This is a real seam in the
ecosystem and one reason OpenInfer treats vLLM as an HTTP backend rather than trying to share
weights with llama.cpp.

## Practical choices in this repo

- Default model: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`, file `qwen2.5-0.5b-instruct-q4_k_m.gguf`.
  Small enough for Hugging Face's free CPU tier and for a developer's first run.
- Oracle free tier (24 GB RAM): a 3B Q4_K_M (2 GB) or 7B Q4_K_M (5 GB) are comfortable.
- `n_ctx` in `openinfer.yaml` sets context length; KV cache memory grows linearly with it, so
  do not set 32k on a 2 GB machine.

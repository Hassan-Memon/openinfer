# Hardware accelerators: CPU, Metal, CUDA, ROCm, NPU

LLM inference is matrix multiplication and memory traffic. Every accelerator below is a
different answer to "how do we do a lot of multiply-adds while reading weights as fast as
possible". `openinfer/hardware/detect.py` probes for each and `recommend()` picks the llama.cpp
build; this lesson explains what it is looking at.

## CPU

Any CPU runs the model; what matters is SIMD width and memory bandwidth. SIMD instructions
process several numbers per instruction: SSE4.2 does 4 floats, AVX2 does 8, AVX-512 does 16,
and ARM NEON does 4 (with newer dot-product instructions that help quantised math a lot).
llama.cpp picks kernels at build time based on these flags, so a wheel compiled on an AVX-512
machine will crash on an AVX2 one, which is why the extra compiles locally.

Detection: Linux `/proc/cpuinfo` flags, macOS `sysctl machdep.cpu.*`, Windows
`IsProcessorFeaturePresent` via ctypes (the constants are in the file). Memory bandwidth is not
probed, but it is the real ceiling: a laptop at 50 GB/s reading a 4 GB model cannot exceed
about 12 tokens/s no matter how many cores it has. That matches the 16 tokens/s the bench
measured here for a 1.5B model.

Thread count: more is not better past the physical core count. The recommendation uses
`cores - 1` to leave the OS and the server a core.

## Apple Metal

Apple silicon has unified memory: CPU and GPU address the same RAM, so a 24 GB MacBook can
load a 20 GB model to the GPU without copying. Metal is Apple's GPU API; ggml has a Metal
backend that runs the matrix kernels there. Build with `CMAKE_ARGS="-DGGML_METAL=on"` and set
`n_gpu_layers=-1`. Detection is by platform (`Darwin` on `arm64`); the report notes
`unified_memory: true` because the fit calculation is different from a discrete GPU.

## NVIDIA CUDA

CUDA is NVIDIA's compute platform and the reason "GPU" and "NVIDIA" are near synonyms in ML.
The driver exposes `nvidia-smi`, which the detector queries for name, memory, driver version and
compute capability (the architecture generation; 8.6 is Ampere consumer, 8.9 Ada, 9.0 Hopper).
llama.cpp builds with `-DGGML_CUDA=on`; vLLM assumes CUDA. The lock-in this project pushes
against is not CUDA's quality, it is that most serving stacks only work there.

## AMD ROCm

ROCm is AMD's answer to CUDA, with HIP as a source-compatible API. Support is narrower: a list
of officially supported cards, Linux first, and a known trick (`HSA_OVERRIDE_GFX_VERSION`) for
consumer cards. ggml builds with `-DGGML_HIPBLAS=on`. The detector shells out to `rocm-smi`.
vLLM runs on ROCm with effort; llama.cpp and Ollama make it routine, which is why the
recommendation sends AMD users to llama.cpp.

## NPUs: Intel, Qualcomm, Apple Neural Engine

Neural processing units are low-power matrix engines designed for always-on inference (camera,
speech). Intel's "AI Boost" NPU ships in Core Ultra laptops, Qualcomm's Hexagon in Snapdragon X
devices, Apple's Neural Engine in every A/M chip. They are efficient but have their own toolchains:
OpenVINO for Intel, Qualcomm's QNN through ONNX Runtime, Core ML for Apple. llama.cpp does not
target them (as of this writing), so the detector reports an NPU and recommends keeping llama.cpp
on CPU while pointing at OpenVINO or ONNX Runtime exports for NPU work. Honesty here beats a
checkbox: an NPU backend would be a separate adapter.

Detection: OpenVINO's `Core().available_devices` if installed, `/dev/accel` on Linux, and the
processor identifier string for Snapdragon.

## What the recommendation produces

```json
{"backend": "llamacpp", "accelerator": "cuda", "n_gpu_layers": -1,
 "build": "CMAKE_ARGS='-DGGML_CUDA=on' uv sync --extra llamacpp",
 "note": "Offload all layers to NVIDIA GeForce RTX 4070; vLLM is the better choice for high concurrency on this hardware."}
```

Order of preference: CUDA, ROCm, Metal, then CPU with a note about which SIMD level was found
and a suggested thread count. `GET /v1/hardware` returns the same JSON so a fleet can be
inventoried over HTTP.

## Where to look

- `openinfer/hardware/detect.py`: every probe has a 3 s timeout and swallows errors; detection
  must never take the server down.
- `tests/test_hardware.py`: synthetic reports exercise each recommendation branch.
- `docs/deploy/oracle-always-free.md`: the ARM NEON path end to end.

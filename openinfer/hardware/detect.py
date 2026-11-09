"""Hardware detection: what can this machine run a model on?

Everything here is best effort and never raises. Each probe has a short timeout, swallows its
own errors and reports what it could learn. The result is plain JSON so `/v1/hardware`,
`openinfer hardware` and the bench can all use it.

See docs/learn/03-concepts/hardware-accelerators-cpu-metal-cuda-rocm-npu.md for what each
accelerator is and why the recommendation logic looks the way it does.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROBE_TIMEOUT_S = 3.0


def _run(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


# -- CPU ----------------------------------------------------------------------------------

_WINDOWS_FEATURES = {
    # Constants from winnt.h for IsProcessorFeaturePresent
    "sse4_2": 38,  # PF_SSE4_2_INSTRUCTIONS_AVAILABLE
    "avx": 39,  # PF_AVX_INSTRUCTIONS_AVAILABLE
    "avx2": 40,  # PF_AVX2_INSTRUCTIONS_AVAILABLE
    "avx512f": 41,  # PF_AVX512F_INSTRUCTIONS_AVAILABLE
    "neon": 19,  # PF_ARM_NEON_INSTRUCTIONS_AVAILABLE
}

_INTERESTING_FLAGS = (
    "sse4_2",
    "avx",
    "avx2",
    "avx512f",
    "avx512_vnni",
    "avx_vnni",
    "amx_tile",
    "fma",
    "f16c",
    "neon",
    "asimd",
    "sve",
)


def cpu_features() -> list[str]:
    system = platform.system()
    flags: set[str] = set()
    try:
        if system == "Linux":
            text = Path("/proc/cpuinfo").read_text("utf-8", "ignore")
            for line in text.splitlines():
                if line.lower().startswith(("flags", "features")):
                    flags.update(line.split(":", 1)[1].split())
                    break
        elif system == "Darwin":
            for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
                out = _run(["sysctl", "-n", key])
                if out:
                    flags.update(f.lower().replace(".", "_") for f in out.split())
            if platform.machine() == "arm64":
                flags.add("neon")
        elif system == "Windows":
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            for name, const in _WINDOWS_FEATURES.items():
                if kernel32.IsProcessorFeaturePresent(const):
                    flags.add(name)
    except Exception:  # noqa: BLE001 - detection must never fail the server
        pass
    return sorted(f for f in flags if f in _INTERESTING_FLAGS)


def cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text("utf-8", "ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
        elif system == "Windows":
            return os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine()


def memory_total_gb() -> float | None:
    system = platform.system()
    try:
        if system == "Linux":
            for line in Path("/proc/meminfo").read_text("utf-8", "ignore").splitlines():
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
        elif system == "Darwin":
            out = _run(["sysctl", "-n", "hw.memsize"])
            return round(int(out) / 1024**3, 1) if out else None
        elif system == "Windows":

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
            return round(status.ullTotalPhys / 1024**3, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


# -- accelerators --------------------------------------------------------------------------


def detect_cuda() -> list[dict[str, Any]]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        gpu: dict[str, Any] = {"type": "cuda", "vendor": "nvidia", "name": parts[0]}
        with contextlib.suppress(ValueError):
            gpu["memory_gb"] = round(float(parts[1]) / 1024, 1)
        if len(parts) > 2:
            gpu["driver"] = parts[2]
        if len(parts) > 3:
            gpu["compute_capability"] = parts[3]
        gpus.append(gpu)
    return gpus


def detect_rocm() -> list[dict[str, Any]]:
    out = _run(["rocm-smi", "--showproductname", "--csv"])
    if not out:
        return []
    gpus = []
    for line in out.splitlines()[1:]:
        if not line.strip() or line.lower().startswith("device"):
            continue
        parts = [p.strip() for p in line.split(",")]
        name = next((p for p in parts[1:] if p and p.lower() not in ("n/a", "")), "AMD GPU")
        gpus.append({"type": "rocm", "vendor": "amd", "name": name})
    return gpus or [{"type": "rocm", "vendor": "amd", "name": "AMD GPU (rocm-smi present)"}]


def detect_metal() -> list[dict[str, Any]]:
    if platform.system() != "Darwin":
        return []
    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple GPU"
    acc: dict[str, Any] = {"type": "metal", "vendor": "apple", "name": chip}
    mem = memory_total_gb()
    if mem is not None:
        # Unified memory: the GPU can address (most of) system RAM.
        acc["memory_gb"] = mem
        acc["unified_memory"] = True
    return [acc]


def detect_npu() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    # Intel NPU through OpenVINO if installed.
    try:
        import openvino as ov  # type: ignore[import-not-found]

        devices = ov.Core().available_devices
        if any(d.startswith("NPU") for d in devices):
            found.append({"type": "npu", "vendor": "intel", "name": "Intel NPU (OpenVINO)"})
    except Exception:  # noqa: BLE001
        pass
    # Linux exposes Intel NPUs as /dev/accel/accel*.
    if platform.system() == "Linux" and not found:
        accel = Path("/dev/accel")
        if accel.exists() and any(accel.iterdir()):
            found.append({"type": "npu", "vendor": "intel", "name": "Intel NPU (/dev/accel)"})
    # Qualcomm Snapdragon X (Windows on ARM) carries a Hexagon NPU.
    ident = (os.environ.get("PROCESSOR_IDENTIFIER", "") + " " + cpu_name()).lower()
    if "snapdragon" in ident or "qualcomm" in ident:
        found.append({"type": "npu", "vendor": "qualcomm", "name": "Qualcomm Hexagon NPU"})
    return found


# -- report ---------------------------------------------------------------------------------


def detect() -> dict[str, Any]:
    features = cpu_features()
    accelerators = detect_cuda() + detect_rocm() + detect_metal() + detect_npu()
    report = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "container": Path("/.dockerenv").exists(),
        },
        "cpu": {
            "name": cpu_name(),
            "arch": platform.machine(),
            "cores_logical": os.cpu_count(),
            "features": features,
            "memory_total_gb": memory_total_gb(),
        },
        "accelerators": accelerators,
    }
    report["recommended"] = recommend(report)
    return report


def recommend(report: dict[str, Any]) -> dict[str, Any]:
    """Pick the llama.cpp build and offload settings that fit the detected hardware."""
    types = {a["type"] for a in report.get("accelerators", [])}
    features = set(report.get("cpu", {}).get("features", []))
    mem = report.get("cpu", {}).get("memory_total_gb") or 0
    threads = report.get("cpu", {}).get("cores_logical") or 4

    if "cuda" in types:
        gpu = next(a for a in report["accelerators"] if a["type"] == "cuda")
        return {
            "backend": "llamacpp",
            "accelerator": "cuda",
            "n_gpu_layers": -1,
            "build": "CMAKE_ARGS='-DGGML_CUDA=on' uv sync --extra llamacpp",
            "note": f"Offload all layers to {gpu.get('name', 'the GPU')}; "
            "vLLM is the better choice for high concurrency on this hardware.",
        }
    if "rocm" in types:
        return {
            "backend": "llamacpp",
            "accelerator": "rocm",
            "n_gpu_layers": -1,
            "build": "CMAKE_ARGS='-DGGML_HIPBLAS=on' uv sync --extra llamacpp",
            "note": "Set HSA_OVERRIDE_GFX_VERSION if your card is not officially supported.",
        }
    if "metal" in types:
        return {
            "backend": "llamacpp",
            "accelerator": "metal",
            "n_gpu_layers": -1,
            "build": "CMAKE_ARGS='-DGGML_METAL=on' uv sync --extra llamacpp",
            "note": f"Unified memory ({mem} GB) is shared between CPU and GPU; "
            "pick a quant that leaves a few GB free.",
        }
    if "npu" in types:
        return {
            "backend": "llamacpp",
            "accelerator": "cpu",
            "n_gpu_layers": 0,
            "build": "uv sync --extra llamacpp",
            "note": "An NPU was detected but llama.cpp does not target it yet; use OpenVINO or "
            "ONNX Runtime (QNN) exports for NPU inference and keep llama.cpp on CPU.",
        }
    simd = (
        "AVX-512"
        if "avx512f" in features
        else "AVX2"
        if "avx2" in features
        else "NEON"
        if {"neon", "asimd"} & features
        else "baseline"
    )
    return {
        "backend": "llamacpp",
        "accelerator": "cpu",
        "n_gpu_layers": 0,
        "n_threads": max(1, threads - 1),
        "build": "uv sync --extra llamacpp",
        "note": f"CPU only ({simd}); use Q4_K_M quantisations of 0.5B to 3B models for "
        "interactive latency, or route heavy work to a cloud alias.",
    }

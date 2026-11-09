from __future__ import annotations

from openinfer.hardware import detect, recommend


def test_detect_returns_report_with_expected_keys():
    report = detect()
    assert report["cpu"]["cores_logical"] >= 1
    assert isinstance(report["cpu"]["features"], list)
    assert isinstance(report["accelerators"], list)
    assert report["recommended"]["backend"] == "llamacpp"
    assert report["recommended"]["accelerator"] in {"cpu", "cuda", "rocm", "metal"}


def base_report(**overrides):
    report = {
        "cpu": {"features": ["avx2"], "memory_total_gb": 16, "cores_logical": 8},
        "accelerators": [],
    }
    report.update(overrides)
    return report


def test_recommend_cuda():
    rec = recommend(base_report(accelerators=[{"type": "cuda", "name": "RTX 4090"}]))
    assert rec["accelerator"] == "cuda" and rec["n_gpu_layers"] == -1
    assert "GGML_CUDA" in rec["build"]


def test_recommend_rocm_metal_npu():
    assert recommend(base_report(accelerators=[{"type": "rocm"}]))["accelerator"] == "rocm"
    metal = recommend(base_report(accelerators=[{"type": "metal"}]))
    assert metal["accelerator"] == "metal" and "GGML_METAL" in metal["build"]
    npu = recommend(base_report(accelerators=[{"type": "npu", "vendor": "intel"}]))
    assert npu["accelerator"] == "cpu" and "OpenVINO" in npu["note"]


def test_recommend_cpu_variants():
    assert (
        "AVX-512"
        in recommend(base_report(cpu={"features": ["avx512f"], "cores_logical": 4}))["note"]
    )
    rec = recommend(base_report(cpu={"features": ["neon"], "cores_logical": 4}))
    assert "NEON" in rec["note"] and rec["n_threads"] == 3
    assert "baseline" in recommend(base_report(cpu={"features": [], "cores_logical": 1}))["note"]

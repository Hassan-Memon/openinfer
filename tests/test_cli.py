from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from openinfer import __version__
from openinfer.cli import app
from tests.conftest import ECHO_YAML

runner = CliRunner()


def write_config(tmp_path: Path) -> str:
    path = tmp_path / "openinfer.yaml"
    path.write_text(ECHO_YAML, "utf-8")
    return str(path)


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_hardware_json():
    result = runner.invoke(app, ["hardware", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "recommended" in data


def test_hardware_human():
    result = runner.invoke(app, ["hardware"])
    assert result.exit_code == 0
    assert "Recommended" in result.output


def test_models_json(tmp_path):
    result = runner.invoke(app, ["models", "--config", write_config(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    ids = [c["id"] for c in json.loads(result.output)]
    assert "fast" in ids and "echo/echo" in ids


def test_models_table(tmp_path):
    result = runner.invoke(app, ["models", "--config", write_config(tmp_path)])
    assert result.exit_code == 0
    assert "echo/echo" in result.output


def test_bench_markdown_and_json(tmp_path):
    cfg = write_config(tmp_path)
    result = runner.invoke(app, ["bench", "--config", cfg, "--runs", "1", "--max-tokens", "8"])
    assert result.exit_code == 0, result.output
    assert "| echo | echo | local |" in result.output
    result = runner.invoke(app, ["bench", "--config", cfg, "--runs", "1", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)[0]["backend"] == "echo"


def test_config_masks_secrets(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "server: {api_key: topsecret}\n"
        "backends:\n  g: {type: cloud, preset: groq, api_key: sk-123}\n",
        "utf-8",
    )
    result = runner.invoke(app, ["config", "--config", str(path)])
    assert result.exit_code == 0
    assert "topsecret" not in result.output and "sk-123" not in result.output
    assert "********" in result.output


def test_missing_config_exits_2(tmp_path):
    result = runner.invoke(app, ["models", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2


def test_download_model_without_extra_is_explicit(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("no hf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = runner.invoke(app, ["download-model"])
    assert result.exit_code == 2
    assert "llamacpp" in result.output

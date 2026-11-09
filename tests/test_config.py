from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from openinfer.config import default_config_text, expand_env, load_config, parse_config

ROOT = Path(__file__).resolve().parents[1]


def test_expand_env_basic_and_defaults():
    env = {"A": "x", "PORT": "9000", "EMPTY": ""}
    assert expand_env("v: ${A}", env) == "v: x"
    assert expand_env("v: ${MISSING:-dflt}", env) == "v: dflt"
    assert expand_env("v: ${MISSING}", env) == "v: "
    assert expand_env("v: ${EMPTY:-fallback}", env) == "v: fallback"
    assert expand_env("port: ${PORT:-8000}", env) == "port: 9000"


def test_expand_env_quotes_yaml_unsafe_values():
    env = {"URL": "sqlite:///:memory:", "FLAG": "true", "N": "42", "PG": "postgresql://u:p@h/db"}
    text = expand_env("url: ${URL}\nflag: ${FLAG}\nn: ${N}\npg: ${PG}\nq: '${URL}'", env)
    data = yaml.safe_load(text)
    assert data["url"] == "sqlite:///:memory:"
    assert data["flag"] is True
    assert data["n"] == 42
    assert data["pg"] == "postgresql://u:p@h/db"
    assert data["q"] == "sqlite:///:memory:"  # already quoted placeholder left verbatim


def test_cloud_backend_without_key_is_inactive():
    cfg = parse_config(
        """
backends:
  echo: {type: echo}
  groq: {type: cloud, preset: groq, api_key_env: GROQ_API_KEY}
  disabled: {type: echo, enabled: false}
""",
        env={},
    )
    assert set(cfg.active_backends({})) == {"echo"}
    assert set(cfg.active_backends({"GROQ_API_KEY": "k"})) == {"echo", "groq"}
    assert cfg.backends["groq"].resolved_api_key({"GROQ_API_KEY": "k"}) == "k"


def test_root_example_matches_packaged_default():
    root = (ROOT / "openinfer.yaml").read_text("utf-8")
    assert root == default_config_text()


def test_packaged_default_is_safe_with_no_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENINFER_CONFIG", raising=False)
    for var in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cfg = load_config()
    assert cfg.source == "<packaged default>"
    active = set(cfg.active_backends())
    assert "echo" in active
    assert not active & {"groq", "openrouter", "gemini", "openai", "llamacpp", "vllm"}
    assert cfg.aliases["private"][-1] == "echo/echo"
    # Every alias chain entry points at a configured backend
    for chain in cfg.aliases.values():
        for item in chain:
            assert item.partition("/")[0] in cfg.backends
    assert cfg.server.api_key is None
    assert cfg.server.port == 8000


def test_env_overrides_in_default_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENINFER_CONFIG", raising=False)
    cfg = load_config(
        env={
            "OPENINFER_PORT": "7860",
            "OPENINFER_API_KEY": "abc",
            "OPENINFER_LEDGER_URL": "sqlite:///:memory:",
            "GROQ_API_KEY": "k",
        }
    )
    assert cfg.server.port == 7860
    assert cfg.server.api_key == "abc"
    assert cfg.ledger.url == "sqlite:///:memory:"
    assert "groq" in cfg.active_backends({"GROQ_API_KEY": "k"})


def test_explicit_path_and_missing_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("backends:\n  echo: {type: echo}\n", "utf-8")
    cfg = load_config(path)
    assert cfg.source == str(path)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_openinfer_config_env_var(monkeypatch, tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text("policy: {retries: 7}\n", "utf-8")
    monkeypatch.setenv("OPENINFER_CONFIG", str(path))
    assert load_config().policy.retries == 7

"""`openinfer` command line: serve, bench, hardware, models, config, download-model."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from openinfer import __version__
from openinfer.backends import build_backends
from openinfer.config import AppConfig, load_config

app = typer.Typer(
    name="openinfer",
    help="One model, any hardware. OpenAI-compatible inference with policy routing.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

ConfigOpt = Annotated[
    str | None,
    typer.Option("--config", "-c", help="Path to openinfer.yaml (default: ./openinfer.yaml)."),
]


def _version(value: bool) -> None:
    if value:
        console.print(f"openinfer {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """OpenInfer CLI."""


def _load(config: str | None) -> AppConfig:
    try:
        return load_config(config)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


# -- serve -----------------------------------------------------------------------------------------


@app.command()
def serve(
    config: ConfigOpt = None,
    host: Annotated[str | None, typer.Option(help="Bind host (env OPENINFER_HOST).")] = None,
    port: Annotated[int | None, typer.Option(help="Bind port (env OPENINFER_PORT).")] = None,
    reload: Annotated[bool, typer.Option(help="Auto reload on code changes (dev only).")] = False,
    log_level: Annotated[str, typer.Option(help="uvicorn log level.")] = "info",
) -> None:
    """Start the OpenAI-compatible HTTP server."""
    import uvicorn

    cfg = _load(config)
    if config:
        os.environ["OPENINFER_CONFIG"] = config
    bind_host = host or os.environ.get("OPENINFER_HOST") or cfg.server.host
    bind_port = port or int(os.environ.get("OPENINFER_PORT") or cfg.server.port)
    console.print(
        f"[bold]OpenInfer {__version__}[/bold] on http://{bind_host}:{bind_port} "
        f"(config: {cfg.source}; backends: {', '.join(cfg.active_backends()) or 'none'})"
    )
    uvicorn.run(
        "openinfer.server.app:create_app",
        factory=True,
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_level=log_level,
    )


# -- hardware --------------------------------------------------------------------------------------


@app.command()
def hardware(
    as_json: Annotated[bool, typer.Option("--json", help="Print raw JSON.")] = False,
) -> None:
    """Detect CPU features and accelerators (CUDA, ROCm, Metal, NPU) and recommend a build."""
    from openinfer.hardware import detect

    report = detect()
    if as_json:
        typer.echo(json.dumps(report, indent=2))
        return
    cpu = report["cpu"]
    osr = report["os"]
    console.print(f"[bold]OS[/bold]        {osr['system']} {osr['release']} ({osr['machine']})")
    console.print(f"[bold]CPU[/bold]       {cpu['name']}")
    console.print(
        f"[bold]Cores[/bold]     {cpu['cores_logical']}   "
        f"[bold]RAM[/bold] {cpu['memory_total_gb']} GB   "
        f"[bold]SIMD[/bold] {', '.join(cpu['features']) or 'none detected'}"
    )
    if report["accelerators"]:
        table = Table(title="Accelerators")
        table.add_column("Type")
        table.add_column("Vendor")
        table.add_column("Name")
        table.add_column("Memory (GB)", justify="right")
        for acc in report["accelerators"]:
            table.add_row(
                acc["type"],
                acc.get("vendor", ""),
                acc.get("name", ""),
                str(acc.get("memory_gb", "")),
            )
        console.print(table)
    else:
        console.print("[yellow]No GPU or NPU detected; CPU inference via llama.cpp.[/yellow]")
    rec = report["recommended"]
    console.print(
        f"\n[bold]Recommended[/bold] {rec['backend']} on {rec['accelerator']}: {rec['note']}"
    )
    console.print(f"[dim]build: {rec['build']}[/dim]")


# -- models ----------------------------------------------------------------------------------------


@app.command()
def models(
    config: ConfigOpt = None, as_json: Annotated[bool, typer.Option("--json")] = False
) -> None:
    """List aliases and the concrete models behind them."""
    from openinfer.ledger.sqlite import SQLiteLedger
    from openinfer.router import Router

    cfg = _load(config)
    backends = build_backends(cfg)
    router = Router(cfg, backends, SQLiteLedger("sqlite:///:memory:"))
    cards = router.model_cards()
    if as_json:
        typer.echo(json.dumps(cards, indent=2))
        return
    table = Table(title=f"Models ({cfg.source})")
    table.add_column("id")
    table.add_column("kind")
    table.add_column("where")
    table.add_column("chain / backend")
    for card in cards:
        meta = card["openinfer"]
        if meta["kind"] == "alias":
            chain = " -> ".join(meta["chain"]) or "[red]no enabled backends[/red]"
            table.add_row(card["id"], "alias", "", chain)
        else:
            table.add_row(
                card["id"],
                meta["kind"],
                "local" if meta["local"] else "cloud",
                meta["backend_type"],
            )
    console.print(table)
    inactive = sorted(set(cfg.backends) - set(backends))
    if inactive:
        console.print(
            f"[dim]Inactive backends (disabled or missing API key): {', '.join(inactive)}[/dim]"
        )


# -- bench -----------------------------------------------------------------------------------------


@app.command()
def bench(
    config: ConfigOpt = None,
    backend: Annotated[
        list[str] | None, typer.Option("--backend", "-b", help="Only these backends (repeatable).")
    ] = None,
    runs: Annotated[int, typer.Option(min=1, help="Runs per backend.")] = 3,
    max_tokens: Annotated[int, typer.Option(min=1, help="max_tokens per run.")] = 128,
    prompt: Annotated[str | None, typer.Option(help="Prompt to send.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of a table.")] = False,
    include_unreachable: Annotated[
        bool, typer.Option(help="Try backends even if their health probe fails.")
    ] = False,
) -> None:
    """Benchmark tokens/s, TTFT and $/1M tokens across configured backends (markdown table)."""
    from openinfer.bench import markdown_table, run_bench
    from openinfer.bench.run import DEFAULT_PROMPT

    cfg = _load(config)
    backends = build_backends(cfg)

    async def go() -> list[Any]:
        try:
            return await run_bench(
                cfg,
                backends,
                only=backend,
                prompt=prompt or DEFAULT_PROMPT,
                runs=runs,
                max_tokens=max_tokens,
                skip_unhealthy=not include_unreachable,
            )
        finally:
            await asyncio.gather(*(b.aclose() for b in backends.values()), return_exceptions=True)

    results = asyncio.run(go())
    if as_json:
        typer.echo(json.dumps([r.to_dict() for r in results], indent=2))
        return
    typer.echo(markdown_table(results))


# -- config ----------------------------------------------------------------------------------------


@app.command(name="config")
def show_config(config: ConfigOpt = None) -> None:
    """Print the effective configuration with secrets masked."""
    cfg = _load(config)
    data = cfg.model_dump(mode="json")
    if data["server"].get("api_key"):
        data["server"]["api_key"] = "********"
    for b in data["backends"].values():
        if b.get("api_key"):
            b["api_key"] = "********"
    typer.echo(json.dumps(data, indent=2))


# -- download-model --------------------------------------------------------------------------------


@app.command(name="download-model")
def download_model(
    repo: Annotated[
        str, typer.Option(help="Hugging Face repo id.")
    ] = "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    filename: Annotated[
        str, typer.Option(help="GGUF file inside the repo.")
    ] = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    dest: Annotated[Path, typer.Option(help="Destination directory.")] = Path("models"),
) -> None:
    """Download a small GGUF (default Qwen2.5-0.5B-Instruct Q4_K_M, ~400 MB) for llama.cpp.

    Requires the llamacpp extra: uv sync --extra llamacpp
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        console.print("[red]huggingface_hub is not installed.[/red] Run: uv sync --extra llamacpp")
        raise typer.Exit(code=2) from exc
    from openinfer.backends.llamacpp import llamacpp_available

    if not llamacpp_available():
        console.print(
            "[yellow]llama-cpp-python is not importable; downloading anyway, but the llamacpp "
            "backend will stay unavailable until `uv sync --extra llamacpp` succeeds.[/yellow]"
        )
    dest.mkdir(parents=True, exist_ok=True)
    console.print(f"Downloading {repo}/{filename} to {dest}/ ...")
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(dest),
        token=os.environ.get("HF_TOKEN") or None,
    )
    console.print(f"[green]Saved[/green] {path}")
    console.print(
        "Next: set LLAMACPP_MODEL_PATH to that file (or edit openinfer.yaml), set the llamacpp "
        "backend `enabled: true`, then `openinfer serve`."
    )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    app()

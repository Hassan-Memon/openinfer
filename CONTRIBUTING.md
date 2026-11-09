# Contributing to OpenInfer

Thanks for taking the time. This is a small project with a clear shape, so contributions
that keep it small and clear are the most welcome.

## Set up

```bash
git clone https://github.com/Hassan-Memon/openinfer
cd openinfer
uv sync                      # core + dev tools
uv run pytest                # all tests run offline on the echo backend
uv run ruff check . && uv run ruff format --check .
uv run openinfer serve       # http://localhost:8000/docs
```

Optional extras: `uv sync --extra llamacpp`, `--extra whisper`, `--extra metrics`, `--extra postgres`.

## Ground rules

- Python 3.12+, type hints everywhere, `ruff` clean. Line length 100.
- Tests must pass with no network and no API keys. Anything that talks to a real backend
  must be mocked (see `tests/test_backends_http.py` for the `httpx.MockTransport` pattern)
  or skipped unless an environment variable opts in.
- New backends implement `openinfer.backends.base.Backend` and register in
  `openinfer/backends/__init__.py`. Keep provider quirks inside the adapter; the router and
  server must never know which vendor they are talking to.
- Routing behaviour changes need a unit test in `tests/test_router.py` and, if they change
  the policy schema, an update to `openinfer.yaml` and the routing lesson in `docs/learn/`.
- Decisions with lasting consequences get an ADR in `docs/adr/` (MADR format, copy the
  latest one as a template).
- Prose: plain English, no em-dashes, no marketing adjectives in code comments.

## Commits and pull requests

- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`.
- One logical change per commit. A PR should be reviewable in under 15 minutes.
- CI (ruff + pytest + docker build) must be green.

## Reporting bugs

Open an issue with: OpenInfer version, OS and accelerator (`openinfer hardware` output),
the relevant part of `openinfer.yaml` with secrets removed, and a request that reproduces it.
For security issues see `SECURITY.md`.

# Security policy

## Supported versions

Only the latest release on `main` receives fixes. This is pre-1.0 software.

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's private
vulnerability reporting on the repository ("Security" tab, "Report a vulnerability").
You should get an acknowledgement within 72 hours and a fix or mitigation plan within
14 days for anything rated high or critical.

## Threat model in one paragraph

OpenInfer sits between your applications and model backends. It holds API keys for cloud
providers (in environment variables only, never on disk by the application), it sees every
prompt, and it writes token counts and costs to a ledger. It does not store prompt or
completion text in the ledger. The main risks are therefore: an exposed server with no
auth, a misconfigured policy that sends private data to a cloud backend, and leaked keys.

## What the project does about it

- `OPENINFER_API_KEY` enables bearer auth on all `/v1/*` routes. Set it for anything
  reachable beyond localhost. `/health` stays open for load balancers.
- Requests tagged `private` are routed to local backends only. Cloud candidates are
  removed before any network call; if nothing local is available the request fails closed
  (HTTP 422) instead of falling through to the cloud. Tests in `tests/test_router.py`
  pin this behaviour.
- Keys are read from the environment at startup. The config file references them by
  variable name (`api_key_env`), so `openinfer.yaml` is safe to commit.
- The ledger stores metadata only: backend, model, token counts, cost, latency, status.
- The Docker image runs as a non-root user and ships without compilers or uv.
- Dependencies are pinned through `uv.lock`; CI runs with `--frozen`.

## Hardening checklist for a public deployment

1. Set `OPENINFER_API_KEY` to a long random value and rotate it like any other credential.
2. Put a TLS-terminating proxy in front (Render, Hugging Face Spaces and Caddy on Oracle all do this).
3. Set `policy.max_cost_usd` in `openinfer.yaml` so a runaway client cannot drain a paid key.
4. Prefer provider keys with spending limits and scoped permissions.
5. Review `docs/learn/03-concepts/routing-policies-and-fallbacks.md` before adding cloud
   backends to the `private` alias chain (do not).

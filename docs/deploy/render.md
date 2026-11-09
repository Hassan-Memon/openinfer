# Deploy to Render (free web service)

Render's free tier runs one Docker web service with 512 MB RAM and 0.1 CPU, spins it down after
15 minutes idle, and gives you HTTPS and a `*.onrender.com` hostname. It is a good fit for the
API gateway role: OpenInfer in the cloud routing to Groq, Gemini and OpenRouter free tiers, with
`private` requests refused because nothing local exists there (fail closed).

Do not expect to run a model on it: 512 MB is below even a 0.5B GGUF plus runtime.

## Steps

1. Push this repo to GitHub.
2. Render dashboard, New, **Web Service**, connect the repo.
3. Runtime: **Docker**. Render detects the `Dockerfile`.
4. Instance type: **Free**.
5. Environment variables:

   | Key | Value |
   |---|---|
   | `OPENINFER_PORT` | `10000` (Render injects `PORT=10000`; both work, this is explicit) |
   | `OPENINFER_API_KEY` | a long random string |
   | `GROQ_API_KEY` | optional |
   | `GEMINI_API_KEY` | optional |
   | `OPENROUTER_API_KEY` | optional |
   | `OPENINFER_LEDGER_URL` | `sqlite:////data/openinfer.db` (ephemeral) or a Neon/Supabase URL |

6. Health check path: `/health?deep=false`.
7. Deploy. First build takes 3 to 5 minutes.

## render.yaml (optional, infrastructure as code)

```yaml
services:
  - type: web
    name: openinfer
    runtime: docker
    plan: free
    healthCheckPath: /health?deep=false
    envVars:
      - key: OPENINFER_PORT
        value: "10000"
      - key: OPENINFER_API_KEY
        generateValue: true
      - key: GROQ_API_KEY
        sync: false
```

## Verify

```bash
curl https://openinfer.onrender.com/v1/models -H "Authorization: Bearer $OPENINFER_API_KEY"
```

## Trade-offs

- Cold starts of 30 to 50 s after idle. Use a free uptime pinger if that matters, or accept it for a demo.
- The disk resets on every deploy; use Postgres for a ledger you want to keep.
- 0.1 CPU is fine for the router (it mostly waits on network) but `openinfer bench` numbers from here reflect the provider, not Render.

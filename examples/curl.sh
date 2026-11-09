#!/usr/bin/env sh
# Smoke test an OpenInfer server with curl. Usage: sh examples/curl.sh [base_url]
set -eu
BASE="${1:-http://localhost:8000}"
AUTH="${OPENINFER_API_KEY:+Authorization: Bearer $OPENINFER_API_KEY}"

echo "# health"
curl -sS "$BASE/health" -H "${AUTH:-X-None: 1}"; echo

echo "# models"
curl -sS "$BASE/v1/models" -H "${AUTH:-X-None: 1}" | head -c 400; echo

echo "# chat (fast alias)"
curl -sS "$BASE/v1/chat/completions" -H 'content-type: application/json' -H "${AUTH:-X-None: 1}" \
  -d '{"model":"fast","messages":[{"role":"user","content":"ping"}]}'; echo

echo "# chat, private, streaming"
curl -sSN "$BASE/v1/chat/completions" -H 'content-type: application/json' -H "${AUTH:-X-None: 1}" \
  -H 'X-OpenInfer-Tags: private' \
  -d '{"model":"private","stream":true,"messages":[{"role":"user","content":"stay local"}]}'; echo

echo "# ledger"
curl -sS "$BASE/v1/ledger?hours=1&limit=3" -H "${AUTH:-X-None: 1}"; echo

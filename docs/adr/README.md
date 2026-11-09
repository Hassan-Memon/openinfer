# Architecture Decision Records

MADR format: context, decision drivers, considered options, decision, consequences. One file per
decision, numbered, never edited after acceptance except to mark them superseded.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-openai-compatible-api.md) | Expose an OpenAI-compatible API rather than a native one | accepted |
| [0002](0002-router-policy-as-yaml.md) | Routing policy as a declarative YAML file | accepted |
| [0003](0003-llamacpp-first.md) | llama.cpp (GGUF) as the first local engine, vLLM and Ollama over HTTP | accepted |
| [0004](0004-sqlite-ledger.md) | SQLite as the default cost ledger, Postgres optional | accepted |
| [0005](0005-privacy-routing-fail-closed.md) | Privacy routing: tag means local only and fails closed | accepted |
| [0006](0006-echo-backend-for-tests-and-demos.md) | A deterministic echo backend as a first-class citizen | accepted |

Template for new records: copy `0006` and renumber.

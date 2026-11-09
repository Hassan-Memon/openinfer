# ADR-0005: Privacy routing: a tag means local only, and it fails closed

- Status: accepted
- Date: 2026-11-09
- Deciders: Hassan Memon

## Context and problem statement

Some requests carry data that must not leave the machine or network: patient intake, legal
matters, source code under NDA. Other requests from the same application are fine to send to a
cloud model and benefit from doing so. The router needs a rule that is simple enough to trust
and strict enough to defend in a compliance review.

## Decision drivers

- A privacy guarantee that depends on "the local model being up" is not a guarantee.
- Developers should express intent once, at the call site, not re-implement provider checks.
- Compliance regimes (HIPAA technical safeguards, GDPR data transfer rules, UAE and KSA PDPL
  localisation) care about where processing happens; the policy must make that auditable.
- The mechanism must be testable offline.

## Considered options

1. **Separate deployments**: one OpenInfer with cloud backends, another with local only. Simple,
   but pushes the decision into the application's URL selection and doubles infrastructure.
2. **Per-alias convention only** (`private` alias lists local backends). Works until someone
   edits the chain and adds a cloud model "just as a fallback".
3. **Request tag enforced by the router**: a `private` tag removes every non-local candidate
   before any network call; if nothing is left the request fails with 422 rather than falling
   through.

## Decision

Option 3, with option 2 as defence in depth. Each backend declares `local: bool` in code, not
configuration (`openinfer/backends/*.py`), so an operator cannot accidentally mark Groq as
local. The router's `plan()` filters on that flag whenever the request carries
`policy.private_tag` (default `private`) or asks for an alias with that name. Tags arrive via
`X-OpenInfer-Tags`, the `openinfer` body object or OpenAI `metadata.privacy`.

When the filter removes every candidate the router raises `NoCandidates` (HTTP 422) with a
message that names the rejected cloud candidates. It never degrades to a cloud backend.
`tests/test_router.py` and `tests/test_server_chat.py` pin both behaviours: cloud is never
contacted, and cloud-only chains fail closed.

The ledger records the tags on every row, so "which requests were private and where did they
run" is one query.

## Consequences

- Positive: the privacy property is a code-level invariant with tests, not a configuration
  habit. Reviewers can point at one function.
- Positive: applications express intent in one header and can use the better cloud model for
  everything else.
- Negative: fail closed means a visible error when the local engine is down. That is the point,
  but it needs monitoring (`/health`, `openinfer_requests_total{status="error"}`).
- Negative: "local" is a coarse notion. A vLLM box in your own VPC is local; a hosted vLLM is
  not, yet both use the same adapter. Operators must configure hosted OpenAI-compatible servers
  as `cloud` with a custom `base_url`, which the config supports. A per-backend `local`
  override is deliberately not offered.
- Neutral: privacy routing does not redact anything. Redaction before a cloud call belongs to
  the application (IntakeAgent does this), OpenInfer only guarantees placement.

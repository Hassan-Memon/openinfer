"""Use OpenInfer from the official openai SDK. Run: uv run python examples/openai_sdk_client.py

Only the base_url differs from talking to OpenAI. The `openinfer` object on the response and
the X-OpenInfer-* headers tell you which backend served the request.
"""

from __future__ import annotations

import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("OPENINFER_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("OPENINFER_API_KEY") or "unused",
)

# 1. Non-streaming, default alias chain.
completion = client.chat.completions.create(
    model="fast",
    messages=[{"role": "user", "content": "Give me one sentence about quantization."}],
)
print(completion.choices[0].message.content)
print("served by:", completion.model_extra.get("openinfer"))  # type: ignore[union-attr]

# 2. Private: local backends only. Fails with 422 if nothing local is available.
private = client.chat.completions.create(
    model="private",
    messages=[{"role": "user", "content": "Patient reports mild headache since Monday."}],
    extra_headers={"X-OpenInfer-Tags": "private"},
)
print("private answer:", private.choices[0].message.content[:80])

# 3. Streaming with a cost ceiling expressed in the body.
stream = client.chat.completions.create(
    model="fast",
    messages=[{"role": "user", "content": "Count to ten."}],
    stream=True,
    extra_body={"openinfer": {"max_cost_usd": 0.0005}},
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

# 4. Embeddings.
emb = client.embeddings.create(model="embed", input=["hello", "world"])
print("embedding dims:", len(emb.data[0].embedding))

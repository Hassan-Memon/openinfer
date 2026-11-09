"""LangChain against OpenInfer. Requires: uv pip install langchain-openai

ChatOpenAI only needs base_url and a dummy key. Route by alias and pass OpenInfer hints through
default_headers or extra_body exactly as you would with the raw SDK.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]

llm = ChatOpenAI(
    model="private",
    base_url=os.environ.get("OPENINFER_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("OPENINFER_API_KEY") or "unused",
    default_headers={"X-OpenInfer-Tags": "private"},
    temperature=0,
)

note = "Summarise this intake note in one line: headache since Monday, no fever."
print(llm.invoke(note).content)

"""Answer generation with model fallback."""

import time
from typing import Dict, List

from openai import OpenAI

from config import config

_client = None

# Transient server-side conditions worth retrying / falling through on.
RETRYABLE = {408, 429, 500, 502, 503, 504}


def client() -> OpenAI:
    global _client
    if _client is None:
        config.validate()
        _client = OpenAI(api_key=config.google_api_key, base_url=config.api_base_url)
    return _client


def build_context(hits: List[Dict]) -> str:
    """Number and label passages so the model can cite them."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        page = f", p.{hit['page']}" if hit.get("page") else ""
        blocks.append(f"[{i}] ({hit.get('source','unknown')}{page})\n{hit['text']}")
    return "\n\n---\n\n".join(blocks)


def generate(query: str, context: str, system_prompt: str = None) -> str:
    """Try each configured model in order; fall through on transient errors."""
    system_prompt = system_prompt or config.system_prompt
    last_error = None

    for model in config.llm_models:
        for attempt in range(config.max_retries_per_model):
            try:
                response = client().chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Question: {query}\n\nContext:\n{context}",
                        },
                    ],
                    temperature=config.temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                status = getattr(e, "status_code", None)
                if status in RETRYABLE:
                    if attempt < config.max_retries_per_model - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    break  # next model
                raise

    raise RuntimeError(
        f"All models failed ({', '.join(config.llm_models)}). Last error: {last_error}"
    )

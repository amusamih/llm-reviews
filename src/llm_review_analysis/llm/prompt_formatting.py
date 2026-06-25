from __future__ import annotations

from typing import Any


def user_chat_messages(prompt: str, *, response_format: str | None = None) -> list[dict[str, str]]:
    """Return a provider-neutral one-turn chat message list."""

    return [{"role": "user", "content": adapted_prompt(prompt, response_format=response_format)}]


def adapted_prompt(prompt: str, *, response_format: str | None = None) -> str:
    if response_format == "json":
        return (
            "Return only a valid JSON object. Do not include Markdown fences or explanatory text.\n\n"
            f"{prompt}"
        )
    return prompt


def text_from_content_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text is not None:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text is not None:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part)
    return str(content or "")

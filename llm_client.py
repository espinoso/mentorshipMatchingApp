"""Claude LLM client: takes a system + user prompt and returns raw text.

The caller parses the JSON (see parse_json). Every LLM call in the app goes through here.
"""

from __future__ import annotations

import json
import re

_JSON_INSTRUCTION = "Respond with a single valid JSON object and nothing else."


class LLMError(Exception):
    """Raised when the model call fails or returns unusable output."""


class ClaudeClient:
    def __init__(self, model: str, api_key: str):
        if not api_key:
            raise LLMError(
                "No Anthropic API key provided. Set ANTHROPIC_API_KEY in the app's "
                "secrets (or environment) before generating groups."
            )
        self.model = model
        self.api_key = api_key

    def complete_json(
        self, system: str, user: str, *, max_tokens: int, cache_system: bool = False
    ) -> str:
        """Stream one request and return the response text (expected to be a JSON object).

        max_tokens must cover any thinking plus the JSON; it is streamed so a large value
        doesn't hit the HTTP timeout. No temperature/top_p: current Claude models reject them.
        cache_system marks the system prompt as an ephemeral prompt-cache prefix.
        """
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise LLMError("The 'anthropic' package is not installed.") from e

        text = f"{system}\n\n{_JSON_INSTRUCTION}"
        system_param = (
            [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
            if cache_system
            else text
        )
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            with client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system_param,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                resp = stream.get_final_message()
        except Exception as e:
            raise LLMError(f"Anthropic request failed: {e}") from e
        if resp.stop_reason == "max_tokens":
            raise LLMError(
                "The model's response was cut off before it finished - the cohort may "
                "be too large for a single run. Try again, and if it persists, split "
                "the group or reduce how many people are matched at once."
            )
        return "".join(block.text for block in resp.content if block.type == "text")


def parse_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from a model response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip ```json fences or surrounding prose, grab the outermost { ... }.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise LLMError(f"Model did not return valid JSON: {e}") from e
    raise LLMError("Model response contained no JSON object.")

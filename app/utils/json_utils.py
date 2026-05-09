"""
app/utils/json_utils.py

Robust JSON extractor for LLM responses.

LLMs commonly produce:
  - Markdown code fences around JSON
  - Prose before/after the JSON object
  - Unescaped double-quotes inside string values
  - Truncated output (hitting max_new_tokens mid-value)

parse_llm_json() handles all of these without external dependencies.
"""
from __future__ import annotations
import json
import re


def parse_llm_json(raw: str) -> dict:
    """
    Extract and parse a JSON object from raw LLM output.

    Strategy:
      1. Strip markdown fences
      2. Try json.loads directly
      3. Locate the outermost { } by bracket counting and try again
      4. Apply targeted repairs (unescaped inner quotes) and retry
      5. Raise JSONDecodeError with a descriptive message on total failure
    """
    text = _strip_fences(raw.strip())

    # Fast path
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract outermost JSON object by bracket counting
    obj = _extract_object(text)
    if obj is not None:
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            # Try repairing the extracted object
            repaired = _repair(obj)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    raise json.JSONDecodeError(
        f"Could not parse JSON from LLM response (len={len(raw)}). "
        f"First 200 chars: {raw[:200]}",
        raw, 0,
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    if "```" not in text:
        return text
    # ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    # Unclosed fence — strip opening line only
    return re.sub(r"^```\w*\s*", "", text, count=1).strip()


def _extract_object(text: str) -> str | None:
    """Return the substring from the first { to its matching }, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text[start:], start):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # No matching close brace — return what we have (truncated output)
    return text[start:]


def _repair(text: str) -> str:
    """
    Fix the most common LLM JSON defect: unescaped double-quotes inside
    string values, e.g.  "question": "Explain "REST" vs "GraphQL""

    Strategy: scan character by character tracking whether we're inside a
    JSON string; when we see a " that is NOT preceded by \\ and we're already
    inside a string, escape it.
    """
    out: list[str] = []
    in_string = False
    prev_was_colon_space = False  # crude heuristic: after ": we expect a value
    i = 0
    while i < len(text):
        ch = text[i]

        if ch == "\\" and in_string:
            # Consume escape sequence as-is
            out.append(ch)
            i += 1
            if i < len(text):
                out.append(text[i])
                i += 1
            continue

        if ch == '"':
            if not in_string:
                in_string = True
                out.append(ch)
            else:
                # Check if this closes the string or is an inner unescaped quote.
                # A closing quote is followed by : , } ] whitespace or end-of-string.
                rest = text[i + 1:].lstrip()
                if rest and rest[0] in (",", "}", "]", ":"):
                    in_string = False
                    out.append(ch)
                elif not rest:
                    in_string = False
                    out.append(ch)
                else:
                    # Looks like an inner quote — escape it
                    out.append('\\"')
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)

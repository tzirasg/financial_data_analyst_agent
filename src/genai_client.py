"""
src/genai_client.py — Ανθεκτικός wrapper γύρω από το Gemini API.

Το gemini-3.6-flash επιστρέφει συχνά 503 (high demand). Εδώ:
  - exponential backoff ΜΕ jitter σε retryable codes (429/500/503/504),
  - pacing (μικρή αναμονή μετά από κάθε επιτυχή κλήση),
  - πλαφόν συνολικού χρόνου ανά κλήση,
  - επιστρέφει και usage tokens (input / thoughts / output) για κόστος & effort.
Το μοντέλο μένει locked (C.MODEL_ID)· δεν κάνουμε σιωπηλό fallback μέσα στο πείραμα.
"""
from __future__ import annotations
import os, time, random
from google import genai
from google.genai import types
import config as C

_client = None


def get_client():
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Λείπει το GEMINI_API_KEY (βάλ' το στο .env).")
        _client = genai.Client(api_key=key)
    return _client


_TRANSPORT_HINTS = ("disconnect", "connection", "timed out", "timeout", "reset by peer",
                    "temporarily", "unavailable", "eof occurred", "broken pipe",
                    "remotedisconnected", "connectionerror", "read timeout")


def _is_transport_error(err) -> bool:
    return any(h in str(err).lower() for h in _TRANSPORT_HINTS)


def _status_code(err) -> int | None:
    """Βγάζει τον HTTP κωδικό από τα διάφορα σχήματα σφαλμάτων του SDK."""
    for attr in ("code", "status_code"):
        v = getattr(err, attr, None)
        if isinstance(v, int):
            return v
    msg = str(err)
    for c in C.RETRYABLE_CODES:
        if str(c) in msg:
            return c
    return None


def generate(contents: str, thinking_level: str, model: str | None = None,
             temperature: float | None = None) -> dict:
    """Μία κλήση generateContent με retries. Επιστρέφει dict με text + usage."""
    model = model or C.MODEL_ID
    client = get_client()
    cfg = types.GenerateContentConfig(
        temperature=C.TEMPERATURE if temperature is None else temperature,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
    )
    start = time.time()
    last_err = None
    for attempt in range(C.MAX_RETRIES + 1):
        try:
            r = client.models.generate_content(model=model, contents=contents, config=cfg)
            time.sleep(C.INTER_CALL_DELAY)  # pacing μετά από επιτυχία
            um = getattr(r, "usage_metadata", None)
            return {
                "text": (r.text or ""),
                "prompt_tokens": int(getattr(um, "prompt_token_count", 0) or 0),
                "thought_tokens": int(getattr(um, "thoughts_token_count", 0) or 0),
                "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0),
                "model": model,
                "thinking_level": thinking_level,
                "attempts": attempt + 1,
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            code = _status_code(e)
            retryable = (code in C.RETRYABLE_CODES) or _is_transport_error(e)
            if not retryable or attempt == C.MAX_RETRIES:
                raise
            delay = min(C.RETRY_BASE_DELAY * (2 ** attempt), C.RETRY_MAX_DELAY)
            if C.RETRY_JITTER:
                delay *= random.uniform(0.5, 1.5)
            if (time.time() - start) + delay > C.PER_CALL_TIMEOUT:
                raise TimeoutError(
                    f"Ξεπεράστηκε PER_CALL_TIMEOUT={C.PER_CALL_TIMEOUT}s "
                    f"(τελευταίο σφάλμα {code})"
                ) from last_err
            time.sleep(delay)
    raise last_err  # pragma: no cover

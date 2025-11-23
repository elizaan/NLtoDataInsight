"""Small helper to estimate and log token usage for chat-style messages.
Uses tiktoken when available; falls back to a simple character-based heuristic.
"""
from datetime import datetime
import os
import json

try:
    import tiktoken
except Exception:
    tiktoken = None


def _get_encoding(model_name: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def estimate_tokens_for_messages(messages, model_name: str = "gpt-4o") -> int:
    """Estimate total tokens for a chat-style messages list.

    messages: list of dicts with at least a 'content' key.
    model_name: model name string to select encoding.
    """
    enc = _get_encoding(model_name)
    total = 0
    if enc is None:
        # fallback heuristic: 1 token ~= 4 chars
        for m in messages:
            content = m.get('content', '') if isinstance(m, dict) else str(m)
            total += max(1, int(len(content) / 4))
        return total

    for m in messages:
        content = m.get('content', '') if isinstance(m, dict) else str(m)
        total += len(enc.encode(content))
    return total


def log_token_usage(model_name: str, messages, label: str = "", out_path: str = None):
    """Append a token usage entry to a log file and return the count.

    If out_path is not provided, writes to ai_data/token_usage.log under repo root (relative).
    """
    try:
        count = estimate_tokens_for_messages(messages, model_name=model_name)
    except Exception:
        count = -1

    entry = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'model': model_name,
        'tokens': count,
        'label': label,
        'messages_preview': [ (m.get('role'), (m.get('content') or '')[:200]) if isinstance(m, dict) else str(m)[:200] for m in (messages or []) ]
    }

    try:
        if out_path is None:
            # locate ai_data directory relative to this file
            here = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.abspath(os.path.join(here, '..', '..'))
            ai_data = os.path.join(repo_root, 'ai_data')
            os.makedirs(ai_data, exist_ok=True)
            out_path = os.path.join(ai_data, 'token_usage.log')

        with open(out_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # best-effort: ignore logging errors
        pass

    # also return the count for immediate inspection
    return count

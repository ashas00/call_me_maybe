"""Table of the text each token adds to the output"""
import json
from typing import Callable


def load_token_vocab(
    token_path: str,
    decode: Callable[[list[int]], str],
) -> dict[int, str]:
    """Map each token id from the vocabulary file to its text"""
    try:
        with open(token_path, encoding="utf-8") as file:
            vocab = json.load(file)
    except OSError as exc:
        raise ValueError(f"cannot read vocabulary {token_path}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"vocabulary {token_path} is not JSON") from exc
    if not isinstance(vocab, dict):
        raise ValueError(f"vocabulary {token_path} is not an object")
    table: dict[int, str] = {}
    for token_id in vocab.values():
        if isinstance(token_id, int):
            text = decode([token_id])
            if text and "\ufffd" not in text:
                table[token_id] = text
    if not table:
        raise ValueError(f"vocabulary {token_path} has no tokens")
    return table

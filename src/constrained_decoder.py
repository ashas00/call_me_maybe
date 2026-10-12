"""Constrained generation of the whole answer object by the model"""
import json
import sys
from typing import Any
import numpy as np
from llm_sdk import Small_LLM_Model
from .answer_grammar import (State, is_complete, forced_text, initial_state,
                             feed_text, feed_char)
from .schemas import FunctionDef

MAX_ANSWER_TOKENS = 200
INSTRUCTIONS = (
    "Turn the request into a JSON call of one function. Copy text "
    "values exactly from the request. If the request describes a value "
    "in words, such as a pattern, write the value itself. Never compute "
    "the result."
)
EXAMPLES = (
    "Examples:\n"
    "Request: In 'room 12, floor 3', replace every number with N\n"
    'Answer: {"name": "fn_replace", "parameters": '
    '{"text": "room 12, floor 3", "pattern": "[0-9]+", "replacement": "N"}}\n'
    "Request: List D:\\games\\saves\n"
    'Answer: {"name": "fn_list", "parameters": '
    '{"path": "D:\\\\games\\\\saves"}}\n'
    'Request: Save: Tom said "go"\n'
    'Answer: {"name": "fn_save", "parameters": '
    '{"note": "Tom said \\"go\\""}}'
)
DEFAULT_VALUES: dict[str, Any] = {
    "number": 0.0,
    "integer": 0,
    "string": "",
    "boolean": False,
}


def build_prompt(request: str, functions: list[FunctionDef]) -> str:
    """Build the model prompt in the Qwen chat format"""
    catalog = "\n".join(
        f"- {fn.name}("
        + ", ".join(
            f"{name}: {param.type}"
            + (f" (one of {', '.join(choices)})"
               if (choices := param.choices()) else "")
            for name, param in fn.parameters.items()
        )
        + f"): {fn.description}"
        for fn in functions
    )
    return (
        "<|im_start|>system\n" + INSTRUCTIONS + "\n\n" + EXAMPLES
        + "<|im_end|>\n<|im_start|>user\nFunctions:\n" + catalog
        + f"\n\nRequest: {request}"
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def group_char(token: dict[int, str]) -> dict[str, list[int]]:
    """Group token ids by the first character of their text."""
    index: dict[str, list[int]] = {}
    for token_id, text in token.items():
        index.setdefault(text[0], []).append(token_id)
    return index


def allowed_tokens(
    state: State,
    token: dict[int, str],
    by_first: dict[str, list[int]],
    functions: dict[str, FunctionDef],
    cache: dict[State, list[int]],
) -> list[int]:
    """Return the token ids that keep the grammar state valid"""
    if state not in cache:
        first_chars = [
            char for char in by_first
            if feed_char(state, char, functions) is not None
        ]
        cache[state] = [
            token_id
            for char in first_chars
            for token_id in by_first[char]
            if feed_text(state, token[token_id], functions) is not None
        ]
    return cache[state]


def forced_tokens(
    model: Small_LLM_Model,
    token: dict[int, str],
    state: State,
    functions: dict[str, FunctionDef],
) -> tuple[list[int], str]:
    """Return text the grammar forces next, with its token ids"""
    forced = forced_text(state)
    if not forced:
        return [], ""
    ids: list[int] = model.encode(forced)[0].tolist()
    state_after = feed_text(state, forced, functions)
    if state_after is None or not is_complete(state_after):
        ids = ids[:-1]
    text = "".join(token.get(token_id, "") for token_id in ids)
    if not text or not forced.startswith(text):
        return [], ""
    return ids, text


def pick_token(
    model: Small_LLM_Model,
    input_ids: list[int],
    allowed: list[int],
) -> int:
    """Return the allowed token the model scores highest"""
    if len(allowed) == 1:
        return allowed[0]
    logits = np.asarray(model.get_logits_from_input_ids(input_ids))
    mask = np.full(logits.shape, -np.inf)
    mask[allowed] = 0.0
    return int(np.argmax(logits + mask))


def generate_call(
    model: Small_LLM_Model,
    token: dict[int, str],
    by_first: dict[str, list[int]],
    functions: list[FunctionDef],
    request: str,
    cache: dict[State, list[int]],
) -> tuple[str, dict[str, Any]]:
    """Let the model write the answer token by token and parse it"""
    by_name = {fn.name: fn for fn in functions}
    input_ids: list[int] = model.encode(
        build_prompt(request, functions)
    )[0].tolist()
    state = initial_state(functions)
    text = ""
    for _ in range(MAX_ANSWER_TOKENS):
        if is_complete(state):
            break
        ids, piece = forced_tokens(model, token, state, by_name)
        if not piece:
            allowed = allowed_tokens(state, token, by_first, by_name, cache)
            ids = [pick_token(model, input_ids, allowed)]
            piece = token[ids[0]]
        next_state = feed_text(state, piece, by_name)
        if next_state is None:
            break
        state = next_state
        input_ids += ids
        text += piece
    if not is_complete(state):
        print(f"Warning: {request!r}: answer not finished, writing a "
              "placeholder call", file=sys.stderr)
        return fallback_call(text, functions)
    answer = json.loads(text)
    return answer["name"], answer["parameters"]


def fallback_call(
    partial: str,
    functions: list[FunctionDef],
) -> tuple[str, dict[str, Any]]:
    """Build a schema-valid placeholder call after a generation failure"""
    head = partial.split('"parameters"', 1)[0]
    fn = next(
        (f for f in functions if json.dumps(f.name) + "," in head),
        functions[0],
    )
    params: dict[str, Any] = {}
    for name, parameter in fn.parameters.items():
        choices = parameter.choices()
        params[name] = (
            json.loads(choices[0]) if choices
            else DEFAULT_VALUES[parameter.type]
        )
    return fn.name, params

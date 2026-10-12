"""Character rules for the whole JSON answer object"""
import json
import os
from typing import Any
from .schemas import FunctionDef

Segment = tuple[Any, ...]
State = tuple[tuple[Segment, ...], int, Any]

DIGITS = "0123456789"
JSON_ESCAPES = '"\\/bfnrt'
NUMBER_FSM: dict[str, dict[str, str]] = {
    "start": {"-": "sign", "0": "zero", "digit": "int"},
    "sign": {"0": "zero", "digit": "int"},
    "zero": {".": "dot"},
    "int": {"digit": "int", ".": "dot"},
    "dot": {"digit": "frac"},
    "frac": {"digit": "frac"},
}
INTEGER_FSM: dict[str, dict[str, str]] = {
    "start": {"-": "sign", "0": "zero", "digit": "int"},
    "sign": {"0": "zero", "digit": "int"},
    "zero": {},
    "int": {"digit": "int"},
}
FLOAT_DONE: tuple[str, ...] = ("frac",)
INTEGER_DONE: tuple[str, ...] = ("zero", "int")
END: Segment = ("lit", "}}")


def segment_start(segment: Segment) -> Any:
    """Return the initial progress value for a segment"""
    kind = segment[0]
    if kind in ("lit", "space"):
        return 0
    if kind == "string":
        return (False, False)
    if kind == "number":
        return "start"
    return ""


def number_step(fsm: dict[str, dict[str, str]], progres: str, ch: str) -> Any:
    """Move a number automaton by one character; `None` if not allowed"""
    moves = fsm.get(progres, {})
    if ch in moves:
        return moves[ch]
    if ch in DIGITS:
        return moves.get("digit")
    return None


def segment_step(segment: Segment, progres: Any, ch: str) -> Any:
    """Feed one character into a segment; return the next state or `None`"""
    kind = segment[0]
    if kind == "lit":
        text: str = segment[1]
        return (progres + 1 if progres < len(text) and
                text[progres] == ch else None)
    if kind == "space":
        return 1 if progres == 0 and ch == " " else None
    if kind in ("names", "choice", "boolean"):
        options = segment[1] if kind != "boolean" else ("true", "false")
        grown = progres + ch
        return (
            grown if any(option.startswith(grown) for option in options)
            else None
        )
    if kind == "string":
        escaped, closed = progres
        if closed or ord(ch) < 0x20:
            return None
        if escaped:
            return (False, False) if ch in JSON_ESCAPES else None
        if ch == "\\":
            return (True, False)
        return (False, ch == '"')
    fsm = INTEGER_FSM if segment[1] else NUMBER_FSM
    return number_step(fsm, progres, ch)


def segment_done(segment: Segment, progres: Any) -> bool:
    """Return whether the current segment may legally end at this point"""
    kind = segment[0]
    if kind == "lit":
        return bool(progres == len(segment[1]))
    if kind == "space":
        return True
    if kind in ("names", "choice"):
        return bool(progres in segment[1])
    if kind == "boolean":
        return bool(progres in ("true", "false"))
    if kind == "string":
        return bool(progres[1])
    done = INTEGER_DONE if segment[1] else FLOAT_DONE
    return bool(progres in done)


def parameter_segments(fn: FunctionDef) -> tuple[Segment, ...]:
    """Build the JSON segments for a function parameter block"""
    segments: list[Segment] = [("lit", "{")]
    for index, (name, parameter) in enumerate(fn.parameters.items()):
        if index:
            segments.extend([("lit", ","), ("space",)])
        segments.extend([("lit", json.dumps(name) + ":"), ("space",)])
        choices = parameter.choices()
        if choices is not None:
            segments.append(("choice", tuple(choices)))
        elif parameter.type == "string":
            segments.extend([("lit", '"'), ("string",)])
        elif parameter.type == "boolean":
            segments.append(("boolean",))
        else:
            segments.append(("number", parameter.type == "integer"))
    segments.append(END)
    return tuple(segments)


def initial_state(functions: list[FunctionDef]) -> State:
    """Return the grammar state before the first answer character"""
    names = tuple(json.dumps(fn.name)[1:] for fn in functions)
    segments: tuple[Segment, ...] = (
        ("space",),
        ("lit", '{"name":'),
        ("space",),
        ("lit", '"'),
        ("names", names),
        ("lit", ","),
        ("space",),
        ("lit", '"parameters":'),
        ("space",),
    )
    return segments, 0, segment_start(segments[0])


def feed_char(
    state: State,
    ch: str,
    functions: dict[str, FunctionDef],
) -> State | None:
    """Feed one character to a grammar state and return the next state"""
    segments, index, progres = state
    while index < len(segments):
        segment = segments[index]
        grown = segment_step(segment, progres, ch)
        if grown is not None:
            return segments, index, grown
        if not segment_done(segment, progres):
            return None
        if segment[0] == "names":
            chosen = json.loads('"' + progres)
            segments += parameter_segments(functions[chosen])
        index += 1
        if index < len(segments):
            progres = segment_start(segments[index])
    return None


def feed_text(
    state: State,
    text: str,
    functions: dict[str, FunctionDef],
) -> State | None:
    """Feed a whole token text through the automaton"""
    current: State | None = state
    for char in text:
        if current is None:
            return None
        current = feed_char(current, char, functions)
    return current


def is_complete(state: State) -> bool:
    """Return True once the answer's closing braces have been written"""
    segments, index, progres = state
    return segments[index] == END and segment_done(END, progres)


def forced_text(state: State) -> str:
    """Return the literal text that must come next, whatever the model wants"""
    segments, index, progres = state
    text = ""
    while index < len(segments):
        segment = segments[index]
        if segment[0] == "lit":
            text += segment[1][progres:]
        elif segment[0] == "names":
            matches = [name for name in segment[1] if name.startswith(progres)]
            if len(matches) != 1:
                text += os.path.commonprefix(matches)[len(progres):]
                break
            text += matches[0][len(progres):]
        else:
            break
        index += 1
        if index < len(segments):
            progres = segment_start(segments[index])
    return text

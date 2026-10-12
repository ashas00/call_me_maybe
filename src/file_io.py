"""Reading the input JSON files and writing the result file"""
import json
import os
import sys
from typing import TypeVar
from pydantic import BaseModel, ValidationError
from .schemas import FunctionCall, FunctionDef

Model = TypeVar("Model", bound=BaseModel)


def load_list(path: str, model: type[Model]) -> list[Model]:
    """Read a JSON array from `path` and validate each item with `model`"""
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"file not found: {path}") from exc
    except IsADirectoryError as exc:
        raise ValueError(f"{path} is a directory, not a file") from exc
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc.strerror}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    except RecursionError as exc:
        raise ValueError(f"{path} is nested too deeply") from exc
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array")
    try:
        return [model.model_validate(item) for item in data]
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "entry"
        raise ValueError(f"{path}: {where}: {first['msg']}") from exc


def load_functions(path: str) -> list[FunctionDef]:
    """Load the function definitions, skipping the ones we cannot generate"""
    usable: list[FunctionDef] = []
    seen: set[str] = set()
    for fn in load_list(path, FunctionDef):
        if fn.name in seen:
            raise ValueError(f"{path}: function {fn.name!r} is defined twice")
        seen.add(fn.name)
        bad = fn.unsupported_parameters()
        if bad:
            print(
                f"Warning: skipped function {fn.name!r}: cannot generate "
                f"parameter(s) {', '.join(bad)}",
                file=sys.stderr,
            )
            continue
        usable.append(fn)
    if not usable:
        raise ValueError(f"{path}: no usable functions defined")
    return usable


def write_results(path: str, results: list[FunctionCall]) -> None:
    """Write result entries to a JSON file, creating folders as needed"""
    payload = [result.model_dump() for result in results]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")

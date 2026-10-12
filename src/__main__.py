"""Command-line entry point: `python -m src`"""
import argparse
import sys
from llm_sdk import Small_LLM_Model
from .constrained_decoder import fallback_call, group_char, generate_call
from .file_io import load_functions, load_list, write_results
from .answer_grammar import State
from .schemas import FunctionCall, PromptItem
from .vocab import load_token_vocab


def parse_args() -> argparse.Namespace:
    """Read command-line arguments; defaults point to the data folder"""
    parser = argparse.ArgumentParser(prog="python -m src")
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calls.json")
    return parser.parse_args()


def save(path: str, results: list[FunctionCall]) -> bool:
    """Write results and report errors; return True on success"""
    try:
        write_results(path, results)
    except OSError as exc:
        print(f"Error: cannot write {path}: {exc.strerror}", file=sys.stderr)
        return False
    print(f"Wrote {len(results)} results to {path}")
    return True


def main() -> int:
    """Run the program and return the process exit code"""
    args = parse_args()
    try:
        functions = load_functions(args.functions_definition)
        prompts = load_list(args.input, PromptItem)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not prompts:
        print("Warning: no prompts to process", file=sys.stderr)
        return 0 if save(args.output, []) else 1
    try:
        model = Small_LLM_Model()
        table = load_token_vocab(model.get_path_to_vocab_file(),
                                 model.decode)
    except Exception as exc:
        print(f"Error: cannot load the language model: {exc}",
              file=sys.stderr)
        return 1
    by_first = group_char(table)
    cache: dict[State, list[int]] = {}
    results: list[FunctionCall] = []
    try:
        for index, item in enumerate(prompts, start=1):
            print(f"[{index}/{len(prompts)}] {item.prompt}", file=sys.stderr)
            try:
                name, parameters = generate_call(
                    model, table, by_first, functions, item.prompt, cache
                )
            except Exception as exc:
                print(f"Warning: {item.prompt!r}: model error ({exc}); "
                      "writing a placeholder call", file=sys.stderr)
                name, parameters = fallback_call("", functions)
            results.append(FunctionCall(prompt=item.prompt, name=name,
                                        parameters=parameters))
    except KeyboardInterrupt:
        print("\nInterrupted: saving the results so far", file=sys.stderr)
        save(args.output, results)
        return 1
    return 0 if save(args.output, results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: unexpected {type(exc).__name__}: {exc}",
              file=sys.stderr)
        sys.exit(1)

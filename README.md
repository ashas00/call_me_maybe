*This project has been created as part of the 42 curriculum by olprymas.*

# call me maybe

## Description

**call me maybe** turns natural-language requests into structured function
calls using a small language model, **Qwen/Qwen3-0.6B**.

Given the request *"What is the sum of 2 and 3?"* and a list of available
functions, the program does not answer `5`. It answers which function to call
and with which arguments:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

Small models are unreliable when they are simply asked to "answer in JSON".
This project uses **constrained decoding** instead: at every generation step,
every token that would break the JSON syntax or the function schema gets a
logit of negative infinity, so the model can only choose among valid
continuations. The output is therefore always parseable JSON that matches the
schema in `functions_definition.json`.

## Instructions

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/).
The `llm_sdk/` folder (provided with the subject) must sit next to `src/`.

```bash
make install        # uv sync: creates .venv, installs numpy, pydantic, llm_sdk
make run            # uv run python -m src (default paths)
make debug          # same, under pdb
make lint           # flake8 + mypy with the flags from the subject
make lint-strict    # flake8 + mypy --strict
make clean          # remove __pycache__ and .mypy_cache
```

The first run downloads the model from Hugging Face (about 1.5 GB).

On 42 machines the Makefile keeps the uv cache inside the project
(`.ft-cache/`) so it does not fill the home quota.

## Example usage

Default paths:

```bash
uv run python -m src
```

reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, and writes
`data/output/function_calls.json` (the folder is created if needed).

Custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Progress and warnings go to stderr, for example:

```
[1/11] What is the sum of 2 and 3?
[2/11] What is the sum of 265 and 345?
...
Wrote 11 results to data/output/function_calls.json
```

Error cases print a clear message and exit with code 1, without a traceback:

```
$ uv run python -m src --input missing.json
Error: file not found: missing.json
```

### Grading with the moulinette

The moulinette (provided with the subject) compares the answers **by
position** with its own test set, so the program must be run on the input
files that the moulinette generates, not on `data/input/`. From the
repository root, with the `moulinette/` folder next to `src/`:

```bash
# 1. Generate the private test set (writes moulinette/data/input/)
cd moulinette
uv sync
uv run python -m moulinette prepare_exercises --set private
cd ..

# 2. Run the program on those files
uv run python -m src \
  --functions_definition moulinette/data/input/functions_definition.json \
  --input moulinette/data/input/function_calling_tests.json \
  --output data/output/private_calls.json

# 3. Grade the answers
cd moulinette
uv run python -m moulinette grade_student_answers --set private \
  --student_answer_path ../data/output/private_calls.json
```

For the public set, replace `--set private` with `--set public` in steps 1
and 3 (and use another output file name, e.g. `public_calls.json`).
`prepare_exercises` rewrites `moulinette/data/input/` each time, so always
run step 2 again after switching sets.

## Algorithm explanation

### 1. The whole answer is one grammar

The model writes the complete object `{"name": "...", "parameters": {...}}`
token by token. The function name and the arguments are generated in one
pass, so the arguments are conditioned on the chosen function.

`src/answer_grammar.py` describes the answer as a sequence of **segments**:

| segment | allows |
|---|---|
| `lit` | exactly a fixed text, e.g. `{"name":` |
| `space` | zero or one space |
| `names` | one of the function names (matched by prefix) |
| `choice` | one of the allowed values of a parameter with an `enum` |
| `string` | the content of a JSON string, with escape rules, up to the closing quote |
| `number` | a JSON number; float literal for `number`, integer for `integer` |
| `boolean` | `true` or `false` |

The initial segments are `{"name": "<one of the names>", "parameters": `.
When the `names` segment is complete, the chosen function is known and the
segments for its parameters are appended:
`{"a": <number>, "b": <number>}}`. The schema is therefore built during
generation, and only the parameters of the chosen function can be written,
all of them, in order.

A **state** is `(segments, current segment index, progress inside it)`.
`feed_char(state, char)` returns the next state, or `None` if the character is
not allowed. `feed_text(state, text)` feeds a whole token text.

**Numbers** use a small finite-state machine:

```
start --"-"--> sign
start --"0"--> zero --"."--> dot --digit--> frac --digit--> frac
start --1-9--> int  --digit--> int
               int  --"."--> dot
```

A `number` value may only end in `frac`, so it is always written as a float
literal (`2.0`, `-0.5`), as required by the subject. Leading zeros (`007`)
are impossible. `integer` uses the same machine without the dot.

**Strings** reject raw control characters (such as a newline) and only
accept valid escapes (`\"`, `\\`, `\n`, ...), so every string is valid JSON.

### 2. From characters to tokens

`src/vocab.py` reads the vocabulary file returned by
`get_path_to_vocab_file()` and builds a table `token id -> text` using the
SDK's `decode`. Tokens that are only a fragment of a multi-byte UTF-8
character decode to the replacement character `"\ufffd"` and are dropped, so
they can never appear in the output.

A token is **allowed** in a state if `feed_text(state, token_text)` does not
return `None`.

### 3. The generation loop

For each prompt (`src/constrained_decoder.py`):

1. Build the prompt: Qwen chat format, the list of functions with their
   parameter types (and allowed values), a few examples with *other*
   functions, the request, and an empty `<think>\n\n</think>\n\n` block.
   This is exactly what Qwen3's own chat template inserts when
   `enable_thinking=False`, so the model answers directly instead of
   reasoning first.
2. Repeat until the grammar reports the answer is finished:
   - **Forced text.** If the grammar allows exactly one continuation (a
     fixed literal such as `"parameters": {`, or the rest of a function name
     once its prefix is unique), it is appended without calling the model.
   - **Otherwise**, compute the allowed tokens, ask the model for logits,
     set every other logit to `-inf`, and take the argmax. If only one token
     is allowed, the model is not called.
3. Parse the finished text with `json.loads`. Thanks to the grammar this
   cannot fail, and `number` values are already floats, `integer` values
   already ints.

## Design decisions

- **One grammar for the whole object** instead of separate steps for the
  name and each argument. One mechanism covers everything, and the model
  sees its own choice of function while writing the arguments.
- **The function is chosen by the model**, through the `names` segment and
  the logits. There are no keyword rules or other heuristics.
- **Character-level grammar, token-level decoding.** The grammar only knows
  characters, so it does not depend on how the tokenizer splits text. Any
  token is checked by running its characters through the automaton.
- **Caching.** The set of allowed tokens depends only on the state, and
  states repeat a lot (for example "inside a string, no escape pending").
  Allowed tokens are cached per state, and tokens are pre-grouped by their
  first character so most of the vocabulary is rejected with one check.
- **Forced tokens** save model calls on everything the grammar already
  knows. The last forced token is held back (unless it ends the answer),
  because the model may prefer a token that merges it with the next
  characters, e.g. `"` and the beginning of a string value.
- **Greedy decoding** (argmax). Results are deterministic and identical
  across runs. The Qwen3 model card recommends sampling (temperature 0.7,
  top-p 0.8, top-k 20) in non-thinking mode and warns against greedy
  decoding in thinking mode, mainly because of repetition in long free
  text. Here thinking is disabled, the answers are short, and every token
  is constrained by the grammar, so greedy decoding is a deliberate choice:
  for function calling, reproducible answers matter more than variety.
- **Allowed values (`enum`).** A parameter with `"enum": [...]` becomes a
  `choice` segment, so only the listed values can be generated.
- **Robust input handling.** Input files are validated with pydantic before
  the model is loaded. Missing files, invalid JSON, wrong shapes and
  duplicate function names give a clear error message. Functions with a
  parameter type the decoder cannot generate (`array`, `object`) are
  skipped with a warning instead of rejecting the whole file.
- **One output entry per prompt, always.** If an answer is not finished
  within 200 tokens, or the model fails on one prompt, a placeholder call is
  written: the function the model had already chosen (if known) with neutral
  values of the right types. The output stays valid and aligned with the
  input, and the program continues with the next prompt.
- **No private SDK attributes.** Only `encode`, `decode`,
  `get_logits_from_input_ids` and `get_path_to_vocab_file` are used.
- **pydantic for every class**; errors use built-in exceptions
  (`ValueError`, `OSError`).

## Performance analysis

- **Validity:** 100% of the outputs are valid JSON and match the schema.
  This is guaranteed by construction: the answer can only end in the
  grammar's final state, and every character on the way is checked.
- **Accuracy on the provided tests:** 11 / 11 (public set),
  11 / 11 (private set, moulinette).
- **Speed:** about 4 minutes 24 seconds for the 11 public prompts on a
  42 school machine (CPU only, 8 threads), about 24 seconds per prompt
  on average, including model loading. The main cost is one forward pass
  per generated token: the SDK has no KV cache, so every call processes
  the whole prompt again. Forced text, skipping the model when only one
  token is allowed, the per-state cache and a short prompt keep the
  number and the cost of model calls low.
- **Reliability:** greedy decoding gives the same output on every run.

## Challenges faced

- **Tokens do not match grammar units.** One token can end one JSON element
  and start the next (`",`). Solved by checking each token character by
  character through the automaton.
- **Too many candidate tokens.** Checking ~150,000 tokens at every step is
  slow. Solved with a first-character index and a per-state cache.
- **Staying under 5 minutes.** Every model call re-reads the whole prompt.
  Shortening the prompt and inserting forced text (including the common
  `fn_` prefix) without calling the model brought the run from ~6.5 to
  ~4.5 minutes.
- **Few-shot examples copied too literally.** With only a "each digit ->
  `[0-9]`" example, the model wrote `[0-9]` for "all numbers". The example
  now shows "every number -> `[0-9]+`".
- **Runaway strings.** A small model can loop inside a string. The answer
  length is capped, and a placeholder call keeps the output valid.

## Testing strategy

- **Lint and types:** `make lint` and `make lint-strict` (flake8, mypy with
  the subject flags and with `--strict`).
- **Grammar checks:** valid answers (floats, negative numbers, empty
  strings, escaped quotes and backslashes, functions without parameters,
  booleans, enum values) were checked to be accepted, and to be valid JSON;
  invalid ones (`2` for a number, leading zeros, `1.`, raw newline in a
  string, invalid escape, unknown function, missing parameter, text after
  the closing brace, value outside an enum) were checked to be rejected.
- **End-to-end with a stub model:** the program was run with a fake SDK
  returning random logits. Even with a model that "talks nonsense", every
  output was valid JSON with exactly the keys `prompt`, `name`,
  `parameters`, correct types, allowed enum values, and the same prompts in
  the same order as the input. A stub that only wants to write digits
  checked that the length cap and the placeholder work.
- **Error cases:** missing input file, invalid JSON, a directory instead of
  a file, wrong field types, empty prompt list, empty function list,
  unsupported parameter types, unwritable output path, model loading
  failure, model error on a single prompt, and Ctrl+C (partial results are
  saved).
- **Moulinette:** `prepare_exercises` and `grade_student_answers` on the
  public and private sets.

## Resources

### Constrained decoding
- [Willard, B. T. and Louf, R. (2023). *Efficient Guided Generation for
  Large Language Models*. arXiv:2307.09702](https://arxiv.org/abs/2307.09702):
  generation guided by a finite-state machine, with invalid tokens masked
  at every step.

### Tokenization
- [Radford, A. et al. (2019). *Language Models are Unsupervised Multitask
  Learners* (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf):
  introduces byte-level BPE, the scheme behind Qwen's `vocab.json`.

### Tooling
- [Pydantic Validation documentation](https://pydantic.dev/docs/validation/latest/get-started/)
- [uv documentation](https://docs.astral.sh/uv/)
- [mypy documentation](https://mypy.readthedocs.io/)

### How AI was used

An AI assistant was used for:

- **Error handling:** the list of failure cases to cover and how to report
  them.
- **Drafting this README** and checking the reference links.

Every change was read, run and checked by me; I can explain every part of
the code.

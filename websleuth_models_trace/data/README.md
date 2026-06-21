# Local benchmark snapshots

Most public benchmarks are loaded through Hugging Face `datasets` and cached by that library. Some suites require a local or licensed snapshot:

- `data/legalbench/`: JSON or JSONL rows with `prompt`, `answer`, and optional `task`.
- `data/livecodebench/`: JSON/JSONL rows with prompt/question, solution, and executable tests.
- `data/toolalpaca/`: JSON/JSONL rows with instruction/prompt and expected tool call.
- `data/infinitebench/`: JSON/JSONL rows with prompt/context and answer.

The runner reports these benchmarks as unavailable rather than silently substituting another dataset.

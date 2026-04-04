# `run_sglang_softthinking_replay.py`

This script runs math-only SGLang soft-thinking inference and evaluation with a single in-file `CONFIG` dict.

It supports two generation modes:

- `standard`: two-stage generation
  - stage 1: generate thinking tokens until `</think>` or the warmup budget
  - stage 2: continue normal decoding with the reserved response budget
- `replay`: STPO-style replay generation
  - stage 1: warm up once per prompt with `return_logprob=True`
  - stage 2: replay the captured soft-thinking trace `n_sampling` times from the original prompt ids
  - fallback: if warmup does not stop at `</think>`, the script falls back to standard continuation for that sample

The script uses the local math evaluator from [matheval.py](/home/james/Projects/Soft-Thinking/matheval.py) and writes STPO-style metrics such as `pass@1`, `pass@5`, `pass@8`, `pass@10`, and `pass@16`.

## Run

Edit the `CONFIG` dict in [run_sglang_softthinking_replay.py](/home/james/Projects/Soft-Thinking/run_sglang_softthinking_replay.py), then run:

```bash
python run_sglang_softthinking_replay.py
```

## `CONFIG` Overview

Top-level keys:

- `dataset`
- `model_name`
- `output_dir`
- `start_idx`
- `end_idx`
- `generation_mode`
- `n_sampling`
- `min_response_budget_tokens`
- `warmup_batch_size`
- `replay_batch_size`
- `engine`
- `sampling`

## Top-Level Keys

### `dataset`

Math dataset name to load from `./datasets/<dataset>.json`.

Supported values:

- `math500`
- `aime2024`
- `aime2025`
- `gpqa_diamond`
- `gsm8k`
- `amc23`

### `model_name`

Hugging Face model name or local model path passed to:

- `AutoTokenizer.from_pretrained(...)`
- `sgl.Engine(model_path=...)`

Example:

```python
"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
```

### `output_dir`

Base output directory for this script.

The script automatically creates:

```text
<output_dir>/results/<dataset>/
```

and writes:

- `<base_filename>.json`: per-sample outputs and scores
- `<base_filename>_metrics.json`: aggregate metrics

### `start_idx`

Inclusive dataset start index.

Use this to evaluate a slice of the dataset.

### `end_idx`

Exclusive dataset end index.

The actual end is clipped to the dataset length.

### `generation_mode`

Controls the inference algorithm.

Allowed values:

- `"standard"`
- `"replay"`

Use:

- `"standard"` if you want budgeted two-stage soft-thinking without replay
- `"replay"` if you want STPO-style warmup + replay

### `n_sampling`

Number of sampled outputs per question.

Important:

- this is the sampling count per prompt
- this is the correct replacement for the old `num_samples` config key
- `pass@1` in STPO-style metrics is effectively average accuracy over these `n_sampling` outputs

### `min_response_budget_tokens`

Reserved token budget for the final response stage.

The script enforces:

```text
0 < min_response_budget_tokens < sampling.max_new_tokens
```

Behavior:

- `standard`
  - warmup budget = `max_new_tokens - min_response_budget_tokens`
  - continuation budget = `min_response_budget_tokens`
- `replay`
  - warmup budget = `max_new_tokens - min_response_budget_tokens`
  - replay response budget = `min_response_budget_tokens`

### `warmup_batch_size`

Warmup chunk size used by replay generation.

This controls how many prompts are processed together in the warmup stage.

Notes:

- mainly relevant for `generation_mode == "replay"`
- kept in the config for alignment with STPO and for consistency with standard-mode setup

### `replay_batch_size`

Batch size for replay stage decoding.

Allowed values:

- `None`: replay all replay requests from the current warmup chunk at once
- positive integer: replay in smaller batches

Only relevant for `generation_mode == "replay"`.

## `engine` Keys

These keys are passed into `sgl.Engine(...)`.

### `engine.tp_size`

Tensor parallel size.

Set this to the number of GPUs you want SGLang to use for tensor parallelism.

### `engine.cuda_graph_max_bs`

Maximum batch size to capture for CUDA graphs inside SGLang.

Use:

- `None` to leave it unset
- a small integer if you want explicit CUDA graph capture control

### `engine.max_running_requests`

Upper bound on concurrent running requests inside the engine.

Use `None` to leave SGLang default behavior.

### `engine.mem_fraction_static`

Fraction of GPU memory reserved by SGLang.

Higher values use more GPU memory and can improve throughput if memory is available.

### `engine.random_seed`

Engine random seed.

Controls sampling reproducibility as far as the backend allows.

### `engine.sampling_backend`

SGLang sampling backend.

Typical values:

- `"flashinfer"`
- `"pytorch"`

### `engine.disable_cuda_graph`

Whether to disable CUDA graphs in SGLang.

Typical tradeoff:

- `False`: better performance if graphs are stable
- `True`: safer when debugging unusual runtime behavior

### `engine.disable_overlap_schedule`

Must be `True` for this runner.

Reason:

- this matches the STPO constraints
- replay mode requires non-overlap scheduling behavior

### `engine.chunked_prefill_size`

SGLang chunked prefill size.

Important:

- replay mode requires `-1`
- this matches the STPO replay constraint

If you switch to `"replay"` and set another value, the script will raise an error.

### `engine.enable_soft_thinking`

Must be `True` for this runner.

This script is specifically for soft-thinking inference.

### `engine.think_end_str`

The marker used to close the thinking phase.

Default:

```python
"</think>"
```

This value is used in:

- warmup stop condition
- replay trace validation
- fallback force-closing when warmup does not stop cleanly

### `engine.max_topk`

Maximum top-k width stored in the replay trace.

Relevant to replay because the warmup output captures:

- `output_topk_idx_list`
- `output_topk_prob_list`

Larger values capture wider replay distributions but increase trace size.

### `engine.add_noise_dirichlet`

Whether to enable Dirichlet noise in the soft-thinking sampler.

Usually `False` unless you are explicitly testing noisy decoding behavior.

### `engine.add_noise_gumbel_softmax`

Whether to enable Gumbel-softmax noise in the soft-thinking sampler.

Usually `False` unless you are explicitly testing noisy decoding behavior.

## `sampling` Keys

These keys are passed as SGLang sampling parameters.

### `sampling.temperature`

Main sampling temperature.

### `sampling.top_p`

Main nucleus sampling threshold.

### `sampling.top_k`

Main top-k sampling threshold.

### `sampling.min_p`

Main minimum probability threshold.

### `sampling.after_thinking_temperature`

Temperature after the thinking phase.

Used by your soft-thinking SGLang fork for response-stage decoding behavior.

### `sampling.after_thinking_top_p`

Top-p after thinking.

### `sampling.after_thinking_top_k`

Top-k after thinking.

### `sampling.after_thinking_min_p`

Min-p after thinking.

### `sampling.repetition_penalty`

Standard repetition penalty.

### `sampling.dirichlet_alpha`

Dirichlet alpha used when `engine.add_noise_dirichlet = True`.

### `sampling.gumbel_softmax_temperature`

Gumbel-softmax temperature used when `engine.add_noise_gumbel_softmax = True`.

### `sampling.max_new_tokens`

Total generation budget.

This is split into:

- thinking-stage budget
- response-stage budget

by `min_response_budget_tokens`.

### `sampling.early_stopping_entropy_threshold`

Entropy threshold used by the soft-thinking implementation for early stopping.

Set to:

- `0.0` to effectively disable entropy-based early stopping
- positive value to enable it

### `sampling.early_stopping_length_threshold`

Minimum generated length before early stopping logic is allowed to trigger.

## Current Default Configuration

The current defaults in the script are aligned with your provided run config:

```python
CONFIG = {
    "dataset": "math500",
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "output_dir": "./outputs/math-eval/DS-R1-Distill-Qwen-7B-sglang",
    "start_idx": 0,
    "end_idx": 500,
    "generation_mode": "standard",
    "n_sampling": 8,
    "min_response_budget_tokens": 5000,
    "warmup_batch_size": 200,
    "replay_batch_size": None,
    "engine": {
        "tp_size": 1,
        "cuda_graph_max_bs": 8,
        "max_running_requests": None,
        "mem_fraction_static": 0.7,
        "random_seed": 0,
        "sampling_backend": "flashinfer",
        "disable_cuda_graph": False,
        "disable_overlap_schedule": True,
        "chunked_prefill_size": -1,
        "enable_soft_thinking": True,
        "think_end_str": "</think>",
        "max_topk": 10,
        "add_noise_dirichlet": False,
        "add_noise_gumbel_softmax": False,
    },
    "sampling": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 30,
        "min_p": 0.001,
        "after_thinking_temperature": 0.6,
        "after_thinking_top_p": 0.95,
        "after_thinking_top_k": 30,
        "after_thinking_min_p": 0.0,
        "repetition_penalty": 1.0,
        "dirichlet_alpha": 1.0,
        "gumbel_softmax_temperature": 1.0,
        "max_new_tokens": 32768,
        "early_stopping_entropy_threshold": 0.1,
        "early_stopping_length_threshold": 256,
    },
}
```

## Practical Examples

### Run standard mode

```python
CONFIG["generation_mode"] = "standard"
```

### Run replay mode

```python
CONFIG["generation_mode"] = "replay"
CONFIG["engine"]["chunked_prefill_size"] = -1
```

### Evaluate only part of a dataset

```python
CONFIG["start_idx"] = 0
CONFIG["end_idx"] = 100
```

### Use more samples per prompt

```python
CONFIG["n_sampling"] = 16
```

### Reduce replay decode batch size to avoid OOM

```python
CONFIG["generation_mode"] = "replay"
CONFIG["replay_batch_size"] = 16
```

## Output Format

Per-sample result file fields include:

- `idx`
- `question`
- `prompt`
- `ground_truth`
- `pred`
- `pred_cot`
- `score`
- `judge_info`
- `finish_generation`
- `full_len`
- `think_len`
- `avg_full_len`
- `avg_think_len`
- `n`
- `generation_mode`

Aggregate metrics file fields include:

- `num_samples`
- `pass@1`
- `pass@5`
- `pass@8`
- `pass@10`
- `pass@16`
- `avg_entropy`
- `avg_num_full_output_tokens`
- `avg_num_think_tokens`
- `evaluation_time_sec`
- `generation_time_min`

## Notes

- This script is math-only. It does not include code datasets or LLM-judge-only dataset handling.
- `pass@1` here follows STPO behavior. With `n_sampling > 1`, it is effectively average accuracy over the sampled outputs, not “at least one correct sample”.
- Replay mode depends on your local SGLang fork exposing replay trace fields in `meta_info`.

import copy
import json
import os
import time
from typing import Any

import sglang as sgl
import torch
from transformers import AutoTokenizer

from matheval import evaluator_map


MATH_DATASETS = [
    "math500",
    "aime2024",
    "aime2025",
    "gpqa_diamond",
    "gsm8k",
    "amc23",
]

MATH_QUERY_TEMPLATE = """
Please reason step by step, and put your final answer within \\boxed{{}}.

{Question}
""".strip()

GPQA_QUERY_TEMPLATE = """
Please solve the following multiple-choice question. Please show your choice in the answer field with only the choice letter, e.g.,"answer": "C".

{Question}
""".strip()


CONFIG = {
    "dataset": "math500",
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "output_dir": "./outputs/math-eval/DS-R1-Distill-Qwen-7B-sglang",
    "start_idx": 0,
    "end_idx": 500,
    "generation_mode": "standard",  # "standard" or "replay"
    "n_sampling": 8,
    "min_response_budget_tokens": 2000,
    "warmup_batch_size": 500,
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


def build_sampling_params(config: dict[str, Any]) -> dict[str, Any]:
    sampling_cfg = config["sampling"]
    engine_cfg = config["engine"]
    return {
        "temperature": sampling_cfg["temperature"],
        "top_p": sampling_cfg["top_p"],
        "top_k": sampling_cfg["top_k"],
        "min_p": sampling_cfg["min_p"],
        "repetition_penalty": sampling_cfg["repetition_penalty"],
        "after_thinking_temperature": sampling_cfg["after_thinking_temperature"],
        "after_thinking_top_p": sampling_cfg["after_thinking_top_p"],
        "after_thinking_top_k": sampling_cfg["after_thinking_top_k"],
        "after_thinking_min_p": sampling_cfg["after_thinking_min_p"],
        "n": 1,
        "gumbel_softmax_temperature": (
            sampling_cfg["gumbel_softmax_temperature"]
        ),
        "dirichlet_alpha": sampling_cfg["dirichlet_alpha"],
        "max_new_tokens": sampling_cfg["max_new_tokens"],
        "think_end_str": engine_cfg["think_end_str"],
        "early_stopping_entropy_threshold": (
            sampling_cfg["early_stopping_entropy_threshold"]
        ),
        "early_stopping_length_threshold": (
            sampling_cfg["early_stopping_length_threshold"]
        ),
    }


def validate_config(config: dict[str, Any], sampling_params: dict[str, Any]) -> None:
    dataset = config["dataset"]
    if dataset not in MATH_DATASETS:
        raise ValueError(
            f"Only math datasets are supported. Got dataset={dataset!r}."
        )

    generation_mode = config["generation_mode"]
    if generation_mode not in {"standard", "replay"}:
        raise ValueError(
            "generation_mode must be 'standard' or 'replay'. "
            f"Got {generation_mode!r}."
        )

    engine_cfg = config["engine"]
    if not engine_cfg["enable_soft_thinking"]:
        raise ValueError("This runner only supports soft-thinking mode.")
    if not engine_cfg["disable_overlap_schedule"]:
        raise ValueError(
            "Replay/standard budgeted generation requires "
            "disable_overlap_schedule=True."
        )

    base_max_new_tokens = int(sampling_params["max_new_tokens"])
    response_budget = int(config["min_response_budget_tokens"])
    if not (0 < response_budget < base_max_new_tokens):
        raise ValueError(
            "Require 0 < min_response_budget_tokens < max_new_tokens. "
            f"Got min_response_budget_tokens={response_budget}, "
            f"max_new_tokens={base_max_new_tokens}."
        )

    if generation_mode == "replay" and engine_cfg["chunked_prefill_size"] != -1:
        raise ValueError(
            "Replay mode requires chunked_prefill_size == -1. "
            f"Got {engine_cfg['chunked_prefill_size']}."
        )


def load_dataset(dataset: str) -> list[dict[str, Any]]:
    path = os.path.join("datasets", f"{dataset}.json")
    with open(path, "r") as f:
        return json.load(f)


def build_prompt(
    dataset: str,
    sample: dict[str, Any],
    tokenizer: AutoTokenizer,
) -> str:
    question = sample["prompt"][0]["value"]
    if dataset == "gpqa_diamond":
        content = GPQA_QUERY_TEMPLATE.format(Question=question)
    else:
        content = MATH_QUERY_TEMPLATE.format(Question=question)
    chat = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
    )


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples < k:
        return 0.0
    if num_correct == num_samples:
        return 1.0
    if num_correct == 0:
        return 0.0
    p_fail = 1.0
    for i in range(k):
        p_fail *= (num_samples - num_correct - i) / (num_samples - i)
    return 1.0 - p_fail


def extract_finish_reason_type(meta_info: dict[str, Any]) -> str | None:
    finish_reason = meta_info.get("finish_reason")
    if isinstance(finish_reason, dict):
        finish_reason_type = finish_reason.get("type")
        return None if finish_reason_type is None else str(finish_reason_type)
    return None if finish_reason is None else str(finish_reason)


def output_hit_token_budget(meta_info: dict[str, Any]) -> bool:
    finish_reason_type = extract_finish_reason_type(meta_info)
    if finish_reason_type is None:
        return False
    finish_reason_type = finish_reason_type.lower()
    return (
        "length" in finish_reason_type
        or "max_tokens" in finish_reason_type
        or "max_new_tokens" in finish_reason_type
    )


def did_stop_on_stop_str(meta_info: dict[str, Any], stop_str: str) -> bool:
    finish_reason = meta_info.get("finish_reason")
    finish_reason_type = extract_finish_reason_type(meta_info)
    if finish_reason_type != "stop":
        return False
    if isinstance(finish_reason, dict):
        return finish_reason.get("matched") == stop_str
    return True


def validate_output_lengths(
    meta_info: dict[str, Any],
    *,
    allow_all_thinking: bool = False,
) -> tuple[int, int]:
    full_len = int(meta_info["full_len"])
    think_len = int(meta_info["think_len"])
    if full_len <= 0:
        raise AssertionError(
            f"`full_len` must be positive, got full_len={full_len}"
        )
    max_think_len = full_len if allow_all_thinking else full_len - 1
    if think_len < 0 or think_len > max_think_len:
        raise AssertionError(
            "Invalid length relation: "
            f"think_len={think_len}, full_len={full_len}"
        )
    return full_len, think_len


def extract_output_entropies(output: dict[str, Any]) -> list[float]:
    entropies = output.get("entropies", [])
    if entropies is None:
        return []
    if not isinstance(entropies, list):
        raise AssertionError(
            "SGLang output has invalid `entropies` type: "
            f"{type(entropies).__name__}"
        )
    return [float(x) for x in entropies]


def extract_sglang_text(output: dict[str, Any], tokenizer: Any) -> str:
    if "text" in output:
        return str(output["text"])
    output_ids = output.get("output_ids")
    if isinstance(output_ids, list) and output_ids:
        if isinstance(output_ids[0], list):
            output_ids = output_ids[0]
        if isinstance(output_ids, list) and all(
            isinstance(x, int) for x in output_ids
        ):
            return tokenizer.decode(output_ids, skip_special_tokens=False)
    raise AssertionError("SGLang output does not contain decodable text.")


def close_thinking_trace(thinking_text: str, think_end_str: str) -> str:
    if thinking_text.endswith(think_end_str):
        return thinking_text
    return thinking_text + think_end_str


def build_generation_summary(
    output_infos: list[dict[str, Any]],
    prompts: list[str],
    num_samples: int,
) -> dict[str, Any]:
    grouped_outputs = [[] for _ in prompts]
    grouped_finish = [[] for _ in prompts]
    grouped_full_lens = [[] for _ in prompts]
    grouped_think_lens = [[] for _ in prompts]
    entropies_list: list[list[float]] = []

    for output_info in output_infos:
        sample_idx = int(output_info["sample_idx"])
        grouped_outputs[sample_idx].append(output_info["text"])
        grouped_finish[sample_idx].append(output_info["finish_generation"])
        grouped_full_lens[sample_idx].append(int(output_info["full_len"]))
        grouped_think_lens[sample_idx].append(int(output_info["think_len"]))
        entropies_list.append(list(output_info["entropies"]))

    expected = len(prompts) * num_samples
    if len(output_infos) != expected:
        raise AssertionError(
            f"Expected {expected} outputs, got {len(output_infos)}."
        )
    for sample_idx in range(len(prompts)):
        if len(grouped_outputs[sample_idx]) != num_samples:
            raise AssertionError(
                "Grouped output count mismatch for sample_idx="
                f"{sample_idx}: expected={num_samples}, "
                f"got={len(grouped_outputs[sample_idx])}"
            )

    full_lens = [float(info["full_len"]) for info in output_infos]
    think_lens = [float(info["think_len"]) for info in output_infos]
    valid_entropies = [ents for ents in entropies_list if ents]
    avg_entropy = -1.0
    if valid_entropies:
        avg_entropy = sum(
            sum(entropies) / len(entropies) for entropies in valid_entropies
        ) / len(valid_entropies)

    return {
        "grouped_outputs": grouped_outputs,
        "grouped_finish_generation": grouped_finish,
        "grouped_full_lens": grouped_full_lens,
        "grouped_think_lens": grouped_think_lens,
        "entropies_list": entropies_list,
        "avg_entropy": avg_entropy,
        "avg_num_full_output_tokens": (
            sum(full_lens) / len(full_lens) if full_lens else -1.0
        ),
        "avg_num_think_tokens": (
            sum(think_lens) / len(think_lens) if think_lens else -1.0
        ),
    }


def run_standard_generation(
    prompts: list[str],
    engine: Any,
    sampling_params: dict[str, Any],
    num_samples: int,
    tokenizer: Any,
    min_response_budget_tokens: int,
) -> dict[str, Any]:
    input_prompts = [prompt for prompt in prompts for _ in range(num_samples)]
    base_max_new_tokens = int(sampling_params["max_new_tokens"])
    think_end_str = str(sampling_params["think_end_str"])
    thinking_max_new_tokens = base_max_new_tokens - min_response_budget_tokens

    warmup_params = copy.deepcopy(sampling_params)
    warmup_params["max_new_tokens"] = thinking_max_new_tokens
    warmup_params["stop"] = think_end_str
    warmup_params["no_stop_trim"] = True
    warmup_out = engine.generate(input_prompts, warmup_params)
    warmup_outputs = warmup_out if isinstance(warmup_out, list) else [warmup_out]
    if len(warmup_outputs) != len(input_prompts):
        raise AssertionError(
            "Thinking-phase output count mismatch: "
            f"expected={len(input_prompts)}, got={len(warmup_outputs)}"
        )

    continuation_units: list[dict[str, Any]] = []
    for output_idx, output in enumerate(warmup_outputs):
        meta_info = output.get("meta_info")
        if not isinstance(meta_info, dict):
            raise AssertionError("Warmup output missing `meta_info`.")
        full_len, think_len = validate_output_lengths(
            meta_info,
            allow_all_thinking=True,
        )
        output_text = extract_sglang_text(output, tokenizer)
        prefill_text = close_thinking_trace(output_text, think_end_str)
        if not did_stop_on_stop_str(meta_info, think_end_str):
            print(
                "Standard warmup did not stop at think-end boundary; "
                f"output_idx={output_idx}, finish_reason={meta_info.get('finish_reason')!r}. "
                "Force-closing and continuing."
            )
        continuation_units.append(
            {
                "sample_idx": output_idx // num_samples,
                "prefill_text": prefill_text,
                "prompt": input_prompts[output_idx] + prefill_text,
                "thinking_full_len": full_len,
                "thinking_think_len": think_len,
                "thinking_entropies": extract_output_entropies(output),
            }
        )

    continuation_params = copy.deepcopy(sampling_params)
    continuation_params.pop("stop", None)
    continuation_params["max_new_tokens"] = min_response_budget_tokens
    continuation_prompts = [unit["prompt"] for unit in continuation_units]
    continuation_out = engine.generate(
        continuation_prompts,
        continuation_params,
    )
    continuation_outputs = (
        continuation_out
        if isinstance(continuation_out, list)
        else [continuation_out]
    )
    if len(continuation_outputs) != len(continuation_units):
        raise AssertionError(
            "Continuation output count mismatch: "
            f"expected={len(continuation_units)}, got={len(continuation_outputs)}"
        )

    output_infos: list[dict[str, Any]] = []
    for output, unit in zip(continuation_outputs, continuation_units, strict=True):
        meta_info = output.get("meta_info")
        if not isinstance(meta_info, dict):
            raise AssertionError("Continuation output missing `meta_info`.")
        hit_token_budget = output_hit_token_budget(meta_info)
        if hit_token_budget:
            print(
                "Standard continuation reached response token budget; "
                f"sample_idx={unit['sample_idx']}, "
                f"finish_reason={meta_info.get('finish_reason')!r}."
            )
        full_len, think_len = validate_output_lengths(
            meta_info,
            allow_all_thinking=hit_token_budget,
        )
        output_infos.append(
            {
                "sample_idx": unit["sample_idx"],
                "text": str(unit["prefill_text"])
                + extract_sglang_text(output, tokenizer),
                "finish_generation": not hit_token_budget,
                "full_len": int(unit["thinking_full_len"]) + full_len,
                "think_len": int(unit["thinking_think_len"]) + think_len,
                "entropies": list(unit["thinking_entropies"])
                + extract_output_entropies(output),
            }
        )

    return build_generation_summary(output_infos, prompts, num_samples)


def extract_replay_trace_from_warmup_output(
    output: dict[str, Any],
    sampling_params: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    meta_info = output.get("meta_info")
    if not isinstance(meta_info, dict):
        raise AssertionError("Warmup output missing `meta_info`.")

    full_len, think_len = validate_output_lengths(
        meta_info,
        allow_all_thinking=True,
    )
    topk_indices = meta_info.get("output_topk_idx_list")
    topk_probs = meta_info.get("output_topk_prob_list")
    if not isinstance(topk_indices, list) or not isinstance(topk_probs, list):
        raise AssertionError(
            "Warmup output missing replay trace fields "
            "(`output_topk_idx_list` / `output_topk_prob_list`)."
        )
    if len(topk_indices) != len(topk_probs):
        raise AssertionError(
            "Warmup trace length mismatch: "
            f"len(topk_idx)={len(topk_indices)}, "
            f"len(topk_prob)={len(topk_probs)}"
        )
    if len(topk_indices) < think_len:
        raise AssertionError(
            "Warmup trace shorter than think_len: "
            f"trace_len={len(topk_indices)}, think_len={think_len}"
        )

    think_end_str = str(sampling_params["think_end_str"])
    if not did_stop_on_stop_str(meta_info, think_end_str):
        raise AssertionError(
            "Warmup did not stop at think-end boundary. "
            f"finish_reason={meta_info.get('finish_reason')}"
        )

    think_end_ids = tokenizer.encode(think_end_str, add_special_tokens=False)
    if not think_end_ids:
        raise AssertionError(
            f"Tokenizer could not encode think_end_str={think_end_str!r}."
        )
    think_end_id = int(think_end_ids[-1])

    trace_len = len(topk_indices)
    if trace_len == think_len:
        replay_topk_indices = copy.deepcopy(topk_indices)
        replay_topk_probs = copy.deepcopy(topk_probs)
        replay_topk_indices.append([think_end_id])
        replay_topk_probs.append([1.0])
    elif trace_len == full_len:
        replay_topk_indices = copy.deepcopy(topk_indices)
        replay_topk_probs = copy.deepcopy(topk_probs)
    else:
        raise AssertionError(
            "Warmup trace length must match think_len or full_len: "
            f"trace_len={trace_len}, think_len={think_len}, full_len={full_len}"
        )

    return {
        "replay_trace": {
            "topk_indices": replay_topk_indices,
            "topk_probs": replay_topk_probs,
        },
        "full_len": full_len,
        "think_len": think_len,
    }


def run_replay_generation(
    prompts: list[str],
    engine: Any,
    sampling_params: dict[str, Any],
    num_samples: int,
    warmup_batch_size: int,
    replay_batch_size: int | None,
    min_response_budget_tokens: int,
    tokenizer: Any,
) -> dict[str, Any]:
    from sglang.srt.managers.io_struct import GenerateReqInput

    base_max_new_tokens = int(sampling_params["max_new_tokens"])
    response_max_new_tokens = min_response_budget_tokens
    warmup_max_new_tokens = base_max_new_tokens - response_max_new_tokens
    think_end_str = str(sampling_params["think_end_str"])

    sample_infos = [
        {
            "sample_idx": sample_idx,
            "prompt": prompt,
            "prompt_ids": tokenizer.encode(prompt),
        }
        for sample_idx, prompt in enumerate(prompts)
    ]

    output_infos: list[dict[str, Any]] = []

    for start_idx in range(0, len(sample_infos), warmup_batch_size):
        chunk_infos = sample_infos[start_idx : start_idx + warmup_batch_size]
        chunk_prompts = [item["prompt"] for item in chunk_infos]

        warmup_params = copy.deepcopy(sampling_params)
        warmup_params["max_new_tokens"] = warmup_max_new_tokens
        warmup_params["stop"] = think_end_str
        warmup_params["no_stop_trim"] = True
        warmup_out = engine.generate(
            prompt=chunk_prompts,
            sampling_params=warmup_params,
            return_logprob=True,
        )
        warmup_outputs = warmup_out if isinstance(warmup_out, list) else [warmup_out]
        if len(warmup_outputs) != len(chunk_infos):
            raise AssertionError(
                "Warmup output count mismatch: "
                f"expected={len(chunk_infos)}, got={len(warmup_outputs)}"
            )

        replay_units: list[dict[str, Any]] = []
        fallback_units: list[dict[str, Any]] = []
        for sample_info, warmup_output in zip(
            chunk_infos,
            warmup_outputs,
            strict=True,
        ):
            meta_info = warmup_output.get("meta_info")
            if not isinstance(meta_info, dict):
                raise AssertionError("Warmup output missing `meta_info`.")

            warmup_full_len, warmup_think_len = validate_output_lengths(
                meta_info,
                allow_all_thinking=True,
            )
            warmup_text = extract_sglang_text(warmup_output, tokenizer)
            warmup_entropies = extract_output_entropies(warmup_output)

            if not did_stop_on_stop_str(meta_info, think_end_str):
                print(
                    "Replay warmup did not stop at think-end boundary; "
                    f"sample_idx={sample_info['sample_idx']}, "
                    f"finish_reason={meta_info.get('finish_reason')!r}. "
                    "Falling back to standard continuation."
                )
                closed_thinking_text = close_thinking_trace(
                    warmup_text,
                    think_end_str,
                )
                for _ in range(num_samples):
                    fallback_units.append(
                        {
                            "sample_idx": int(sample_info["sample_idx"]),
                            "prompt": sample_info["prompt"] + closed_thinking_text,
                            "prefill_text": closed_thinking_text,
                            "warmup_full_len": warmup_full_len,
                            "warmup_think_len": warmup_think_len,
                            "warmup_entropies": warmup_entropies,
                        }
                    )
                continue

            closed_warmup_text = close_thinking_trace(warmup_text, think_end_str)
            warmup_info = extract_replay_trace_from_warmup_output(
                output=warmup_output,
                sampling_params=sampling_params,
                tokenizer=tokenizer,
            )
            for _ in range(num_samples):
                replay_units.append(
                    {
                        "sample_idx": int(sample_info["sample_idx"]),
                        "prompt_ids": sample_info["prompt_ids"],
                        "prefill_text": closed_warmup_text,
                        "replay_trace": warmup_info["replay_trace"],
                        "warmup_full_len": int(warmup_info["full_len"]),
                        "warmup_think_len": int(warmup_info["think_len"]),
                        "warmup_entropies": warmup_entropies,
                    }
                )

        if replay_units:
            replay_step = replay_batch_size or len(replay_units)
            for replay_batch_start in range(0, len(replay_units), replay_step):
                replay_batch = replay_units[
                    replay_batch_start : replay_batch_start + replay_step
                ]
                replay_input_ids = [
                    copy.deepcopy(unit["prompt_ids"]) for unit in replay_batch
                ]
                replay_traces = [
                    copy.deepcopy(unit["replay_trace"]) for unit in replay_batch
                ]

                replay_params = copy.deepcopy(sampling_params)
                replay_params.pop("stop", None)
                replay_params["max_new_tokens"] = response_max_new_tokens
                replay_obj = GenerateReqInput(
                    input_ids=replay_input_ids,
                    sampling_params=replay_params,
                    return_logprob=False,
                    soft_thinking_trace=replay_traces,
                )
                replay_out = engine.generate(
                    prompt=replay_obj.text,
                    input_ids=replay_obj.input_ids,
                    sampling_params=replay_obj.sampling_params,
                    image_data=replay_obj.image_data,
                    return_logprob=replay_obj.return_logprob,
                    logprob_start_len=replay_obj.logprob_start_len,
                    top_logprobs_num=replay_obj.top_logprobs_num,
                    token_ids_logprob=replay_obj.token_ids_logprob,
                    lora_path=replay_obj.lora_path,
                    custom_logit_processor=replay_obj.custom_logit_processor,
                    return_hidden_states=replay_obj.return_hidden_states,
                    stream=replay_obj.stream,
                    soft_thinking_trace=replay_obj.soft_thinking_trace,
                )
                replay_outputs = (
                    replay_out if isinstance(replay_out, list) else [replay_out]
                )
                if len(replay_outputs) != len(replay_batch):
                    raise AssertionError(
                        "Replay output count mismatch: "
                        f"expected={len(replay_batch)}, got={len(replay_outputs)}"
                    )

                for output, replay_unit in zip(
                    replay_outputs,
                    replay_batch,
                    strict=True,
                ):
                    meta_info = output.get("meta_info")
                    if not isinstance(meta_info, dict):
                        raise AssertionError("Replay output missing `meta_info`.")
                    replay_hit_token_budget = output_hit_token_budget(meta_info)
                    if replay_hit_token_budget:
                        print(
                            "Replay output reached response token budget; "
                            f"sample_idx={replay_unit['sample_idx']}, "
                            f"finish_reason={meta_info.get('finish_reason')!r}."
                        )
                    full_len, think_len = validate_output_lengths(
                        meta_info,
                        allow_all_thinking=replay_hit_token_budget,
                    )
                    output_infos.append(
                        {
                            "sample_idx": replay_unit["sample_idx"],
                            "text": str(replay_unit["prefill_text"])
                            + extract_sglang_text(output, tokenizer),
                            "finish_generation": not replay_hit_token_budget,
                            "full_len": int(replay_unit["warmup_full_len"])
                            + full_len,
                            "think_len": int(replay_unit["warmup_think_len"])
                            + think_len,
                            "entropies": list(replay_unit["warmup_entropies"])
                            + extract_output_entropies(output),
                        }
                    )

        if fallback_units:
            fallback_prompts = [str(unit["prompt"]) for unit in fallback_units]
            fallback_params = copy.deepcopy(sampling_params)
            fallback_params.pop("stop", None)
            fallback_params["max_new_tokens"] = response_max_new_tokens
            fallback_out = engine.generate(
                prompt=fallback_prompts,
                sampling_params=fallback_params,
            )
            fallback_outputs = (
                fallback_out if isinstance(fallback_out, list) else [fallback_out]
            )
            if len(fallback_outputs) != len(fallback_units):
                raise AssertionError(
                    "Fallback output count mismatch: "
                    f"expected={len(fallback_units)}, got={len(fallback_outputs)}"
                )

            for output, fallback_unit in zip(
                fallback_outputs,
                fallback_units,
                strict=True,
            ):
                meta_info = output.get("meta_info")
                if not isinstance(meta_info, dict):
                    raise AssertionError("Fallback output missing `meta_info`.")
                fallback_hit_token_budget = output_hit_token_budget(meta_info)
                if fallback_hit_token_budget:
                    print(
                        "Fallback output reached response token budget; "
                        f"sample_idx={fallback_unit['sample_idx']}, "
                        f"finish_reason={meta_info.get('finish_reason')!r}."
                    )
                full_len, think_len = validate_output_lengths(
                    meta_info,
                    allow_all_thinking=fallback_hit_token_budget,
                )
                output_infos.append(
                    {
                        "sample_idx": fallback_unit["sample_idx"],
                        "text": str(fallback_unit["prefill_text"])
                        + extract_sglang_text(output, tokenizer),
                        "finish_generation": not fallback_hit_token_budget,
                        "full_len": int(fallback_unit["warmup_full_len"])
                        + full_len,
                        "think_len": int(fallback_unit["warmup_think_len"])
                        + think_len,
                        "entropies": list(fallback_unit["warmup_entropies"])
                        + extract_output_entropies(output),
                    }
                )

    return build_generation_summary(output_infos, prompts, num_samples)


def build_engine(config: dict[str, Any]) -> Any:
    engine_cfg = config["engine"]
    return sgl.Engine(
        model_path=config["model_name"],
        tp_size=engine_cfg["tp_size"],
        log_level="info",
        trust_remote_code=True,
        random_seed=engine_cfg["random_seed"],
        max_running_requests=engine_cfg["max_running_requests"],
        mem_fraction_static=engine_cfg["mem_fraction_static"],
        disable_cuda_graph=engine_cfg["disable_cuda_graph"],
        disable_overlap_schedule=engine_cfg["disable_overlap_schedule"],
        chunked_prefill_size=engine_cfg["chunked_prefill_size"],
        enable_soft_thinking=engine_cfg["enable_soft_thinking"],
        think_end_str=engine_cfg["think_end_str"],
        add_noise_dirichlet=engine_cfg["add_noise_dirichlet"],
        add_noise_gumbel_softmax=engine_cfg["add_noise_gumbel_softmax"],
        max_topk=engine_cfg["max_topk"],
        cuda_graph_max_bs=engine_cfg["cuda_graph_max_bs"],
        sampling_backend=engine_cfg["sampling_backend"],
    )


def evaluate_outputs(
    dataset: str,
    samples: list[dict[str, Any]],
    prompts: list[str],
    generation_summary: dict[str, Any],
    start_idx: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_start_time = time.time()
    results: list[dict[str, Any]] = []
    pass_1_list: list[float] = []
    pass_5_list: list[float] = []
    pass_8_list: list[float] = []
    pass_10_list: list[float] = []
    pass_16_list: list[float] = []

    grouped_outputs = generation_summary["grouped_outputs"]
    grouped_finish = generation_summary["grouped_finish_generation"]
    grouped_full_lens = generation_summary["grouped_full_lens"]
    grouped_think_lens = generation_summary["grouped_think_lens"]

    for local_idx, sample in enumerate(samples):
        pred_cot = grouped_outputs[local_idx]
        finish_generation = grouped_finish[local_idx]
        full_lens = grouped_full_lens[local_idx]
        think_lens = grouped_think_lens[local_idx]

        pred_answers: list[str] = []
        score: list[float] = []
        judge_info: list[dict[str, Any]] = []
        for output_text, finished in zip(
            pred_cot,
            finish_generation,
            strict=True,
        ):
            rule_judge_result, extracted_answer = evaluator_map[dataset].rule_judge(
                output_text,
                sample["final_answer"],
                finished,
            )
            pred_answers.append(extracted_answer)
            score.append(1.0 if rule_judge_result else 0.0)
            judge_info.append(
                {
                    "rule_judge_result": bool(rule_judge_result),
                    "extracted_answer": extracted_answer,
                }
            )

        num_correct = int(sum(score))
        n_preds = len(score)
        pass_1_list.append(estimate_pass_at_k(n_preds, num_correct, 1))
        pass_5_list.append(estimate_pass_at_k(n_preds, num_correct, 5))
        pass_8_list.append(estimate_pass_at_k(n_preds, num_correct, 8))
        pass_10_list.append(estimate_pass_at_k(n_preds, num_correct, 10))
        pass_16_list.append(estimate_pass_at_k(n_preds, num_correct, 16))

        results.append(
            {
                "idx": start_idx + local_idx,
                "question": sample["prompt"][0]["value"],
                "prompt": prompts[local_idx],
                "ground_truth": sample["final_answer"],
                "pred": pred_answers,
                "pred_cot": pred_cot,
                "score": score,
                "judge_info": judge_info,
                "finish_generation": finish_generation,
                "full_len": full_lens,
                "think_len": think_lens,
                "avg_full_len": sum(full_lens) / len(full_lens),
                "avg_think_len": sum(think_lens) / len(think_lens),
                "n": n_preds,
                "generation_mode": CONFIG["generation_mode"],
            }
        )

    eval_time = time.time() - eval_start_time
    num_samples = len(results)

    def pct(values: list[float]) -> float:
        return round((sum(values) / len(values)) * 100, 1) if values else 0.0

    metrics = {
        "num_samples": num_samples,
        "pass@1": pct(pass_1_list),
        "pass@5": pct(pass_5_list),
        "pass@8": pct(pass_8_list),
        "pass@10": pct(pass_10_list),
        "pass@16": pct(pass_16_list),
        "avg_entropy": generation_summary["avg_entropy"],
        "avg_num_full_output_tokens": (
            generation_summary["avg_num_full_output_tokens"]
        ),
        "avg_num_think_tokens": generation_summary["avg_num_think_tokens"],
        "evaluation_time_sec": eval_time,
    }
    return results, metrics


def build_output_paths(config: dict[str, Any]) -> tuple[str, str]:
    dataset = config["dataset"]
    output_dir = os.path.join(config["output_dir"], "results", dataset)
    os.makedirs(output_dir, exist_ok=True)

    sampling_cfg = config["sampling"]
    engine_cfg = config["engine"]
    replay_bs = config["replay_batch_size"]
    replay_bs_str = "all" if replay_bs is None else str(replay_bs)
    noise_suffix = (
        (
            f"_gumbel_{sampling_cfg['gumbel_softmax_temperature']}"
            if engine_cfg["add_noise_gumbel_softmax"]
            else ""
        )
        + (
            f"_dirichlet_{sampling_cfg['dirichlet_alpha']}"
            if engine_cfg["add_noise_dirichlet"]
            else ""
        )
    )
    base_filename = (
        f"{config['model_name'].split('/')[-1]}_{dataset}_"
        f"{config['generation_mode']}_{engine_cfg['enable_soft_thinking']}_"
        f"{config['n_sampling']}_{sampling_cfg['temperature']}_"
        f"{sampling_cfg['top_p']}_{sampling_cfg['top_k']}_{sampling_cfg['min_p']}_"
        f"{sampling_cfg['repetition_penalty']}_{engine_cfg['max_topk']}_"
        f"{sampling_cfg['max_new_tokens']}_{config['min_response_budget_tokens']}_"
        f"{config['warmup_batch_size']}_{replay_bs_str}_"
        f"{config['start_idx']}_{config['end_idx']}"
        f"{noise_suffix}"
    )
    results_file = os.path.join(output_dir, f"{base_filename}.json")
    metrics_file = os.path.join(output_dir, f"{base_filename}_metrics.json")
    return results_file, metrics_file


def main() -> None:
    sampling_params = build_sampling_params(CONFIG)
    validate_config(CONFIG, sampling_params)

    dataset = CONFIG["dataset"]
    raw_samples = load_dataset(dataset)
    start_idx = CONFIG["start_idx"]
    end_idx = min(CONFIG["end_idx"], len(raw_samples))
    samples = raw_samples[start_idx:end_idx]

    print(json.dumps(CONFIG, indent=2), flush=True)
    print(
        f"Loaded dataset={dataset} with sample range [{start_idx}, {end_idx}).",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_name"],
        trust_remote_code=True,
    )
    prompts = [build_prompt(dataset, sample, tokenizer) for sample in samples]

    engine = None
    generation_start = time.time()
    try:
        engine = build_engine(CONFIG)
        if CONFIG["generation_mode"] == "replay":
            generation_summary = run_replay_generation(
                prompts=prompts,
                engine=engine,
                sampling_params=sampling_params,
                num_samples=CONFIG["n_sampling"],
                warmup_batch_size=CONFIG["warmup_batch_size"],
                replay_batch_size=CONFIG["replay_batch_size"],
                min_response_budget_tokens=(
                    CONFIG["min_response_budget_tokens"]
                ),
                tokenizer=tokenizer,
            )
        else:
            generation_summary = run_standard_generation(
                prompts=prompts,
                engine=engine,
                sampling_params=sampling_params,
                num_samples=CONFIG["n_sampling"],
                tokenizer=tokenizer,
                min_response_budget_tokens=(
                    CONFIG["min_response_budget_tokens"]
                ),
            )
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if engine is not None:
            engine.shutdown()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generation_time_min = (time.time() - generation_start) / 60.0
    results, metrics = evaluate_outputs(
        dataset=dataset,
        samples=samples,
        prompts=prompts,
        generation_summary=generation_summary,
        start_idx=start_idx,
    )
    metrics["generation_time_min"] = generation_time_min

    results_file, metrics_file = build_output_paths(CONFIG)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved results to {results_file}", flush=True)
    print(f"Saved metrics to {metrics_file}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

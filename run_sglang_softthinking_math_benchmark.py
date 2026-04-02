from __future__ import annotations

import argparse
import copy
import json
import logging
import logging.config
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sglang as sgl
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from matheval import evaluator_map
from sglang.srt.managers.io_struct import GenerateReqInput


MATH_DATASETS = [
    "math500",
    "aime2024",
    "aime2025",
    "gpqa_diamond",
    "gsm8k",
    "amc23",
]

DATASET_FILES = {
    "math500": "./datasets/math500.json",
    "aime2024": "./datasets/aime2024.json",
    "aime2025": "./datasets/aime2025.json",
    "gpqa_diamond": "./datasets/gpqa_diamond.json",
    "gsm8k": "./datasets/gsm8k.json",
    "amc23": "./datasets/amc23.json",
}

MATH_QUERY_TEMPLATE = """
Please reason step by step, and put your final answer within \\boxed{{}}.

{Question}
""".strip()

GPQA_QUERY_TEMPLATE = """
Please solve the following multiple-choice question. Please show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".

{Question}
""".strip()

_THINK_MODE_CACHE_ATTR = "_soft_thinking_supports_thinking_mode_cached"


@dataclass
class GenerationResult:
    grouped_outputs: list[list[str]]
    grouped_generated_tokens: list[list[int]]
    grouped_think_tokens: list[list[int]]
    grouped_finish_reasons: list[list[dict[str, Any] | None]]
    avg_num_full_output_tokens: float = -1.0
    avg_num_think_tokens: float = -1.0
    avg_entropy: float = -1.0
    entropies_list: list[list[float]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Math-only Soft-Thinking benchmark with STPO-style replay inference."
    )

    parser.add_argument(
        "--logging_config_file",
        type=str,
        default="./configs/loggers.yml",
        help="Optional logging config path. Falls back to basic logging if missing.",
    )
    parser.add_argument(
        "--logger_name",
        type=str,
        default="MATH_EVAL",
        help="Logger name.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=MATH_DATASETS,
        default="math500",
        help="Math dataset to evaluate.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="general-cot",
        help="Prompt type label. The benchmark currently implements general-cot style prompts.",
    )
    parser.add_argument(
        "--apply_chat_template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply tokenizer chat template before generation.",
    )
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable thinking mode in chat templating and SGLang generation.",
    )
    parser.add_argument(
        "--generation_mode",
        type=str,
        choices=["standard", "replay"],
        default="standard",
        help="Inference mode. Replay matches the two-stage STPO replay flow.",
    )
    parser.add_argument(
        "--warmup_batch_size",
        type=int,
        default=200,
        help="Warmup chunk size used by replay generation.",
    )
    parser.add_argument(
        "--min_response_budget_tokens",
        type=int,
        default=5000,
        help="Reserved response budget used by standard/replay thinking flows.",
    )
    parser.add_argument(
        "--overwrite_outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start dataset index.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End dataset index (exclusive). Defaults to the dataset length.",
    )

    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES value. Default matches GPU_MESH [[0]].",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Model name or path.",
    )
    parser.add_argument(
        "--model_backend",
        type=str,
        choices=["sglang"],
        default="sglang",
        help="Model backend. This benchmark is SGLang-only.",
    )
    parser.add_argument(
        "--model_nickname",
        type=str,
        default=None,
        help="Optional model nickname used in the output directory name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base output directory. Defaults to ./outputs/math-eval/<model_nickname>.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Tensor parallel size.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of sampled completions per question.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--max_running_requests",
        type=int,
        default=None,
        help="Optional SGLang max_running_requests override.",
    )
    parser.add_argument(
        "--mem_fraction_static",
        type=float,
        default=0.7,
        help="Static GPU memory fraction.",
    )
    parser.add_argument(
        "--sampling_backend",
        type=str,
        choices=["pytorch", "flashinfer"],
        default="flashinfer",
        help="SGLang sampling backend.",
    )
    parser.add_argument(
        "--cuda_graph_max_bs",
        type=int,
        default=8,
        help="SGLang cuda_graph_max_bs.",
    )
    parser.add_argument(
        "--disable_cuda_graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable CUDA graph in SGLang.",
    )
    parser.add_argument(
        "--disable_overlap_schedule",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable overlap schedule in SGLang. Required for replay.",
    )
    parser.add_argument(
        "--chunked_prefill_size",
        type=int,
        default=-1,
        help="SGLang chunked_prefill_size. Replay requires -1.",
    )
    parser.add_argument(
        "--enable_soft_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SGLang soft thinking.",
    )
    parser.add_argument(
        "--add_noise_dirichlet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Dirichlet noise.",
    )
    parser.add_argument(
        "--add_noise_gumbel_softmax",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Gumbel softmax noise.",
    )
    parser.add_argument(
        "--max_topk",
        type=int,
        default=10,
        help="SGLang max_topk for soft thinking trace capture.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="Top-k.",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.001,
        help="Min-p.",
    )
    parser.add_argument(
        "--after_thinking_temperature",
        type=float,
        default=0.6,
        help="Temperature after thinking.",
    )
    parser.add_argument(
        "--after_thinking_top_p",
        type=float,
        default=0.95,
        help="Top-p after thinking.",
    )
    parser.add_argument(
        "--after_thinking_top_k",
        type=int,
        default=30,
        help="Top-k after thinking.",
    )
    parser.add_argument(
        "--after_thinking_min_p",
        type=float,
        default=0.0,
        help="Min-p after thinking.",
    )
    parser.add_argument(
        "--gumbel_softmax_temperature",
        type=float,
        default=1.0,
        help="Gumbel-softmax temperature.",
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=1.0,
        help="Dirichlet alpha.",
    )
    parser.add_argument(
        "--max_generated_tokens",
        type=int,
        default=32768,
        help="Max new tokens.",
    )
    parser.add_argument(
        "--think_end_str",
        type=str,
        default="</think>",
        help="Thinking end string.",
    )
    parser.add_argument(
        "--early_stopping_entropy_threshold",
        type=float,
        default=0.1,
        help="Early stopping entropy threshold.",
    )
    parser.add_argument(
        "--early_stopping_length_threshold",
        type=int,
        default=256,
        help="Early stopping length threshold.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty.",
    )

    return parser.parse_args()


def resolve_model_nickname(args: argparse.Namespace) -> str:
    if args.model_nickname:
        return args.model_nickname
    layer_weights = os.getenv("LAYER_WEIGHTS", "")
    return f"DS-R1-Distill-Qwen-7B-{args.model_backend}-logit-{layer_weights}"


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path("./outputs/math-eval") / resolve_model_nickname(args)


def configure_logger(args: argparse.Namespace) -> logging.Logger:
    config_candidates = [Path(args.logging_config_file)]
    default_cfg = Path("./configs/loggers.yml")
    if default_cfg not in config_candidates:
        config_candidates.append(default_cfg)
    stpo_cfg = Path("../STPO/configs/examples/loggers.yml")
    if stpo_cfg not in config_candidates:
        config_candidates.append(stpo_cfg)

    for cfg_path in config_candidates:
        if not cfg_path.is_file():
            continue
        try:
            import yaml

            with cfg_path.open("r", encoding="utf-8") as f:
                logging_config = yaml.safe_load(f)
            logging.config.dictConfig(logging_config)
            logger = logging.getLogger(args.logger_name)
            logger.info("Loaded logging config from %s", cfg_path)
            return logger
        except Exception as exc:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )
            logger = logging.getLogger(args.logger_name)
            logger.warning(
                "Failed to load logging config from %s: %s. Falling back to basic logging.",
                cfg_path,
                exc,
            )
            return logger

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(args.logger_name)


def tokenizer_supports_thinking_mode(
    tokenizer: PreTrainedTokenizerBase,
) -> bool:
    if hasattr(tokenizer, _THINK_MODE_CACHE_ATTR):
        return getattr(tokenizer, _THINK_MODE_CACHE_ATTR)

    vocab = tokenizer.get_vocab()
    result = "<think>" in vocab and "</think>" in vocab
    setattr(tokenizer, _THINK_MODE_CACHE_ATTR, result)
    return result


def apply_chat_template_for_thinking_mode(
    tokenizer: PreTrainedTokenizerBase,
    user_input: str,
    force_thinking: bool,
) -> str:
    if force_thinking and not tokenizer_supports_thinking_mode(tokenizer):
        raise ValueError(
            "The tokenizer does not expose <think> / </think> tokens required for thinking mode."
        )

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_input}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=force_thinking,
    )
    if not prompt:
        raise ValueError("Tokenizer returned an empty chat template prompt.")
    if force_thinking and not prompt.rstrip().endswith("<think>"):
        prompt += "<think>"
    return prompt


def ensure_output_list(outputs: Any) -> list[dict[str, Any]]:
    if isinstance(outputs, list):
        return outputs
    return [outputs]


def extract_finish_reason_type(meta_info: dict[str, Any]) -> str | None:
    finish_reason = meta_info.get("finish_reason")
    if isinstance(finish_reason, dict):
        finish_reason_type = finish_reason.get("type")
        return None if finish_reason_type is None else str(finish_reason_type)
    if finish_reason is None:
        return None
    return str(finish_reason)


def serialize_finish_reason(meta_info: dict[str, Any]) -> dict[str, Any] | None:
    finish_reason = meta_info.get("finish_reason")
    if finish_reason is None:
        return None
    if isinstance(finish_reason, dict):
        return dict(finish_reason)
    return {"type": str(finish_reason)}


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
        raise AssertionError(f"`full_len` must be positive, got {full_len}.")
    max_think_len = full_len if allow_all_thinking else full_len - 1
    if think_len < 0 or think_len > max_think_len:
        raise AssertionError(
            f"Invalid full_len/think_len relation: full_len={full_len}, think_len={think_len}."
        )
    return full_len, think_len


def extract_output_entropies(output: dict[str, Any]) -> list[float]:
    entropies = output.get("entropies", [])
    if entropies is None:
        return []
    if not isinstance(entropies, list):
        raise AssertionError(
            f"Output entropies must be a list, got {type(entropies).__name__}."
        )
    return [float(x) for x in entropies]


def extract_sglang_text(output: dict[str, Any], tokenizer: Any) -> str:
    if "text" in output:
        return str(output["text"])

    output_ids = output.get("output_ids")
    if isinstance(output_ids, list) and output_ids:
        if isinstance(output_ids[0], list):
            output_ids = output_ids[0]
        if all(isinstance(x, int) for x in output_ids):
            return tokenizer.decode(output_ids, skip_special_tokens=False)
    raise AssertionError("SGLang output does not contain decodable text.")


def close_thinking_trace(thinking_text: str, think_end_str: str) -> str:
    if thinking_text.endswith(think_end_str):
        return thinking_text
    return thinking_text + think_end_str


def extract_replay_trace_from_warmup_output(
    output: dict[str, Any],
    sampling_params: dict[str, Any],
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
            "Warmup output missing replay trace fields (`output_topk_idx_list` / `output_topk_prob_list`)."
        )
    if len(topk_indices) != len(topk_probs):
        raise AssertionError(
            "Warmup trace length mismatch: "
            + f"len(topk_idx)={len(topk_indices)}, len(topk_prob)={len(topk_probs)}"
        )
    if len(topk_indices) < think_len:
        raise AssertionError(
            f"Warmup trace shorter than think_len: trace_len={len(topk_indices)}, think_len={think_len}"
        )

    think_end_str = str(sampling_params["think_end_str"])
    if not did_stop_on_stop_str(meta_info, think_end_str):
        raise AssertionError(
            "Warmup did not stop at think-end boundary. "
            + f"finish_reason={meta_info.get('finish_reason')}"
        )

    return {
        "replay_trace": {
            "topk_indices": copy.deepcopy(topk_indices[:think_len]),
            "topk_probs": copy.deepcopy(topk_probs[:think_len]),
        },
        "full_len": full_len,
        "think_len": think_len,
    }


def init_grouped_lists(num_samples: int) -> tuple[
    list[list[str]],
    list[list[int]],
    list[list[int]],
    list[list[dict[str, Any] | None]],
]:
    return (
        [[] for _ in range(num_samples)],
        [[] for _ in range(num_samples)],
        [[] for _ in range(num_samples)],
        [[] for _ in range(num_samples)],
    )


def finalize_generation_result(
    grouped_outputs: list[list[str]],
    grouped_generated_tokens: list[list[int]],
    grouped_think_tokens: list[list[int]],
    grouped_finish_reasons: list[list[dict[str, Any] | None]],
    entropies_list: list[list[float]],
) -> GenerationResult:
    flat_generated_tokens = [
        float(token_count)
        for sample_tokens in grouped_generated_tokens
        for token_count in sample_tokens
    ]
    flat_think_tokens = [
        float(token_count)
        for sample_tokens in grouped_think_tokens
        for token_count in sample_tokens
    ]
    valid_entropies = [ents for ents in entropies_list if ents]
    avg_entropy = -1.0
    if valid_entropies:
        avg_entropy = sum(
            sum(entropies) / len(entropies) for entropies in valid_entropies
        ) / len(valid_entropies)

    avg_num_full_output_tokens = (
        sum(flat_generated_tokens) / len(flat_generated_tokens)
        if flat_generated_tokens
        else -1.0
    )
    avg_num_think_tokens = (
        sum(flat_think_tokens) / len(flat_think_tokens)
        if flat_think_tokens
        else -1.0
    )
    return GenerationResult(
        grouped_outputs=grouped_outputs,
        grouped_generated_tokens=grouped_generated_tokens,
        grouped_think_tokens=grouped_think_tokens,
        grouped_finish_reasons=grouped_finish_reasons,
        avg_num_full_output_tokens=avg_num_full_output_tokens,
        avg_num_think_tokens=avg_num_think_tokens,
        avg_entropy=avg_entropy,
        entropies_list=entropies_list,
    )


def build_sampling_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "after_thinking_temperature": args.after_thinking_temperature,
        "after_thinking_top_p": args.after_thinking_top_p,
        "after_thinking_top_k": args.after_thinking_top_k,
        "after_thinking_min_p": args.after_thinking_min_p,
        "gumbel_softmax_temperature": args.gumbel_softmax_temperature,
        "dirichlet_alpha": args.dirichlet_alpha,
        "max_new_tokens": args.max_generated_tokens,
        "think_end_str": args.think_end_str,
        "early_stopping_entropy_threshold": args.early_stopping_entropy_threshold,
        "early_stopping_length_threshold": args.early_stopping_length_threshold,
    }


def build_engine_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_path": args.model_name,
        "tp_size": args.num_gpus,
        "log_level": "info",
        "trust_remote_code": True,
        "random_seed": args.random_seed,
        "max_running_requests": args.max_running_requests,
        "mem_fraction_static": args.mem_fraction_static,
        "disable_cuda_graph": args.disable_cuda_graph,
        "disable_overlap_schedule": args.disable_overlap_schedule,
        "chunked_prefill_size": args.chunked_prefill_size,
        "enable_soft_thinking": args.enable_soft_thinking,
        "add_noise_dirichlet": args.add_noise_dirichlet,
        "add_noise_gumbel_softmax": args.add_noise_gumbel_softmax,
        "max_topk": args.max_topk,
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
        "sampling_backend": args.sampling_backend,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.generation_mode == "replay" and args.model_backend != "sglang":
        raise ValueError("Replay mode is only supported by SGLang.")
    if args.model_backend != "sglang":
        raise ValueError("This benchmark only supports the SGLang backend.")
    if not args.disable_overlap_schedule:
        raise ValueError("disable_overlap_schedule must be True.")
    if args.enable_soft_thinking and not args.enable_thinking:
        raise ValueError(
            "enable_soft_thinking=True requires --enable_thinking."
        )
    if args.enable_thinking and not (
        0 < args.min_response_budget_tokens < args.max_generated_tokens
    ):
        raise ValueError(
            "Thinking mode requires 0 < min_response_budget_tokens < max_generated_tokens."
        )
    if args.generation_mode == "replay" and args.chunked_prefill_size != -1:
        raise ValueError(
            "Replay mode requires chunked_prefill_size == -1."
        )
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")
    if args.start_idx < 0:
        raise ValueError("--start_idx must be >= 0.")
    if args.end_idx is not None and args.end_idx <= args.start_idx:
        raise ValueError("--end_idx must be greater than --start_idx.")


def load_samples(dataset: str) -> list[dict[str, Any]]:
    dataset_path = Path(DATASET_FILES[dataset])
    with dataset_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_user_question(dataset: str, sample: dict[str, Any]) -> str:
    question = sample["prompt"][0]["value"]
    if dataset == "gpqa_diamond":
        return GPQA_QUERY_TEMPLATE.format(Question=question)
    return MATH_QUERY_TEMPLATE.format(Question=question)


def build_prompt(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    apply_chat_template: bool,
    enable_thinking: bool,
) -> str:
    if not apply_chat_template:
        return question
    return apply_chat_template_for_thinking_mode(
        tokenizer=tokenizer,
        user_input=question,
        force_thinking=enable_thinking,
    )


def run_sglang_standard_generation(
    all_samples: list[dict[str, Any]],
    model: Any,
    sampling_params: dict[str, Any],
    n_sampling: int,
    tokenizer: Any,
    logger: logging.Logger,
    enable_thinking: bool,
    min_response_budget_tokens: int,
) -> GenerationResult:
    (
        grouped_outputs,
        grouped_generated_tokens,
        grouped_think_tokens,
        grouped_finish_reasons,
    ) = init_grouped_lists(len(all_samples))
    entropies_list: list[list[float]] = []

    input_units = [
        {"sample_idx": sample_idx, "prompt": sample["prompt"]}
        for sample_idx, sample in enumerate(all_samples)
        for _ in range(n_sampling)
    ]
    input_prompts = [unit["prompt"] for unit in input_units]

    if not enable_thinking:
        outputs = ensure_output_list(model.generate(input_prompts, sampling_params))
        if len(outputs) != len(input_units):
            raise AssertionError(
                f"SGLang output count mismatch: expected={len(input_units)}, got={len(outputs)}"
            )
        for unit, output in zip(input_units, outputs, strict=True):
            meta_info = output.get("meta_info")
            if not isinstance(meta_info, dict):
                raise AssertionError("SGLang output missing `meta_info`.")
            full_len, think_len = validate_output_lengths(
                meta_info,
                allow_all_thinking=output_hit_token_budget(meta_info),
            )
            sample_idx = int(unit["sample_idx"])
            grouped_outputs[sample_idx].append(extract_sglang_text(output, tokenizer))
            grouped_generated_tokens[sample_idx].append(full_len)
            grouped_think_tokens[sample_idx].append(think_len)
            grouped_finish_reasons[sample_idx].append(
                serialize_finish_reason(meta_info)
            )
            entropies_list.append(extract_output_entropies(output))
        return finalize_generation_result(
            grouped_outputs,
            grouped_generated_tokens,
            grouped_think_tokens,
            grouped_finish_reasons,
            entropies_list,
        )

    base_max_new_tokens = int(sampling_params["max_new_tokens"])
    thinking_max_new_tokens = base_max_new_tokens - min_response_budget_tokens
    think_end_str = str(sampling_params["think_end_str"])
    logger.info(
        "Standard SGLang generation reserving %s response tokens; thinking phase cap=%s.",
        min_response_budget_tokens,
        thinking_max_new_tokens,
    )

    thinking_params = copy.deepcopy(sampling_params)
    thinking_params["max_new_tokens"] = thinking_max_new_tokens
    thinking_params["stop"] = think_end_str
    thinking_params["no_stop_trim"] = True
    thinking_outputs = ensure_output_list(model.generate(input_prompts, thinking_params))
    if len(thinking_outputs) != len(input_units):
        raise AssertionError(
            "Thinking-phase output count mismatch: "
            + f"expected={len(input_units)}, got={len(thinking_outputs)}"
        )

    continuation_units: list[dict[str, Any]] = []
    for unit, output in zip(input_units, thinking_outputs, strict=True):
        meta_info = output.get("meta_info")
        if not isinstance(meta_info, dict):
            raise AssertionError("Thinking output missing `meta_info`.")
        full_len, think_len = validate_output_lengths(
            meta_info,
            allow_all_thinking=True,
        )
        output_text = extract_sglang_text(output, tokenizer)
        entropies = extract_output_entropies(output)
        prefill_text = close_thinking_trace(output_text, think_end_str)
        if not did_stop_on_stop_str(meta_info, think_end_str):
            logger.warning(
                "Standard warmup sample_idx=%s did not finish thinking in %s tokens. finish_reason=%r. "
                + "Force-closing with %r and continuing with response budget=%s.",
                unit["sample_idx"],
                thinking_max_new_tokens,
                meta_info.get("finish_reason"),
                think_end_str,
                min_response_budget_tokens,
            )
        continuation_units.append(
            {
                "sample_idx": int(unit["sample_idx"]),
                "prompt": str(unit["prompt"]) + prefill_text,
                "prefill_text": prefill_text,
                "thinking_full_len": full_len,
                "thinking_think_len": think_len,
                "thinking_entropies": entropies,
            }
        )

    continuation_params = copy.deepcopy(sampling_params)
    continuation_params.pop("stop", None)
    continuation_params["max_new_tokens"] = min_response_budget_tokens
    continuation_outputs = ensure_output_list(
        model.generate([unit["prompt"] for unit in continuation_units], continuation_params)
    )
    if len(continuation_outputs) != len(continuation_units):
        raise AssertionError(
            "Continuation output count mismatch: "
            + f"expected={len(continuation_units)}, got={len(continuation_outputs)}"
        )

    for output, unit in zip(continuation_outputs, continuation_units, strict=True):
        meta_info = output.get("meta_info")
        if not isinstance(meta_info, dict):
            raise AssertionError("Continuation output missing `meta_info`.")
        hit_token_budget = output_hit_token_budget(meta_info)
        if hit_token_budget:
            logger.warning(
                "Standard continuation output reached response token budget. sample_idx=%s, "
                + "response_budget=%s, finish_reason=%r.",
                unit["sample_idx"],
                min_response_budget_tokens,
                meta_info.get("finish_reason"),
            )
        full_len, think_len = validate_output_lengths(
            meta_info,
            allow_all_thinking=hit_token_budget,
        )
        sample_idx = int(unit["sample_idx"])
        grouped_outputs[sample_idx].append(
            str(unit["prefill_text"]) + extract_sglang_text(output, tokenizer)
        )
        grouped_generated_tokens[sample_idx].append(
            int(unit["thinking_full_len"]) + full_len
        )
        grouped_think_tokens[sample_idx].append(
            int(unit["thinking_think_len"]) + think_len
        )
        grouped_finish_reasons[sample_idx].append(
            serialize_finish_reason(meta_info)
        )
        entropies_list.append(
            list(unit["thinking_entropies"]) + extract_output_entropies(output)
        )

    return finalize_generation_result(
        grouped_outputs,
        grouped_generated_tokens,
        grouped_think_tokens,
        grouped_finish_reasons,
        entropies_list,
    )


def run_sglang_replay_generation(
    all_samples: list[dict[str, Any]],
    model: Any,
    sampling_params: dict[str, Any],
    n_sampling: int,
    warmup_batch_size: int,
    min_response_budget_tokens: int,
    tokenizer: Any,
    logger: logging.Logger,
) -> GenerationResult:
    if not all_samples:
        return GenerationResult([], [], [], [])

    (
        grouped_outputs,
        grouped_generated_tokens,
        grouped_think_tokens,
        grouped_finish_reasons,
    ) = init_grouped_lists(len(all_samples))
    entropies_list: list[list[float]] = []

    base_max_new_tokens = int(sampling_params["max_new_tokens"])
    response_max_new_tokens = min_response_budget_tokens
    warmup_max_new_tokens = base_max_new_tokens - response_max_new_tokens
    think_end_str = str(sampling_params["think_end_str"])
    logger.info(
        "Replay SGLang generation reserving %s response tokens; thinking phase cap=%s.",
        response_max_new_tokens,
        warmup_max_new_tokens,
    )

    sample_infos = [
        {
            "sample_idx": sample_idx,
            "prompt": sample["prompt"],
            "prompt_ids": tokenizer.encode(sample["prompt"]),
        }
        for sample_idx, sample in enumerate(all_samples)
    ]

    for start_idx in range(0, len(sample_infos), warmup_batch_size):
        chunk_infos = sample_infos[start_idx : start_idx + warmup_batch_size]
        warmup_params = copy.deepcopy(sampling_params)
        warmup_params["max_new_tokens"] = warmup_max_new_tokens
        warmup_params["stop"] = think_end_str
        warmup_params["no_stop_trim"] = True
        warmup_outputs = ensure_output_list(
            model.generate(
                prompt=[item["prompt"] for item in chunk_infos],
                sampling_params=warmup_params,
                return_logprob=True,
            )
        )
        if len(warmup_outputs) != len(chunk_infos):
            raise AssertionError(
                "Warmup output count mismatch: "
                + f"expected={len(chunk_infos)}, got={len(warmup_outputs)}"
            )

        replay_units: list[dict[str, Any]] = []
        fallback_units: list[dict[str, Any]] = []
        for sample_info, warmup_output in zip(chunk_infos, warmup_outputs, strict=True):
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
                logger.error(
                    "Replay warmup sample_idx=%s did not finish thinking in %s tokens. finish_reason=%r. "
                    + "Force-closing with %r and continuing with response budget=%s.",
                    sample_info["sample_idx"],
                    warmup_max_new_tokens,
                    meta_info.get("finish_reason"),
                    think_end_str,
                    response_max_new_tokens,
                )
                closed_thinking_text = close_thinking_trace(warmup_text, think_end_str)
                for _ in range(n_sampling):
                    fallback_units.append(
                        {
                            "sample_idx": int(sample_info["sample_idx"]),
                            "prompt": str(sample_info["prompt"]) + closed_thinking_text,
                            "prefill_text": closed_thinking_text,
                            "warmup_full_len": warmup_full_len,
                            "warmup_think_len": warmup_think_len,
                            "warmup_entropies": warmup_entropies,
                        }
                    )
                continue

            closed_warmup_text = close_thinking_trace(
                warmup_text, think_end_str
            )
            warmup_info = extract_replay_trace_from_warmup_output(
                output=warmup_output,
                sampling_params=sampling_params,
            )
            for _ in range(n_sampling):
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
            replay_obj = GenerateReqInput(
                input_ids=[
                    copy.deepcopy(unit["prompt_ids"]) for unit in replay_units
                ],
                sampling_params={
                    **copy.deepcopy(sampling_params),
                    "max_new_tokens": response_max_new_tokens,
                },
                return_logprob=False,
                soft_thinking_trace=[
                    copy.deepcopy(unit["replay_trace"]) for unit in replay_units
                ],
            )
            replay_obj.sampling_params.pop("stop", None)
            replay_outputs = ensure_output_list(
                model.generate(
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
            )
            if len(replay_outputs) != len(replay_units):
                raise AssertionError(
                    "Replay output count mismatch: "
                    + f"expected={len(replay_units)}, got={len(replay_outputs)}"
                )

            for output, replay_unit in zip(replay_outputs, replay_units, strict=True):
                meta_info = output.get("meta_info")
                if not isinstance(meta_info, dict):
                    raise AssertionError("Replay output missing `meta_info`.")
                hit_token_budget = output_hit_token_budget(meta_info)
                if hit_token_budget:
                    logger.warning(
                        "Replay output reached response token budget in chunk [%s, %s). sample_idx=%s, "
                        + "response_budget=%s, finish_reason=%r.",
                        start_idx,
                        start_idx + len(chunk_infos),
                        replay_unit["sample_idx"],
                        response_max_new_tokens,
                        meta_info.get("finish_reason"),
                    )
                full_len, think_len = validate_output_lengths(
                    meta_info,
                    allow_all_thinking=hit_token_budget,
                )
                sample_idx = int(replay_unit["sample_idx"])
                grouped_outputs[sample_idx].append(
                    str(replay_unit["prefill_text"])
                    + extract_sglang_text(output, tokenizer)
                )
                grouped_generated_tokens[sample_idx].append(
                    int(replay_unit["warmup_full_len"]) + full_len
                )
                grouped_think_tokens[sample_idx].append(
                    int(replay_unit["warmup_think_len"]) + think_len
                )
                grouped_finish_reasons[sample_idx].append(
                    serialize_finish_reason(meta_info)
                )
                entropies_list.append(
                    list(replay_unit["warmup_entropies"])
                    + extract_output_entropies(output)
                )

        if fallback_units:
            fallback_params = copy.deepcopy(sampling_params)
            fallback_params.pop("stop", None)
            fallback_params["max_new_tokens"] = response_max_new_tokens
            fallback_outputs = ensure_output_list(
                model.generate(
                    prompt=[unit["prompt"] for unit in fallback_units],
                    sampling_params=fallback_params,
                )
            )
            if len(fallback_outputs) != len(fallback_units):
                raise AssertionError(
                    "Fallback output count mismatch: "
                    + f"expected={len(fallback_units)}, got={len(fallback_outputs)}"
                )

            for output, fallback_unit in zip(
                fallback_outputs,
                fallback_units,
                strict=True,
            ):
                meta_info = output.get("meta_info")
                if not isinstance(meta_info, dict):
                    raise AssertionError("Fallback output missing `meta_info`.")
                hit_token_budget = output_hit_token_budget(meta_info)
                if hit_token_budget:
                    logger.warning(
                        "Fallback continuation reached response token budget. sample_idx=%s, "
                        + "response_budget=%s, finish_reason=%r.",
                        fallback_unit["sample_idx"],
                        response_max_new_tokens,
                        meta_info.get("finish_reason"),
                    )
                full_len, think_len = validate_output_lengths(
                    meta_info,
                    allow_all_thinking=hit_token_budget,
                )
                sample_idx = int(fallback_unit["sample_idx"])
                grouped_outputs[sample_idx].append(
                    str(fallback_unit["prefill_text"])
                    + extract_sglang_text(output, tokenizer)
                )
                grouped_generated_tokens[sample_idx].append(
                    int(fallback_unit["warmup_full_len"]) + full_len
                )
                grouped_think_tokens[sample_idx].append(
                    int(fallback_unit["warmup_think_len"]) + think_len
                )
                grouped_finish_reasons[sample_idx].append(
                    serialize_finish_reason(meta_info)
                )
                entropies_list.append(
                    list(fallback_unit["warmup_entropies"])
                    + extract_output_entropies(output)
                )

    for sample_idx, sample_outputs in enumerate(grouped_outputs):
        if len(sample_outputs) != n_sampling:
            raise AssertionError(
                f"Replay output count mismatch for sample {sample_idx}: expected={n_sampling}, got={len(sample_outputs)}"
            )

    return finalize_generation_result(
        grouped_outputs,
        grouped_generated_tokens,
        grouped_think_tokens,
        grouped_finish_reasons,
        entropies_list,
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


def build_output_paths(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path]:
    model_label = resolve_model_nickname(args)
    dataset_dir = output_dir / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    noise_suffix = (
        (f"_gumbel_{args.gumbel_softmax_temperature}" if args.add_noise_gumbel_softmax else "")
        + (f"_dirichlet_{args.dirichlet_alpha}" if args.add_noise_dirichlet else "")
    )
    base_filename = (
        f"{model_label}_{args.dataset}_{args.generation_mode}_{args.enable_soft_thinking}_"
        + f"{args.num_samples}_{args.temperature}_{args.top_p}_{args.top_k}_{args.min_p}_"
        + f"{args.repetition_penalty}_{args.max_topk}_{args.max_generated_tokens}_"
        + f"{args.early_stopping_entropy_threshold}_{args.early_stopping_length_threshold}_"
        + f"warmup_{args.warmup_batch_size}_budget_{args.min_response_budget_tokens}{noise_suffix}"
    )
    return (
        dataset_dir / f"{base_filename}.json",
        dataset_dir / f"{base_filename}_statistics.json",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    logger = configure_logger(args)
    logger.info("Arguments: %s", vars(args))

    output_dir = resolve_output_dir(args)
    results_file, statistics_file = build_output_paths(args, output_dir)
    if not args.overwrite_outputs and (results_file.exists() or statistics_file.exists()):
        raise FileExistsError(
            f"Output exists and overwrite_outputs is disabled: {results_file}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    sampling_params = build_sampling_params(args)
    samples = load_samples(args.dataset)
    end_idx = len(samples) if args.end_idx is None else min(args.end_idx, len(samples))
    selected_samples = samples[args.start_idx : end_idx]

    if not selected_samples:
        raise ValueError("No samples selected for evaluation.")

    prepared_samples: list[dict[str, Any]] = []
    for idx, sample in enumerate(
        tqdm(selected_samples, desc="Preparing Prompts"),
        start=args.start_idx,
    ):
        question = build_user_question(args.dataset, sample)
        prompt = build_prompt(
            tokenizer=tokenizer,
            question=question,
            apply_chat_template=args.apply_chat_template,
            enable_thinking=args.enable_thinking,
        )
        prepared_samples.append(
            {
                "idx": idx,
                "prompt": prompt,
                "question": sample["prompt"][0]["value"],
                "ground_truth": sample["final_answer"],
            }
        )

    for i in range(min(3, len(prepared_samples))):
        logger.info("Example prompt %s:\n%s", i, prepared_samples[i]["prompt"])

    logger.info("Starting SGLang inference on %s samples.", len(prepared_samples))
    llm = sgl.Engine(**build_engine_args(args))
    benchmark_start_time = time.time()
    try:
        if args.generation_mode == "replay":
            generation_result = run_sglang_replay_generation(
                all_samples=prepared_samples,
                model=llm,
                sampling_params=sampling_params,
                n_sampling=args.num_samples,
                warmup_batch_size=args.warmup_batch_size,
                min_response_budget_tokens=args.min_response_budget_tokens,
                tokenizer=tokenizer,
                logger=logger,
            )
        else:
            generation_result = run_sglang_standard_generation(
                all_samples=prepared_samples,
                model=llm,
                sampling_params=sampling_params,
                n_sampling=args.num_samples,
                tokenizer=tokenizer,
                logger=logger,
                enable_thinking=args.enable_thinking,
                min_response_budget_tokens=args.min_response_budget_tokens,
            )
        torch.cuda.synchronize()
    finally:
        llm.shutdown()
        torch.cuda.empty_cache()
    generation_end_time = time.time()
    generation_time_min = (generation_end_time - benchmark_start_time) / 60.0
    logger.info("Inference completed in %.4f minutes.", generation_time_min)

    results: list[dict[str, Any]] = []
    pass_at_k_lists = {1: [], 5: [], 8: [], 10: [], 16: []}

    eval_start_time = time.time()
    for sample, completions, generated_tokens, think_tokens, finish_reasons in zip(
        prepared_samples,
        generation_result.grouped_outputs,
        generation_result.grouped_generated_tokens,
        generation_result.grouped_think_tokens,
        generation_result.grouped_finish_reasons,
        strict=True,
    ):
        judge_info = []
        sample_scores = []
        for completion, finish_reason in zip(
            completions,
            finish_reasons,
            strict=True,
        ):
            finish_generation = (
                finish_reason is not None
                and str(finish_reason.get("type", "")).lower() == "stop"
            )
            rule_judge_result, extracted_answer = evaluator_map[args.dataset].rule_judge(
                completion,
                sample["ground_truth"],
                finish_generation,
            )
            is_correct = bool(rule_judge_result)
            judge_info.append(
                {
                    "rule_judge_result": is_correct,
                    "extracted_answer": extracted_answer,
                    "finish_reason": finish_reason,
                }
            )
            sample_scores.append(1.0 if is_correct else 0.0)

        num_correct = int(sum(sample_scores))
        sample_pass_at_1 = estimate_pass_at_k(args.num_samples, num_correct, 1)
        for k in pass_at_k_lists:
            pass_at_k_lists[k].append(
                estimate_pass_at_k(args.num_samples, num_correct, k)
            )

        results.append(
            {
                "hyperparams": vars(args),
                "idx": sample["idx"],
                "prompt": sample["question"],
                "model_input": sample["prompt"],
                "completion": completions,
                "ground_truth": sample["ground_truth"],
                "generated_tokens": generated_tokens,
                "think_tokens": think_tokens,
                "avg_generated_tokens": (
                    sum(generated_tokens) / len(generated_tokens)
                    if generated_tokens
                    else 0.0
                ),
                "avg_think_tokens": (
                    sum(think_tokens) / len(think_tokens)
                    if think_tokens
                    else 0.0
                ),
                "n": args.num_samples,
                "finish_generation": [
                    bool(
                        finish_reason is not None
                        and str(finish_reason.get("type", "")).lower() == "stop"
                    )
                    for finish_reason in finish_reasons
                ],
                "finish_reasons": finish_reasons,
                "judge_info": judge_info,
                "passat1": sample_pass_at_1,
                "passat1_list": sample_scores,
            }
        )

    evaluation_time_sec = time.time() - eval_start_time
    logger.info("Evaluation completed in %.2f seconds.", evaluation_time_sec)

    results.sort(key=lambda item: item["idx"])
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total_num = len(results)
    mean_accuracy = sum(item["passat1"] for item in results) / total_num
    correct_results = [item for item in results if item["passat1"] > 0]
    results_statistics = {
        "dataset": args.dataset,
        "generation_mode": args.generation_mode,
        "num_samples": total_num,
        "n_sampling": args.num_samples,
        "pass@1": round(sum(pass_at_k_lists[1]) / total_num * 100, 1),
        "pass@5": round(sum(pass_at_k_lists[5]) / total_num * 100, 1),
        "pass@8": round(sum(pass_at_k_lists[8]) / total_num * 100, 1),
        "pass@10": round(sum(pass_at_k_lists[10]) / total_num * 100, 1),
        "pass@16": round(sum(pass_at_k_lists[16]) / total_num * 100, 1),
        "mean_accuracy": mean_accuracy,
        "avg_token_length-all": (
            sum(item["avg_generated_tokens"] for item in results) / total_num
        ),
        "avg_token_length-correct": (
            sum(item["avg_generated_tokens"] for item in correct_results)
            / len(correct_results)
            if correct_results
            else 0.0
        ),
        "avg_num_full_output_tokens": generation_result.avg_num_full_output_tokens,
        "avg_num_think_tokens": generation_result.avg_num_think_tokens,
        "avg_entropy": generation_result.avg_entropy,
        "generation_time_min": generation_time_min,
        "evaluation_time_sec": evaluation_time_sec,
        "time_taken_h": (time.time() - benchmark_start_time) / 3600.0,
        "all_idx": {item["idx"]: item["passat1"] for item in results},
    }
    with statistics_file.open("w", encoding="utf-8") as f:
        json.dump(results_statistics, f, indent=2)

    logger.info("Saved results to %s", results_file)
    logger.info("Saved statistics to %s", statistics_file)
    logger.info("Statistics: %s", results_statistics)


if __name__ == "__main__":
    main()

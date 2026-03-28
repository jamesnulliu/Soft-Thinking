import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sglang as sgl
from datasets import load_dataset
from transformers import AutoTokenizer

from sglang.srt.managers.io_struct import GenerateReqInput


def _generate_with_obj(llm: sgl.Engine, obj: GenerateReqInput) -> Any:
    return llm.generate(
        prompt=obj.text,
        input_ids=obj.input_ids,
        sampling_params=obj.sampling_params,
        image_data=obj.image_data,
        return_logprob=obj.return_logprob,
        logprob_start_len=obj.logprob_start_len,
        top_logprobs_num=obj.top_logprobs_num,
        token_ids_logprob=obj.token_ids_logprob,
        lora_path=obj.lora_path,
        custom_logit_processor=obj.custom_logit_processor,
        return_hidden_states=obj.return_hidden_states,
        stream=obj.stream,
        soft_thinking_trace=obj.soft_thinking_trace,
    )


def _ensure_output_list(outputs: Any) -> List[Dict[str, Any]]:
    if isinstance(outputs, list):
        return outputs
    return [outputs]


def _extract_text(output: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    if "text" in output:
        return output["text"]
    if "output_ids" in output:
        return tokenizer.decode(output["output_ids"], skip_special_tokens=False)
    return ""


def _build_engine_args(
    model_path: str,
    tp_size: int,
    max_topk: int,
    mem_fraction_static: float,
    random_seed: int,
    sampling_backend: str,
    disable_think_prefix_cache: bool,
) -> Dict[str, Any]:
    return {
        "model_path": model_path,
        "tp_size": tp_size,
        "log_level": "info",
        "trust_remote_code": True,
        "random_seed": random_seed,
        "max_running_requests": None,
        "mem_fraction_static": mem_fraction_static,
        "disable_cuda_graph": False,
        "disable_overlap_schedule": True,
        "chunked_prefill_size": -1,
        "enable_soft_thinking": True,
        "add_noise_dirichlet": False,
        "add_noise_gumbel_softmax": False,
        "max_topk": max_topk,
        "disable_think_prefix_cache": disable_think_prefix_cache,
        "cuda_graph_max_bs": 8,
        "sampling_backend": sampling_backend,
    }


def _build_sampling_params(args: argparse.Namespace) -> Dict[str, Any]:
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
        "gumbel_softmax_temperature": 1.0,
        "dirichlet_alpha": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "think_end_str": args.think_end_str,
        "early_stopping_entropy_threshold": args.early_stopping_entropy_threshold,
        "early_stopping_length_threshold": args.early_stopping_length_threshold,
    }


def _build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    message = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=False,
    )


def _load_aime24_samples(limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        raise ValueError("--n must be a positive integer.")

    dataset = load_dataset("math-ai/aime24")
    if isinstance(dataset, dict):
        if not dataset:
            raise ValueError("math-ai/aime24 returned no splits.")
        first_split = next(iter(dataset))
        samples = list(dataset[first_split])
    else:
        samples = list(dataset)

    return samples[: min(limit, len(samples))]


def _find_subsequence_start(haystack: List[int], needle: List[int]) -> Optional[int]:
    if not needle or len(needle) > len(haystack):
        return None
    end = len(haystack) - len(needle) + 1
    for start in range(end):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


def _normalize_output_ids(raw_output_ids: Any) -> Optional[List[int]]:
    if raw_output_ids is None:
        return None
    if not isinstance(raw_output_ids, list):
        return None
    if raw_output_ids and isinstance(raw_output_ids[0], list):
        raw_output_ids = raw_output_ids[0]
    if not all(isinstance(x, int) for x in raw_output_ids):
        return None
    return [int(x) for x in raw_output_ids]


def _find_think_end_step(
    topk_indices: List[List[int]],
    think_end_ids: List[int],
    output_ids: Optional[List[int]],
) -> Optional[int]:
    # Prefer searching generated token IDs so we do not depend on think-end being top-1.
    if output_ids:
        think_end_start = _find_subsequence_start(output_ids, think_end_ids)
        if think_end_start is not None:
            return think_end_start

    # Fallback: search top-k rows for the final think-end token.
    think_end_id = think_end_ids[-1]
    for step, row in enumerate(topk_indices):
        if row and any(int(idx) == think_end_id for idx in row):
            return step
    return None


def _serialize_finish_reason(meta_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    finish_reason = meta_info.get("finish_reason")
    if finish_reason is None:
        return None
    if isinstance(finish_reason, dict):
        return dict(finish_reason)
    return {"type": str(finish_reason)}


def _build_repetition_records(
    outputs: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    batch_elapsed_sec: float,
) -> List[Dict[str, Any]]:
    repetitions = []
    for repeat_idx, out in enumerate(outputs):
        meta_info = out["meta_info"]
        repetitions.append(
            {
                "repeat_idx": repeat_idx,
                "elapsed_sec": batch_elapsed_sec,
                "completion_tokens": int(meta_info.get("completion_tokens", 0)),
                "cached_tokens": int(meta_info.get("cached_tokens", 0)),
                "finish_reason": _serialize_finish_reason(meta_info),
                "text": _extract_text(out, tokenizer),
            }
        )
    return repetitions


def _capture_trace_until_think_end(
    llm: sgl.Engine,
    tokenizer: AutoTokenizer,
    prompt: str,
    prompt_ids: List[int],
    sampling_params: Dict[str, Any],
) -> Dict[str, Any]:
    warmup_params = copy.deepcopy(sampling_params)
    warmup_params["stop"] = sampling_params["think_end_str"]

    start = time.perf_counter()
    warmup_out = llm.generate(
        prompt=prompt,
        sampling_params=warmup_params,
        return_logprob=True,
    )
    elapsed = time.perf_counter() - start

    meta_info = warmup_out["meta_info"]
    topk_indices = meta_info.get("output_topk_idx_list", [])
    topk_probs = meta_info.get("output_topk_prob_list", [])
    if not topk_indices or not topk_probs:
        raise AssertionError(
            "Warmup generation did not return a soft-thinking trace. "
            "Ensure enable_soft_thinking=True and return_logprob=True."
        )

    think_end_ids = tokenizer.encode(
        sampling_params["think_end_str"], add_special_tokens=False
    )
    if not think_end_ids:
        raise AssertionError(
            f"Tokenizer could not encode think_end_str={sampling_params['think_end_str']}"
        )

    finish_reason = _serialize_finish_reason(meta_info)
    think_end_str = sampling_params["think_end_str"]
    if (
        finish_reason is None
        or finish_reason.get("type") != "stop"
        or finish_reason.get("matched") != think_end_str
    ):
        raise AssertionError(
            "Warmup consistency check failed: expected finish_reason to match "
            f"think_end_str={think_end_str!r}, got finish_reason={finish_reason}."
        )

    think_end_step: Optional[int] = None

    # Prefer server-side think_len/full_len metadata. It is computed from real
    # output_ids in scheduler and remains valid even if the last stop token is
    # not present in output_topk_idx_list.
    think_len_raw = meta_info.get("think_len")
    full_len_raw = meta_info.get("full_len")
    try:
        think_len = int(think_len_raw) if think_len_raw is not None else None
        full_len = int(full_len_raw) if full_len_raw is not None else None
    except (TypeError, ValueError):
        think_len = None
        full_len = None

    if think_len is None or full_len is None:
        raise AssertionError(
            "Warmup consistency check failed: missing think_len/full_len in meta_info. "
            f"think_len={think_len_raw}, full_len={full_len_raw}, finish_reason={finish_reason}"
        )
    if not (0 <= think_len < full_len):
        raise AssertionError(
            "Warmup consistency check failed: invalid think_len/full_len relation. "
            f"think_len={think_len}, full_len={full_len}, finish_reason={finish_reason}"
        )
    if think_len != full_len - 1:
        raise AssertionError(
            "Warmup consistency check failed: expected </think> to be the last generated token "
            f"under stop='</think>', got think_len={think_len}, full_len={full_len}."
        )

    topk_len = len(topk_indices)
    if topk_len not in (think_len, full_len):
        raise AssertionError(
            "Warmup consistency check failed: unexpected top-k trace length. "
            f"topk_len={topk_len}, think_len={think_len}, full_len={full_len}"
        )
    if len(topk_probs) != topk_len:
        raise AssertionError(
            "Warmup consistency check failed: top-k indices/probs length mismatch. "
            f"len(topk_indices)={topk_len}, len(topk_probs)={len(topk_probs)}"
        )

    if think_len <= len(topk_indices):
        think_end_step = think_len
    else:
        output_ids = _normalize_output_ids(warmup_out.get("output_ids"))
        think_end_step = _find_think_end_step(
            topk_indices=topk_indices,
            think_end_ids=think_end_ids,
            output_ids=output_ids,
        )

    if think_end_step is None:
        text_preview = _extract_text(warmup_out, tokenizer)[:200]
        output_ids = _normalize_output_ids(warmup_out.get("output_ids"))
        output_ids_tail = output_ids[-16:] if output_ids else None
        raise AssertionError(
            "Warmup run did not reach the think-end boundary. "
            f"finish_reason={finish_reason}, "
            f"think_end_ids={think_end_ids}, "
            f"think_len={think_len_raw}, "
            f"full_len={full_len_raw}, "
            f"output_ids_tail={output_ids_tail}, "
            f"preview={text_preview!r}"
        )

    replay_trace = {
        "topk_indices": copy.deepcopy(topk_indices[:think_end_step]),
        "topk_probs": copy.deepcopy(topk_probs[:think_end_step]),
    }

    return {
        "elapsed_sec": elapsed,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": int(meta_info.get("completion_tokens", len(topk_indices))),
        "cached_tokens": int(meta_info.get("cached_tokens", 0)),
        "finish_reason": _serialize_finish_reason(meta_info),
        "trace_steps_total": len(topk_indices),
        "replay_trace_len": len(replay_trace["topk_indices"]),
        "think_end_step": think_end_step,
        "text": _extract_text(warmup_out, tokenizer),
        "replay_trace": replay_trace,
    }


def _run_normal_softthinking(
    llm: sgl.Engine,
    tokenizer: AutoTokenizer,
    samples: List[Dict[str, Any]],
    sampling_params: Dict[str, Any],
    k: int,
) -> Dict[str, Any]:
    per_sample = []
    method_start = time.perf_counter()

    for sample_idx, sample in enumerate(samples):
        question = sample["problem"]
        prompt = _build_prompt(tokenizer, question)
        prompt_batch = [prompt] * k

        sample_start = time.perf_counter()
        batch_out = llm.generate(
            prompt=prompt_batch,
            sampling_params=copy.deepcopy(sampling_params),
            return_logprob=False,
        )
        sample_elapsed = time.perf_counter() - sample_start
        outputs = _ensure_output_list(batch_out)
        if len(outputs) != k:
            raise AssertionError(
                f"Expected {k} outputs for batched normal generation, got {len(outputs)}"
            )

        repetitions = _build_repetition_records(outputs, tokenizer, sample_elapsed)
        per_sample.append(
            {
                "sample_idx": sample_idx,
                "question": question,
                "ground_truth": sample.get("answer", sample.get("final_answer")),
                "warmup_elapsed_sec": 0.0,
                "generation_elapsed_sec": sample_elapsed,
                "total_elapsed_sec": sample_elapsed,
                "replay_trace_len": 0,
                "think_end_step": None,
                "batched_request_size": k,
                "repetitions": repetitions,
            }
        )
        print(
            f"[normal_soft_thinking] sample={sample_idx} total={sample_elapsed:.3f}s "
            f"effective_avg_per_repeat={sample_elapsed / k:.3f}s batch_size={k}"
        )

    total_elapsed = time.perf_counter() - method_start
    return _build_method_summary("normal_soft_thinking", per_sample, total_elapsed)


def _run_replay_method(
    llm: sgl.Engine,
    tokenizer: AutoTokenizer,
    samples: List[Dict[str, Any]],
    sampling_params: Dict[str, Any],
    k: int,
    method_name: str,
) -> Dict[str, Any]:
    per_sample = []
    method_start = time.perf_counter()

    for sample_idx, sample in enumerate(samples):
        question = sample["problem"]
        prompt = _build_prompt(tokenizer, question)
        prompt_ids = tokenizer.encode(prompt)

        warmup = _capture_trace_until_think_end(
            llm=llm,
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )

        replay_obj = GenerateReqInput(
            input_ids=[copy.deepcopy(prompt_ids) for _ in range(k)],
            sampling_params=copy.deepcopy(sampling_params),
            return_logprob=False,
            soft_thinking_trace=copy.deepcopy(warmup["replay_trace"]),
        )
        replay_start = time.perf_counter()
        replay_out = _generate_with_obj(llm, replay_obj)
        replay_elapsed = time.perf_counter() - replay_start
        outputs = _ensure_output_list(replay_out)
        if len(outputs) != k:
            raise AssertionError(
                f"Expected {k} outputs for batched replay generation, got {len(outputs)}"
            )

        repetitions = _build_repetition_records(outputs, tokenizer, replay_elapsed)
        total_elapsed = warmup["elapsed_sec"] + replay_elapsed
        per_sample.append(
            {
                "sample_idx": sample_idx,
                "question": question,
                "ground_truth": sample.get("answer", sample.get("final_answer")),
                "warmup_elapsed_sec": warmup["elapsed_sec"],
                "generation_elapsed_sec": replay_elapsed,
                "total_elapsed_sec": total_elapsed,
                "replay_trace_len": warmup["replay_trace_len"],
                "think_end_step": warmup["think_end_step"],
                "warmup_finish_reason": warmup["finish_reason"],
                "warmup_completion_tokens": warmup["completion_tokens"],
                "warmup_cached_tokens": warmup["cached_tokens"],
                "warmup_text": warmup["text"],
                "batched_request_size": k,
                "repetitions": repetitions,
            }
        )
        print(
            f"[{method_name}] sample={sample_idx} warmup={warmup['elapsed_sec']:.3f}s "
            f"replay_total={replay_elapsed:.3f}s total={total_elapsed:.3f}s "
            f"effective_avg_per_repeat={replay_elapsed / k:.3f}s batch_size={k}"
        )

    total_elapsed = time.perf_counter() - method_start
    return _build_method_summary(method_name, per_sample, total_elapsed)


def _build_method_summary(
    method_name: str,
    per_sample: List[Dict[str, Any]],
    wall_clock_elapsed_sec: float,
) -> Dict[str, Any]:
    total_repetitions = sum(len(sample["repetitions"]) for sample in per_sample)
    total_warmup = sum(float(sample.get("warmup_elapsed_sec", 0.0)) for sample in per_sample)
    total_generation = sum(
        float(sample.get("generation_elapsed_sec", 0.0)) for sample in per_sample
    )
    total_accounted = sum(
        float(sample.get("total_elapsed_sec", 0.0)) for sample in per_sample
    )
    total_completion_tokens = sum(
        int(rep.get("completion_tokens", 0))
        for sample in per_sample
        for rep in sample["repetitions"]
    )
    cached_token_values = [
        int(rep.get("cached_tokens", 0))
        for sample in per_sample
        for rep in sample["repetitions"]
    ]

    return {
        "method": method_name,
        "num_dataset_samples": len(per_sample),
        "num_repetitions": total_repetitions,
        "wall_clock_elapsed_sec": wall_clock_elapsed_sec,
        "accounted_elapsed_sec": total_accounted,
        "warmup_elapsed_sec": total_warmup,
        "generation_elapsed_sec": total_generation,
        "avg_elapsed_per_sample_sec": (
            total_accounted / len(per_sample) if per_sample else 0.0
        ),
        "avg_elapsed_per_repeat_sec": (
            total_generation / total_repetitions if total_repetitions else 0.0
        ),
        "total_completion_tokens": total_completion_tokens,
        "completion_tokens_per_sec": (
            total_completion_tokens / total_accounted if total_accounted > 0 else 0.0
        ),
        "avg_cached_tokens_per_repeat": (
            sum(cached_token_values) / len(cached_token_values)
            if cached_token_values
            else 0.0
        ),
        "samples": per_sample,
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    print()
    print(f"Method: {summary['method']}")
    print(f"  dataset_samples={summary['num_dataset_samples']}")
    print(f"  repetitions={summary['num_repetitions']}")
    print(f"  warmup_elapsed_sec={summary['warmup_elapsed_sec']:.3f}")
    print(f"  generation_elapsed_sec={summary['generation_elapsed_sec']:.3f}")
    print(f"  accounted_elapsed_sec={summary['accounted_elapsed_sec']:.3f}")
    print(f"  wall_clock_elapsed_sec={summary['wall_clock_elapsed_sec']:.3f}")
    print(f"  avg_elapsed_per_sample_sec={summary['avg_elapsed_per_sample_sec']:.3f}")
    print(f"  avg_elapsed_per_repeat_sec={summary['avg_elapsed_per_repeat_sec']:.3f}")
    print(f"  total_completion_tokens={summary['total_completion_tokens']}")
    print(f"  completion_tokens_per_sec={summary['completion_tokens_per_sec']:.3f}")
    print(
        f"  avg_cached_tokens_per_repeat={summary['avg_cached_tokens_per_repeat']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare normal soft-thinking against replay with and without cached thinking on AIME24."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--n", type=int, default=1, help="Use the first n AIME24 samples.")
    parser.add_argument("--k", type=int, default=4, help="Repeat each sample k times.")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--sampling-backend",
        type=str,
        choices=["pytorch", "flashinfer"],
        default="flashinfer",
    )
    parser.add_argument("--max-topk", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--after-thinking-temperature", type=float, default=0.6)
    parser.add_argument("--after-thinking-top-p", type=float, default=0.95)
    parser.add_argument("--after-thinking-top-k", type=int, default=30)
    parser.add_argument("--after-thinking-min-p", type=float, default=0.0)
    parser.add_argument("--think-end-str", type=str, default="</think>")
    parser.add_argument(
        "--early-stopping-entropy-threshold", type=float, default=0.0
    )
    parser.add_argument("--early-stopping-length-threshold", type=int, default=256)
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save the full benchmark results as JSON.",
    )
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be a positive integer.")

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    sampling_params = _build_sampling_params(args)
    samples = _load_aime24_samples(args.n)

    methods = [
        ("normal_soft_thinking", False),
        ("replay_without_cached_thinking", True),
        ("replay_with_cached_thinking", False),
    ]

    all_results = {
        "config": {
            "model_path": args.model_path,
            "tokenizer_path": tokenizer_path,
            "dataset": "math-ai/aime24",
            "n": args.n,
            "k": args.k,
            "sampling_params": sampling_params,
            "engine": {
                "tp_size": args.tp_size,
                "mem_fraction_static": args.mem_fraction_static,
                "random_seed": args.random_seed,
                "sampling_backend": args.sampling_backend,
                "max_topk": args.max_topk,
            },
        },
        "results": [],
    }

    for method_name, disable_think_prefix_cache in methods:
        print()
        print(
            f"[INFO] Starting {method_name} "
            f"(disable_think_prefix_cache={disable_think_prefix_cache})"
        )
        llm = sgl.Engine(
            **_build_engine_args(
                model_path=args.model_path,
                tp_size=args.tp_size,
                max_topk=args.max_topk,
                mem_fraction_static=args.mem_fraction_static,
                random_seed=args.random_seed,
                sampling_backend=args.sampling_backend,
                disable_think_prefix_cache=disable_think_prefix_cache,
            )
        )

        try:
            if method_name == "normal_soft_thinking":
                summary = _run_normal_softthinking(
                    llm=llm,
                    tokenizer=tokenizer,
                    samples=samples,
                    sampling_params=sampling_params,
                    k=args.k,
                )
            else:
                summary = _run_replay_method(
                    llm=llm,
                    tokenizer=tokenizer,
                    samples=samples,
                    sampling_params=sampling_params,
                    k=args.k,
                    method_name=method_name,
                )
        finally:
            llm.shutdown()

        all_results["results"].append(summary)
        _print_summary(summary)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print()
        print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    main()

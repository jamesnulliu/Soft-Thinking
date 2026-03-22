import argparse
import copy
from typing import Any, Dict, List, Tuple

import sglang as sgl
from transformers import AutoTokenizer

from sglang.srt.managers.io_struct import GenerateReqInput


def _generate_with_obj(llm: sgl.Engine, obj: GenerateReqInput) -> Dict[str, Any]:
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


def _extract_text(output: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    if "text" in output:
        return output["text"]
    if "output_ids" in output:
        return tokenizer.decode(output["output_ids"], skip_special_tokens=False)
    return ""


def _build_engine_args(
    model_path: str,
    max_topk: int,
    disable_overlap_schedule: bool,
    chunked_prefill_size: int,
    disable_think_prefix_cache: bool,
) -> Dict[str, Any]:
    return {
        "model_path": model_path,
        "tp_size": 1,
        "log_level": "info",
        "trust_remote_code": True,
        "random_seed": 42,
        "max_running_requests": None,
        "mem_fraction_static": 0.8,
        "disable_cuda_graph": False,
        "disable_overlap_schedule": disable_overlap_schedule,
        "chunked_prefill_size": chunked_prefill_size,
        "enable_soft_thinking": True,
        "add_noise_dirichlet": False,
        "add_noise_gumbel_softmax": False,
        "max_topk": max_topk,
        "disable_think_prefix_cache": disable_think_prefix_cache,
        "cuda_graph_max_bs": 8,
        "sampling_backend": "flashinfer",
    }


def _build_sampling_params(max_new_tokens: int) -> Dict[str, Any]:
    return {
        "n": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "after_thinking_temperature": 0.0,
        "after_thinking_top_p": 1.0,
        "after_thinking_top_k": -1,
        "after_thinking_min_p": 0.0,
        "gumbel_softmax_temperature": 1.0,
        "dirichlet_alpha": 1.0,
        "max_new_tokens": max_new_tokens,
        "think_end_str": "</think>",
        "early_stopping_entropy_threshold": 0.0,
        "early_stopping_length_threshold": 256,
    }


def _assert_raises_value_error(func, contains: str) -> None:
    try:
        func()
    except ValueError as exc:
        msg = str(exc)
        if contains not in msg:
            raise AssertionError(
                f"Expected ValueError containing '{contains}', got: {msg}"
            )
        return
    raise AssertionError("Expected ValueError but no exception was raised.")


def _build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    message = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=False,
    )


def _get_resp_len_from_meta(meta_info: Dict[str, Any]) -> Tuple[int, str]:
    if "resp_len" in meta_info:
        return int(meta_info["resp_len"]), "resp_len"
    if "full_len" in meta_info:
        return int(meta_info["full_len"]), "full_len"
    raise AssertionError(
        "Missing response length metadata. Expected `resp_len` or `full_len` in meta_info."
    )


def _assert_len_metadata(
    meta_info: Dict[str, Any],
    topk_indices: List[List[int]],
    think_end_id: int,
    label: str,
) -> Tuple[int, int, str]:
    if "think_len" not in meta_info:
        raise AssertionError(f"[{label}] Missing `think_len` in meta_info.")

    think_len = int(meta_info["think_len"])
    resp_len, resp_len_key = _get_resp_len_from_meta(meta_info)

    if think_len < 0 or think_len > resp_len:
        raise AssertionError(
            f"[{label}] Invalid think_len={think_len} for response length {resp_len}."
        )

    completion_tokens = meta_info.get("completion_tokens")
    if completion_tokens is not None and int(completion_tokens) != resp_len:
        raise AssertionError(
            f"[{label}] {resp_len_key} mismatch with completion_tokens. "
            f"{resp_len_key}={resp_len}, completion_tokens={completion_tokens}"
        )

    think_end_step = next(
        (i for i, row in enumerate(topk_indices) if row and int(row[0]) == think_end_id),
        None,
    )
    expected_think_len = resp_len if think_end_step is None else think_end_step
    if think_len != expected_think_len:
        raise AssertionError(
            f"[{label}] think_len mismatch. expected={expected_think_len}, got={think_len}"
        )

    return think_len, resp_len, resp_len_key


def run_replay_happy_path(
    llm: sgl.Engine,
    tokenizer: AutoTokenizer,
    sampling_params: Dict[str, Any],
    question: str,
    mode_label: str,
    expect_prefix_cache: bool,
    run_validation: bool,
) -> None:
    prompt = _build_prompt(tokenizer, question)
    prompt_ids = tokenizer.encode(prompt)

    baseline_out = llm.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        return_logprob=True,
    )
    baseline_meta = baseline_out["meta_info"]

    baseline_text = _extract_text(baseline_out, tokenizer)
    print(f"[{mode_label}] Baseline output:\n{baseline_text}\n")

    full_trace = {
        "topk_indices": baseline_meta.get("output_topk_idx_list", []),
        "topk_probs": baseline_meta.get("output_topk_prob_list", []),
    }

    if not full_trace["topk_indices"] or not full_trace["topk_probs"]:
        raise AssertionError(
            "Captured soft-thinking trace is empty. Ensure enable_soft_thinking=True and return_logprob=True."
        )

    think_end_str = sampling_params["think_end_str"]
    think_end_ids = tokenizer.encode(think_end_str, add_special_tokens=False)
    if not think_end_ids:
        raise AssertionError(f"Tokenizer could not encode think_end_str={think_end_str}")
    think_end_id = think_end_ids[-1]

    compare_steps = 2

    baseline_think_end_step = None
    for step, row in enumerate(full_trace["topk_indices"]):
        if row and int(row[0]) == think_end_id:
            baseline_think_end_step = step
            break

    replay_boundary_reason = "at_</think>"
    if baseline_think_end_step is None:
        if len(full_trace["topk_indices"]) <= compare_steps:
            raise AssertionError(
                "Baseline trace is too short for replay boundary comparison when </think> "
                f"is absent. trace_len={len(full_trace['topk_indices'])}, need>{compare_steps}"
            )
        baseline_think_end_step = len(full_trace["topk_indices"]) - compare_steps
        replay_boundary_reason = "tail_fallback_no_</think>"

    if baseline_think_end_step + compare_steps > len(full_trace["topk_indices"]):
        raise AssertionError(
            "Baseline trace does not contain enough tokens from replay boundary onward "
            f"to compare {compare_steps} steps."
        )

    baseline_think_len, baseline_resp_len, baseline_resp_len_key = _assert_len_metadata(
        baseline_meta,
        full_trace["topk_indices"],
        think_end_id,
        f"{mode_label}/baseline",
    )

    replay_trace = {
        "topk_indices": copy.deepcopy(
            full_trace["topk_indices"][:baseline_think_end_step]
        ),
        "topk_probs": copy.deepcopy(full_trace["topk_probs"][:baseline_think_end_step]),
    }

    if baseline_meta.get("prompt_tokens") != len(prompt_ids):
        raise AssertionError(
            "Baseline prompt_tokens mismatch. "
            "expected={}, got={}".format(len(prompt_ids), baseline_meta.get("prompt_tokens"))
        )

    replay_obj = GenerateReqInput(
        input_ids=prompt_ids,
        sampling_params=copy.deepcopy(sampling_params),
        return_logprob=True,
        soft_thinking_trace=replay_trace,
    )
    replay_out = _generate_with_obj(llm, replay_obj)
    replay_meta = replay_out["meta_info"]

    expected_prompt_tokens = len(prompt_ids) + len(replay_trace["topk_indices"])
    if replay_meta.get("prompt_tokens") != expected_prompt_tokens:
        raise AssertionError(
            "Replay prompt_tokens mismatch. "
            "expected={}, got={}".format(expected_prompt_tokens, replay_meta.get("prompt_tokens"))
        )

    replay_topk_indices = replay_meta.get("output_topk_idx_list", [])
    replay_topk_probs = replay_meta.get("output_topk_prob_list", [])
    if len(replay_topk_indices) < compare_steps or len(replay_topk_probs) < compare_steps:
        raise AssertionError(
            "Replay output does not contain enough steps for boundary comparison. "
            f"need={compare_steps}, got={len(replay_topk_indices)}"
        )

    replay_think_len, replay_resp_len, replay_resp_len_key = _assert_len_metadata(
        replay_meta,
        replay_topk_indices,
        think_end_id,
        f"{mode_label}/replay",
    )

    baseline_first2 = [
        int(full_trace["topk_indices"][baseline_think_end_step + i][0])
        for i in range(compare_steps)
    ]
    replay_first2 = [int(replay_topk_indices[i][0]) for i in range(compare_steps)]
    if replay_first2 != baseline_first2:
        raise AssertionError(
            f"[{mode_label}] replay first 2 tokens mismatch. "
            f"baseline={baseline_first2}, replay={replay_first2}"
        )

    prob_tol = 5e-3
    compare_k = min(10, llm.tokenizer_manager.max_topk)
    for step_offset in range(compare_steps):
        base_idx_row = full_trace["topk_indices"][baseline_think_end_step + step_offset]
        base_prob_row = full_trace["topk_probs"][baseline_think_end_step + step_offset]
        replay_idx_row = replay_topk_indices[step_offset]
        replay_prob_row = replay_topk_probs[step_offset]

        k = min(
            compare_k,
            len(base_idx_row),
            len(base_prob_row),
            len(replay_idx_row),
            len(replay_prob_row),
        )
        if k == 0:
            raise AssertionError("Encountered empty top-k row during boundary comparison.")

        if [int(x) for x in base_idx_row[:k]] != [int(x) for x in replay_idx_row[:k]]:
            raise AssertionError(
                "Top-k token indices mismatch at boundary step "
                f"{step_offset} from </think>."
            )

        max_delta = max(
            abs(float(bp) - float(rp))
            for bp, rp in zip(base_prob_row[:k], replay_prob_row[:k])
        )
        if max_delta > prob_tol:
            raise AssertionError(
                "Top-k probabilities mismatch at boundary step "
                f"{step_offset} from </think>. max_delta={max_delta:.6f}, tol={prob_tol:.6f}"
            )

    print(f"[PASS] Replay happy path ({mode_label})")
    print("  baseline_prompt_tokens={}".format(baseline_meta.get("prompt_tokens")))
    print("  replay_prompt_tokens={}".format(replay_meta.get("prompt_tokens")))
    print("  replay_trace_len={}".format(len(replay_trace["topk_indices"])))
    print(f"  baseline_think_end_step={baseline_think_end_step}")
    print(
        f"  baseline_think_len={baseline_think_len}, "
        f"baseline_{baseline_resp_len_key}={baseline_resp_len}"
    )
    print(
        f"  replay_think_len={replay_think_len}, "
        f"replay_{replay_resp_len_key}={replay_resp_len}"
    )
    print(f"  compared_steps={compare_steps}, compared_topk={compare_k}")
    print(f"  first2_tokens={replay_first2}")

    if expect_prefix_cache:
        run_replay_cache_matching_tests(llm, prompt_ids, sampling_params, replay_trace)
    else:
        run_replay_no_prefix_cache_check(llm, prompt_ids, sampling_params, replay_trace)

    if run_validation:
        run_validation_failures(llm, prompt_ids, sampling_params, replay_trace)


def run_replay_cache_matching_tests(
    llm: sgl.Engine,
    prompt_ids: List[int],
    sampling_params: Dict[str, Any],
    replay_trace: Dict[str, Any],
) -> None:
    if not replay_trace["topk_indices"]:
        raise AssertionError("Replay trace is empty; cannot validate cache matching.")

    cache_sampling_params = copy.deepcopy(sampling_params)
    cache_sampling_params["max_new_tokens"] = 1

    def run_with_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
        obj = GenerateReqInput(
            input_ids=prompt_ids,
            sampling_params=copy.deepcopy(cache_sampling_params),
            return_logprob=False,
            soft_thinking_trace=trace,
        )
        return _generate_with_obj(llm, obj)

    replay_a = run_with_trace(copy.deepcopy(replay_trace))
    replay_b = run_with_trace(copy.deepcopy(replay_trace))
    cached_a = int(replay_a["meta_info"].get("cached_tokens", 0))
    cached_b = int(replay_b["meta_info"].get("cached_tokens", 0))

    vocab_size = llm.tokenizer_manager.model_config.vocab_size
    idx_mismatch_trace = copy.deepcopy(replay_trace)
    original_idx = int(idx_mismatch_trace["topk_indices"][0][0])
    replacement_idx = (original_idx + 1) % vocab_size
    if replacement_idx == original_idx:
        replacement_idx = (original_idx + 2) % vocab_size
    idx_mismatch_trace["topk_indices"][0][0] = int(replacement_idx)
    replay_idx_mismatch = run_with_trace(idx_mismatch_trace)
    cached_idx_mismatch = int(replay_idx_mismatch["meta_info"].get("cached_tokens", 0))

    if cached_b <= cached_idx_mismatch:
        raise AssertionError(
            "Replay cache key mismatch on token ids was not reflected in cached_tokens. "
            f"matched={cached_b}, idx_mismatch={cached_idx_mismatch}"
        )

    prob_mismatch_trace = copy.deepcopy(replay_trace)
    row_with_multiple_probs = next(
        (i for i, row in enumerate(prob_mismatch_trace["topk_probs"]) if len(row) >= 2),
        None,
    )

    cached_prob_mismatch = None
    if row_with_multiple_probs is None:
        print("[WARN] Skipped probability-mismatch cache check: top-k rows have k=1 only.")
    else:
        orig_row = [float(x) for x in prob_mismatch_trace["topk_probs"][row_with_multiple_probs]]
        k = len(orig_row)
        new_row = [0.0] * k
        new_row[0] = 0.75
        new_row[1] = 0.25
        if all(abs(a - b) < 1e-6 for a, b in zip(new_row, orig_row)):
            new_row[0] = 0.625
            new_row[1] = 0.375
        prob_mismatch_trace["topk_probs"][row_with_multiple_probs] = new_row

        replay_prob_mismatch = run_with_trace(prob_mismatch_trace)
        cached_prob_mismatch = int(
            replay_prob_mismatch["meta_info"].get("cached_tokens", 0)
        )

        if cached_b <= cached_prob_mismatch:
            raise AssertionError(
                "Replay cache key mismatch on probabilities was not reflected in cached_tokens. "
                f"matched={cached_b}, prob_mismatch={cached_prob_mismatch}"
            )

    print("[PASS] Replay cache matching checks")
    print(
        "  cached_tokens: "
        f"first={cached_a}, matched={cached_b}, "
        f"idx_mismatch={cached_idx_mismatch}, prob_mismatch={cached_prob_mismatch}"
    )


def run_replay_no_prefix_cache_check(
    llm: sgl.Engine,
    prompt_ids: List[int],
    sampling_params: Dict[str, Any],
    replay_trace: Dict[str, Any],
) -> None:
    cache_sampling_params = copy.deepcopy(sampling_params)
    cache_sampling_params["max_new_tokens"] = 1

    def run_with_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
        obj = GenerateReqInput(
            input_ids=prompt_ids,
            sampling_params=copy.deepcopy(cache_sampling_params),
            return_logprob=False,
            soft_thinking_trace=trace,
        )
        return _generate_with_obj(llm, obj)

    replay_a = run_with_trace(copy.deepcopy(replay_trace))
    replay_b = run_with_trace(copy.deepcopy(replay_trace))
    cached_a = int(replay_a["meta_info"].get("cached_tokens", 0))
    cached_b = int(replay_b["meta_info"].get("cached_tokens", 0))

    if cached_a != 0 or cached_b != 0:
        raise AssertionError(
            "Expected cached_tokens to stay 0 when disable_think_prefix_cache=True for replay requests. "
            f"first={cached_a}, second={cached_b}"
        )

    print("[PASS] Replay no-prefix-cache checks")
    print(f"  cached_tokens: first={cached_a}, second={cached_b}")


def run_validation_failures(
    llm: sgl.Engine,
    prompt_ids: List[int],
    sampling_params: Dict[str, Any],
    valid_trace: Dict[str, Any],
) -> None:
    vocab_size = llm.tokenizer_manager.model_config.vocab_size
    max_topk = llm.tokenizer_manager.max_topk

    def run_with_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
        obj = GenerateReqInput(
            input_ids=prompt_ids,
            sampling_params=copy.deepcopy(sampling_params),
            return_logprob=False,
            soft_thinking_trace=trace,
        )
        return _generate_with_obj(llm, obj)

    bad_t_mismatch = {
        "topk_indices": [[1, 2], [3, 4]],
        "topk_probs": [[0.5, 0.5]],
    }
    _assert_raises_value_error(
        lambda: run_with_trace(bad_t_mismatch),
        "must have the same length",
    )

    bad_k_inconsistent = {
        "topk_indices": [[1, 2], [3]],
        "topk_probs": [[0.5, 0.5], [1.0]],
    }
    _assert_raises_value_error(
        lambda: run_with_trace(bad_k_inconsistent),
        "does not match the first row top-k",
    )

    big_k = max_topk + 1
    bad_k_exceed = {
        "topk_indices": [list(range(big_k))],
        "topk_probs": [[1.0 / big_k for _ in range(big_k)]],
    }
    _assert_raises_value_error(
        lambda: run_with_trace(bad_k_exceed),
        "exceeds max_topk",
    )

    bad_prob_sum = {
        "topk_indices": [[1, 2]],
        "topk_probs": [[0.9, 0.9]],
    }
    _assert_raises_value_error(
        lambda: run_with_trace(bad_prob_sum),
        "must sum to 1",
    )

    bad_oov = {
        "topk_indices": [[vocab_size]],
        "topk_probs": [[1.0]],
    }
    _assert_raises_value_error(
        lambda: run_with_trace(bad_oov),
        "out of vocab range",
    )

    _ = run_with_trace(valid_trace)
    print("[PASS] Validation failure cases")


def run_overlap_guard_test(
    model_path: str,
    tokenizer: AutoTokenizer,
    sampling_params: Dict[str, Any],
    question: str,
    max_topk: int,
) -> None:
    prompt = _build_prompt(tokenizer, question)
    prompt_ids = tokenizer.encode(prompt)

    overlap_engine = sgl.Engine(
        **_build_engine_args(
            model_path=model_path,
            max_topk=max_topk,
            disable_overlap_schedule=False,
            chunked_prefill_size=-1,
            disable_think_prefix_cache=False,
        )
    )

    try:
        baseline_out = overlap_engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            return_logprob=True,
        )
        trace = {
            "topk_indices": baseline_out["meta_info"].get("output_topk_idx_list", []),
            "topk_probs": baseline_out["meta_info"].get("output_topk_prob_list", []),
        }
        if not trace["topk_indices"]:
            raise AssertionError("Failed to capture trace for overlap guard test.")

        replay_obj = GenerateReqInput(
            input_ids=prompt_ids,
            sampling_params=copy.deepcopy(sampling_params),
            return_logprob=False,
            soft_thinking_trace=trace,
        )

        _assert_raises_value_error(
            lambda: _generate_with_obj(overlap_engine, replay_obj),
            "not supported with overlap or mixed-chunk scheduling in v1",
        )
        print("[PASS] Overlap/mixed-chunk mode guard")
    finally:
        overlap_engine.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone replay test for soft-thinking trace support"
    )
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--tokenizer-path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-topk", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=10000)
    parser.add_argument(
        "--question",
        type=str,
        default="If f(x)=x^2+3x+2, what is f(5)?",
    )
    parser.add_argument(
        "--test-overlap-guard",
        action="store_true",
        help="Also start an overlap-enabled engine and verify replay is rejected in v1.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
    )
    sampling_params = _build_sampling_params(args.max_new_tokens)

    modes = [
        ("prefix_cache_enabled", False, True),
        ("prefix_cache_disabled", True, False),
    ]

    for mode_label, disable_think_prefix_cache, expect_prefix_cache in modes:
        print(
            f"\n[INFO] Running mode={mode_label}, "
            f"disable_think_prefix_cache={disable_think_prefix_cache}"
        )
        llm = sgl.Engine(
            **_build_engine_args(
                model_path=args.model_path,
                max_topk=args.max_topk,
                disable_overlap_schedule=True,
                chunked_prefill_size=-1,
                disable_think_prefix_cache=disable_think_prefix_cache,
            )
        )

        try:
            run_replay_happy_path(
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                question=args.question,
                mode_label=mode_label,
                expect_prefix_cache=expect_prefix_cache,
                run_validation=expect_prefix_cache,
            )
        finally:
            llm.shutdown()

    if args.test_overlap_guard:
        run_overlap_guard_test(
            model_path=args.model_path,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            question=args.question,
            max_topk=args.max_topk,
        )

    print("[PASS] All replay-soft-thinking checks completed")


if __name__ == "__main__":
    main()

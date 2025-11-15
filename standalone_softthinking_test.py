import sglang as sgl
from transformers import AutoTokenizer
from datasets import load_dataset

slg_engine_args = {
    "model_path": "Qwen/Qwen3-4B",
    "tp_size": 1,
    "log_level": "info",
    "trust_remote_code": True,
    "random_seed": 42,
    "max_running_requests": None,
    "mem_fraction_static": 0.5,
    "disable_cuda_graph": False,
    "disable_overlap_schedule": True,
    "enable_soft_thinking": True,
    "add_noise_dirichlet": True,
    "add_noise_gumbel_softmax": True,
    "max_topk": 10,
    "cuda_graph_max_bs": 8,
    "sampling_backend": "flash_infer",
}

tokenizer_args = {
    "pretrained_model_name_or_path": "Qwen/Qwen3-4B",
    "trust_remote_code": True,
}

sampling_params = {
    "n": 1,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 30,
    "min_p": 0.0,
    "repetition_penalty": 1.2,
    "after_thinking_temperature": 0.6,
    "after_thinking_top_p": 0.95,
    "after_thinking_top_k": 30,
    "after_thinking_min_p": 0.0,
    "gumbel_softmax_temperature": 1.0,
    "dirichlet_alpha": 1.0,
    "max_new_tokens": 32768,
    "think_end_str": "</think>",
    "early_stopping_entropy_threshold": 0.0,
    "early_stopping_length_threshold": 256,
}

prompt_template = (
    "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
)

dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]

tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)
llm = sgl.Engine(**slg_engine_args)

input_prompts: list[str] = []

for sample in enumerate(dataset):
    problem = sample["problem"]
    message = [{"role": "user", "content": prompt_template.replace("{input}", problem)}]
    prompt = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=False,
    )
    input_prompts.append(prompt)

outputs = llm.generate(
    prompt=input_prompts,
    sampling_params=sampling_params,
)

decoded_text = [o["text"] for o in outputs]
entropies = [o["entropies"] for o in outputs]

print(f"Length of decoded_text: {len(decoded_text)}")
print(f"Length of entropies: {len(entropies)}")

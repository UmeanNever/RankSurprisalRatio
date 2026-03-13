import json
import time
from pathlib import Path
from typing import List, Tuple, Dict

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
import fire
from rsr_utils import (
    _scan_response_positions, 
    _patch_multimodal_messages, 
    save_inference_results, 
    load_inference_results,
    _iter_json_files, 
    _data_name_from,
    upsert_tsv,
)

ASSISTANT_MARKERS = {
    "qwen": ("<|im_start|>assistant\n", "<|im_end|>"),
    "llama3": ("<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>"),
}  # Must be consistent with the chat template used in the tokenizer, used for locating response tokens

# ----------------------------- Inference Engine ------------------------------

@torch.inference_mode()
def infer_dataset(
    model,
    tokenizer,
    json_path: Path,
    batch_size: int,
    max_model_len: int = None,
    input_format: str = "messages",
    rank_clip_r: int = 100,
    chat_template: str = "qwen",
    media_root: str = "",
) -> List[Dict]:
    """
    Perform inference and return sample-level result list:
    [
      {
        "sample_id": int or str,
        "resp_token_positions": [...],
        "probs": [...],
        "nlls": [...],
        "ranks": [...],
        "metadata": {...}
      }, ...
    ]
    """
    items = []

    if json_path.suffix.lower() == ".jsonl":
        data = []
        with json_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        data = json.loads(json_path.read_text(encoding="utf-8"))

    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    pad_token_id = text_tokenizer.pad_token_id if text_tokenizer.pad_token_id is not None else text_tokenizer.eos_token_id
    assistant_header, assistant_end = ASSISTANT_MARKERS[chat_template]
    assistant_header_ids = text_tokenizer(assistant_header, add_special_tokens=False)["input_ids"]
    assistant_end_ids = text_tokenizer(assistant_end, add_special_tokens=False)["input_ids"]
    print(f"assistant_header_ids: {assistant_header_ids}, assistant_end_ids: {assistant_end_ids}")

    if input_format in ("messages", "multimodal_messages"):
        for i, sample in enumerate(data):
            if input_format == "multimodal_messages":
                sample = _patch_multimodal_messages(sample, media_root)

            # print(f"\nProcessing sample {i} (id: {sample.get('id', sample.get('_id', 'N/A'))})...")
            if "messages" not in sample:
                continue

            input_ids, response_positions, mm_extras = _process_messages(
                tokenizer,
                sample["messages"],
                max_model_len=max_model_len,
                assistant_header_ids=assistant_header_ids,
                assistant_end_ids=assistant_end_ids,
            )
            if not input_ids or not response_positions:
                continue

            sample_id = sample.get("id", sample.get("_id", i))
            item = {
                "id": sample_id,
                "input_ids": input_ids,
                "response_positions": response_positions,
                "metadata": {},
            }
            item.update(mm_extras)
            items.append(item)

    elif input_format == "other":
        # Placeholder for custom format implementation
        # Expected output: items list with same structure as above
        raise NotImplementedError(
            "Custom input format not implemented. "
            "Please modify this section to parse your custom format and populate 'items' list."
        )
    else:
        raise ValueError(f"Unknown input_format: {input_format}")

    if not items:
        print(f"[SKIP] {json_path}: no valid samples")
        return []

    inferred_samples: List[Dict] = []

    for st in tqdm(range(0, len(items), batch_size), desc="Inference", file=sys.stdout):
        batch = items[st:st + batch_size]
        input_ids_list = [it["input_ids"] for it in batch]
        max_len = max(len(ids) for ids in input_ids_list)

        input_ids_batch, attention_mask_batch = [], []
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            input_ids_batch.append(ids + [pad_token_id] * pad_len)
            attention_mask_batch.append([1] * len(ids) + [0] * pad_len)

        input_ids_tensor = torch.tensor(input_ids_batch, dtype=torch.long, device=model.device)
        attention_mask_tensor = torch.tensor(attention_mask_batch, dtype=torch.long, device=model.device)

        mm_kwargs = {}  # Additional multimodal inputs
        for k in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
            vals = [it.get(k) for it in batch]
            if any(v is None for v in vals):
                if not all(v is None for v in vals):
                    raise ValueError(f"Mixed presence of {k} in one batch, which may not be supported by the model. You can revise the code here to deal with mixed batch.")
                continue
            if all(torch.is_tensor(v) for v in vals):
                vals = [v.to(model.device) for v in vals]
                try:
                    mm_kwargs[k] = torch.cat(vals, dim=0)
                except Exception:
                    mm_kwargs[k] = vals
            else:
                mm_kwargs[k] = vals

        outputs = model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            use_cache=False,
            **mm_kwargs,
        )
        logits = outputs.logits  # [B, T, V]

        for bi, it in enumerate(batch):
            seq_len = len(it["input_ids"])
            response_positions = sorted(set(p for p in it["response_positions"] if 0 < p < seq_len))
            logit_pos = [p - 1 for p in response_positions]
            target_ids = [it["input_ids"][p] for p in response_positions]
            if not logit_pos:
                continue

            valid_logits = logits[bi, logit_pos]  # [T, V]
            targets = torch.tensor(target_ids, device=valid_logits.device, dtype=torch.long)

            # ---------- NLL, prob (exact, vectorized) ----------
            vl32 = valid_logits.float()
            logZ = torch.logsumexp(vl32, dim=-1)  # [T]
            tgt = vl32.gather(1, targets[:, None]).squeeze(1)  # [T]
            nll = logZ - tgt
            prob = torch.exp(-nll)

            # ---------- rank clip @ r (vectorized) ----------
            topv = torch.topk(valid_logits, k=min(rank_clip_r, valid_logits.shape[-1]), dim=-1).values  # [T, r]
            rank = 1 + (topv.float() > tgt[:, None]).sum(dim=-1)  # [T]
            rank = torch.clamp(rank, max=rank_clip_r)

            nll_cpu = nll.detach().cpu().tolist()
            prob_cpu = prob.detach().cpu().tolist()
            rank_cpu = rank.detach().cpu().tolist()

            inferred_samples.append({
                "sample_id": it["id"],
                "resp_token_positions": [int(p) for p in response_positions],
                "probs": [float(x) for x in prob_cpu],
                "nlls": [float(x) for x in nll_cpu],
                "ranks": [int(x) for x in rank_cpu],
                "metadata": it["metadata"],
            })

        sys.stdout.flush()

    return inferred_samples

# --------------------------- Metrics Computation ---------------------------

def compute_sample_metrics(inferred_samples: List[Dict], rank_clip_r: int = 100) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Input: list of sample-level inference results (see infer_dataset output).
    Returns:
      - aggregated metrics
      - per-sample metrics with metadata
    """
    if not inferred_samples:
        return {}, []

    eps = 1e-12
    sample_metrics = []
    for s in inferred_samples:
        ranks = s.get("ranks", [])
        nlls = s.get("nlls", [])
        if not ranks or not nlls:
            print(f"[WARN] Sample {s['sample_id']} missing ranks or nlls, skipping...")
            continue

        assert len(ranks) == len(nlls), f"Sample {s['sample_id']} has mismatched ranks and nlls lengths"
        token_length = len(ranks)

        ranks_clip = [min(r, rank_clip_r) for r in ranks]

        avg_rank = sum(ranks_clip) / token_length
        avg_nll = sum(nlls) / token_length
        sum_rank = sum(ranks_clip)
        sum_nll = sum(nlls)
        ratio = sum_rank / max(sum_nll, eps)

        sample_metrics.append({
            "sample_id": s["sample_id"],
            "resp_token_length": token_length,
            "avg_rank_clip": avg_rank,
            "avg_surprisal": avg_nll,
            "rank_surprisal_ratio": ratio,
            **s.get("metadata", {})
        })

    if not sample_metrics:
        return {}, []

    n = len(sample_metrics)
    sum_avg_rank = sum(x["avg_rank_clip"] for x in sample_metrics)
    sum_avg_surprisal = sum(x["avg_surprisal"] for x in sample_metrics)
    sum_resp_token_length = sum(x["resp_token_length"] for x in sample_metrics)
    dataset_agg = {
        "avg_resp_token_length": sum_resp_token_length / n,
        "avg_rank_clip": sum_avg_rank / n,
        "avg_surprisal": sum_avg_surprisal / n,
        "rank_surprisal_ratio": sum_avg_rank / max(sum_avg_surprisal, eps),
    }  # Refer to our paper for detailed explanations of the dataset-level RSR computation (Appendix A.8)
    return dataset_agg, sample_metrics

# --------------------------- Preprocessing Messages ---------------------------

def _process_messages(
    tokenizer,
    messages,
    max_model_len: int,
    assistant_header_ids: List[int],
    assistant_end_ids: List[int],
) -> Tuple[List[int], List[int], Dict]:
    """Unified chat-template encode, truncate, and assistant span scan."""
    kwargs = dict(
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )
    if max_model_len:
        kwargs.update(dict(truncation=True, max_length=max_model_len))

    enc_messages = tokenizer.apply_chat_template(messages, **kwargs)

    input_ids = enc_messages.get("input_ids", [])
    if torch.is_tensor(input_ids):
        input_ids = input_ids[0].tolist() if input_ids.ndim > 1 else input_ids.tolist()
    elif isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if not input_ids:
        return [], [], {}

    # Get response token positions based on assistant header/end markers
    response_positions = _scan_response_positions(input_ids, assistant_header_ids, assistant_end_ids)
    
    extras = {}
    for k in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if k in enc_messages:
            extras[k] = enc_messages[k]
    return input_ids, response_positions, extras

# ---------------------------------- CLI -------------------------------------

def run(
    model_path: str,
    model_name: str,
    data_root: str,
    data_path: str,
    output_root: str,
    batch_size: int = 8,
    dtype: str = "bfloat16",
    tag: str = "",
    run_mode: str = "together",
    filter_file_suffix: str = "",
    input_format: str = "messages",
    rank_clip_r: int = 100,
    use_flash_attn: bool = False,
    max_model_len: int = None,
    chat_template: str = "qwen",
    infer_save_format: str = "parquet",
    media_root: str = "",
):
    """
    Main entry point for RSR calculation.
    
    Args:
    - model_path: HF model name or local path
    - model_name: Model identifier for output files
    - data_root: Root directory for input data
    - data_path: JSON file or directory
    - output_root: Root directory for all outputs, outputs will be organized by model_name and relative data path
    - batch_size: Batch size for inference
    - dtype: Model precision (float16, bfloat16, float32, auto)
    - tag: Custom tag for distinguishing runs
    - run_mode: "infer_only", "metrics_only", or "together" (default)
    - input_format: "messages"  or "multimodal_messages" or "other" (custom, needs implementation)
    - filter_file_suffix: Only process data file names (w/o extension) ending with this suffix, e.g. "_gen3" to include only files matching "*_gen3.*"
    - max_model_len: Maximum sequence length, will truncate inputs if specified
    - infer_save_format: "" or "jsonl" or "parquet" for intermediate inference results, set to "" to disable saving
    - media_root: Root directory for media files referenced in the multimodal data, used for resolving relative paths
    """
    valid_modes = ["infer_only", "metrics_only", "together"]
    if run_mode not in valid_modes:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be one of {valid_modes}")
    if input_format not in ("messages", "multimodal_messages", "other"):
        raise ValueError(f"Invalid input_format: {input_format}.")
    if chat_template not in ASSISTANT_MARKERS:
        raise ValueError(f"Invalid chat_template: {chat_template}. Must be one of {list(ASSISTANT_MARKERS.keys())}")
    if infer_save_format not in ("", "jsonl", "parquet"):
        raise ValueError(f"Invalid infer_save_format: {infer_save_format}. Must be empty, 'jsonl', or 'parquet'.")
    if run_mode != "together" and not infer_save_format:
        raise ValueError("infer_save_format can only be empty when run_mode is 'together'.") 
    if dtype not in ("float16", "bfloat16", "float32", "auto"):
        raise ValueError(f"Invalid dtype: {dtype}. Must be 'float16', 'bfloat16', 'float32', or 'auto'.")

    model_name_with_tag = f"{model_name}_{tag}" if tag else model_name
    data_root = Path(data_root).resolve()
    data_path = Path(data_path).resolve()
    output_root = Path(output_root).resolve()
    files = _iter_json_files(data_path, input_format)
    if not files:
        print("[Error] No input files found")
        return

    # Setup output directories
    infer_dir = output_root / "_inferred_data" / model_name_with_tag
    sample_metrics_dir = output_root / "_sample_metrics" / model_name_with_tag
    
    # TSV path
    tsv_path = output_root / f"rsr_{model_name_with_tag}.tsv"
    infer_ext = infer_save_format if infer_save_format else None
    
    # Reuse model and tokenizer across files to save time
    model = None
    tokenizer = None
    
    print(f"# Model: {model_name_with_tag} ({model_path})")
    print(f"# Input format: {input_format}, Infer format: {infer_save_format}")
    print(f"# Outputs:")
    print(f"  - Inference: {infer_dir}")
    print(f"  - Sample metrics: {sample_metrics_dir}")
    print(f"  - TSV: {tsv_path}")
    print()

    for fp in files:
        try:
            rel_stem, data_name = _data_name_from(fp, data_root)
        except ValueError:
            data_name = fp.stem
            rel_stem = Path(fp.stem)

        infer_path = infer_dir / rel_stem.parent / f"{rel_stem.name}.{infer_ext}" if infer_ext else None
        if filter_file_suffix and not data_name.endswith(filter_file_suffix):
            print(f"[SKIP] {data_name}: does not match filter suffix '{filter_file_suffix}'")
            continue

        inferred_samples = None
        infer_seconds = None
        should_run_inference = (run_mode in ["infer_only", "together"]) and (infer_path is None or not infer_path.exists())
        if should_run_inference:
            print(f"## Running inference: {data_name}")
            t0 = time.perf_counter()
            if model is None:
                if input_format == "multimodal_messages":
                    tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

                if not getattr(tokenizer, "chat_template", None):
                    if chat_template == "llama3":
                        # A patch for llama3 base models that are missing the chat template in their tokenizer config
                        print("[Warning] chat_template missing in tokenizer; applying built-in LLaMA3 template.")
                        tokenizer.chat_template = (
                            "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"
                        )
                    else:
                        raise ValueError("Tokenizer/Processor has no chat_template. Check the tokenizer config of the model or add the appropriate template manually.")

                dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                    "auto": "auto"
                }
                torch_dtype = dtype_map[dtype]

                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2" if use_flash_attn else None,
                )
                model.eval()

            inferred_samples = infer_dataset(
                model,
                tokenizer,
                fp,
                batch_size,
                max_model_len=max_model_len,
                input_format=input_format,
                rank_clip_r=rank_clip_r,
                chat_template=chat_template,
                media_root=media_root,
            )
            infer_seconds = time.perf_counter() - t0
            if not inferred_samples:
                continue
            if infer_save_format:
                save_inference_results(inferred_samples, infer_path, fmt=infer_save_format)
                print(f"[OK] Saved inference results: {infer_path}")

        if run_mode == "infer_only":
            print(f"[SKIP] Skip metrics for infer_only mode.")
            continue

        if inferred_samples is None:
            print(f"## Inference is skipped. Loading cached inference: {data_name}")
            if not infer_path.exists():
                print(f"[ERROR] {data_name}: No cached inference results at {infer_path}. Run inference first or check the path.")
                continue
            inferred_samples = load_inference_results(infer_path, fmt=infer_save_format)

        print(f"## Computing metrics: {data_name}")
        t1 = time.perf_counter()
        ds_metrics, sample_metrics = compute_sample_metrics(inferred_samples, rank_clip_r=rank_clip_r)
        metrics_seconds = time.perf_counter() - t1

        sample_metrics_jsonl = sample_metrics_dir / rel_stem.parent / f"{rel_stem.name}.jsonl"
        sample_metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with sample_metrics_jsonl.open("w", encoding="utf-8") as f:
            for s in sample_metrics:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[OK] Saved sample metrics: {sample_metrics_jsonl}")

        row = {"data_name": data_name, **ds_metrics, "infer_seconds": infer_seconds, "metrics_seconds": metrics_seconds}
        upsert_tsv(tsv_path, [row])
        print(f"[OK] Updated TSV: {tsv_path}")
        print()

    print(f"\n[OK] Processing complete. Results in: {output_root}")


if __name__ == "__main__":
    fire.Fire(run)

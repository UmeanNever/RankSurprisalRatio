import json
from pathlib import Path
from typing import List, Tuple, Dict

import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import fire

# ----------------------------- Inference Engine ------------------------------

@torch.inference_mode()
def infer_dataset(
    model, tokenizer, json_path: Path, batch_size: int, max_model_len: int = None, input_format: str = "messages", rank_clip_r: int = 100
) -> List[Dict]:
    """
    Perform inference and return sample-level list:
    [
      {
        "sample_idx": int,
        "tokens": [{"position": int, "prob": float, "nll": float, "rank": int}, ...],
        "metadata": {...}
      }, ...
    ]
    """
    items = []
    if input_format == "messages":
        # Standard format: JSON array of objects with "messages" field
        # Example:
        # [
        #   {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]},
        #   ...
        # ]
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for i, sample in enumerate(data):
            prompt_ids, resp_ids = _apply_prompt_and_response(tokenizer, sample["messages"])
            if not (prompt_ids and resp_ids):
                continue
            full_ids = prompt_ids + resp_ids
            if max_model_len and len(full_ids) >= max_model_len:
                if len(prompt_ids) < max_model_len:
                    max_resp_len = max_model_len - 1 - len(prompt_ids)
                    resp_ids = resp_ids[:max_resp_len]
                    full_ids = prompt_ids + resp_ids
                else:
                    prompt_ids = prompt_ids[:max_model_len-1]
                    resp_ids = []
                    full_ids = prompt_ids
            if not resp_ids:
                continue
            items.append({"idx": i, "prompt": prompt_ids, "resp": resp_ids, "full_ids": full_ids, "metadata": {}})
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

    all_samples: List[Dict] = []

    for st in tqdm(range(0, len(items), batch_size), desc="Inference", file=sys.stdout):
        batch = items[st:st + batch_size]
        input_ids_list = [it["full_ids"] for it in batch]
        max_len = max(len(ids) for ids in input_ids_list)

        input_ids_batch, attention_mask_batch = [], []
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            input_ids_batch.append(ids + [tokenizer.pad_token_id] * pad_len)
            attention_mask_batch.append([1] * len(ids) + [0] * pad_len)

        input_ids_tensor = torch.tensor(input_ids_batch, dtype=torch.long, device=model.device)
        attention_mask_tensor = torch.tensor(attention_mask_batch, dtype=torch.long, device=model.device)

        outputs = model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            use_cache=False,
        )
        logits = outputs.logits  # [B, T, V]

        for bi, it in enumerate(batch):
            pr_len = len(it["prompt"])
            seq_len = len(it["full_ids"])
            resp_start = pr_len - 1

            valid_logits = logits[bi, resp_start:seq_len-1]  # [T, V]
            targets = torch.tensor(
                it["full_ids"][resp_start+1:],
                device=valid_logits.device,
                dtype=torch.long,
            )

            if valid_logits.numel() == 0:
                continue

            # ---------- NLL / prob (exact, vectorized) ----------
            vl32 = valid_logits.float()
            logZ = torch.logsumexp(vl32, dim=-1)                 # [T]
            tgt = vl32.gather(1, targets[:, None]).squeeze(1)   # [T]
            nll = logZ - tgt                                  # [T]
            prob = torch.exp(-nll)                            # [T]

            # ---------- rank clip @ r (vectorized) ----------
            topv = torch.topk(valid_logits, k=rank_clip_r, dim=-1).values  # [T, r]
            rank = 1 + (topv.float() > tgt[:, None]).sum(dim=-1)           # [T]
            rank = torch.clamp(rank, max=rank_clip_r)                      # [T]

            # ---------- pack results ----------
            nll_cpu = nll.detach().cpu()
            prob_cpu = prob.detach().cpu()
            rank_cpu = rank.detach().cpu()

            sample_tokens = []
            for j in range(nll_cpu.numel()):
                sample_tokens.append({
                    "position": resp_start + j,
                    "prob": float(prob_cpu[j].item()),
                    "nll": float(nll_cpu[j].item()),
                    "rank": int(rank_cpu[j].item()),
                })

            all_samples.append({
                "sample_idx": it["idx"],
                "tokens": sample_tokens,
                "metadata": it["metadata"],
            })

        sys.stdout.flush()

    return all_samples


def save_inference_results(samples: List[Dict], output_path: Path):
    """Save sample-level results (one JSON per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def load_inference_results(input_path: Path) -> List[Dict]:
    """Load sample-level results (JSONL)."""
    out = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

# --------------------------- Metrics Computation ---------------------------

def compute_sample_metrics(samples: List[Dict], rank_clip_r: int = 100) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Input: sample-level list (see infer_dataset output).
    Returns:
      - aggregated metrics
      - per-sample metrics with metadata
    """
    if not samples:
        return {}, []

    sample_results = []
    for s in samples:
        tokens = s.get("tokens", [])
        if not tokens:
            continue
        ranks_clip = [min(t["rank"], rank_clip_r) for t in tokens]
        nlls = [t["nll"] for t in tokens]
        if not ranks_clip or not nlls:
            continue

        avg_rank = sum(ranks_clip) / len(ranks_clip)
        avg_nll = sum(nlls) / len(nlls)
        sum_rank = sum(ranks_clip)
        sum_nll = sum(nlls)
        ratio = sum_rank / sum_nll if sum_nll > 0 else 0.0

        sample_results.append({
            "sample_idx": s["sample_idx"],
            "avg_rank_clip": avg_rank,
            "avg_surprisal": avg_nll,
            "rank_surprisal_ratio": ratio,
            **s.get("metadata", {})
        })

    if not sample_results:
        return {}, []

    n = len(sample_results)
    sum_avg_rank = sum(x["avg_rank_clip"] for x in sample_results)
    sum_avg_surprisal = sum(x["avg_surprisal"] for x in sample_results)
    agg = {
        "avg_rank_clip": sum_avg_rank / n,
        "avg_surprisal": sum_avg_surprisal / n,
        "rank_surprisal_ratio": (sum_avg_rank / sum_avg_surprisal) if sum_avg_surprisal > 0 else 0.0,
    }  # Refer to our paper for detailed explanations of the dataset-level RSR computation (Appendix A.8)
    return agg, sample_results

# --------------------------- Utilities ---------------------------

def _apply_prompt_and_response(tokenizer, messages) -> Tuple[List[int], List[int]]:
    """Extract prompt and response token IDs."""
    assistant = ""
    msgs_wo_assistant = []
    for m in messages:
        if m["role"] == "assistant":
            assistant = m["content"]
        else:
            msgs_wo_assistant.append(m)
    if not assistant:
        return [], []

    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            msgs_wo_assistant, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    else:
        raise NotImplementedError(
            "Tokenizer has no chat_template attribute. "
            "Please implement custom prompt formatting for your model."
        )

    resp_ids = tokenizer(assistant, add_special_tokens=False)["input_ids"]
    return prompt_ids, resp_ids


def _iter_json_files(data: Path, input_format: str) -> List[Path]:
    """Iterate dataset files by format."""
    if data.is_file():
        return [data]
    return sorted(data.rglob("*.json"))


def _data_name_from(path: Path, data_root: Path) -> str:
    """Generate data name from path."""
    rel = path.relative_to(data_root).with_suffix("")
    return "_".join(rel.parts)


def upsert_tsv(tsv_path: Path, rows: List[Dict[str, object]]):
    """Update or insert rows in TSV file based on data_name."""
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if tsv_path.exists():
        df_old = pd.read_csv(tsv_path, sep="\t")
    else:
        df_old = pd.DataFrame(columns=["data_name"])

    df_new = pd.DataFrame(rows)
    
    # Align columns
    for c in set(df_new.columns) - set(df_old.columns):
        df_old[c] = pd.NA
    for c in set(df_old.columns) - set(df_new.columns):
        df_new[c] = pd.NA

    # Merge
    df = pd.concat([df_new, df_old[~df_old["data_name"].isin(df_new["data_name"])]], ignore_index=True)
    
    # Ensure data_name is first column
    cols = ["data_name"] + [c for c in df.columns if c != "data_name"]
    df = df[cols]
    df.sort_values("data_name", inplace=True)
    df.to_csv(tsv_path, sep="\t", index=False, float_format="%.5f")


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
    max_model_len: int = None,
    run_mode: str = "together",
    filter_file_suffix: str = "",
    input_format: str = "messages",
    rank_clip_r: int = 100,
    use_flash_attn: bool = False,
):
    """
    Main entry point for RSR calculation.
    
    Args:
    - input_format: "messages" (standard) or "other" (custom, needs implementation)
    - model_path: HF model name or local path
    - model_name: Model identifier for output files
    - data_root: Root directory for input data
    - data_path: JSON file or directory
    - output_root: Root directory for all outputs
    - batch_size: Batch size for inference
    - dtype: Model precision (float16, bfloat16, float32, auto)
    - tag: Custom tag for distinguishing runs
    - max_model_len: Maximum sequence length
    - run_mode: "infer_only", "metrics_only", or "together" (default)
    - filter_file_suffix: Only process files ending with this suffix
    """
    valid_modes = ["infer_only", "metrics_only", "together"]
    if run_mode not in valid_modes:
        print(f"[Error] Invalid run_mode: {run_mode}. Must be one of {valid_modes}")
        return

    model_name_with_tag = f"{model_name}_{tag}" if tag else model_name
    data_root = Path(data_root).resolve()
    data_path = Path(data_path).resolve()
    output_root = Path(output_root).resolve()
    files = _iter_json_files(data_path, input_format)
    if not files:
        print("[Error] No input files found")
        return

    # Setup output directories
    infer_dir = output_root / "_infered_data" / model_name_with_tag
    sample_metrics_dir = output_root / "_sample_metrics" / model_name_with_tag
    
    infer_dir.mkdir(parents=True, exist_ok=True)
    sample_metrics_dir.mkdir(parents=True, exist_ok=True)

    # TSV path
    tsv_path = output_root / f"rsr_{model_name_with_tag}.tsv"

    # Initialize model if needed
    model = None
    tokenizer = None
    
    print(f"# Model: {model_name_with_tag} ({model_path})")
    print(f"# Input format: {input_format}")
    print(f"# Outputs:")
    print(f"  - Inference: {infer_dir}")
    print(f"  - Sample metrics: {sample_metrics_dir}")
    print(f"  - TSV: {tsv_path}")
    print()

    for fp in files:
        try:
            data_name = _data_name_from(fp, data_root)
        except ValueError:
            data_name = fp.stem

        infer_path = infer_dir / f"{data_name}.jsonl"
        if filter_file_suffix and not data_name.endswith(filter_file_suffix):
            print(f"[SKIP] {data_name}: does not match filter suffix '{filter_file_suffix}'")
            continue

        should_run_inference = (run_mode in ["infer_only", "together"]) and not infer_path.exists()
        if should_run_inference:
            print(f"## Running inference: {data_name}")
            if model is None:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                if (not getattr(tokenizer, "chat_template", None)) and "llama" in model_name.lower():
                    print(f"[Warning] Tokenizer has no chat_template attribute; attempting to set LLaMA3 template.")
                    tokenizer.chat_template = "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"
                
                dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                    "auto": "auto"
                }
                torch_dtype = dtype_map.get(dtype, torch.bfloat16)
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2" if use_flash_attn else None,
                )
                model.eval()
            
            samples = infer_dataset(model, tokenizer, fp, batch_size, max_model_len, input_format, rank_clip_r)
            if not samples:
                continue
            save_inference_results(samples, infer_path)
            print(f"[OK] Saved inference results: {infer_path}")

        if run_mode == "infer_only":
            print(f"[SKIP] Skip metrics for infer_only mode.")
            continue

        if not infer_path.exists():
            print(f"[ERROR] {data_name}: No cached inference results at {infer_path}")
            print(f"        Run with --run_mode=together or --run_mode=infer_only first.")
            continue

        print(f"## Loading cached inference: {data_name}")
        samples = load_inference_results(infer_path)

        print(f"## Computing metrics: {data_name}")
        metrics, sample_results = compute_sample_metrics(samples, rank_clip_r=rank_clip_r)

        sample_metrics_jsonl = sample_metrics_dir / f"{data_name}.jsonl"
        with sample_metrics_jsonl.open("w", encoding="utf-8") as f:
            for s in sample_results:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[OK] Saved sample metrics: {sample_metrics_jsonl}")

        row = {"data_name": data_name, **metrics}
        upsert_tsv(tsv_path, [row])
        print(f"[OK] Updated TSV: {tsv_path}")
        print()

    print(f"\n[OK] Processing complete. Results in: {output_root}")


if __name__ == "__main__":
    fire.Fire(run)

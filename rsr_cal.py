import math
import json
from pathlib import Path
from typing import List, Tuple, Dict

import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import fire
import torch.nn.functional as F


# ----------------------------- Inference Engine ------------------------------

@torch.inference_mode()
def infer_dataset(
    model, tokenizer, json_path: Path, batch_size: int, max_model_len: int = None, input_format: str = "messages"
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
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
        logits = outputs.logits

        for bi, it in enumerate(batch):
            pr_len = len(it["prompt"])
            seq_len = len(it["full_ids"])
            valid_logits = logits[bi, :seq_len-1]
            valid_targets = it["full_ids"][1:]
            log_probs = F.log_softmax(valid_logits, dim=-1)

            sample_tokens = []
            for i, token_id in enumerate(valid_targets):
                # only response tokens
                if i < (pr_len - 1):
                    continue
                token_log_prob = log_probs[i, token_id].item()
                nll = -token_log_prob
                prob = math.exp(token_log_prob)
                rank = (log_probs[i] > token_log_prob).sum().item() + 1
                sample_tokens.append({
                    "position": i,
                    "prob": prob,
                    "nll": nll,
                    "rank": rank
                })

            if sample_tokens:
                all_samples.append({
                    "sample_idx": it["idx"],
                    "tokens": sample_tokens,
                    "metadata": it["metadata"]
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

def compute_sample_metrics(samples: List[Dict]) -> Tuple[Dict[str, float], List[Dict]]:
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
        ranks_t100 = [min(t["rank"], 100) for t in tokens]
        nlls = [t["nll"] for t in tokens]
        if not ranks_t100 or not nlls:
            continue

        avg_rank = sum(ranks_t100) / len(ranks_t100)
        avg_nll = sum(nlls) / len(nlls)
        sum_rank = sum(ranks_t100)
        sum_nll = sum(nlls)
        ratio = sum_rank / sum_nll if sum_nll > 0 else 0.0

        sample_results.append({
            "sample_idx": s["sample_idx"],
            "avg_rank_clip100": avg_rank,
            "avg_surprisal": avg_nll,
            "rank_surprisal_ratio": ratio,
            **s.get("metadata", {})
        })

    if not sample_results:
        return {}, []

    n = len(sample_results)
    agg = {
        "avg_rank_clip100": sum(x["avg_rank_clip100"] for x in sample_results) / n,
        "avg_surprisal": sum(x["avg_surprisal"] for x in sample_results) / n,
        "rank_surprisal_ratio": sum(x["rank_surprisal_ratio"] for x in sample_results) / n,
    }
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
                    attn_implementation="flash_attention_2",
                )
                model.eval()
            
            samples = infer_dataset(model, tokenizer, fp, batch_size, max_model_len, input_format)
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
        metrics, sample_results = compute_sample_metrics(samples)

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

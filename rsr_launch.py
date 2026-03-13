#!/usr/bin/env python3

import subprocess
from pathlib import Path

run_mode = "together"  # Run mode: "infer_only" | "metrics_only" | "together"
input_format = "messages"  # Supporting "messages" and "multimodal_messages" format now

# Example model configuration. Will run each model separately. Please ensure that the chat template is also configured correctly.
models = [
    ("/path/to/your/model", "model-name"),
]
chat_template = "qwen" # Currently supports "qwen" or "llama3", used only for locating response tokens, won't override the model’s tokenizer configuration, please set it according to the model you are using.

# Path configuration - customize these for your setup
data_root = "/path/to/data/root"  # "/path/to/RSR_data"
output_root = "/path/to/output/root"  # will output metric results to output_root under subdirectories named by model and data
data_paths = [
    "/path/to/data/file1.json",
    "/path/to/data/directory2",
]  # List of data files or directories to compute RSR on, should be under data_root
filter_file_suffix = ""  # Optional filter for data file names (w/o extension), e.g. "_gen3" to include only files matching "*_gen3.*"
tag = ""  # Optional tag / run name for saving outputs, e.g. "dev0311"
infer_save_format = "parquet"  # "" or "jsonl" or "parquet" for intermediate inference results, set to "" to disable saving

# Inference configuration
batch_size = 4  # Adjust based on your GPU memory
dtype = "bfloat16"
max_model_len = 32768
rank_clip_r = 100  # Clip rank values at a threshold for numerical stability, as illustrated in the paper.
use_flash_attn = True  # Whether to use flash attention (boosts speed if installed)

def main(model_path, model_name, data_path):
    script_path = str(Path(__file__).parent / "rsr_cal.py")
    
    cmd_parts = [
        # "set -ex &&",
        # "conda activate rsr &&",
        # "export PYTHONUNBUFFERED=1 &&",
        "python", script_path,
        f"--model_path={model_path}",
        f"--model_name={model_name}",
        f"--data_root={data_root}", 
        f"--data_path={data_path}",
        f"--output_root={output_root}",
        f"--batch_size={batch_size}",
        f"--dtype={dtype}",
        f"--tag={tag}",
        f"--run_mode={run_mode}",
        f"--filter_file_suffix={filter_file_suffix}",
        f"--input_format={input_format}",
        f"--rank_clip_r={rank_clip_r}",
        f"--chat_template={chat_template}",
        f"--infer_save_format={infer_save_format}",
    ]
    if use_flash_attn:
        cmd_parts.append("--use_flash_attn")
    if max_model_len is not None:
        cmd_parts.append(f"--max_model_len={max_model_len}")
    
    print(f"Running: {model_name} on {Path(data_path).name}")
    print(f"Command: {' '.join(cmd_parts)}")
    
    subprocess.run(" ".join(cmd_parts), shell=True, executable="/bin/bash", check=True)
    print(f"Completed: {model_name} on {Path(data_path).name}\n")


if __name__ == "__main__":
    for model_path, model_name in models:
        for data_path in data_paths:
            main(model_path, model_name, data_path)

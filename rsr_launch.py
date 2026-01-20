#!/usr/bin/env python3

import subprocess
from pathlib import Path

run_mode = "together"  # Run mode: "infer_only" | "metrics_only" | "together"
input_format = "messages"  # Only supporting "messages" format now

# Example model configuration
models = [
    ("/path/to/your/model", "model-name"),
]

# Path configuration - customize these for your setup
data_root = "/path/to/data/root"
output_root = "/path/to/output/root"
data_paths = [
    "/path/to/data/file1.json",
    "/path/to/data/file2.json",
]  # List of data files to compute RSR on, should be under data_root
filter_file_suffix = ""  # Optional filter for data files
tag = ""  # Optional tag for the run

# Inference configuration
batch_size = 4
dtype = "bfloat16"
max_model_len = 32768

# GPU configuration
num_gpus = 8 if run_mode in ["infer_only", "together"] else 0


def main(model_path, model_name, data_path):
    script_path = str(Path(__file__).parent / "rsr_cal.py")
    
    cmd_parts = [
        "set -ex &&",
        # "conda activate torch27 &&",
        # "export PYTHONUNBUFFERED=1 &&",
        "python", script_path,
        f"--model_path={model_path}",
        f"--model_name={model_name}",
        f"--data_root={data_root}", 
        f"--data_path={data_path}",
        f"--output_root={output_root}",
        f"--batch_size={batch_size}",
        f"--dtype={dtype}",
        f"--run_mode={run_mode}",
        f"--input_format={input_format}",
    ]
    
    if max_model_len is not None:
        cmd_parts.append(f"--max_model_len={max_model_len}")
    if tag:
        cmd_parts.append(f"--tag={tag}")
    if filter_file_suffix:
        cmd_parts.append(f"--filter_file_suffix={filter_file_suffix}")
    
    print(f"Running: {model_name} on {Path(data_path).name}")
    print(f"Command: {' '.join(cmd_parts)}")
    
    subprocess.run(cmd_parts, check=True)
    print(f"Completed: {model_name} on {Path(data_path).name}\n")


if __name__ == "__main__":
    for model_path, model_name in models:
        for data_path in data_paths:
            main(model_path, model_name, data_path)

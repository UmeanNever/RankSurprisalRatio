import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

def _scan_response_positions(input_ids: List[int], header_ids: List[int], end_ids: List[int]) -> List[int]:
    pos = []
    n, h, e = len(input_ids), len(header_ids), len(end_ids)
    if h == 0:
        return pos

    i = 0
    while i <= n - h:
        if input_ids[i:i+h] == header_ids:
            st = i + h
            j = st
            found_end = False
            while j <= n - e:
                if e > 0 and input_ids[j:j+e] == end_ids:
                    found_end = True
                    break
                j += 1
            ed = j if found_end else n
            if ed > st:
                pos.extend(range(st, ed))
            i = (j + e) if found_end else n
        else:
            i += 1
    return pos

def _patch_multimodal_messages(sample: Dict, media_root: str) -> Dict:
    """Normalize non-HF multimodal blocks to HF-style image/video blocks."""
    if "messages" not in sample:
        return sample

    root = media_root.rstrip("/") if media_root else ""

    def complete_url(url: str) -> str:
        if root and url and not url.startswith(("http://", "https://", "file://", "/")):
            return f"{root}/{url.lstrip('/')}"
        return url

    patched = dict(sample)
    patched_msgs = []

    for m in sample["messages"]:
        mm = dict(m)
        c = mm.get("content")
        if isinstance(c, list):
            new_c = []
            for blk in c:
                if not isinstance(blk, dict):
                    new_c.append(blk)
                    continue

                t = blk.get("type")
                if t == "image_url":
                    url = complete_url(blk.get("image_url", {}).get("url", ""))
                    new_c.append({"type": "image", "image": url})
                elif t == "video_url":
                    url = complete_url(blk.get("video_url", {}).get("url", ""))
                    new_c.append({"type": "video", "video": url})
                elif t in ("image", "video"):
                    key = "image" if t == "image" else "video"
                    v = blk.get(key, "")
                    if isinstance(v, str):
                        blk = dict(blk)
                        blk[key] = complete_url(v)
                    new_c.append(blk)
                else:
                    new_c.append(blk)
            mm["content"] = new_c
        else:
            mm["content"] = [{"type": "text", "text": c}] if isinstance(c, str) else c
        patched_msgs.append(mm)

    patched["messages"] = patched_msgs
    return patched

def save_inference_results(samples: List[Dict], output_path: Path, fmt: str = "parquet"):
    """Save sample-level results as parquet or jsonl."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        rows = []
        for s in samples:
            rows.append({
                "sample_id": s["sample_id"],
                "resp_token_positions": s["resp_token_positions"],
                "probs": s["probs"],
                "nlls": s["nlls"],
                "ranks": s["ranks"],
                **s.get("metadata", {}),
            })
        pd.DataFrame(rows).to_parquet(output_path, index=False)
    else:
        with output_path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

def load_inference_results(input_path: Path, fmt: str = "parquet") -> List[Dict]:
    """Load sample-level results from parquet or jsonl."""
    if fmt == "parquet":
        df = pd.read_parquet(input_path)
        out = []
        core_cols = {"sample_id", "resp_token_positions", "probs", "nlls", "ranks"}
        for _, row in df.iterrows():
            d = row.to_dict()
            sample_id = d.pop("sample_id")
            resp_token_positions = list(d.pop("resp_token_positions"))
            probs = list(d.pop("probs"))
            nlls = list(d.pop("nlls"))
            ranks = list(d.pop("ranks"))
            metadata = {k: v for k, v in d.items() if k not in core_cols}
            out.append({
                "sample_id": sample_id,
                "resp_token_positions": resp_token_positions,
                "probs": probs,
                "nlls": nlls,
                "ranks": ranks,
                "metadata": metadata,
            })
        return out
    else:
        out = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

def _iter_json_files(data: Path, input_format: str) -> List[Path]:
    """Iterate dataset files by format."""
    if data.is_file():
        return [data]
    return sorted(list(data.rglob("*.json")) + list(data.rglob("*.jsonl")))

def _data_name_from(path: Path, data_root: Path) -> Tuple[Path, str]:
    """Generate data name from path."""
    rel = path.relative_to(data_root).with_suffix("")
    return rel, "__".join(rel.parts)

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
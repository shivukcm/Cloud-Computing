#!/usr/bin/env python3
"""
Task 1: Sharded Worker for Teacher Labelling
Reads SHARD_INDEX and SHARD_COUNT from environment.
Selects commands where line_number % SHARD_COUNT == SHARD_INDEX.
Sends each command to the teacher (Ollama / phi3.5) with temperature=0 and seed=42.
Writes outputs to datasets/teacher/shard-{SHARD_INDEX}.jsonl
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_prompt_template(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def query_teacher(
    ollama_host: str,
    model_name: str,
    prompt: str,
    retries: int = 5,
    delay: float = 2.0,
) -> str:
    url = f"{ollama_host.rstrip('/')}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 42,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"Failed to query teacher at {url} after {retries} attempts: {e}"
                )
            time.sleep(delay * (attempt + 1))
    return ""


def main():
    shard_index = int(os.environ.get("SHARD_INDEX", "0"))
    shard_count = int(os.environ.get("SHARD_COUNT", "2"))
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model_name = os.environ.get("MODEL_NAME", "phi3.5")
    commands_file = os.environ.get("COMMANDS_FILE", "data/commands_pool.jsonl")
    prompt_file = os.environ.get("PROMPT_FILE", "prompts/teacher_prompt.txt")
    output_dir = os.environ.get("OUTPUT_DIR", "datasets/teacher")
    output_file = os.environ.get(
        "OUTPUT_FILE", f"{output_dir}/shard-{shard_index}.jsonl"
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    prompt_template = load_prompt_template(prompt_file)

    print(
        f"[Worker {shard_index}/{shard_count}] Starting labeling using model {model_name} at {ollama_host}..."
    )

    # Read commands
    assigned_commands = []
    with open(commands_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if line_no % shard_count == shard_index:
                data = json.loads(line)
                assigned_commands.append(data)

    print(
        f"[Worker {shard_index}] Assigned {len(assigned_commands)} commands. Writing to {output_file}..."
    )

    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, cmd in enumerate(assigned_commands):
            cmd_id = cmd["id"]
            text = cmd["text"]
            full_prompt = prompt_template.replace("{command}", text)

            raw_reply = query_teacher(ollama_host, model_name, full_prompt)
            record = {"id": cmd_id, "text": text, "teacher_raw": raw_reply}
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if (i + 1) % 50 == 0 or (i + 1) == len(assigned_commands):
                print(
                    f"[Worker {shard_index}] Processed {i + 1}/{len(assigned_commands)} commands"
                )

    print(
        f"[Worker {shard_index}] Finished! Output saved to {output_file}"
    )


if __name__ == "__main__":
    main()

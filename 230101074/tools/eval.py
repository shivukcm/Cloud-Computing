#!/usr/bin/env python3
"""
Evaluation script: Generates predictions on a dataset using a HuggingFace model.
Outputs JSON Lines format required by tools/score.py:
{"id": "V001", "prediction": "<raw output text>"}
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on dataset")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path or HuggingFace ID of model/tokenizer",
    )
    parser.add_argument(
        "--dataset",
        default="data/validation.jsonl",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--out", required=True, help="Path to write predictions.jsonl"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens to generate",
    )
    args = parser.parse_args()

    print(f"Loading model and tokenizer from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.to("cpu")
    model.eval()

    records = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    print(
        f"Evaluating {len(records)} examples from {args.dataset}..."
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, rec in enumerate(records):
            rec_id = rec["id"]
            text = rec["text"]
            prompt = f"Command: {text}\nJSON: "

            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )

            # Extract generated portion
            gen_text = tokenizer.decode(
                out_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            # If model produced newline or extra characters after the JSON closing brace, extract the JSON object
            clean_pred = gen_text.strip()
            first_brace = clean_pred.find("{")
            last_brace = clean_pred.rfind("}")
            if (
                first_brace != -1
                and last_brace != -1
                and last_brace >= first_brace
            ):
                clean_pred = clean_pred[first_brace : last_brace + 1]

            pred_record = {"id": rec_id, "prediction": clean_pred}
            out_f.write(json.dumps(pred_record) + "\n")
            out_f.flush()

            if (i + 1) % 20 == 0 or (i + 1) == len(records):
                print(f"Processed {i + 1}/{len(records)} examples")

    print(f"Done! Predictions saved to {args.out}")


if __name__ == "__main__":
    main()

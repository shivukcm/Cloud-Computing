#!/usr/bin/env python3
"""
Task 2: Student Model Distillation Training
Fine-tunes EleutherAI/pythia-70m on CPU using Sequence-Level Knowledge Distillation.
Reads hyperparameters exclusively from training/config.yaml.
Supports --subset / SUBSET=50 for verification runs.
Saves per-epoch loss to training/loss.csv and computes SHA-256 checksum.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CommandDataset(Dataset):

    def __init__(self, records, tokenizer, max_length=128):
        self.examples = []
        for r in records:
            cmd = r["text"]
            # Format: Command: <cmd>\nJSON: <teacher_json>
            if "json_reply" in r:
                json_str = r["json_reply"]
            else:
                json_obj = {
                    "intent": r.get("intent"),
                    "date": r.get("date"),
                    "time": r.get("time"),
                    "place": r.get("place"),
                }
                json_str = json.dumps(json_obj)

            prompt_text = f"Command: {cmd}\nJSON: "
            full_text = f"{prompt_text}{json_str}{tokenizer.eos_token}"

            prompt_enc = tokenizer(
                prompt_text, add_special_tokens=False
            )
            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = full_enc["input_ids"].squeeze(0)
            attention_mask = full_enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            # Mask out the prompt tokens in labels so loss is only calculated on the JSON completion
            prompt_len = min(
                len(prompt_enc["input_ids"]), max_length
            )
            labels[:prompt_len] = -100

            # Mask padding in labels
            labels[attention_mask == 0] = -100

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def compute_dir_sha256(dir_path: str, out_checksum_path: str):
    """Computes a combined sha256 checksum for the model files in directory."""
    files = sorted(
        [
            p
            for p in Path(dir_path).glob("*")
            if p.is_file() and not p.name.endswith(".sha256")
        ]
    )
    combined = hashlib.sha256()
    file_hashes = []

    for f in files:
        h = hashlib.sha256()
        with open(f, "rb") as fp:
            while chunk := fp.read(65536):
                h.update(chunk)
                combined.update(chunk)
        file_hashes.append(f"{h.hexdigest()}  {f.name}")

    overall_hash = combined.hexdigest()
    with open(out_checksum_path, "w", encoding="utf-8") as out:
        out.write(f"{overall_hash}\n")
        for line in file_hashes:
            out.write(f"{line}\n")
    return overall_hash


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune pythia-70m on CPU"
    )
    parser.add_argument(
        "--config", default="training/config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Train on subset of records (e.g. 50)",
    )
    parser.add_argument(
        "--version",
        default="v1",
        choices=["v1", "v2"],
        help="Model version to train (v1 or v2)",
    )
    args = parser.parse_args()

    # Allow SUBSET env variable if flag not passed
    subset = (
        args.subset
        if args.subset is not None
        else (
            int(os.environ["SUBSET"]) if "SUBSET" in os.environ else None
        )
    )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("random_seed", 42)
    lr = float(cfg.get("learning_rate", 3e-4))
    batch_size = int(cfg.get("batch_size", 8))
    epochs = int(cfg.get("epochs", 5))
    max_seq_len = int(cfg.get("max_sequence_length", 128))
    model_name = cfg.get("model_name", "EleutherAI/pythia-70m")
    data_path = cfg.get("data_path", "datasets/v1/train.jsonl")
    loss_csv = cfg.get("loss_csv", "training/loss.csv")

    output_dir = (
        cfg.get("output_dir_v1", "models/v1")
        if args.version == "v1"
        else cfg.get("output_dir_v2", "models/v2")
    )
    if args.version == "v2":
        epochs = epochs + 1  # v2 trains for 1 additional epoch

    set_seed(seed)
    device = torch.device("cpu")
    print(
        f"Training {model_name} (Version: {args.version}) on CPU with seed={seed}, lr={lr}, epochs={epochs}, batch_size={batch_size}"
    )

    # Load data
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    # Apply 1,500 record limit according to specification, or subset
    records = records[:1500]
    if subset:
        records = records[:subset]
        print(f"Running in verification mode with SUBSET={len(records)}")
    else:
        print(f"Training on {len(records)} cleaned examples")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = CommandDataset(records, tokenizer, max_length=max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )

    Path(loss_csv).parent.mkdir(parents=True, exist_ok=True)
    epoch_losses = []

    print(
        f"Starting training for {epochs} epochs ({total_steps} steps total)..."
    )
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        epoch_losses.append((epoch, avg_loss))
        print(f"Epoch {epoch}/{epochs} - Average Loss: {avg_loss:.4f}")

    # Write loss.csv (epoch, loss)
    with open(loss_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for ep, l in epoch_losses:
            writer.writerow([ep, f"{l:.6f}"])

    print(f"Loss history saved to {loss_csv}")

    # Save model and tokenizer
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

    # Compute checksum
    checksum_file = f"{output_dir}/model.sha256"
    combined_hash = compute_dir_sha256(output_dir, checksum_file)
    print(
        f"Model SHA-256 checksum saved to {checksum_file}: {combined_hash}"
    )


if __name__ == "__main__":
    main()

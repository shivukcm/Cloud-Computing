#!/usr/bin/env python3
"""
Task 1: Data Cleaning and Checksum Verification
Reads all teacher shard outputs from datasets/teacher/, removes invalid records,
deduplicates, sorts by id, and outputs datasets/v1/train.jsonl and datasets/v1/train.sha256.
Prints the complete rejection counts table.
"""

import glob
import hashlib
import json
import os
import sys
from pathlib import Path

VALID_INTENTS = {
    "alarm_set",
    "alarm_query",
    "calendar_set",
    "calendar_query",
    "calendar_remove",
    "weather_query",
    "datetime_query",
    "recommendation_events",
}

EXPECTED_KEYS = {"intent", "date", "time", "place"}


def extract_json(raw_text: str):
    """Attempts to parse raw_text as JSON. Handles potential markdown code fences."""
    raw = raw_text.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    # Find first { and last }
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace >= first_brace:
        raw = raw[first_brace : last_brace + 1]

    return json.loads(raw)


def clean_dataset(
    input_pattern="datasets/teacher/shard-*.jsonl",
    output_file="datasets/v1/train.jsonl",
    checksum_file="datasets/v1/train.sha256",
):
    shard_files = sorted(glob.glob(input_pattern))
    if not shard_files:
        print(f"Error: No shard files found matching {input_pattern}")
        sys.exit(1)

    print(
        f"Reading shard files: {', '.join(shard_files)}"
    )

    all_records = []
    seen_ids = set()

    for sf in shard_files:
        with open(sf, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rec_id = record["id"]
                if rec_id in seen_ids:
                    print(
                        f"Warning: Duplicate ID {rec_id} across shard files!"
                    )
                seen_ids.add(rec_id)
                all_records.append(record)

    total_input = len(all_records)
    print(f"Total raw input records loaded: {total_input}")

    # Sort all input records by ID first to guarantee deterministic evaluation
    all_records.sort(key=lambda x: x["id"])

    count_rejected_a = 0  # Not valid JSON
    count_rejected_b = 0  # Keys not exactly intent, date, time, place
    count_rejected_c = 0  # Intent not in the 8 valid intents
    count_rejected_d = 0  # Slot value not null and not appearing in command text
    count_duplicates = 0  # Duplicate command text
    count_kept = 0

    seen_command_texts = {}  # text -> smallest id
    valid_candidates = []

    for rec in all_records:
        rec_id = rec["id"]
        text = rec["text"]
        raw = rec.get("teacher_raw", "")

        # (a) Check valid JSON
        try:
            parsed = extract_json(raw)
            if not isinstance(parsed, dict):
                count_rejected_a += 1
                continue
        except Exception:
            count_rejected_a += 1
            continue

        # (b) Check keys are exactly intent, date, time, place
        if set(parsed.keys()) != EXPECTED_KEYS:
            count_rejected_b += 1
            continue

        # (c) Check intent is one of the eight
        intent = parsed.get("intent")
        if intent not in VALID_INTENTS:
            count_rejected_c += 1
            continue

        # (d) Check slots: either null or text appears in command
        text_lower = text.lower()
        slots_valid = True
        for slot in ["date", "time", "place"]:
            val = parsed.get(slot)
            if val is not None:
                if not isinstance(val, str):
                    slots_valid = False
                    break
                if val.lower().strip() not in text_lower:
                    slots_valid = False
                    break

        if not slots_valid:
            count_rejected_d += 1
            continue

        # Record is structurally valid! Now check (e) duplicates:
        # Keep the one with the smallest id
        normalized_text = " ".join(text_lower.split())
        if normalized_text in seen_command_texts:
            count_duplicates += 1
            continue

        seen_command_texts[normalized_text] = rec_id
        valid_candidates.append(
            {
                "id": rec_id,
                "text": text,
                "intent": parsed["intent"],
                "date": parsed["date"],
                "time": parsed["time"],
                "place": parsed["place"],
                "json_reply": json.dumps(
                    parsed, separators=(", ", ": ")
                ),
            }
        )

    count_kept = len(valid_candidates)

    # Sort kept records in id order
    valid_candidates.sort(key=lambda x: x["id"])

    # Output to datasets/v1/train.jsonl
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in valid_candidates:
            # We output clean jsonl with the parsed fields
            clean_obj = {
                "id": rec["id"],
                "text": rec["text"],
                "intent": rec["intent"],
                "date": rec["date"],
                "time": rec["time"],
                "place": rec["place"],
            }
            f.write(
                json.dumps(clean_obj, ensure_ascii=False) + "\n"
            )

    # Calculate SHA-256
    sha256 = hashlib.sha256()
    with open(output_file, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    digest = sha256.hexdigest()

    with open(checksum_file, "w", encoding="utf-8") as f:
        f.write(digest + "\n")

    # Print summary table
    print("\n" + "=" * 50)
    print("DATA CLEANING COUNTS SUMMARY")
    print("=" * 50)
    print(f"Total Input Records      : {total_input}")
    print(f"Rejected (a) Invalid JSON: {count_rejected_a}")
    print(f"Rejected (b) Key Mismatch: {count_rejected_b}")
    print(f"Rejected (c) Bad Intent  : {count_rejected_c}")
    print(f"Rejected (d) Bad Slot    : {count_rejected_d}")
    print(f"Duplicates (e) Removed   : {count_duplicates}")
    print(f"Records Kept             : {count_kept}")
    print("-" * 50)
    calc_sum = (
        count_rejected_a
        + count_rejected_b
        + count_rejected_c
        + count_rejected_d
        + count_duplicates
        + count_kept
    )
    print(f"Sum (a + b + c + d + dup + kept) = {calc_sum}")
    print(f"Equation Holds (Sum == Input)   : {calc_sum == total_input}")
    print(f"SHA-256 Checksum: {digest}")
    print(f"Saved to: {output_file} and {checksum_file}")
    print("=" * 50 + "\n")

    return {
        "input": total_input,
        "rejected_a": count_rejected_a,
        "rejected_b": count_rejected_b,
        "rejected_c": count_rejected_c,
        "rejected_d": count_rejected_d,
        "duplicates": count_duplicates,
        "kept": count_kept,
        "checksum": digest,
    }


if __name__ == "__main__":
    clean_dataset()

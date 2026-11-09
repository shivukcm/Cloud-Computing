#!/usr/bin/env python3
"""
Task 3: Production Inference Service in FastAPI
Endpoints:
  POST /parse   -> Extracts 4-key JSON, sets X-Model-Version and X-Served-By
  GET  /healthz -> Process liveness probe (200 OK)
  GET  /readyz  -> Model readiness probe (503 until loaded, 200 once ready)
Configuration:
  MODEL_URI, MODEL_VERSION, S3_ENDPOINT, MAX_NEW_TOKENS, FAULT_INJECT
"""

import json
import os
import re
import socket
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Header, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Schema definition and validation rules
SCHEMA_PATH = os.environ.get("SCHEMA_PATH", "data/schema.json")
try:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        SCHEMA_DATA = json.load(f)
        VALID_INTENTS = set(SCHEMA_DATA.get("intents", []))
        REQUIRED_KEYS = set(SCHEMA_DATA.get("keys", []))
except Exception:
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
    REQUIRED_KEYS = {"intent", "date", "time", "place"}

# Environment Configuration
MODEL_URI = os.environ.get("MODEL_URI", "models/v1")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
FAULT_INJECT = os.environ.get("FAULT_INJECT", "0")
HOSTNAME = os.environ.get("HOSTNAME", socket.gethostname())

model = None
tokenizer = None
is_ready = False
model_lock = threading.Lock()


class ParseRequest(BaseModel):
    text: str


def download_model_from_s3(s3_uri: str, endpoint_url: str, dest_dir: str):
    import boto3
    from botocore.client import Config
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url if endpoint_url else None,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "seaweedfs"),
        aws_secret_access_key=os.environ.get(
            "AWS_SECRET_ACCESS_KEY", "seaweedfs"
        ),
        config=Config(signature_version="s3v4"),
    )

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = os.path.relpath(key, prefix)
            target_path = Path(dest_dir) / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading s3://{bucket}/{key} to {target_path}...")
            s3.download_file(bucket, key, str(target_path))


def load_model_pipeline():
    global model, tokenizer, is_ready
    local_path = MODEL_URI

    print(f"Initializing model from MODEL_URI={MODEL_URI}...")
    if MODEL_URI.startswith("s3://"):
        cache_dir = f"/tmp/models/{MODEL_VERSION}"
        try:
            download_model_from_s3(MODEL_URI, S3_ENDPOINT, cache_dir)
            local_path = cache_dir
        except Exception as e:
            print(f"Warning: S3 download failed: {e}. Checking local fallback...")
            fallback = f"models/{MODEL_VERSION}"
            if os.path.exists(fallback):
                local_path = fallback

    print(f"Loading model weights from {local_path}...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(local_path)
        model.to(torch.device("cpu"))
        model.eval()
        is_ready = True
        print(f"Model {MODEL_VERSION} successfully loaded and ready.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # If model loading failed, readiness remains False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model in separate thread to allow fast process startup for /healthz
    t = threading.Thread(target=load_model_pipeline, daemon=True)
    t.start()
    yield


app = FastAPI(title="Command Parser Inference Service", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    """Liveness probe: returns 200 OK as long as server is alive."""
    return Response(content="OK", status_code=status.HTTP_200_OK)


@app.get("/readyz")
def readyz():
    """Readiness probe: returns 503 until model is loaded, then 200."""
    if not is_ready:
        return Response(
            content="Model Not Ready",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return Response(content="Ready", status_code=status.HTTP_200_OK)


def validate_schema(output_text: str):
    """
    Validates output text against schema.
    Returns (valid: bool, parsed_obj: dict).
    """
    try:
        raw = output_text.strip()
        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
            return False, None
        extracted = raw[first_brace : last_brace + 1]
        obj = json.loads(extracted)
    except Exception:
        return False, None

    if not isinstance(obj, dict):
        return False, None

    if set(obj.keys()) != REQUIRED_KEYS:
        return False, None

    if obj.get("intent") not in VALID_INTENTS:
        return False, None

    for slot in ["date", "time", "place"]:
        val = obj.get(slot)
        if val is not None and not isinstance(val, str):
            return False, None

    return True, obj


@app.post("/parse")
def parse(req: ParseRequest):
    """
    Parses a single command text into 4-key JSON.
    Returns 200 with JSON body or 500 with {"error": "invalid_model_output"}.
    Sets X-Model-Version and X-Served-By response headers.
    """
    headers = {
        "X-Model-Version": MODEL_VERSION,
        "X-Served-By": HOSTNAME,
    }

    if not is_ready or model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready",
            headers=headers,
        )

    # Prompt format
    prompt_text = f"Command: {req.text}\nJSON: "
    inputs = tokenizer(prompt_text, return_tensors="pt")

    try:
        import torch
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
    except ImportError:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    generated_text = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    # Check FAULT_INJECT
    current_fault = os.environ.get("FAULT_INJECT", FAULT_INJECT)
    if current_fault == "1":
        generated_text = "not json"

    # Schema validation
    is_valid, parsed_json = validate_schema(generated_text)
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "invalid_model_output"},
            headers=headers,
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=parsed_json,
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

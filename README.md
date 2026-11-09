# Cloud Computing (CS3081) Mini Project

**Knowledge Distillation and Deployment of a Lightweight Command-Parsing Language Model on a Local Cloud**

* **Author:** Shivu K C M  
* **Roll Number:** 230101074  
* **GitHub Repository:** [https://github.com/shivukcm/Cloud-Computing](https://github.com/shivukcm/Cloud-Computing)  
* **Release Assets:** [v1.0.0 Releases](https://github.com/shivukcm/Cloud-Computing/releases/tag/v1.0.0)  

---

## 1. System Architecture Overview

This project implements an end-to-end cloud-native machine learning pipeline:
1. **Teacher Distillation:** Sharded parallel worker jobs extract structured 4-key JSON (`intent`, `date`, `time`, `place`) from 3,000 SLURP natural language commands using Microsoft Phi-3.5-mini-instruct (3.8B parameters) via Ollama (`temperature=0`, `seed=42`).
2. **Data Pipeline & Quality Filter:** Automated cleaning script (`worker/clean.py`) enforces strict JSON schema adherence, eliminates hallucinations, removes duplicate commands, and generates SHA-256 integrity hashes.
3. **Student Distillation:** Fine-tunes `EleutherAI/pythia-70m` on CPU using Sequence-Level Knowledge Distillation on the first 1,500 cleaned records.
4. **Cloud-Native Storage:** SeaweedFS deployed in Kubernetes namespace `data` as an S3-compatible object store for datasets and model weights.
5. **Containerized Inference Service:** Production FastAPI microservice running under strict resource constraints (1 CPU, 1 GiB memory), featuring decoupled weight loading, `/healthz` and `/readyz` probes, and schema-checking middleware.
6. **Kubernetes Deployment & Orchestration:** Deployed in namespace `app` with Traefik Ingress routing `/parse`, Horizontal Pod Autoscaling (HPA), rolling updates from v1 to v2, and recovery procedures.
7. **Stateless Subnetting & NACL:** Four-tier CIDR subnetting of `10.20.0.0/16` and stateless NACL protecting public ingress against abusive ranges.
8. **Automated CI/CD:** GitHub Actions workflow executing code linting, unit testing, model evaluation from Release assets, and a 95% JSON validity quality gate.

---

## 2. Directory Structure

```
├── cluster/
│   └── k3d.yaml                  # Local k3d Kubernetes cluster configuration
├── k8s/
│   ├── data/
│   │   └── seaweedfs.yaml        # SeaweedFS S3 object store deployment & service
│   ├── pipeline/
│   │   └── worker-jobs.yaml      # Parallel sharded teacher labeling worker jobs
│   └── app/
│       └── deployment.yaml       # Deployment, Service, Ingress, HPA, Secret
├── prompts/
│   └── teacher_prompt.txt        # Four-key JSON teacher extraction prompt
├── worker/
│   ├── worker.py                 # Parallel sharded worker script
│   └── clean.py                  # Rule-based dataset cleaner & checksum generator
├── datasets/
│   └── v1/
│       ├── train.jsonl           # Cleaned training dataset (1,500 records)
│       └── train.sha256          # SHA-256 integrity hash
├── training/
│   ├── config.yaml               # Reproducible training configuration & pinned versions
│   ├── train.py                  # Pythia-70M CPU sequence distillation script
│   └── loss.csv                  # Epoch loss tracking
├── models/
│   ├── v1/
│   │   └── model.sha256          # Model artifact checksum (matches GitHub Release)
│   └── v2/                       # Second version for rolling update
├── service/
│   ├── app.py                    # FastAPI inference microservice
│   ├── Dockerfile                # Pinned CPU-only container (< 2GB, no weights)
│   └── requirements.txt          # Pinned service dependencies
├── tests/
│   └── test_service.py           # Unit test suite (5 test cases)
├── tools/
│   ├── eval.py                   # Model evaluation harness
│   ├── score.py                  # Evaluation scoring script (provided)
│   ├── locustfile.py             # Locust load-testing script (provided)
│   └── nacl_sim.py               # Stateless NACL simulator (provided)
├── network/
│   ├── subnets.md                # 10.20.0.0/16 subnet calculations & NACL reasoning
│   ├── nacl.json                 # Stateless NACL rules for public subnet
│   └── packets_sample.jsonl      # Test packet traces
├── results/                      # Evaluation reports, load test stats, headers
├── report/
│   └── report.pdf                # Comprehensive 8-page technical report
├── .github/workflows/
│   └── ci.yml                    # Complete CI/CD workflow
├── Makefile                      # One-command orchestration
├── AI_USE.md                     # Disclosure of AI tools and verification procedures
└── 230101074.pdf                 # Scanned 3-page handwritten reflection notes
```

---

## 3. Quickstart: Build, Deploy, Test, and Teardown

### Prerequisites
* Python 3.11+
* Docker / Colima / OrbStack
* `k3d` (v5.x) and `kubectl`
* `ollama` (for running the Phi-3.5 teacher)

### 3.1 System Startup (`make up`)
To build and deploy the complete infrastructure in at most three commands:
```bash
make up
```
This automatically:
1. Provisions the local Kubernetes cluster (`k3d cluster create --config cluster/k3d.yaml`).
2. Deploys SeaweedFS object storage in namespace `data` (`kubectl apply -f k8s/data/seaweedfs.yaml`).
3. Deploys the inference service, Traefik ingress, and HPA in namespace `app` (`kubectl apply -f k8s/app/deployment.yaml`).

### 3.2 Model Training (`make train`)
To fine-tune `EleutherAI/pythia-70m` on CPU using hyperparameters defined in `training/config.yaml`:
```bash
# Full training on the first 1,500 records
make train

# Verification mode on the first 50 records
make train SUBSET=50

# Train version 2 for rolling update demonstration
make train-v2
```

### 3.3 Running Unit Tests & Linting
```bash
# Run unit test suite (5 unit tests covering endpoints, headers, schema validation, FAULT_INJECT)
make test

# Run code linter
make lint
```

### 3.4 Simulating Stateless NACL
```bash
python3 tools/nacl_sim.py network/nacl.json network/packets_sample.jsonl --check
```

### 3.5 Load Testing with Locust
Run the standardized 2-minute load test with 4 concurrent users:
```bash
locust -f tools/locustfile.py --headless --users 4 --spawn-rate 4 --run-time 2m --host http://localhost:8080 --csv results/loadtest
```

### 3.6 System Teardown (`make down`)
To completely delete the cluster and release all local system resources:
```bash
make down
```

---

## 4. GitHub Release Assets & Checksum Verification

Model weights are strictly excluded from the Git repository and hosted as GitHub Release assets:
* **Release Asset URL:** `https://github.com/shivukcm/Cloud-Computing/releases/download/v1.0.0/model_v1.tar.gz`
* **Integrity Check:** Checksums are verified via `models/v1/model.sha256`.

---

## 5. Verification Checklist

* [x] Task 1: `cluster/k3d.yaml`, SeaweedFS in `data`, sharded workers in `pipeline`, `worker/clean.py` equation verified.
* [x] Task 2: `training/config.yaml` with all 6 items, `make train SUBSET=50` support, `models/v1/model.sha256`.
* [x] Task 3: Pinned Docker image < 2GB (CPU-only PyTorch), external model loading, `/parse`, `/healthz`, `/readyz`, `FAULT_INJECT=1`.
* [x] Task 4: Ingress `/parse`, 2 replicas, HPA 70%, rolling update v1 $\to$ v2, pod recovery, cold start, load balancing.
* [x] Task 5: Contiguous subnetting table in `network/subnets.md`, 100% score on `tools/nacl_sim.py`.
* [x] Task 6: `Makefile` (`make up`, `make down`), GitHub Actions CI pipeline with secret scan, quality gate, `AI_USE.md`.
* [x] Task 7: Locust results, cost calculation from `rate_card.json`, SLO & error budget, evidence review table.
* [x] Deliverables: `report/report.pdf` (max 8 pages), `230101074.pdf` (max 3 pages), `230101074.zip`.

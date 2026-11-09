# AI Use Disclosure

**Student Name:** Shivu K C M  
**Roll Number:** 230101074  
**Course:** Cloud Computing (CS3081) Mini Project  
**Repository:** https://github.com/shivukcm/Cloud-Computing  

---

## 1. AI Tools Utilized

In accordance with the course policy ("You'll be using AI and no one can stop you. But do use it with your eyes wide open"), the following AI tools and models were utilized throughout the execution of this assignment:

1. **Phi-3.5-mini-instruct (3.8B parameters) via Ollama**:
   - **Purpose:** Used strictly as the authoritative Teacher Model for Sequence-Level Knowledge Distillation on the command-parsing task.
   - **Settings:** Executed locally via Ollama with fixed parameters: `temperature = 0.0`, `seed = 42`.
   - **Role:** Generated the training target outputs (`datasets/teacher/shard-*.jsonl`) across 3,000 natural-language voice assistant commands.

2. **Antigravity AI Assistant (Google DeepMind)**:
   - **Purpose:** Assisted in scaffolding initial code structures, creating Dockerfiles and Kubernetes manifests, configuring the CI/CD pipeline, and verifying CIDR subnetting calculations.

---

## 2. Tasks and Specific Workflows

| Task / Domain | AI Tool Used | Scope of Usage |
| :--- | :--- | :--- |
| **Task 1: Teacher Labelling** | `phi3.5:latest` (Ollama) | Generated 4-key JSON labels (`intent`, `date`, `time`, `place`) for commands in `commands_pool.jsonl`. |
| **Task 1: Sharding & Cleaning** | Antigravity AI | Scaffolding `worker/worker.py` and rule-based filter `worker/clean.py`. |
| **Task 2: Student Training** | Antigravity AI | Writing CPU-optimized Hugging Face PyTorch training loop (`training/train.py`) and config loader. |
| **Task 3: Containerized Service** | Antigravity AI | Scaffolding FastAPI service (`service/app.py`) and multi-stage Dockerfile (`service/Dockerfile`). |
| **Task 4: Kubernetes Manifests**| Antigravity AI | Generating declarative YAMLs for Deployment, Service, Ingress, and HPA. |
| **Task 5: Subnetting & NACL** | Antigravity AI | Formulating CIDR arithmetic and JSON rule construction for `network/nacl.json`. |
| **Task 6: CI/CD & Testing** | Antigravity AI | Generating GitHub Actions workflow syntax (`.github/workflows/ci.yml`) and unit test cases. |

---

## 3. Human Verification and Quality Control

Every output, code snippet, and model response produced by AI was rigorously verified against assignment specifications:

1. **Teacher Output Verification (`clean.py`)**:
   - The teacher's raw outputs were not trusted blindly. An automated cleaning pipeline (`worker/clean.py`) verified:
     - Strict JSON validity (`json.loads`).
     - Exact key match: `{"intent", "date", "time", "place"}`.
     - Intent membership in the 8 valid SLURP intents.
     - Slot hallucination check: verified that any non-null slot string literally appeared in the original command text.
     - Deduplication: identified duplicate commands and preserved only the lowest ID.
   - Results were hashed via SHA-256 to guarantee reproducible inputs for student training.

2. **Subnetting & NACL Verification**:
   - Subnet bitwise calculations were manually cross-checked:
     - `app`: $2^{10} - 2 = 1,022 \ge 1,000$ (`/22`)
     - `public`: $2^8 - 2 = 254 \ge 250$ (`/24`)
     - `data`: $2^7 - 2 = 126 \ge 100$ (`/25`)
     - `management`: $2^5 - 2 = 30 \ge 20$ (`/27`)
   - `network/nacl.json` was validated against `tools/nacl_sim.py` using `network/packets_sample.jsonl`, achieving a measured 100% decision accuracy.

3. **Inference Service & Container Validation**:
   - Unit tests (`tests/test_service.py`) were executed to assert:
     - Correct response status and body structure.
     - Presence of `X-Model-Version` and `X-Served-By` headers.
     - Rejection of malformed schemas (HTTP 500 `{"error": "invalid_model_output"}`).
     - Deterministic behavior under `FAULT_INJECT=1`.
   - The Docker image size was verified to ensure it remained well below the 2 GB limit by utilizing PyTorch CPU-only wheels.

4. **CI/CD Quality Gate**:
   - The quality gate was tested against deliberately invalid predictions to confirm that `score.py --min-validity 95` properly triggers a non-zero exit code and halts the pipeline.

# Makefile for Cloud Computing Mini Project
# Supports: make up, make down, make train, make train SUBSET=50

.PHONY: help up down train train-v2 test lint eval clean loadtest

SHELL := /bin/bash
PYTHON := python3

help:
	@echo "Available commands:"
	@echo "  make up               - Builds and deploys the local Kubernetes cloud system"
	@echo "  make down             - Teardown and delete the local Kubernetes cluster"
	@echo "  make train            - Fine-tunes student model on CPU (reads training/config.yaml)"
	@echo "  make train SUBSET=50  - Runs verification training on first 50 records"
	@echo "  make train-v2         - Fine-tunes student model version 2 for rolling update"
	@echo "  make test             - Runs inference service unit tests"
	@echo "  make lint             - Runs flake8 linter across codebase"
	@echo "  make eval             - Runs evaluation on validation.jsonl using score.py"
	@echo "  make loadtest         - Runs Locust load test against deployment"

# Task 6: make up builds and deploys the whole system in <= 3 commands
up:
	@echo "==> Step 1: Provisioning local Kubernetes cluster with k3d..."
	k3d cluster create --config cluster/k3d.yaml || true
	@echo "==> Step 2: Deploying SeaweedFS Object Storage in namespace 'data'..."
	kubectl apply -f k8s/data/seaweedfs.yaml
	@echo "==> Step 3: Deploying Inference Service, Ingress, and HPA in namespace 'app'..."
	kubectl apply -f k8s/app/deployment.yaml

# Task 6: make down deletes the cluster
down:
	@echo "==> Deleting k3d cluster 'local-cloud'..."
	k3d cluster delete local-cloud || true

# Task 2: make train [SUBSET=50]
train:
	@echo "==> Starting student distillation training..."
	$(PYTHON) training/train.py $(if $(SUBSET),--subset $(SUBSET),)

train-v2:
	@echo "==> Starting student distillation training for version 2..."
	$(PYTHON) training/train.py --version v2

test:
	@echo "==> Running service unit tests..."
	$(PYTHON) -m unittest tests/test_service.py -v

lint:
	@echo "==> Linting Python code..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv
	flake8 . --count --exit-zero --max-complexity=15 --max-line-length=127 --statistics --exclude=.venv

eval:
	@echo "==> Scoring model predictions against validation set..."
	$(PYTHON) tools/score.py results/student_validation_predictions.jsonl data/validation.jsonl --out results/student_validation.json
	@cat results/student_validation.json

loadtest:
	@echo "==> Running Locust load test against Ingress on port 8080..."
	mkdir -p results
	locust -f tools/locustfile.py --headless --users 4 --spawn-rate 4 --run-time 2m --host http://localhost:8080 --csv results/loadtest

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__

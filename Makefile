.PHONY: install test notebook lint \
       run-training-local run-scoring-local \
       submit-training submit-scoring \
       schedule-training schedule-scoring \
       setup-data setup-dev-env setup-prod \
       build-image setup-ar-python publish-wheel

REGION ?= us-central1
IMAGE_TAG ?= latest

# ── Install ──────────────────────────────────────────────────────────────────
install:
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv sync --all-extras

# ── Test & Lint ──────────────────────────────────────────────────────────────
test:
	uv run pytest tests/unit -v
	uv run pytest tests/integration -v

test-unit:
	uv run pytest tests/unit -v

test-integration:
	uv run pytest tests/integration -v

lint:
	uv run ruff check fraud_detector/ tests/
	uv run ruff format --check fraud_detector/ tests/

format:
	uv run ruff check --fix fraud_detector/ tests/
	uv run ruff format fraud_detector/ tests/

# ── Notebook ─────────────────────────────────────────────────────────────────
notebook:
	uv run jupyter lab notebooks/

# ── Local Pipeline Runs ──────────────────────────────────────────────────────
run-training-local:
	uv run python -m fraud_detector.pipelines.submit_pipeline --pipeline training --local

run-scoring-local:
	uv run python -m fraud_detector.pipelines.submit_pipeline --pipeline scoring --local

# ── Submit Pipelines to Vertex AI ────────────────────────────────────────────
submit-training:
	uv run python -m fraud_detector.pipelines.submit_pipeline --pipeline training

submit-scoring:
	uv run python -m fraud_detector.pipelines.submit_pipeline --pipeline scoring

# ── Schedule Pipelines ───────────────────────────────────────────────────────
schedule-training:
	uv run python -m fraud_detector.pipelines.submit_pipeline --pipeline training --schedule-only

schedule-scoring:
	uv run python -m fraud_detector.pipelines.submit_pipeline --pipeline scoring --schedule-only

# ── Data Setup ───────────────────────────────────────────────────────────────
setup-data:
	uv run python scripts/setup_data.py --source synthetic --n-transactions 10000 $(ARGS)

setup-data-gcs:
	uv run python scripts/setup_data.py $(ARGS)

setup-data-full:
	uv run python scripts/setup_data.py --full

# ── Container Image (deps-only) ─────────────────────────────────────────────
build-image:
	@CICD_PROJECT=$${CICD_PROJECT_ID:-$$PROJECT_ID}; \
	IMAGE_URI=$(REGION)-docker.pkg.dev/$$CICD_PROJECT/fraud-detector-docker/fraud-detector:$(IMAGE_TAG); \
	echo "Building deps-only image $$IMAGE_URI..."; \
	docker buildx build --platform linux/amd64 -t $$IMAGE_URI --push . && \
	echo "Done: $$IMAGE_URI"

# ── Artifact Registry Python Repo ──────────────────────────────────────────
setup-ar-python:
	@CICD_PROJECT=$${CICD_PROJECT_ID:-$$PROJECT_ID}; \
	gcloud artifacts repositories create fraud-detector-python \
		--repository-format=python \
		--location=$(REGION) \
		--project=$$CICD_PROJECT \
		--description="fraud-detector Python wheels" 2>/dev/null || \
	echo "Repository fraud-detector-python already exists"

publish-wheel:
	@uv build --wheel --out-dir dist && \
	CICD_PROJECT=$${CICD_PROJECT_ID:-$$PROJECT_ID}; \
	TOKEN=$$(gcloud auth print-access-token); \
	uv publish \
		--publish-url "https://$(REGION)-python.pkg.dev/$$CICD_PROJECT/fraud-detector-python/" \
		--username oauth2accesstoken \
		--password "$$TOKEN" \
		dist/fraud_detector-*.whl

# ── Infrastructure ───────────────────────────────────────────────────────────
setup-dev-env:
	uv run python scripts/setup_dev_env.py

setup-dev-env-terraform:
	cd deployment/terraform/dev && terraform init && terraform apply -var-file=vars/env.tfvars

setup-prod:
	cd deployment/terraform && terraform init && terraform apply -var-file=vars/env.tfvars

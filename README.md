# Predictive MLOps Starter Pack

Clone. Run. Ship to production.

A ready-to-use template for building production ML systems on Google Cloud. Comes with a working fraud detection pipeline you can run end-to-end in under an hour, then swap in your own model.

```
Data (BigQuery)  -->  Feature Engineering  -->  Train (XGBoost)  -->  Evaluate
                                                                        |
                                              Monitor (drift)  <--  Register (Model Registry)
                                                                        |
                                              Score (batch)    -->  Predictions (BigQuery)
```

## What's in the box

- **Working ML pipeline** — feature engineering, training, evaluation, model registration, batch scoring
- **Model monitoring** — automated drift detection comparing training vs serving data
- **Local-first development** — run the full pipeline on your laptop before touching the cloud
- **Production CI/CD** — GitHub Actions with Workload Identity Federation (no service account keys)
- **Multi-environment infra** — Terraform for dev / staging / prod, one command each
- **Scheduled retraining** — weekly training, 6-hourly scoring, all configurable via YAML
- **AI coding agent support** — [`GEMINI.md`](GEMINI.md) gives Gemini Code Assist (and other coding agents) full project context so they can make changes that actually work

## Get running in 4 commands

```bash
make install                 # Install dependencies
gcloud auth application-default login
make setup-data              # Load 100K transactions into BigQuery (~10s)
make run-training-local      # Run the full pipeline on your machine
```

That's it. You'll see feature engineering, model training (AUC ~0.89), evaluation, and registration — all running locally via KFP's subprocess runner.

## Ship to Vertex AI

```bash
export PROJECT_ID=<your-project>
make submit-training         # Builds container, submits to Vertex AI Pipelines
make submit-scoring          # Score all transactions, write predictions to BigQuery
```

The training pipeline registers the model to Vertex AI Model Registry and sets up weekly drift monitoring automatically.

## Set up production

```bash
make setup-prod              # Terraform: staging + prod infra, WIF, GitHub secrets
git push origin main         # Triggers CI/CD: lint -> test -> deploy to staging
```

Production deployment is a manual approval away.

| Trigger | Action |
|---------|--------|
| PR to main | Lint + unit tests |
| Push to main | Deploy to staging |
| Manual dispatch | Deploy to production + create schedules |

## Make it yours

| Swap this | Edit here |
|-----------|-----------|
| Dataset / tables | `fraud_detector/config/*.yaml` |
| Model algorithm | `fraud_detector/model.py` |
| Features | `fraud_detector/model.py` (`compute_features`) |
| Pipeline steps | `fraud_detector/pipelines/components/` |
| Drift thresholds | `fraud_detector/config/monitoring.yaml` |
| Schedules | `fraud_detector/config/training.yaml`, `scoring.yaml` |
| Infrastructure | `deployment/terraform/` |
| CI/CD | `.github/workflows/` |

## Project layout

```
fraud_detector/
  model.py                       # FraudDetector class — features, training, evaluation, scoring
  config.py                      # YAML config loader with ${VAR} resolution
  config/
    training.yaml                # Training pipeline config + XGBoost params
    scoring.yaml                 # Scoring pipeline config
    monitoring.yaml              # Drift thresholds + alert emails
  pipelines/
    training_pipeline.py         # FE -> Train -> Evaluate -> Register -> Monitor
    scoring_pipeline.py          # FE -> Predict -> Write
    submit_pipeline.py           # CLI: --local / --compile-only / --schedule-only
    components/                  # @dsl.component definitions (one per step)

scripts/                         # Data setup, e2e test
tests/                           # Unit + integration tests
deployment/terraform/            # Multi-project infra (dev, staging, prod)
.github/workflows/               # CI/CD (PR checks, staging deploy, prod deploy)
```

## Commands

| Command | What it does |
|---------|-------------|
| `make install` | Install all dependencies |
| `make setup-data` | Load sample data into BigQuery |
| `make run-training-local` | Run training pipeline locally |
| `make run-scoring-local` | Run scoring pipeline locally |
| `make submit-training` | Submit training to Vertex AI |
| `make submit-scoring` | Submit scoring to Vertex AI |
| `make test-unit` | Run unit tests |
| `make lint` | Check code style |
| `make notebook` | Launch Jupyter Lab |
| `make setup-dev-env` | Provision dev infrastructure |
| `make setup-prod` | Provision staging + prod + CI/CD |

## Google Cloud services used

| Service | Role |
|---------|------|
| **BigQuery** | Data warehouse — raw data, features, predictions |
| **Vertex AI Pipelines** | Orchestrate training, scoring, monitoring |
| **Vertex AI Model Registry** | Version and manage trained models |
| **Vertex AI Model Monitoring** | Detect feature drift (Jensen-Shannon divergence) |
| **Cloud Storage** | Model artifacts and pipeline staging |
| **Artifact Registry** | Container images for pipeline steps |
| **Workload Identity Federation** | Keyless GitHub Actions -> GCP auth |

## Develop with AI coding agents

This repo ships with [`GEMINI.md`](GEMINI.md) — a context file that teaches coding agents (Gemini Code Assist, Claude Code, Copilot, Cursor, etc.) how this project works. It includes:

- **Architecture and design decisions** — so the agent understands *why* the code is structured this way
- **20+ gotchas and learnings** — BigQuery decimal types, timezone handling, KFP caching quirks, cross-project IAM, ARM-to-AMD64 builds, and more
- **Code patterns** — the right way to prepare features, handle model artifacts, check pipeline status
- **Development workflow** — edit, test, run locally, submit to Vertex AI

When you ask an agent to add a feature, fix a bug, or extend a pipeline, it reads `GEMINI.md` first and avoids the pitfalls that would otherwise cost you hours of debugging. The file is designed for [Gemini Code Assist](https://cloud.google.com/products/gemini/code-assist) but works with any agent that reads project-level context files.

## Prerequisites

- Google Cloud project with billing enabled
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- `gcloud` CLI installed and authenticated
- Docker (for building pipeline container images)
- Terraform >= 1.0 (for infrastructure provisioning)

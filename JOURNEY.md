# Demo Journey

End-to-end walkthrough: from clone to production deployment.

## Prerequisites

- Google Cloud project with billing enabled
- `gcloud` CLI installed and authenticated
- `uv` package manager (auto-installed by `make install` if missing)

## Step 1: Clone & Install

```bash
git clone <repo-url>
cd predictive_mlops_demo
make install
```

This installs all dependencies (including pipelines, notebook, and dev extras) via `uv`.

## Step 2: Authenticate to GCP

```bash
gcloud auth application-default login
```

`PROJECT_ID` is auto-detected from your `gcloud` config. You can override it with `export PROJECT_ID=<your-gcp-project-id>` if needed.

## Step 3: Load Data into BigQuery

```bash
make setup-data              # 10K synthetic rows (fast demo)
make setup-data-gcs          # ~100K rows from GCS
make setup-data-full         # ~3.1M rows full dataset
```

Default uses synthetic data (10K rows, ~2% fraud rate) for fast iteration. Use `setup-data-gcs` or `setup-data-full` for real FraudFinder data from `gs://fraudfinder-public-data/`.

### Verified output

| Table | Rows |
|-------|------|
| `tx` | 10,000 |
| `txlabels` | 10,000 |
| `fraud_features` | 10,000 (written by feature engineering) |
| `fraud_scores` | 10,000 (written by scoring pipeline) |

## Step 4: Run Training Pipeline Locally

```bash
make run-training-local
```

This runs the full KFP training pipeline locally using `kfp.local.SubprocessRunner`:

```
feature-engineering-op  ->  Reads raw BQ data, computes 25 rolling features, writes to BQ
train-op                ->  Trains XGBoost model (6 depth, 200 estimators), uploads artifact
evaluate-op             ->  Evaluates on holdout set, returns AUC-ROC
register-op             ->  Registers model in Vertex AI if AUC >= threshold (skipped locally)
setup-monitoring-op     ->  Sets up Model Monitoring v2 (skipped locally: SKIPPED:LOCAL_ONLY)
```

### Verified output

- **AUC-ROC**: 0.8880
- **register-op**: `LOCAL_ONLY` (expected — model URI is local, skips Vertex registration)
- **setup-monitoring-op**: `SKIPPED:LOCAL_ONLY` (expected — no model to monitor)
- All 5 steps: SUCCESS

## Step 5: Run Tests

```bash
make test-unit
make lint
```

### Verified output

- **15 unit tests passed** (feature engineering: 5, monitoring: 6, training: 4)
- **Lint**: all checks passed, 22 files formatted

## Step 6: Submit Training Pipeline to Vertex AI

```bash
make submit-training
```

This builds the container image (content-hash tagged), compiles the KFP pipeline, and submits to Vertex AI.

### Verified output

All 5 pipeline steps succeeded on Vertex AI:

| Step | Status | Output |
|------|--------|--------|
| `feature-engineering-op` | SUCCEEDED | `asp-test-dev.fraud_detection.fraud_features` |
| `train-op` | SUCCEEDED | Model artifact uploaded to GCS |
| `evaluate-op` | SUCCEEDED | AUC-ROC: **0.8880** |
| `register-op` | SUCCEEDED | `projects/901701644605/locations/us-central1/models/1220914204156887040` |
| `setup-monitoring-op` | SUCCEEDED | Monitor ID: `4170724681084567552` |

- **Model registered** in Vertex AI Model Registry as `fraud-detector-xgb` (version 1)
- **Model Monitor created** with weekly drift detection schedule (Monday 8am)
- **Drift detection**: Jensen-Shannon divergence, threshold 0.3, comparing `fraud_features` vs `fraud_scores`

## Step 7: Submit Scoring Pipeline to Vertex AI

```bash
make submit-scoring
```

### Verified output

All 3 scoring pipeline steps succeeded:

| Step | Status |
|------|--------|
| `feature-engineering-op` | SUCCEEDED |
| `predict-op` | SUCCEEDED |
| `write-predictions-op` | SUCCEEDED |

- **101,490 transactions scored** to `fraud_detection.fraud_scores`
- **1,744 predicted fraud** (1.7%), avg fraud probability 0.061

## Step 8: Deploy Infrastructure + CI/CD

Edit `deployment/terraform/vars/env.tfvars` with your values:

```hcl
project_name           = "fraud-detector"
prod_project_id        = "<your-prod-project>"
staging_project_id     = "<your-staging-project>"
cicd_runner_project_id = "<your-dev-project>"
region                 = "us-central1"
repository_owner       = "<your-github-org-or-user>"
repository_name        = "predictive_mlops_demo"
```

Then apply:

```bash
make setup-prod
```

This creates:
- Workload Identity Federation (WIF) pool + OIDC provider for GitHub Actions
- CI/CD runner service account with cross-project permissions
- Pipeline service accounts for staging and prod
- GCS buckets (pipeline root, model artifacts) per environment
- BigQuery datasets per environment
- GitHub Actions secrets (`WIF_POOL_ID`, `WIF_PROVIDER_ID`, `GCP_SERVICE_ACCOUNT`)
- GitHub Actions variables (`GCP_PROJECT_NUMBER`, `STAGING_PROJECT_ID`, `PROD_PROJECT_ID`, etc.)
- GitHub production environment with branch protection

## Step 9: Push to Main

```bash
git add -A && git commit -m "feat: initial fraud detection pipeline"
git push origin main
```

This triggers the CI/CD pipeline:

1. **PR checks** (`pr_checks.yaml`): lint + unit tests on pull requests
2. **Staging deploy** (`staging.yaml`): on merge to main — compiles and submits training + scoring pipelines to staging
3. **Prod deploy** (`deploy-to-prod.yaml`): manual dispatch — deploys to prod + creates pipeline schedules

## Production Checklist

- [x] Data loaded into BigQuery (101K transactions)
- [x] Unit tests passing (15/15)
- [x] Lint passing
- [x] Local pipeline run validated
- [x] Training pipeline on Vertex AI: all 5 steps succeeded
- [x] Model registered in Vertex AI Model Registry (AUC 0.888 >= 0.85 threshold)
- [x] Model Monitoring v2 configured (weekly drift detection)
- [x] Scoring pipeline on Vertex AI: all 3 steps succeeded
- [x] Predictions written to BigQuery (101K rows scored)
- [ ] Update `monitoring.yaml` alert emails from `team@example.com` to real addresses
- [ ] Terraform applied for staging/prod environments
- [ ] CI/CD pipeline triggered via push to main

## Useful Commands

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies |
| `make test-unit` | Run unit tests |
| `make run-training-local` | Run training pipeline locally (KFP) |
| `make run-scoring-local` | Run scoring pipeline locally (KFP) |
| `make submit-training` | Submit training pipeline to Vertex AI |
| `make submit-scoring` | Submit scoring pipeline to Vertex AI |
| `make setup-data` | Load 10K synthetic data into BigQuery |
| `make setup-data-gcs` | Load ~100K GCS sample into BigQuery |
| `make setup-data-full` | Load full dataset into BigQuery |
| `make setup-dev-env` | Deploy dev infrastructure (optional) |
| `make setup-prod` | Deploy staging/prod infrastructure + CI/CD |
| `make notebook` | Launch Jupyter Lab |
| `make lint` | Run linter |
| `make format` | Auto-format code |

## Project Structure

```
fraud_detector/                    # Single source code package
  model.py                         # FraudDetector class (pure ML: features, training, scoring)
  config.py                        # YAML config loader with ${VAR} resolution
  config/                          # YAML pipeline configs
    training.yaml
    scoring.yaml
    monitoring.yaml
  pipelines/
    training_pipeline.py            # KFP: FE -> Train -> Evaluate -> Register -> Monitor
    scoring_pipeline.py             # KFP: FE -> Predict -> Write
    submit_pipeline.py              # CLI: --local, --compile-only, --schedule-only
    components/                     # Individual @dsl.component definitions
      feature_engineering_op.py
      train_op.py
      evaluate_op.py
      register_op.py
      monitoring_op.py

scripts/                           # setup_data.py, test_e2e.py
tests/                             # unit + integration tests
deployment/terraform/              # Multi-project Terraform + GitHub CI/CD
notebooks/                         # Exploratory notebook
```

variable "project_name" {
  description = "The name of the project"
  type        = string
  default     = "fraud-detector"
}

variable "prod_project_id" {
  description = "The GCP project ID for the production environment"
  type        = string
}

variable "staging_project_id" {
  description = "The GCP project ID for the staging environment"
  type        = string
}

variable "cicd_runner_project_id" {
  description = "The GCP project ID for the CICD runner"
  type        = string
}

variable "region" {
  description = "The GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "repository_owner" {
  description = "The GitHub repository owner (organization or user)"
  type        = string
}

variable "repository_name" {
  description = "The GitHub repository name"
  type        = string
}

variable "create_repository" {
  description = "Whether to create the GitHub repository"
  type        = bool
  default     = false
}

variable "app_sa_roles" {
  description = "Roles to assign to the application service account"
  type        = list(string)
  default = [
    "roles/aiplatform.user",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/storage.admin",
    "roles/serviceusage.serviceUsageConsumer",
  ]
}

variable "cicd_roles" {
  description = "Roles to assign to the CICD service account in the CICD project"
  type        = list(string)
  default = [
    "roles/storage.admin",
    "roles/aiplatform.user",
    "roles/artifactregistry.writer",
    "roles/cloudbuild.builds.builder",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/iam.serviceAccountUser",
    "roles/serviceusage.serviceUsageConsumer",
    "roles/logging.viewer",
  ]
}

variable "cicd_sa_deployment_required_roles" {
  description = "Roles to assign to the CICD service account in deployment projects"
  type        = list(string)
  default = [
    "roles/aiplatform.user",
    "roles/storage.admin",
    "roles/iam.serviceAccountUser",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
  ]
}

variable "pipelines_roles" {
  description = "Roles to assign to the pipeline service account in deployment projects"
  type        = list(string)
  default = [
    "roles/storage.admin",
    "roles/aiplatform.user",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/bigquery.readSessionUser",
    "roles/artifactregistry.writer",
  ]
}

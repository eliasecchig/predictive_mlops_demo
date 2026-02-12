"""KFP component -- conditional model registration."""

from kfp import dsl

from fraud_detector.pipelines import pipeline_component


@pipeline_component()
def register_op(
    project_id: str,
    region: str,
    model_display_name: str,
    model: dsl.Input[dsl.Model],
    auc_roc: float,
    threshold_auc: float,
) -> str:
    """Register model to Vertex AI Model Registry if AUC exceeds threshold."""
    import logging

    logger = logging.getLogger(__name__)

    logger.info("-" * 60)
    logger.info("[REG] STEP: Model Registration")
    logger.info("-" * 60)

    if auc_roc < threshold_auc:
        logger.warning("[WARN] AUC %.4f < threshold %.4f -- model NOT registered", auc_roc, threshold_auc)
        return "NOT_REGISTERED"

    # Local runs: model.uri is a local path, skip Vertex registration
    if not model.uri.startswith("gs://"):
        logger.info(
            "[LOCAL] Local run -- skipping Vertex registration (AUC %.4f, model at %s)",
            auc_roc,
            model.uri,
        )
        return "LOCAL_ONLY"

    from google.cloud import aiplatform

    # artifact_uri must be a directory; model.uri points to the file inside it
    artifact_dir = model.uri.rsplit("/", 1)[0]

    aiplatform.init(project=project_id, location=region)
    registered = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_dir,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
        labels={"auc_roc": str(round(auc_roc, 4)).replace(".", "_")},
    )
    logger.info("[OK] Model registered: %s (resource: %s)", model_display_name, registered.resource_name)
    return registered.resource_name

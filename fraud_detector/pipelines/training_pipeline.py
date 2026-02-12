"""KFP Training Pipeline definition."""

from kfp import dsl

from fraud_detector.pipelines.components.data_profile_op import data_profile_op
from fraud_detector.pipelines.components.evaluate_op import evaluate_op
from fraud_detector.pipelines.components.feature_engineering_op import feature_engineering_op
from fraud_detector.pipelines.components.monitoring_op import setup_monitoring_op
from fraud_detector.pipelines.components.register_op import register_op
from fraud_detector.pipelines.components.train_op import train_op


@dsl.pipeline(
    name="fraud-detector-training",
    description="Training pipeline: feature engineering → train → evaluate → conditional register → monitoring",
)
def training_pipeline(
    project_id: str,
    region: str = "us-central1",
    bq_dataset: str = "fraud_detection",
    feature_table: str = "fraud_features",
    model_display_name: str = "fraud-detector-xgb",
    split_date: str = "2023-06-01",
    threshold_auc: float = 0.85,
    read_raw_sql: str = "",
    read_features_sql: str = "",
    max_depth: int = 6,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    scale_pos_weight: float = 10.0,
    alert_emails: str = "",
    default_drift_threshold: float = 0.3,
    predictions_table: str = "fraud_scores",
    monitoring_schedule: str = "0 8 * * 1",
    skip_profiling: bool = False,
):
    # Step 1: Feature engineering
    fe_task = feature_engineering_op(
        project_id=project_id,
        bq_dataset=bq_dataset,
        feature_table=feature_table,
        read_raw_sql=read_raw_sql,
    )

    # Step 2a: Data profiling (runs in parallel with training, skipped locally)
    if not skip_profiling:
        data_profile_op(
            project_id=project_id,
            bq_dataset=bq_dataset,
            feature_table=feature_table,
            split_date=split_date,
            read_features_sql=read_features_sql,
        ).after(fe_task)

    # Step 2b: Train model (outputs dsl.Model artifact, stored by Vertex)
    train_task = train_op(
        project_id=project_id,
        bq_dataset=bq_dataset,
        feature_table=feature_table,
        split_date=split_date,
        read_features_sql=read_features_sql,
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
    ).after(fe_task)

    # Step 3: Evaluate model (receives model artifact from train)
    eval_task = evaluate_op(
        project_id=project_id,
        bq_dataset=bq_dataset,
        feature_table=feature_table,
        split_date=split_date,
        read_features_sql=read_features_sql,
        model=train_task.outputs["model"],
    ).after(train_task)

    # Step 4: Conditional registration (receives model artifact from train)
    register_task = register_op(
        project_id=project_id,
        region=region,
        model_display_name=model_display_name,
        model=train_task.outputs["model"],
        auc_roc=eval_task.outputs["Output"],
        threshold_auc=threshold_auc,
    ).after(eval_task)

    # Step 5: Set up model monitoring (conditional: only if registered)
    setup_monitoring_op(
        project_id=project_id,
        region=region,
        bq_dataset=bq_dataset,
        feature_table=feature_table,
        predictions_table=predictions_table,
        model_resource_name=register_task.output,
        alert_emails=alert_emails,
        default_drift_threshold=default_drift_threshold,
        monitoring_schedule=monitoring_schedule,
    ).after(register_task)

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import package_model_card, tune_threshold_for_f1


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=tune_threshold_for_f1,
                inputs=dict(
                    model="toxicity_model",
                    X_val="X_val",
                    y_val="y_val",
                ),
                outputs="toxicity_threshold",
                name="tune_threshold_for_f1_node",
            ),
            node(
                func=package_model_card,
                inputs=dict(
                    metrics="toxicity_metrics",
                    threshold_info="toxicity_threshold",
                ),
                outputs="toxicity_model_card",
                name="package_model_card_node",
            ),
        ]
    )

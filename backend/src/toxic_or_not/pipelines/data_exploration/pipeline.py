from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import add_length_features, compute_basic_summary, label_distribution


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=compute_basic_summary,
                inputs="toxicity_raw",
                outputs="toxicity_summary",
                name="compute_basic_summary",
            ),
            node(
                func=label_distribution,
                inputs="toxicity_raw",
                outputs="toxicity_label_distribution",
                name="label_distribution",
            ),
            node(
                func=add_length_features,
                inputs="toxicity_raw",
                outputs="toxicity_with_length_features",
                name="add_length_features",
            ),
        ]
    )

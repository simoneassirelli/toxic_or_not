from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_tfidf_logreg


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=dict(
                    df="toxicity_raw",
                    test_size="params:split.test_size",
                    val_size="params:split.val_size",
                    random_state="params:split.random_state",
                    stratify="params:split.stratify",
                ),
                outputs=["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_tfidf_logreg,
                inputs=dict(
                    X_train="X_train",
                    y_train="y_train",
                    tfidf_params="params:tfidf",
                    logreg_params="params:logreg",
                ),
                outputs="toxicity_model",
                name="train_tfidf_logreg_node",
            ),
            node(
                func=evaluate_model,
                inputs=dict(
                    model="toxicity_model",
                    X_val="X_val",
                    y_val="y_val",
                    X_test="X_test",
                    y_test="y_test",
                ),
                outputs="toxicity_metrics",
                name="evaluate_model_node",
            ),
        ]
    )

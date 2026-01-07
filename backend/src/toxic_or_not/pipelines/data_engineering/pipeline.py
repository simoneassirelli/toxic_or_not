from kedro.pipeline import Pipeline, node, pipeline

from .nodes import download_toxicity_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=download_toxicity_dataset,
                inputs=dict(
                    data_dir="params:data_dir",
                    max_rows="params:max_rows",
                ),
                outputs="toxicity_raw",
                name="load_jigsaw_toxicity_csv",
            )
        ]
    )

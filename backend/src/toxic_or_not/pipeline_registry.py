from __future__ import annotations

from kedro.pipeline import Pipeline

from toxic_or_not.pipelines.data_engineering.pipeline import create_pipeline as de
from toxic_or_not.pipelines.data_exploration.pipeline import create_pipeline as dx
from toxic_or_not.pipelines.modeling.pipeline import create_pipeline as mdl
from toxic_or_not.pipelines.inference.pipeline import create_pipeline as inf


def register_pipelines() -> dict[str, Pipeline]:
    data_engineering = de()
    data_exploration = dx()
    modeling = mdl()
    inference = inf()

    return {
        "data_engineering": data_engineering,
        "data_exploration": data_exploration,
        "modeling": modeling,
        "inference": inference,
        "__default__": data_engineering + data_exploration + modeling + inference,
    }

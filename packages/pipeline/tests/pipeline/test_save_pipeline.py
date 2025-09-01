from pyearthtools.pipeline import controller
from pyearthtools.pipeline._save_pipeline import save_pipeline
from pyearthtools.pipeline._save_pipeline import load_pipeline


def test_save_and_load_pipeline():

    p = controller.Pipeline()
    yaml = save_pipeline(p)

    _p2 = load_pipeline(yaml)

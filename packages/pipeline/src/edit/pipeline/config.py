"""Setup config"""

import os

import pyearthtools.utils
import yaml


def reconfigure():
    fn = os.path.join(os.path.dirname(__file__), "pipeline.yaml")
    pyearthtools.utils.config.ensure_file(source=fn)

    with open(fn) as f:
        defaults = yaml.safe_load(f)

    pyearthtools.utils.config.update_defaults(defaults)


reconfigure()

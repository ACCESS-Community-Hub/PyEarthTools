"""Setup config"""

import os

import edit.utils
import yaml


def reconfigure():
    fn = os.path.join(os.path.dirname(__file__), "pipeline.yaml")
    # edit.utils.config.ensure_file(source=fn)

    with open(fn) as f:
        defaults = yaml.safe_load(f)

    edit.utils.config.update_defaults(defaults)


reconfigure()

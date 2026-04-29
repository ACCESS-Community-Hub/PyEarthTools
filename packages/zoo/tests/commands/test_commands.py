from collections import namedtuple
import pytest
from pyearthtools.zoo.commands import _commands as cmd
from unittest.mock import patch

from pyearthtools.zoo.exceptions import ModelException

# def test_entry_point():

#     with pytest.raises(SystemExit):
#         ep = cmd.entry_point(None, None)


def test_models():

    with pytest.raises(SystemExit):
        m = cmd.list_models()


def test_run_predict():

    ctx = namedtuple('Any', ['args', 'kwargs'])([], [])
    model = 'nonexistent'
    time = '2020T010100'
    output = 'tbc'
    pipeline_name = 'tbc'
    data_cache = 'tbc'
    config_path = 'tbc'

    with pytest.raises(ModelException):
        cmd.cmd_run_predict(ctx, model, time, output, pipeline_name, data_cache, config_path)


def test_interactive():

    ctx = namedtuple('Any', ['args', 'kwargs'])([], [])
    model = 'nonexistent'
    time = '2020T010100'
    output = 'tbc'
    pipeline_name = 'tbc'
    data_cache = 'tbc'
    config_path = 'tbc'    

    with pytest.raises(AttributeError):
        with patch('pyearthtools.zoo.available_models', return_value='fake_model'):
            cmd.cmd_interactive(ctx, model, time, pipeline_name, output, data_cache, config_path)     


def test_data():

    ctx = namedtuple('Any', ['args', 'kwargs'])([], [])
    model = 'nonexistent'
    time = '2020T010100'
    output = 'tbc'
    pipeline_name = 'tbc'
    data_cache = 'tbc'
    config_path = 'tbc'

    with pytest.raises(ModelException):
        cmd.cmd_data(ctx, model, time, pipeline_name, data_cache, config_path)
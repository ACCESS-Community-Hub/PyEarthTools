import pytest

from pyearthtools.data.archive import extensions
import pyearthtools.data


def test_register_archive():
    @extensions.register_archive("NewData")
    class NewData:
        def __init__(self, args):
            self.args = args

    # If registering over the top of an existing name, confirm warning occurs
    with pytest.warns(pyearthtools.data.AccessorRegistrationWarning):

        @extensions.register_archive("NewData")
        class NewData:

            _pyearthtools_initialisation = {"class": {}}

            def __init__(self, args):
                self.args = args

            def __call__(self, *args, **kwargs):
                pass

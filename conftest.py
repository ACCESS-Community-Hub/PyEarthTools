import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow-tests", action="store_true", default=False, help="Run slow tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow-tests"):
        skip_slow = pytest.mark.skip(reason="need --run-slow-tests option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

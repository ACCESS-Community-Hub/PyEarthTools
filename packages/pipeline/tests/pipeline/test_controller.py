import pytest

from pyearthtools.pipeline import controller

# def test_PipelineIndex():


def test_Pipeline():
    # TODO: Break this test out into more individual test functions which are more isolated, making
    # the pipelines into fixtures perhaps

    # Test basic creation
    p = controller.Pipeline()

    # Smoke test for making a graph from the pipeline
    _graph = p.graph()

    # Smoke test for adding two pipelines together
    # TODO: test this with non-empty pipelines
    _pp = p + p

    # Smoke test can't access anything in an empty pipeline
    # TODO: test access with a non-empty pipeline
    with pytest.raises(ValueError):
        p[0]

    # Smoke test conversion to "steps"
    # TODO: test with a non-empty pipeline
    steps = p.as_steps()
    assert steps is not None
    _s0 = steps[0]  # Should this be an index error (see consistency below with the .steps method)

    # Test can't access a string-based nonpresent index
    # TODO: test string index with string-indexed values present
    with pytest.raises(ValueError):
        _s0 = steps["first"]

    # Test the "contains" method
    "first" in p

    # Test step-based access
    with pytest.raises(IndexError):
        p.step(0)

    # Test the exception-ignoring capabilities of the pipeline
    p.get_and_catch
    p.exceptions_to_ignore = NotImplementedError
    assert p.exceptions_to_ignore is not None
    p.get_and_catch

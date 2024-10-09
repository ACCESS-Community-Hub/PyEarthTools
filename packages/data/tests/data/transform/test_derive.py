# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import pytest
import math

from edit.data.transforms.derive import EquationException, evaluate


@pytest.mark.parametrize(
    "eq, result",
    [
        ("2 + 7", 9),
        ("2 +7", 9),
        ("2+7", 9),
        ("2 - 7", -5),
        ("2 - 7", -5),
        ("2 + -7", -5),
        ("-2 +5", 3),
        ("2 + (5 * 5)", 27),
        ("2 + 5 * 5", 35),
        ("2 + (5 * (7 - 5))", 12),
        ("2 + 5 * 7 - 5", 44),
        ("sqrt(4)", 2),
        ("sqrt 4 * 4", 8),
        ("sqrt(4 * 4)", 4),
        ("cos(pi) + 2 * 5", 5.0),
        ("cos(pi) + (2 * 5)", 9.0),
        ("cos(pi) * (2 * 5)", -10.0),
        ("cos(sin(pi)) * (2 * 5)", 10.0),
    ],
)
def test_evaluate_only_eq(eq, result):
    assert evaluate(eq) == float(result)


@pytest.mark.parametrize(
    "eq, result",
    [
        ("pi", math.pi),
        ("2 * pi", 2 * math.pi),
    ],
)
def test_constants(eq, result):
    assert evaluate(eq) == float(result)

from pyearthtools.data.derived import insolation


def test_array():

    a = insolation.array([1, 2, 3, 4, 5])
    assert all(a == [1, 2, 3, 4, 5])


def test_Insolation():

    insol = insolation.Insolation(
        [-10],
        [40],
    )

    at_time = ["2021-01-01"]

    result = insol.derive(at_time)
    assert result is not None

    # TODO: Check the insolation calculation
    # TODO: Check the result is the right type/contents

import numpy as np
import pytest

from krippendorff import alpha, bipolar_metric, circular_metric


def test_circular_metric_exact_hand_calculation() -> None:
    data = [
        [0, 1, 2, 3],
        [0, 2, 2, 0],
    ]
    res_str = alpha(data, level_of_measurement="circular")
    assert round(res_str, 6) == 0.5625

    res_metric = alpha(data, level_of_measurement=circular_metric(4))
    assert round(res_metric, 6) == 0.5625

    # Check with default circular_metric() factory
    res_default_factory = alpha(data, level_of_measurement=circular_metric())
    assert round(res_default_factory, 6) == 0.5625


def test_circular_metric_value_counts() -> None:
    # 4 units, 4 categories (0, 1, 2, 3)
    vc = np.array(
        [
            [2, 0, 0, 0],  # unit 0: both assigned 0
            [0, 1, 1, 0],  # unit 1: assigned 1 and 2
            [0, 0, 2, 0],  # unit 2: both assigned 2
            [1, 0, 0, 1],  # unit 3: assigned 0 and 3
        ]
    )
    res = alpha(value_counts=vc, level_of_measurement="circular")
    assert round(res, 6) == 0.5625


def test_circular_metric_degrees() -> None:
    # Compass directions in degrees: 0, 90, 180, 270
    # Two coders:
    # unit 0: 0 and 90 -> diff 90 deg -> sin^2(pi*90/360) = 0.5
    # unit 1: 0 and 270 -> diff 270 deg (wraps around to 90 deg) -> sin^2(pi*270/360) = 0.5
    # unit 2: 180 and 180 -> diff 0 deg -> 0
    data = [
        [0, 0, 180],
        [90, 270, 180],
    ]
    metric_360 = circular_metric(circumference=360)
    res = alpha(data, level_of_measurement=metric_360)
    assert np.isfinite(res)
    assert res < 1.0


def test_circular_metric_clock_hours() -> None:
    # 12-hour clock: 1 to 12
    # Coder A and B agree closely: hour 12 vs 1 (diff 1 hour) and hour 1 vs 2 (diff 1 hour)
    data = [
        [12, 1, 6],
        [1, 2, 6],
    ]
    # Default circumference for 1..12 is 12 - 1 + 1 = 12
    res = alpha(data, level_of_measurement="circular")
    assert np.isfinite(res)


def test_bipolar_metric_exact_hand_calculation() -> None:
    data = [
        [-1, 0, 1],
        [-1, 1, 1],
    ]
    res_str = alpha(data, level_of_measurement="bipolar")
    assert round(res_str, 6) == round(18 / 23, 6)

    res_metric = alpha(data, level_of_measurement=bipolar_metric(-1, 1))
    assert round(res_metric, 6) == round(18 / 23, 6)

    res_default_factory = alpha(data, level_of_measurement=bipolar_metric())
    assert round(res_default_factory, 6) == round(18 / 23, 6)


def test_bipolar_metric_value_counts() -> None:
    # 3 units, values -1, 0, 1 mapped to columns 0, 1, 2
    vc = np.array(
        [
            [2, 0, 0],
            [0, 1, 1],
            [0, 0, 2],
        ]
    )
    res = alpha(value_counts=vc, value_domain=[-1, 0, 1], level_of_measurement="bipolar")
    assert round(res, 6) == round(18 / 23, 6)


def test_bipolar_metric_custom_bounds() -> None:
    # Data only contains values 2 and 3, but the scale is from 1 to 5
    data = [
        [2, 3],
        [3, 3],
    ]
    # Using explicit low and high
    metric_1_to_5 = bipolar_metric(low=1, high=5)
    res_custom = alpha(data, level_of_measurement=metric_1_to_5)

    # Using value_domain
    res_domain = alpha(data, value_domain=[1, 2, 3, 4, 5], level_of_measurement="bipolar")
    assert round(res_custom, 6) == round(res_domain, 6)


def test_circular_bipolar_dict_input() -> None:
    data = [
        {"u1": 0, "u2": 1, "u3": 2, "u4": 3},
        {"u1": 0, "u2": 2, "u3": 2, "u4": 0},
    ]
    res_circ = alpha(data, level_of_measurement="circular")
    assert round(res_circ, 6) == 0.5625

    bipolar_data = [
        {"u1": -1, "u2": 0, "u3": 1},
        {"u1": -1, "u2": 1, "u3": 1},
    ]
    res_bip = alpha(bipolar_data, level_of_measurement="bipolar")
    assert round(res_bip, 6) == round(18 / 23, 6)


def test_circular_bipolar_subsamples() -> None:
    full_data = [
        [0, 1, 2, 3, 0, 1],
        [0, 2, 2, 0, 0, 2],
    ]
    subsample = [
        [0, 1, 2, 3],
        [0, 2, 2, 0],
    ]
    res_circ = alpha(subsample, level_of_measurement="circular", all_reliability_data=full_data)
    assert np.isfinite(res_circ)

    bipolar_full = [
        [-1, 0, 1, -1, 0],
        [-1, 1, 1, -1, 1],
    ]
    bipolar_sub = [
        [-1, 0, 1],
        [-1, 1, 1],
    ]
    res_bip = alpha(bipolar_sub, level_of_measurement="bipolar", all_reliability_data=bipolar_full)
    assert np.isfinite(res_bip)


def test_invalid_parameters_and_data() -> None:
    with pytest.raises(ValueError, match="finite, positive number"):
        circular_metric(circumference=0)

    with pytest.raises(ValueError, match="finite, positive number"):
        circular_metric(circumference=-10)

    with pytest.raises(ValueError, match="low must be strictly less than high"):
        bipolar_metric(low=5, high=5)

    with pytest.raises(ValueError, match="low must be strictly less than high"):
        bipolar_metric(low=5, high=2)

    # Values exceeding custom low/high in bipolar_metric
    data = [[0, 10], [0, 10]]
    metric = bipolar_metric(low=0, high=5)
    with pytest.raises(ValueError, match="out-of-bounds"):
        alpha(data, level_of_measurement=metric)

    # Complex values with circular / bipolar
    complex_dict_data = [
        {"u1": 1 + 2j, "u2": 3 + 4j},
        {"u1": 1 + 2j, "u2": 3 + 4j},
    ]
    complex_domain = [1 + 2j, 3 + 4j]
    with pytest.raises(ValueError, match="does not support complex"):
        alpha(reliability_data=complex_dict_data, value_domain=complex_domain, level_of_measurement="circular")

    with pytest.raises(ValueError, match="does not support complex"):
        alpha(reliability_data=complex_dict_data, value_domain=complex_domain, level_of_measurement="bipolar")

    # low greater than max of data
    metric_bip_low = bipolar_metric(low=10)
    with pytest.raises(ValueError, match="low must be strictly less than high"):
        alpha([[1, 2], [1, 2]], level_of_measurement=metric_bip_low)

    # Non-integer circular data without explicit circumference
    float_data = [[0.5, 1.2], [0.5, 1.2]]
    with pytest.raises(ValueError, match="explicit circumference must be provided"):
        alpha(float_data, level_of_measurement="circular")

    # Non-finite circumference, low, high
    with pytest.raises(ValueError, match="finite, positive number"):
        circular_metric(circumference=float("inf"))

    with pytest.raises(ValueError, match="low must be a finite number"):
        bipolar_metric(low=float("nan"))

    with pytest.raises(ValueError, match="high must be a finite number"):
        bipolar_metric(high=float("inf"))

    # Direct calls to metric callable with complex or non-finite inputs
    circ_fn = circular_metric(4)
    bip_fn = bipolar_metric(-1, 1)
    dummy_idx = np.array([0])
    dummy_nv = np.array([1.0])

    with pytest.raises(ValueError, match="does not support complex"):
        circ_fn(np.array([1 + 2j]), np.array([1]), dummy_idx, dummy_idx, dummy_nv)

    with pytest.raises(ValueError, match="requires finite values"):
        circ_fn(np.array([np.nan]), np.array([1]), dummy_idx, dummy_idx, dummy_nv)

    with pytest.raises(ValueError, match="does not support complex"):
        bip_fn(np.array([1 + 2j]), np.array([1]), dummy_idx, dummy_idx, dummy_nv)

    with pytest.raises(ValueError, match="requires finite values"):
        bip_fn(np.array([np.nan]), np.array([1]), dummy_idx, dummy_idx, dummy_nv)

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


def test_circular_and_bipolar_symmetry_across_differing_domains() -> None:
    circ_fn = circular_metric()
    bip_fn = bipolar_metric()
    dummy_idx = np.array([0, 0])
    dummy_nv = np.array([1.0, 1.0])

    a = np.array([0, 1])
    b = np.array([1, 3])

    circ_ab = circ_fn(a, b, dummy_idx, dummy_idx, dummy_nv)
    circ_ba = circ_fn(b, a, dummy_idx, dummy_idx, dummy_nv)
    assert np.allclose(circ_ab, circ_ba)

    bip_ab = bip_fn(a, b, dummy_idx, dummy_idx, dummy_nv)
    bip_ba = bip_fn(b, a, dummy_idx, dummy_idx, dummy_nv)
    assert np.allclose(bip_ab, bip_ba)


def test_circular_metric_uint_subtraction_overflow() -> None:
    circ_fn = circular_metric(circumference=360)
    dummy_idx = np.array([0])
    dummy_nv = np.array([1.0])

    u1 = np.array([0], dtype=np.uint8)
    u2 = np.array([5], dtype=np.uint8)

    d1 = circ_fn(u1, u2, dummy_idx, dummy_idx, dummy_nv)
    d2 = circ_fn(u2, u1, dummy_idx, dummy_idx, dummy_nv)
    assert np.allclose(d1, d2)
    assert np.allclose(d1, np.sin(np.pi * 5 / 360) ** 2)


def test_object_dtype_complex_values_rejected() -> None:
    circ_fn = circular_metric(4)
    bip_fn = bipolar_metric(-1, 1)
    dummy_idx = np.array([0])
    dummy_nv = np.array([1.0])

    obj_complex = np.array([1 + 2j], dtype=object)
    obj_real = np.array([1.0], dtype=object)

    with pytest.raises(ValueError, match="does not support complex"):
        circ_fn(obj_complex, obj_real, dummy_idx, dummy_idx, dummy_nv)

    with pytest.raises(ValueError, match="does not support complex"):
        bip_fn(obj_complex, obj_real, dummy_idx, dummy_idx, dummy_nv)

    data = [{"u1": 1 + 2j, "u2": 2}, {"u1": 1 + 2j, "u2": 2}]
    complex_domain = [1 + 2j, 2]
    with pytest.raises(ValueError, match="does not support complex"):
        alpha(reliability_data=data, value_domain=complex_domain, level_of_measurement=circ_fn)

    with pytest.raises(ValueError, match="does not support complex"):
        alpha(reliability_data=data, value_domain=complex_domain, level_of_measurement=bip_fn)


def test_numeric_object_dtype_domain_and_operands_supported() -> None:
    data_circ = [
        [0, 1, 2, 3],
        [0, 2, 2, 0],
    ]
    domain_circ = np.array([0, 1, 2, 3], dtype=object)
    res_circ_str = alpha(data_circ, level_of_measurement="circular", value_domain=domain_circ)
    res_circ_fn = alpha(data_circ, level_of_measurement=circular_metric(), value_domain=domain_circ)
    assert round(res_circ_str, 6) == 0.5625
    assert round(res_circ_fn, 6) == 0.5625

    data_bip = [
        [-1, 0, 1],
        [-1, 1, 1],
    ]
    domain_bip = np.array([-1, 0, 1], dtype=object)
    res_bip_str = alpha(data_bip, level_of_measurement="bipolar", value_domain=domain_bip)
    res_bip_fn = alpha(data_bip, level_of_measurement=bipolar_metric(), value_domain=domain_bip)
    assert np.isfinite(res_bip_str)
    assert np.isfinite(res_bip_fn)
    assert res_bip_str == res_bip_fn

    # Direct metric invocation with object arrays
    circ_fn = circular_metric()
    bip_fn = bipolar_metric()
    dummy_idx = np.array([0])
    dummy_nv = np.array([1.0])
    o1 = np.array([0, 2], dtype=object)
    o2 = np.array([1, 2], dtype=object)
    d_circ = circ_fn(o1, o2, dummy_idx, dummy_idx, dummy_nv)
    assert np.all(np.isfinite(d_circ))
    d_bip = bip_fn(o1, o2, dummy_idx, dummy_idx, dummy_nv)
    assert np.all(np.isfinite(d_bip))


def test_non_finite_object_dtype_rejected() -> None:
    circ_fn = circular_metric(4)
    bip_fn = bipolar_metric(-1, 1)
    dummy_idx = np.array([0])
    dummy_nv = np.array([1.0])

    for invalid_val in [np.nan, np.inf, -np.inf, "not-a-number", None]:
        obj_arr = np.array([invalid_val], dtype=object)
        valid_arr = np.array([1.0], dtype=object)

        with pytest.raises(ValueError, match="Circular metric requires finite values"):
            circ_fn(obj_arr, valid_arr, dummy_idx, dummy_idx, dummy_nv)

        with pytest.raises(ValueError, match="Bipolar metric requires finite values"):
            bip_fn(obj_arr, valid_arr, dummy_idx, dummy_idx, dummy_nv)


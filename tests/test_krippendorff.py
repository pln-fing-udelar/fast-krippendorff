import re

import numpy as np
import numpy.typing as npt
import pytest

import krippendorff
from krippendorff import DEFAULT_DTYPE, alpha


def test_nominal_metric_reliability_data() -> None:
    data = [
        [np.nan, np.nan, np.nan, np.nan, np.nan, 3, 4, 1, 2, 1, 1, 3, 3, np.nan, 3],
        [1, np.nan, 2, 1, 3, 3, 4, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, 2, 1, 3, 4, 4, np.nan, 2, 1, 1, 3, 3, np.nan, 4],
    ]
    res = alpha(reliability_data=data, level_of_measurement="nominal")
    assert round(res, 6) == 0.691358


def test_nominal_metric_value_counts() -> None:
    value_counts = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 2, 1],
            [0, 0, 0, 3],
            [1, 0, 1, 0],
            [0, 2, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
        ]
    )
    res = alpha(value_counts=value_counts, level_of_measurement="nominal")
    assert round(res, 6) == 0.691358


def test_ordinal_metric() -> None:
    data = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
        [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan],
    ]
    res = alpha(data, level_of_measurement="ordinal")
    assert round(res, 3) == 0.815


def test_interval_metric() -> None:
    data = [
        [np.nan, np.nan, np.nan, np.nan, np.nan, 3, 4, 1, 2, 1, 1, 3, 3, np.nan, 3],
        [1, np.nan, 2, 1, 3, 3, 4, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, 2, 1, 3, 4, 4, np.nan, 2, 1, 1, 3, 3, np.nan, 4],
    ]
    res = alpha(reliability_data=data, level_of_measurement="interval")
    assert round(res, 6) == 0.810845


def test_ratio_metric() -> None:
    data = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
        [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan],
    ]
    res = alpha(data, level_of_measurement="ratio")
    assert round(res, 3) == 0.797


def test_ratio_metric_with_zeros() -> None:
    data = [
        [0, 1, 2],
        [0, 1, 2],
    ]
    res = alpha(data, level_of_measurement="ratio")
    assert res == 1.0


def test_string_labels_nominal() -> None:
    data = [
        ["pos", "neg", "neu", np.nan],
        ["pos", "neg", "neu", "pos"],
        ["pos", "neg", "pos", "pos"],
    ]
    res = alpha(data, level_of_measurement="nominal")
    assert isinstance(res, float)
    assert 0.0 < res < 1.0


def test_string_labels_ordinal_with_domain() -> None:
    data = [
        ["very low", "low", "mid", "mid", "low", "very low", "high", "very low", "low", np.nan, np.nan, np.nan],
        ["very low", "low", "mid", "mid", "low", "low", "high", "very low", "low", "very high", np.nan, "mid"],
        [np.nan, "mid", "mid", "mid", "low", "mid", "high", "low", "low", "very high", "very low", np.nan],
        ["very low", "low", "mid", "mid", "low", "high", "high", "very low", "low", "very high", "very low", np.nan],
    ]
    domain = ["very low", "low", "mid", "high", "very high"]
    res = alpha(data, level_of_measurement="ordinal", value_domain=domain)
    assert round(res, 3) == 0.815


def test_custom_distance_metric() -> None:
    def absolute_distance(
        v1: npt.NDArray[np.generic],
        v2: npt.NDArray[np.generic],
        i1: npt.NDArray[np.int_],
        i2: npt.NDArray[np.int_],
        n_v: npt.NDArray[np.inexact],
        dtype: np.dtype[np.inexact] = DEFAULT_DTYPE,  # ty:ignore[invalid-parameter-default]
    ) -> npt.NDArray[np.inexact]:
        return np.abs(v1.astype(float) - v2.astype(float)).astype(dtype)

    data = [
        [1, 2, 3],
        [1, 2, 4],
    ]
    res = alpha(data, level_of_measurement=absolute_distance)
    assert isinstance(res, float)
    assert 0.0 < res < 1.0


def test_perfect_agreement() -> None:
    data = [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ]
    assert alpha(data, level_of_measurement="nominal") == 1.0
    assert alpha(data, level_of_measurement="interval") == 1.0


def test_float_dtype() -> None:
    data = [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]
    res_f32 = alpha(data, dtype=np.float32)
    res_f64 = alpha(data, dtype=np.float64)
    assert res_f32 == 1.0
    assert res_f64 == 1.0


def test_error_both_reliability_and_value_counts() -> None:
    with pytest.raises(
        ValueError, match=re.escape("Either reliability_data or value_counts must be provided, but not both.")
    ):
        alpha(reliability_data=[[1, 2], [1, 2]], value_counts=[[2, 0], [0, 2]])


def test_error_neither_reliability_nor_value_counts() -> None:
    with pytest.raises(
        ValueError, match=re.escape("Either reliability_data or value_counts must be provided, but not both.")
    ):
        alpha()


def test_error_single_value_in_domain() -> None:
    with pytest.raises(ValueError, match=re.escape("There has to be more than one value in the domain.")):
        alpha(reliability_data=[[1, 1], [1, 1]])


def test_error_insufficient_coders_per_unit() -> None:
    data = [
        [1, np.nan],
        [np.nan, 2],
    ]
    with pytest.raises(
        ValueError,
        match=re.escape("There has to be at least one unit with values assigned by at least two coders."),
    ):
        alpha(reliability_data=data)


def test_error_invalid_dtype() -> None:
    with pytest.raises(ValueError, match=re.escape("`dtype` must be an inexact type.")):
        alpha(reliability_data=[[1, 2], [1, 2]], dtype=int)


def test_error_out_of_domain() -> None:
    data = [
        [1, 2],
        [1, 3],
    ]
    with pytest.raises(ValueError, match=re.escape("The reliability data contains out-of-domain values.")):
        alpha(reliability_data=data, value_domain=[1, 2])


def test_error_mismatched_value_counts_and_domain() -> None:
    counts = [
        [2, 0, 0],
        [0, 2, 0],
    ]
    with pytest.raises(
        ValueError,
        match=re.escape("The value domain should be equal to the number of columns of value_counts."),
    ):
        alpha(value_counts=counts, value_domain=[0, 1])


def test_error_strings_without_ordered_domain_non_nominal() -> None:
    data = [
        ["low", "high"],
        ["low", "high"],
    ]
    with pytest.raises(ValueError, match="When using strings, an ordered value_domain is required"):
        alpha(data, level_of_measurement="ordinal")


def test_error_unsupported_dtype_kind() -> None:
    data = [
        [object(), object()],
        [object(), object()],
    ]
    with pytest.raises(ValueError, match="Don't know how to construct value domain for dtype kind"):
        alpha(data)  # ty:ignore[invalid-argument-type]


def test_package_exports() -> None:
    for name in [
        "DEFAULT_DTYPE",
        "DistanceMetric",
        "LevelOfMeasurement",
        "MetricResultScalarType",
        "ValueScalarType",
        "alpha",
    ]:
        assert hasattr(krippendorff, name)

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import krippendorff


def test_dict_input_sequence_of_dicts_matches_dense_matrix() -> None:
    data_matrix = [
        [np.nan, np.nan, 3, 4, 1],
        [1, np.nan, 3, 4, 3],
        [np.nan, 2, 3, 4, 2],
    ]
    data_dicts = [
        {2: 3, 3: 4, 4: 1},
        {0: 1, 2: 3, 3: 4, 4: 3},
        {1: 2, 2: 3, 3: 4, 4: 2},
    ]

    for level in ("nominal", "ordinal", "interval", "ratio"):
        alpha_matrix = krippendorff.alpha(reliability_data=data_matrix, level_of_measurement=level)
        alpha_dicts = krippendorff.alpha(reliability_data=data_dicts, level_of_measurement=level)
        assert np.isclose(alpha_matrix, alpha_dicts)


def test_dict_input_dict_of_dicts() -> None:
    data_dict_of_dicts = {
        "coder1": {"unitA": 1, "unitB": 2, "unitC": 3},
        "coder2": {"unitA": 1, "unitB": 2, "unitC": 4},
        "coder3": {"unitB": 2, "unitC": 3},
    }
    # Equivalent to a matrix with coders 1, 2, 3 and units A, B, C
    matrix = [
        [1, 2, 3],
        [1, 2, 4],
        [np.nan, 2, 3],
    ]

    alpha_matrix = krippendorff.alpha(reliability_data=matrix, level_of_measurement="interval")
    alpha_dicts = krippendorff.alpha(reliability_data=data_dict_of_dicts, level_of_measurement="interval")
    assert np.isclose(alpha_matrix, alpha_dicts)


def test_dict_input_with_explicit_value_domain() -> None:
    data = [
        {"u1": 1, "u2": 2},
        {"u1": 1, "u2": 3},
    ]
    domain = [1, 2, 3, 4, 5]
    alpha_val = krippendorff.alpha(reliability_data=data, value_domain=domain, level_of_measurement="ordinal")
    assert isinstance(alpha_val, float)
    assert not np.isnan(alpha_val)


def test_dict_input_with_none_and_nan_values() -> None:
    data_with_missing = [
        {"u1": 1, "u2": None, "u3": 3},
        {"u1": 1, "u2": 2, "u3": np.nan},
        {"u1": np.nan, "u2": 2, "u3": 3},
    ]
    data_clean = [
        {"u1": 1, "u3": 3},
        {"u1": 1, "u2": 2},
        {"u2": 2, "u3": 3},
    ]
    alpha_with_missing = krippendorff.alpha(reliability_data=data_with_missing, level_of_measurement="nominal")
    alpha_clean = krippendorff.alpha(reliability_data=data_clean, level_of_measurement="nominal")
    assert np.isclose(alpha_with_missing, alpha_clean)


def test_dict_input_string_labels() -> None:
    data = {
        "c1": {"doc1": "pos", "doc2": "neg"},
        "c2": {"doc1": "pos", "doc2": "pos"},
    }
    alpha_val = krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")
    assert np.isclose(alpha_val, 0.0)


def test_dict_input_errors() -> None:
    # Insufficient pairable units
    data_single = [{"u1": 1}, {"u2": 2}]
    with pytest.raises(ValueError, match="at least one unit with values assigned by at least two coders"):
        krippendorff.alpha(reliability_data=data_single)

    # Only 1 unique value
    data_constant = [{"u1": 1, "u2": 1}, {"u1": 1, "u2": 1}]
    with pytest.raises(ValueError, match="more than one value in the domain"):
        krippendorff.alpha(reliability_data=data_constant)

    # Out of domain
    data_out = [{"u1": 1, "u2": 2}, {"u1": 1, "u2": 99}]
    with pytest.raises(ValueError, match="out-of-domain"):
        krippendorff.alpha(reliability_data=data_out, value_domain=[1, 2])

    # String dict without domain when level != nominal
    with pytest.raises(ValueError, match="ordered value_domain is required"):
        krippendorff.alpha(
            reliability_data=[{"u1": "a", "u2": "b"}, {"u1": "a", "u2": "b"}],
            level_of_measurement="ordinal",
        )


def test_standard_array_and_value_counts() -> None:
    # Test array path
    arr = np.array([[1, 2], [1, 2]])
    res_arr = krippendorff.alpha(reliability_data=arr, level_of_measurement="nominal")
    assert np.isclose(res_arr, 1.0)

    # Test value counts path
    vc = np.array([[2, 0], [0, 2]])
    res_vc = krippendorff.alpha(value_counts=vc, level_of_measurement="nominal")
    assert np.isclose(res_vc, 1.0)

    # Both provided raises error
    with pytest.raises(ValueError, match="Either reliability_data or value_counts"):
        krippendorff.alpha(reliability_data=arr, value_counts=vc)

    # String array without domain when level != nominal
    with pytest.raises(ValueError, match="ordered value_domain is required"):
        krippendorff.alpha(reliability_data=[["a", "b"], ["a", "b"]], level_of_measurement="ordinal")

    # String array with nominal
    res_str = krippendorff.alpha(reliability_data=[["a", "b"], ["a", "b"]], level_of_measurement="nominal")
    assert np.isclose(res_str, 1.0)

    # Array out of domain
    with pytest.raises(ValueError, match="out-of-domain"):
        krippendorff.alpha(reliability_data=arr, value_domain=[1])

    # Value counts column mismatch
    with pytest.raises(ValueError, match="number of columns of value_counts"):
        krippendorff.alpha(value_counts=vc, value_domain=[1, 2, 3])

    # Non-inexact dtype
    with pytest.raises(ValueError, match="must be an inexact type"):
        krippendorff.alpha(reliability_data=arr, dtype=int)


def test_dict_input_user_list_sequence() -> None:
    from collections import UserList

    data = UserList(
        [
            {"u1": 1, "u2": 2},
            {"u1": 1, "u2": 2},
        ]
    )
    res = krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")
    assert np.isclose(res, 1.0)


def test_dict_input_literal_nan_string() -> None:
    data = [
        {"u1": "nan", "u2": "val"},
        {"u1": "nan", "u2": "val"},
    ]
    res = krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")
    assert np.isclose(res, 1.0)


def test_dict_input_mixed_nominal_labels() -> None:
    data = [
        {"u1": 1, "u2": "a"},
        {"u1": 1, "u2": "a"},
    ]
    res = krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")
    assert np.isclose(res, 1.0)


def test_dict_input_malformed_flat_dict() -> None:
    flat_data: Any = {"u1": 1, "u2": 2}
    with pytest.raises(ValueError, match="Expected a mapping of coders to unit annotations"):
        krippendorff.alpha(reliability_data=flat_data, level_of_measurement="nominal")


def test_dict_input_mixed_numeric_types_interval() -> None:
    data = [
        {"u1": 1, "u2": 2.5},
        {"u1": 1.0, "u2": 2.5},
    ]
    res = krippendorff.alpha(reliability_data=data, level_of_measurement="interval")
    assert np.isclose(res, 1.0)


def test_dict_input_explicit_mixed_value_domain_nominal() -> None:
    data = [
        {"u1": 1, "u2": "a"},
        {"u1": 1, "u2": "a"},
    ]
    res = krippendorff.alpha(reliability_data=data, value_domain=[1, "a", 2], level_of_measurement="nominal")
    assert np.isclose(res, 1.0)


def test_dict_input_sequence_with_non_mapping_element() -> None:
    data: Any = [{"u1": 1}, [1, 2]]
    with pytest.raises(ValueError, match="Expected a sequence of mappings for coder annotations, but element 1"):
        krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")


def test_dict_input_tuple_labels() -> None:
    data = [
        {"u1": (1, 2), "u2": (3, 4)},
        {"u1": (1, 2), "u2": (3, 4)},
    ]
    res = krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")
    assert np.isclose(res, 1.0)


def test_dict_input_bytes_and_str_labels() -> None:
    data = [
        {"u1": b"a", "u2": "a"},
        {"u1": b"a", "u2": "a"},
    ]
    res = krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")
    assert np.isclose(res, 1.0)


def test_dict_input_large_int_and_float_labels() -> None:
    v1 = 2**53 + 1
    v2 = float(2**53)
    data = [
        {"u1": v1, "u2": v2},
        {"u1": v1, "u2": v2},
    ]
    res = krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")
    assert np.isclose(res, 1.0)


def test_value_counts_tuple_value_domain() -> None:
    vc = np.array([[2, 0], [0, 2]])
    res = krippendorff.alpha(value_counts=vc, value_domain=[(1, 2), (3, 4)], level_of_measurement="nominal")
    assert np.isclose(res, 1.0)


def test_dict_input_complex_labels_nominal() -> None:
    data = [
        {"u1": 1 + 2j, "u2": 3 + 4j},
        {"u1": 1 + 2j, "u2": 3 + 4j},
    ]
    res = krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")
    assert np.isclose(res, 1.0)


def test_dict_input_complex_labels_interval_rejected() -> None:
    data = [
        {"u1": 1 + 2j, "u2": 3 + 4j},
        {"u1": 1 + 2j, "u2": 3 + 4j},
    ]
    with pytest.raises(ValueError, match="ordered value_domain is required"):
        krippendorff.alpha(reliability_data=data, level_of_measurement="interval")


def test_dict_input_numpy_complex_labels_interval_rejected() -> None:
    data = [
        {"u1": np.complex128(1 + 2j), "u2": np.complex128(3 + 4j)},
        {"u1": np.complex128(1 + 2j), "u2": np.complex128(3 + 4j)},
    ]
    with pytest.raises(ValueError, match="ordered value_domain is required"):
        krippendorff.alpha(reliability_data=data, level_of_measurement="interval")

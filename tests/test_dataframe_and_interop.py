from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

import krippendorff
from krippendorff import (
    AlphaResult,
    BootstrapResult,
    alpha,
    alpha_table,
    bootstrap_ci,
    bootstrap_ci_table,
    table_to_value_counts,
)


class MockTorchTensor:
    """Mock PyTorch tensor supporting .detach(), .cpu(), and .numpy()."""

    def __init__(self, data: Any, requires_grad: bool = False, device: str = "cpu") -> None:
        self._data = np.asarray(data)
        self.requires_grad = requires_grad
        self.device = device
        self.detached = False
        self.cpu_called = False

    def detach(self) -> MockTorchTensor:
        res = MockTorchTensor(self._data.copy(), requires_grad=False, device=self.device)
        res.detached = True
        res.cpu_called = self.cpu_called
        return res

    def cpu(self) -> MockTorchTensor:
        res = MockTorchTensor(self._data.copy(), requires_grad=self.requires_grad, device="cpu")
        res.detached = self.detached
        res.cpu_called = True
        return res

    def numpy(self) -> np.ndarray:
        if self.requires_grad:
            raise RuntimeError("Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.")
        if self.device != "cpu":
            raise RuntimeError("Can't call numpy() on CUDA tensor. Use var.cpu().numpy() instead.")
        return self._data


class MockJAXArray:
    """Mock JAX / Array API object supporting __array__."""

    def __init__(self, data: Any) -> None:
        self._data = np.asarray(data)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self._data, dtype=dtype)


class MockArrayInterface:
    """Mock object implementing __array_interface__."""

    def __init__(self, data: Any) -> None:
        self._arr = np.asarray(data)

    @property
    def __array_interface__(self) -> dict[str, Any]:
        return self._arr.__array_interface__


class MockPolarsArray:
    """Mock object implementing .to_numpy()."""

    def __init__(self, data: Any) -> None:
        self._arr = np.asarray(data)

    def to_numpy(self) -> np.ndarray:
        return self._arr


class MockSeries:
    """Mock pandas Series supporting iteration, indexing, and .to_numpy()."""

    def __init__(self, values: list[Any], name: Any = None) -> None:
        self._values = list(values)
        self.name = name

    def __iter__(self) -> Any:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, idx: Any) -> Any:
        return self._values[idx]

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self._values)

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self._values)


class MockDataFrame:
    """Mock pandas/polars DataFrame supporting columns, indexing, and .to_numpy()."""

    def __init__(self, data: dict[str, list[Any]]) -> None:
        self._data = {k: list(v) for k, v in data.items()}
        self.columns = list(data.keys())

    def __getitem__(self, key: str) -> MockSeries:
        return MockSeries(self._data[key], name=key)

    def to_numpy(self) -> np.ndarray:
        if not self.columns:
            return np.empty((0, 0))
        return np.array([self._data[k] for k in self.columns]).T

    @property
    def values(self) -> np.ndarray:
        return self.to_numpy()

    def __len__(self) -> int:
        if not self.columns:
            return 0
        return len(self._data[self.columns[0]])


class MockDataFrameWithValues:
    """Mock DataFrame with non-callable .values attribute and no to_numpy method."""

    def __init__(self, data: dict[str, list[Any]]) -> None:
        self.columns = list(data.keys())
        self.values = np.array([data[k] for k in self.columns]).T


class MockNA:
    """Simulates pandas.NA or pandas.NaT."""

    def __repr__(self) -> str:
        return "<NA>"


class NAType:
    """Simulates pd.NA type."""

    def __repr__(self) -> str:
        return "<NAType>"


class NaTType:
    """Simulates pd.NaT type."""

    def __repr__(self) -> str:
        return "<NaTType>"


# ---------------------------------------------------------------------------
# Framework Interoperability (PyTorch, JAX, Polars)
# ---------------------------------------------------------------------------


def test_torch_tensor_cpu_and_cuda_and_grad() -> None:
    data = [
        [1.0, 2.0, np.nan, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [np.nan, 2.0, 3.0, 5.0],
    ]
    expected_alpha = alpha(data, level_of_measurement="interval")

    # Standard CPU tensor without grad
    t_cpu = MockTorchTensor(data)
    assert np.isclose(alpha(t_cpu, level_of_measurement="interval"), expected_alpha)

    # Tensor requiring grad: ensures .detach() is called
    t_grad = MockTorchTensor(data, requires_grad=True)
    assert np.isclose(alpha(t_grad, level_of_measurement="interval"), expected_alpha)

    # CUDA tensor requiring grad: ensures .detach().cpu() is called
    t_cuda = MockTorchTensor(data, requires_grad=True, device="cuda:0")
    assert np.isclose(alpha(t_cuda, level_of_measurement="interval"), expected_alpha)

    # bootstrap_ci with CUDA tensor requiring grad
    res_np = bootstrap_ci(data, level_of_measurement="interval", random_state=42)
    res_cuda = bootstrap_ci(t_cuda, level_of_measurement="interval", random_state=42)
    assert np.isclose(res_cuda.alpha, res_np.alpha)
    assert np.isclose(res_cuda.standard_error, res_np.standard_error)


def test_jax_and_array_interface_interop() -> None:
    data = [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 4.0],
    ]
    expected = alpha(data, level_of_measurement="interval")

    # __array__ protocol
    jax_arr = MockJAXArray(data)
    assert np.isclose(alpha(jax_arr, level_of_measurement="interval"), expected)

    # __array_interface__ protocol
    iface_arr = MockArrayInterface(data)
    assert np.isclose(alpha(iface_arr, level_of_measurement="interval"), expected)

    # .to_numpy() method (Polars)
    polars_arr = MockPolarsArray(data)
    assert np.isclose(alpha(polars_arr, level_of_measurement="interval"), expected)

    # Non-callable .values attribute
    df_values = MockDataFrameWithValues({"u1": [1.0, 1.0], "u2": [2.0, 2.0], "u3": [3.0, 4.0]})
    assert np.isclose(alpha(df_values, level_of_measurement="interval"), expected)


# ---------------------------------------------------------------------------
# Wide DataFrames and coders_axis
# ---------------------------------------------------------------------------


def test_dataframe_wide_coders_axis() -> None:
    # 2 coders, 3 units: rows are coders, columns are units
    matrix_coders_as_rows = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 4.0],
        ]
    )
    expected_alpha = alpha(matrix_coders_as_rows, level_of_measurement="interval")

    # Wide DataFrame where rows are coders, columns are units
    df_rows = MockDataFrame(
        {
            "unit1": [1.0, 1.0],
            "unit2": [2.0, 2.0],
            "unit3": [3.0, 4.0],
        }
    )
    assert np.isclose(alpha(df_rows, coders_axis="rows", level_of_measurement="interval"), expected_alpha)
    assert np.isclose(alpha(df_rows, coders_axis=0, level_of_measurement="interval"), expected_alpha)
    assert np.isclose(alpha(df_rows, coders_axis="row", level_of_measurement="interval"), expected_alpha)

    # Wide DataFrame where columns are coders, rows are units
    df_cols = MockDataFrame(
        {
            "coder1": [1.0, 2.0, 3.0],
            "coder2": [1.0, 2.0, 4.0],
        }
    )
    assert np.isclose(alpha(df_cols, coders_axis="columns", level_of_measurement="interval"), expected_alpha)
    assert np.isclose(alpha(df_cols, coders_axis=1, level_of_measurement="interval"), expected_alpha)
    assert np.isclose(alpha(df_cols, coders_axis="col", level_of_measurement="interval"), expected_alpha)
    assert np.isclose(alpha(df_cols, coders_axis="cols", level_of_measurement="interval"), expected_alpha)

    # bootstrap_ci with coders_axis="columns"
    ci_rows = bootstrap_ci(matrix_coders_as_rows, level_of_measurement="interval", random_state=42)
    ci_cols = bootstrap_ci(df_cols, coders_axis="columns", level_of_measurement="interval", random_state=42)
    assert np.isclose(ci_cols.alpha, ci_rows.alpha)
    assert np.isclose(ci_cols.confidence_interval[0], ci_rows.confidence_interval[0])
    assert np.isclose(ci_cols.confidence_interval[1], ci_rows.confidence_interval[1])


def test_dict_of_columns_as_wide_data() -> None:
    # A mapping of coder names to sequences of unit ratings
    col_dict = {
        "coder1": [1.0, 2.0, 3.0],
        "coder2": [1.0, 2.0, 4.0],
    }
    expected_alpha = alpha([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]], level_of_measurement="interval")
    assert np.isclose(alpha(col_dict, coders_axis="rows", level_of_measurement="interval"), expected_alpha)


def test_coders_axis_subsample_all_reliability_data() -> None:
    subsample = [[1.0, 2.0], [1.0, 3.0]]
    reference = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]])

    alpha_sub_ref = alpha(subsample, all_reliability_data=reference, level_of_measurement="interval")

    # Transposed reference with coders_axis="columns"
    ref_cols = reference.T
    alpha_sub_ref_rows = alpha(
        subsample,
        all_reliability_data=ref_cols.T,
        coders_axis="rows",
        level_of_measurement="interval",
    )
    assert np.isclose(alpha_sub_ref, alpha_sub_ref_rows)

    alpha_both_cols = alpha(
        np.array(subsample).T,
        all_reliability_data=ref_cols,
        coders_axis="columns",
        level_of_measurement="interval",
    )
    assert np.isclose(alpha_sub_ref, alpha_both_cols)


def test_invalid_coders_axis_error() -> None:
    data = [[1, 2], [1, 2]]
    with pytest.raises(ValueError, match="Invalid coders_axis 'diagonal'"):
        alpha(data, coders_axis="diagonal")  # ty:ignore[no-matching-overload]

    with pytest.raises(ValueError, match="Invalid coders_axis 'diagonal'"):
        alpha(data, all_reliability_data=data, coders_axis="diagonal")  # ty:ignore[no-matching-overload]


# ---------------------------------------------------------------------------
# Missing Value Sentinels & Object Array Cleaning
# ---------------------------------------------------------------------------


def test_missing_value_sentinels_numeric() -> None:
    # Test pandas NA, NAType, None, np.nan in numeric object arrays
    data_with_na = np.array(
        [
            [1, 2, MockNA()],
            [1, None, 2],
            [NAType(), 2, 2],
        ],
        dtype=object,
    )
    data_equivalent = [
        [1.0, 2.0, np.nan],
        [1.0, np.nan, 2.0],
        [np.nan, 2.0, 2.0],
    ]
    alpha_na = alpha(data_with_na, level_of_measurement="interval")
    alpha_eq = alpha(data_equivalent, level_of_measurement="interval")
    assert np.isclose(alpha_na, alpha_eq)


def test_missing_value_sentinels_string() -> None:
    # Test pandas NA and None in string object arrays
    data_with_na = np.array(
        [
            ["yes", "no", MockNA()],
            ["yes", None, "no"],
            [NaTType(), "no", "no"],
        ],
        dtype=object,
    )
    data_equivalent = [
        ["yes", "no", "nan"],
        ["yes", "nan", "no"],
        ["nan", "no", "no"],
    ]
    alpha_na = alpha(data_with_na, level_of_measurement="nominal")
    alpha_eq = alpha(data_equivalent, level_of_measurement="nominal")
    assert np.isclose(alpha_na, alpha_eq)


# ---------------------------------------------------------------------------
# Long / Tidy Tabular Format (alpha_table, bootstrap_ci_table, table_to_value_counts)
# ---------------------------------------------------------------------------


def test_table_to_value_counts_and_alpha_table_dict_columns() -> None:
    table = {
        "rater": ["c1", "c1", "c1", "c2", "c2", "c2"],
        "item": ["u1", "u2", "u3", "u1", "u2", "u3"],
        "score": [1, 2, 3, 1, 2, 4],
    }
    vc, dom = table_to_value_counts(table, coder_col="rater", unit_col="item", value_col="score")
    assert vc.shape == (3, 4)
    assert list(dom) == [1, 2, 3, 4]

    expected_alpha = alpha([[1, 2, 3], [1, 2, 4]], level_of_measurement="interval")
    res_table = alpha_table(
        table, coder_col="rater", unit_col="item", value_col="score", level_of_measurement="interval"
    )
    assert np.isclose(res_table, expected_alpha)

    # Calling alpha directly with table column args
    res_direct = alpha(table, coder_col="rater", unit_col="item", value_col="score", level_of_measurement="interval")
    assert np.isclose(res_direct, expected_alpha)

    # Calling bootstrap_ci_table
    res_boot_table = bootstrap_ci_table(
        table,
        coder_col="rater",
        unit_col="item",
        value_col="score",
        level_of_measurement="interval",
        random_state=42,
        return_distribution=True,
    )
    res_boot_direct = bootstrap_ci(
        table,
        coder_col="rater",
        unit_col="item",
        value_col="score",
        level_of_measurement="interval",
        random_state=42,
        return_distribution=True,
    )
    assert isinstance(res_boot_table, BootstrapResult)
    assert np.isclose(res_boot_table.alpha, expected_alpha)
    assert np.isclose(res_boot_direct.alpha, expected_alpha)
    assert res_boot_table.resamples is not None


def test_alpha_table_dataframe_mock() -> None:
    df = MockDataFrame(
        {
            "annotator": ["A", "B", "A", "B"],
            "document": ["d1", "d1", "d2", "d2"],
            "label": ["pos", "pos", "neg", "pos"],
        }
    )
    expected_alpha = alpha([["pos", "neg"], ["pos", "pos"]], level_of_measurement="nominal")
    res = alpha_table(df, coder_col="annotator", unit_col="document", value_col="label", level_of_measurement="nominal")
    assert np.isclose(res, expected_alpha)

    # return_components=True
    comp = alpha_table(
        df,
        coder_col="annotator",
        unit_col="document",
        value_col="label",
        level_of_measurement="nominal",
        return_components=True,
    )
    assert isinstance(comp, AlphaResult)
    assert np.isclose(comp.alpha, expected_alpha)


def test_alpha_table_records_and_tuples() -> None:
    # List of dict records
    records = [
        {"c": 1, "u": "x", "v": 10},
        {"c": 2, "u": "x", "v": 10},
        {"c": 1, "u": "y", "v": 20},
        {"c": 2, "u": "y", "v": 30},
    ]
    res_records = alpha_table(records, coder_col="c", unit_col="u", value_col="v", level_of_measurement="interval")
    expected = alpha([[10, 20], [10, 30]], level_of_measurement="interval")
    assert np.isclose(res_records, expected)

    # List of tuples / rows
    tuples = [
        (1, "x", 10),
        (2, "x", 10),
        (1, "y", 20),
        (2, "y", 30),
    ]
    res_tuples = alpha_table(tuples, coder_col=0, unit_col=1, value_col=2, level_of_measurement="interval")
    assert np.isclose(res_tuples, expected)


def test_table_missing_values_ignored() -> None:
    table = {
        "coder": ["c1", "c2", "c3", "c1", "c2", "c3"],
        "unit": ["u1", "u1", "u1", "u2", "u2", "u2"],
        "value": [1, 1, MockNA(), None, 2, 2],
    }
    vc, dom = table_to_value_counts(table, coder_col="coder", unit_col="unit", value_col="value")
    # Unit u1: c1 and c2 rated 1, c3 is missing -> count for 1 is 2
    # Unit u2: c1 is missing, c2 and c3 rated 2 -> count for 2 is 2
    assert vc.tolist() == [[2, 0], [0, 2]]
    assert list(dom) == [1, 2]


def test_table_explicit_value_domain() -> None:
    table = {
        "coder": ["c1", "c2", "c1", "c2"],
        "unit": ["u1", "u1", "u2", "u2"],
        "value": [1, 2, 1, 2],
    }
    # Explicit domain [1, 2, 3] includes unobserved value 3
    vc, dom = table_to_value_counts(
        table, coder_col="coder", unit_col="unit", value_col="value", value_domain=[1, 2, 3]
    )
    assert list(dom) == [1, 2, 3]
    assert vc.shape == (2, 3)

    # Out-of-domain error
    with pytest.raises(ValueError, match=re.escape("The reliability data contains out-of-domain values.")):
        table_to_value_counts(table, coder_col="coder", unit_col="unit", value_col="value", value_domain=[1])


# ---------------------------------------------------------------------------
# Error Validations & Edge Cases
# ---------------------------------------------------------------------------


def test_table_duplicate_ratings_error() -> None:
    table = {
        "coder": ["c1", "c1"],
        "unit": ["u1", "u1"],
        "value": [1, 2],
    }
    with pytest.raises(ValueError, match=re.escape("Coder 'c1' has duplicate ratings for unit 'u1'.")):
        table_to_value_counts(table, coder_col="coder", unit_col="unit", value_col="value")


def test_table_empty_error() -> None:
    with pytest.raises(ValueError, match=re.escape("The provided table is empty.")):
        table_to_value_counts([], coder_col="c", unit_col="u", value_col="v")


def test_table_mismatched_lengths_error() -> None:
    table = {
        "coder": ["c1", "c2"],
        "unit": ["u1"],
        "value": [1, 2],
    }
    with pytest.raises(
        ValueError, match=re.escape("Columns coder_col, unit_col, and value_col must all have the same length.")
    ):
        table_to_value_counts(table, coder_col="coder", unit_col="unit", value_col="value")


def test_table_all_missing_error() -> None:
    table = {
        "coder": ["c1", "c2"],
        "unit": ["u1", "u1"],
        "value": [None, np.nan],
    }
    with pytest.raises(ValueError, match=re.escape("No valid non-missing ratings found in the table.")):
        table_to_value_counts(table, coder_col="coder", unit_col="unit", value_col="value")


def test_table_insufficient_values_or_coders() -> None:
    # Domain with only 1 value
    table_single_val = {
        "coder": ["c1", "c2"],
        "unit": ["u1", "u1"],
        "value": [1, 1],
    }
    with pytest.raises(ValueError, match=re.escape("There has to be more than one value in the domain.")):
        alpha(table_single_val, coder_col="coder", unit_col="unit", value_col="value")

    # No unit rated by >= 2 coders
    table_single_coder = {
        "coder": ["c1", "c2"],
        "unit": ["u1", "u2"],
        "value": [1, 2],
    }
    with pytest.raises(
        ValueError,
        match=re.escape("There has to be at least one unit with values assigned by at least two coders."),
    ):
        alpha(table_single_coder, coder_col="coder", unit_col="unit", value_col="value")


def test_alpha_table_column_args_validation() -> None:
    # Partial column arguments
    with pytest.raises(
        ValueError, match=re.escape("All three of coder_col, unit_col, and value_col must be specified together.")
    ):
        alpha([[1, 2], [1, 2]], coder_col="c")

    # Column arguments with reliability_data=None
    with pytest.raises(ValueError, match="A tabular dataset must be provided as reliability_data"):
        alpha(None, coder_col="c", unit_col="u", value_col="v")

    # Column arguments with value_counts
    with pytest.raises(ValueError, match="Cannot provide value_counts when coder_col"):
        alpha(reliability_data=[(1, 2, 3)], value_counts=[[1, 1]], coder_col=0, unit_col=1, value_col=2)


def test_2d_array_validation() -> None:
    # 1D array
    with pytest.raises(ValueError, match="reliability_data must be a 2D array"):
        alpha([1, 2, 3])

    # 3D array
    with pytest.raises(ValueError, match="reliability_data must be a 2D array"):
        alpha([[[1], [2]], [[3], [4]]])

    # 1D all_reliability_data
    with pytest.raises(ValueError, match=re.escape("The all_reliability_data must be a 2D array.")):
        alpha([[1, 2], [1, 2]], all_reliability_data=[1, 2, 3])


def test_to_domain_array_types() -> None:
    # bytes domain
    arr_b = krippendorff.krippendorff._to_domain_array([b"x", b"y"])
    assert arr_b.dtype.kind in {"S", "V"}

    # float domain
    arr_f = krippendorff.krippendorff._to_domain_array([1.5, 2.5])
    assert arr_f.dtype == float


def test_reference_data_out_of_domain() -> None:
    data = [[1, 2], [1, 2]]
    ref_data = [[1, 99], [1, 99]]
    with pytest.raises(ValueError, match=re.escape("The reference reliability data contains out-of-domain values.")):
        alpha(data, all_reliability_data=ref_data, value_domain=[1, 2])


class CustomRowIndexedTable:
    """Table implementing __getitem__ for integer row indexing and raising KeyError for string column lookup."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, int):
            return self.rows[idx]
        raise KeyError(f"Invalid column access {idx}")

    def __iter__(self) -> Any:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)


def test_custom_row_indexed_table() -> None:
    table = CustomRowIndexedTable(
        [
            {"rater": "c1", "item": "u1", "score": 1},
            {"rater": "c2", "item": "u1", "score": 1},
            {"rater": "c1", "item": "u2", "score": 2},
            {"rater": "c2", "item": "u2", "score": 2},
        ]
    )
    res = alpha_table(table, coder_col="rater", unit_col="item", value_col="score")
    assert np.isclose(res, 1.0)


def test_dict_reliability_data_with_all_reliability_data_error() -> None:
    dict_data = [{"u1": 1}, {"u1": 2}]
    ref_data = [[1, 2], [1, 2]]
    with pytest.raises(ValueError, match=re.escape("Subsample with dict reliability_data is not supported.")):
        alpha(reliability_data=dict_data, all_reliability_data=ref_data)


def test_value_counts_not_2d_error() -> None:
    with pytest.raises(ValueError, match=re.escape("The value_counts must be a 2D array.")):
        alpha(value_counts=[1, 2, 3])


def test_bootstrap_ci_table_with_value_domain() -> None:
    table = {
        "rater": ["c1", "c1", "c2", "c2"],
        "item": ["u1", "u2", "u1", "u2"],
        "score": [1, 2, 1, 2],
    }
    res = bootstrap_ci_table(
        table,
        coder_col="rater",
        unit_col="item",
        value_col="score",
        value_domain=[1, 2, 3],
        random_state=42,
    )
    assert np.isclose(res.alpha, 1.0)

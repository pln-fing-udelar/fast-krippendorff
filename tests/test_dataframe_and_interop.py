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
    def dtypes(self) -> list[np.dtype]:
        res = []
        for k in self.columns:
            types = {type(x) for x in self._data[k]}
            if len(types) > 1:
                res.append(np.dtype("O"))
            else:
                res.append(np.array(self._data[k]).dtype)
        return res

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
    # A mapping of coder names to sequences of unit ratings (columns are coders)
    col_dict = {
        "coder1": [1.0, 2.0, 3.0],
        "coder2": [1.0, 2.0, 4.0],
    }
    expected_alpha = alpha([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]], level_of_measurement="interval")
    df = MockDataFrame(col_dict)

    res_dict = alpha(col_dict, coders_axis="columns", level_of_measurement="interval")
    res_df = alpha(df, coders_axis="columns", level_of_measurement="interval")
    assert np.isclose(res_dict, expected_alpha)
    assert np.isclose(res_df, expected_alpha)

    # bootstrap_ci equivalence between mapping and DataFrame
    res_ci_dict = bootstrap_ci(col_dict, coders_axis="columns", level_of_measurement="interval", random_state=42)
    res_ci_df = bootstrap_ci(df, coders_axis="columns", level_of_measurement="interval", random_state=42)
    assert np.isclose(res_ci_dict.alpha, expected_alpha)
    assert np.isclose(res_ci_dict.alpha, res_ci_df.alpha)
    assert np.isclose(res_ci_dict.confidence_interval[0], res_ci_df.confidence_interval[0])
    assert np.isclose(res_ci_dict.confidence_interval[1], res_ci_df.confidence_interval[1])


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


def test_extract_table_columns_2d_numpy_array() -> None:
    # 2D NumPy array where rows represent records
    table_arr = np.array(
        [
            [1, 10, 1],
            [2, 10, 2],
            [1, 20, 1],
            [2, 20, 2],
        ]
    )
    res = alpha_table(table_arr, coder_col=0, unit_col=1, value_col=2, level_of_measurement="interval")
    expected = alpha([[1, 1], [2, 2]], level_of_measurement="interval")
    assert np.isclose(res, expected)


def test_table_all_missing_unit_retained_in_counts() -> None:
    table = {
        "coder": ["c1", "c2", "c1", "c2"],
        "unit": ["u1", "u1", "u2", "u2"],
        "value": [1, 2, np.nan, None],
    }
    vc, dom = table_to_value_counts(table, coder_col="coder", unit_col="unit", value_col="value")
    # Unit u2 has only missing ratings, but must still be retained as a row with count 0
    assert vc.shape == (2, 2)
    assert vc[0].sum() == 2
    assert vc[1].sum() == 0
    assert list(dom) == [1, 2]


def test_table_string_ratings_require_domain_for_non_nominal() -> None:
    table = {
        "coder": ["c1", "c2", "c1", "c2"],
        "unit": ["u1", "u1", "u2", "u2"],
        "value": ["low", "med", "low", "high"],
    }
    # Non-nominal level of measurement without explicit value_domain must raise ValueError
    with pytest.raises(ValueError, match="When using strings, an ordered value_domain is required"):
        alpha_table(table, coder_col="coder", unit_col="unit", value_col="value", level_of_measurement="ordinal")

    with pytest.raises(ValueError, match="When using strings, an ordered value_domain is required"):
        table_to_value_counts(
            table, coder_col="coder", unit_col="unit", value_col="value", level_of_measurement="ordinal"
        )


def test_table_with_all_reliability_data_unions_unseen_reference_categories() -> None:
    # Subsample table only contains ratings 1 and 2
    subsample_table = {
        "coder": ["c1", "c2", "c1", "c2"],
        "unit": ["u1", "u1", "u2", "u2"],
        "value": [1, 2, 1, 2],
    }
    # Reference data contains category 3, which never appears in the subsample table
    reference_matrix = np.array(
        [
            [1, 2, 3],
            [1, 2, 3],
        ]
    )
    # Without explicit value_domain, domain should union categories from table and reference data
    res_table = alpha_table(
        subsample_table,
        coder_col="coder",
        unit_col="unit",
        value_col="value",
        all_reliability_data=reference_matrix,
        level_of_measurement="interval",
    )
    res_direct = alpha(
        subsample_table,
        coder_col="coder",
        unit_col="unit",
        value_col="value",
        all_reliability_data=reference_matrix,
        level_of_measurement="interval",
    )
    expected = alpha(
        [[1, 1], [2, 2]],
        all_reliability_data=reference_matrix,
        level_of_measurement="interval",
    )
    assert np.isclose(res_table, expected)
    assert np.isclose(res_direct, expected)


def test_bootstrap_ci_with_reference_matrix_and_coders_axis_columns() -> None:
    matrix = np.array(
        [
            [1.0, 2.0, 1.0, 2.0],
            [2.0, 2.0, 1.0, 1.0],
        ]
    )
    reference = np.array(
        [
            [1.0, 2.0, 3.0, 1.0, 2.0],
            [2.0, 2.0, 3.0, 1.0, 1.0],
        ]
    )
    # Rows as coders:
    res_rows = bootstrap_ci(
        matrix,
        all_reliability_data=reference,
        coders_axis="rows",
        level_of_measurement="interval",
        random_state=42,
    )
    # Columns as coders (transposed inputs):
    res_cols = bootstrap_ci(
        matrix.T,
        all_reliability_data=reference.T,
        coders_axis="columns",
        level_of_measurement="interval",
        random_state=42,
    )
    assert np.isclose(res_rows.alpha, res_cols.alpha)
    assert np.isclose(res_rows.confidence_interval[0], res_cols.confidence_interval[0])
    assert np.isclose(res_rows.confidence_interval[1], res_cols.confidence_interval[1])
    assert np.isclose(res_rows.standard_error, res_cols.standard_error)


def test_table_and_dict_uncovered_error_branches() -> None:
    # 1. No valid units in table
    with pytest.raises(ValueError, match=re.escape("No valid units found in the table.")):
        table_to_value_counts(
            {"coder": [1, 2], "unit": [None, np.nan], "value": [10, 20]},
            coder_col="coder",
            unit_col="unit",
            value_col="value",
        )

    # 2. No valid ratings in table passed to alpha
    with pytest.raises(ValueError, match=re.escape("No valid non-missing ratings found in the table.")):
        alpha(
            {"coder": ["c1"], "unit": ["u1"], "value": [None]},
            coder_col="coder",
            unit_col="unit",
            value_col="value",
        )

    # 3. Empty table passed to alpha
    with pytest.raises(ValueError, match=re.escape("The provided table is empty.")):
        alpha(
            {"coder": [], "unit": [], "value": []},
            coder_col="coder",
            unit_col="unit",
            value_col="value",
        )

    # 4. Unequal length columns passed to alpha
    with pytest.raises(
        ValueError, match=re.escape("Columns coder_col, unit_col, and value_col must all have the same length.")
    ):
        alpha(
            {"coder": ["c1"], "unit": ["u1", "u2"], "value": [1]},
            coder_col="coder",
            unit_col="unit",
            value_col="value",
        )

    # 5. Table out-of-domain with explicit value_domain in alpha
    with pytest.raises(ValueError, match=re.escape("The reliability data contains out-of-domain values.")):
        alpha(
            {"coder": [1, 2], "unit": ["u1", "u1"], "value": [1, 99]},
            coder_col="coder",
            unit_col="unit",
            value_col="value",
            value_domain=[1, 2],
        )

    # 6. Reference data out-of-domain with explicit value_domain in table alpha
    with pytest.raises(ValueError, match=re.escape("The reference reliability data contains out-of-domain values.")):
        alpha(
            {"coder": [1, 2], "unit": ["u1", "u1"], "value": [1, 2]},
            coder_col="coder",
            unit_col="unit",
            value_col="value",
            value_domain=[1, 2],
            all_reliability_data=[[1, 99], [2, 99]],
        )

    # 7. Dict reliability data with out-of-domain values
    with pytest.raises(ValueError, match=re.escape("The reliability data contains out-of-domain values.")):
        alpha(
            [{"u1": 1, "u2": 2}, {"u1": 1, "u2": 99}],
            value_domain=[1, 2],
        )

    # 8. Column-accessible table propagates KeyError when column is missing
    with pytest.raises(KeyError, match=re.escape("missing_col")):
        alpha_table(
            {"coder": [1, 2], "unit": ["u1", "u2"], "value": [10, 20]},
            coder_col="missing_col",
            unit_col="unit",
            value_col="value",
        )


def test_preserve_exact_integer_labels_in_object_array() -> None:
    # Copilot review finding:
    # Nominal ratings [[0, 2**53, 2**53 + 1], [0, 2**53 + 1, 2**53]] with dtype=object
    # must not merge 2**53 and 2**53 + 1 into the same float64 (which would produce alpha=1.0 instead of 1/6)
    data = np.array([[0, 2**53, 2**53 + 1], [0, 2**53 + 1, 2**53]], dtype=object)

    res_alpha = alpha(data, level_of_measurement="nominal")
    assert np.isclose(res_alpha, 1.0 / 6.0)

    res_ci = bootstrap_ci(data, level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_ci.alpha, 1.0 / 6.0)

    # With missing values / None
    data_with_none = np.array([[0, 2**53, 2**53 + 1], [0, None, 2**53]], dtype=object)
    res_alpha_none = alpha(data_with_none, level_of_measurement="nominal")
    res_ci_none = bootstrap_ci(data_with_none, level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_alpha_none, res_ci_none.alpha)

    # Arbitrarily large integers (> 2**64 and > 2**1024 float overflow)
    data_huge = np.array([[0, 2**100, 2**100 + 1], [0, 2**100 + 1, 2**100]], dtype=object)
    res_alpha_huge = alpha(data_huge, level_of_measurement="nominal")
    assert np.isclose(res_alpha_huge, 1.0 / 6.0)
    res_ci_huge = bootstrap_ci(data_huge, level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_ci_huge.alpha, 1.0 / 6.0)

    data_overflow = np.array([[0, 2**2000, 2**2000 + 1], [0, 2**2000 + 1, 2**2000]], dtype=object)
    res_alpha_overflow = alpha(data_overflow, level_of_measurement="nominal")
    assert np.isclose(res_alpha_overflow, 1.0 / 6.0)

    # Object array with strings and None
    data_str_obj = np.array([["a", None, "b"], ["a", "b", "b"]], dtype=object)
    res_str_obj = alpha(data_str_obj, level_of_measurement="nominal")
    assert np.isclose(res_str_obj, 1.0)


def test_preserve_missing_sentinels_with_string_nan_category() -> None:
    # Copilot review finding:
    # The string "nan" can be a valid category label (e.g. "not applicable" or literal "nan").
    # Missing values (None/NaN) must not be converted to the string "nan" and merged with that category.
    # For matrix [["nan", "yes", None], ["nan", "no", "no"]] with domain ["nan", "yes", "no"]:
    # Unit 0: coders agree on "nan" (2 ratings)
    # Unit 1: "yes" vs "no" (2 ratings)
    # Unit 2: coder 0 is missing, coder 1 is "no" (1 rating, not pairable)
    # Expected alpha is 0.4 (if None was converted to string "nan", alpha would be 1/11).
    data = np.array([["nan", "yes", None], ["nan", "no", "no"]], dtype=object)
    domain = ["nan", "yes", "no"]

    res_alpha = alpha(data, value_domain=domain, level_of_measurement="nominal")
    assert np.isclose(res_alpha, 0.4)

    res_ci = bootstrap_ci(data, value_domain=domain, level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_ci.alpha, 0.4)


def test_preserve_literal_nan_string_inferred_domain_object_array() -> None:
    # Copilot review finding (discussion_r4278635337):
    # Downcasting a string object array to Unicode changes how the literal "nan" label is handled:
    # _domain_from_reliability_data excludes "nan" from Unicode arrays but preserves it in object arrays.
    # Without an explicit domain or missing values, np.array([["nan", "yes"], ["nan", "no"]], dtype=object)
    # must retain object dtype and infer domain ["nan", "no", "yes"], yielding nominal alpha 0.4 instead of 0.0.
    data = np.array([["nan", "yes"], ["nan", "no"]], dtype=object)

    res_alpha = alpha(data, level_of_measurement="nominal")
    assert np.isclose(res_alpha, 0.4)

    res_ci = bootstrap_ci(data, level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_ci.alpha, 0.4)

    # Wide DataFrame returning object array with "nan" string labels
    class MockWideDF:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        def to_numpy(self) -> np.ndarray:
            return self._arr

    df_obj = MockWideDF(data)
    res_df = alpha(df_obj, level_of_measurement="nominal")
    assert np.isclose(res_df, 0.4)

    res_df_ci = bootstrap_ci(df_obj, level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_df_ci.alpha, 0.4)

    # Adding an all-missing unit to the object array must not change the alpha of valid units
    data_with_missing_unit = np.array([["nan", "yes", None], ["nan", "no", None]], dtype=object)
    res_with_missing = alpha(data_with_missing_unit, level_of_measurement="nominal")
    assert np.isclose(res_with_missing, 0.4)
    res_with_missing_ci = bootstrap_ci(data_with_missing_unit, level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_with_missing_ci.alpha, 0.4)


def test_preserve_wide_integer_labels_in_mapping_input() -> None:
    # Copilot review finding (discussion_r4278700654):
    # Mapping with wide columns {'c1': [-1, 2**63, 2**63 + 1], 'c2': [-1, 2**63 + 1, 2**63]}
    # must construct an object array first so large integers are not coerced to float64,
    # which would merge 2**63 and 2**63 + 1. Nominal alpha with coders_axis='columns'
    # must be 1/6 (not 1.0).
    data = {"c1": [-1, 2**63, 2**63 + 1], "c2": [-1, 2**63 + 1, 2**63]}
    res_alpha = alpha(data, coders_axis="columns", level_of_measurement="nominal")
    assert np.isclose(res_alpha, 1.0 / 6.0)

    res_ci = bootstrap_ci(data, coders_axis="columns", level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_ci.alpha, 1.0 / 6.0)


def test_preserve_extended_precision_floats_object_array() -> None:
    # Copilot review finding (discussion_r4278700684):
    # On platforms where np.longdouble has higher precision than float64,
    # object arrays with ratings [[0, a, b], [0, b, a]] (where b = a + eps)
    # must retain object dtype so a and b are not merged into the same float64.
    if np.finfo(np.longdouble).precision <= np.finfo(np.float64).precision:
        pytest.skip("Platform does not support extended precision longdouble.")

    a = np.longdouble(1)
    eps = np.finfo(np.longdouble).eps
    b = a + eps
    data = np.array([[0, a, b], [0, b, a]], dtype=object)

    res_alpha = alpha(data, level_of_measurement="nominal")
    assert np.isclose(res_alpha, 1.0 / 6.0)

    res_ci = bootstrap_ci(data, level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_ci.alpha, 1.0 / 6.0)


def test_prevent_float64_coercion_merging_large_integer_domain_values() -> None:
    # Copilot review finding (discussion_r4278700709):
    # For B = 2**53, nullable integer ratings [[0, B, None], [0, B, None]] with domain [0, B, B + 1]
    # must not be converted to float64, which would cause B in ratings to match both B and B + 1
    # in the domain and produce alpha = 4/9 instead of 1.0.
    b_val = 2**53
    data = np.array([[0, b_val, None], [0, b_val, None]], dtype=object)
    domain = [0, b_val, b_val + 1]

    res_alpha = alpha(data, value_domain=domain, level_of_measurement="nominal")
    assert np.isclose(res_alpha, 1.0)

    res_ci = bootstrap_ci(data, value_domain=domain, level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_ci.alpha, 1.0)


def test_reject_nested_mapping_reference_data() -> None:
    # Copilot review finding (discussion_r4278767326):
    # Reject nested mappings as reference data (all_reliability_data) instead of treating
    # keys as ratings, which would silently produce incorrect agreement values.
    ref = {"c1": {1: 0, 2: 1}, "c2": {1: 0, 2: 0}}
    data = [[0, 1], [0, 0]]
    with pytest.raises(ValueError, match=re.escape("Subsample with dict all_reliability_data is not supported.")):
        alpha(data, all_reliability_data=ref, level_of_measurement="nominal")

    with pytest.raises(ValueError, match=re.escape("Subsample with dict all_reliability_data is not supported.")):
        bootstrap_ci(data, all_reliability_data=ref, level_of_measurement="nominal", random_state=42)


def test_table_with_iterator_columns() -> None:
    # Copilot review finding (discussion_r4278767388):
    # For a mapping whose columns are single-use iterators, extracting columns must not exhaust
    # them before table_to_value_counts() is called.
    table = {
        "coder": iter(["c1", "c2", "c1", "c2"]),
        "unit": iter(["u1", "u1", "u2", "u2"]),
        "val": iter([1, 1, 2, 3]),
    }
    res_table = alpha_table(table, coder_col="coder", unit_col="unit", value_col="val", level_of_measurement="nominal")
    assert np.isclose(res_table, 0.4)

    table_boot = {
        "coder": iter(["c1", "c2", "c1", "c2"]),
        "unit": iter(["u1", "u1", "u2", "u2"]),
        "val": iter([1, 1, 2, 3]),
    }
    res_boot = bootstrap_ci_table(
        table_boot,
        coder_col="coder",
        unit_col="unit",
        value_col="val",
        level_of_measurement="nominal",
        random_state=42,
    )
    assert np.isclose(res_boot.alpha, 0.4)


def test_mixed_numeric_and_string_labels_nominal() -> None:
    # Copilot review finding:
    # Allow mixed numeric and string labels for nominal measurements in wide mappings and object arrays.
    data = {"c1": [1, "a"], "c2": [1, "a"]}
    res = alpha(data, coders_axis="columns", level_of_measurement="nominal")
    assert np.isclose(res, 1.0)

    res_ci = bootstrap_ci(data, coders_axis="columns", level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_ci.alpha, 1.0)

    # Object array input directly
    arr = np.array([[1, "a"], [1, "a"]], dtype=object)
    res_arr = alpha(arr, level_of_measurement="nominal")
    assert np.isclose(res_arr, 1.0)

    # Non-nominal with mixed labels without explicit domain should raise ValueError
    with pytest.raises(ValueError, match=re.escape("When using mixed types, an ordered value_domain is required")):
        alpha(arr, level_of_measurement="ordinal")


def test_numeric_ratings_subsample_skips_reference_scan() -> None:
    # Ordinary numeric float/int arrays bypass reference scan and clean_object_array
    sample = np.array([[1.0, 2.0], [1.0, 2.0]])
    ref = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    res = alpha(sample, all_reliability_data=ref, level_of_measurement="interval")
    assert np.isclose(res, 1.0)


def test_integer_downcast_preserves_precision_with_large_domain() -> None:
    # Copilot review finding (discussion_r4278867396):
    # Downcasting object ratings [[0, B], [0, B]] to signed int when domain is np.uint64 [0, B, B + 1] (B = 2**53)
    # causes float64 promotion during comparison, collapsing B and B + 1 and yielding nominal alpha = 4/9 instead of
    # 1.0. Preserve exact comparisons against the domain and reference data.
    b_val = 2**53
    ratings = np.array([[0, b_val], [0, b_val]], dtype=object)
    domain = np.array([0, b_val, b_val + 1], dtype=np.uint64)

    res_alpha = alpha(ratings, value_domain=domain, level_of_measurement="nominal")
    assert np.isclose(res_alpha, 1.0)

    res_ci = bootstrap_ci(ratings, value_domain=domain, level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_ci.alpha, 1.0)
    assert np.isclose(res_ci.confidence_interval[0], 1.0)
    assert np.isclose(res_ci.confidence_interval[1], 1.0)

    # Subsample reference data with large integers
    all_ratings = np.array([[0, b_val, b_val + 1], [0, b_val, b_val + 1]], dtype=np.uint64)
    res_subsample_alpha = alpha(ratings, all_reliability_data=all_ratings, level_of_measurement="nominal")
    assert np.isclose(res_subsample_alpha, 1.0)

    res_subsample_ci = bootstrap_ci(
        ratings, all_reliability_data=all_ratings, level_of_measurement="nominal", random_state=42
    )
    assert np.isclose(res_subsample_ci.alpha, 1.0)


def test_string_object_array_retains_object_dtype_avoiding_unicode_blowup() -> None:
    # Copilot review finding:
    # Converting object-backed string ratings to fixed-width Unicode reserves space for the longest label
    # in every cell, causing severe memory blowup for long strings. Retain object array.
    long_label = "x" * 1000
    ratings = np.array([[long_label, "short"], [long_label, "short"]], dtype=object)

    from krippendorff.krippendorff import _clean_object_array

    cleaned = _clean_object_array(ratings)
    assert cleaned.dtype == object

    res_alpha = alpha(ratings, level_of_measurement="nominal")
    assert np.isclose(res_alpha, 1.0)

    res_ci = bootstrap_ci(ratings, level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_ci.alpha, 1.0)


def test_object_arrays_normalized_in_count_input_paths() -> None:
    # Copilot review finding (discussion_r4278950260):
    # Numeric column mappings such as {"a": [2, 0], "b": [0, 2]} and object-backed DataFrames
    # become object arrays in _to_numpy_array, but must be normalized by _clean_object_array
    # in both value_counts and all_value_counts count-input paths.
    vc_dict = {"a": [2, 0], "b": [0, 2]}
    res_a = alpha(value_counts=vc_dict, level_of_measurement="nominal")
    assert np.isclose(res_a, 1.0)

    res_ci = bootstrap_ci(value_counts=vc_dict, level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_ci.alpha, 1.0)

    # Reference all_value_counts as column mapping
    sample = [[0, 1], [0, 1]]
    all_vc_dict = {"a": [2, 0], "b": [0, 2]}
    res_ref_a = alpha(sample, all_value_counts=all_vc_dict, level_of_measurement="nominal")
    assert np.isclose(res_ref_a, 1.0)

    res_ref_ci = bootstrap_ci(sample, all_value_counts=all_vc_dict, level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_ref_ci.alpha, 1.0)

    # Mock object-backed DataFrame for direct and reference counts
    class MockObjCountsDF:
        def __init__(self, arr: list[list[int]]) -> None:
            self._arr = np.asarray(arr, dtype=object)

        def to_numpy(self) -> np.ndarray:
            return self._arr

    df_vc = MockObjCountsDF([[2, 0], [0, 2]])
    assert np.isclose(alpha(value_counts=df_vc, level_of_measurement="nominal"), 1.0)
    assert np.isclose(bootstrap_ci(value_counts=df_vc, level_of_measurement="nominal", random_state=42).alpha, 1.0)
    assert np.isclose(alpha(sample, all_value_counts=df_vc, level_of_measurement="nominal"), 1.0)
    assert np.isclose(
        bootstrap_ci(sample, all_value_counts=df_vc, level_of_measurement="nominal", random_state=42).alpha, 1.0
    )


def test_bootstrap_ci_reusable_iterator_reference_inputs() -> None:
    # Copilot review finding (discussion_r4279006146):
    # Materialize iterator-backed reference inputs (all_value_counts and random_coincidences)
    # once in bootstrap_ci() so they are not exhausted when reused for expected coincidences and distance calculations.
    sample = [[0, 1], [0, 1]]
    all_vc_iter = {"a": iter([2, 0]), "b": iter([0, 2])}
    res_boot_vc = bootstrap_ci(sample, all_value_counts=all_vc_iter, level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_boot_vc.alpha, 1.0)

    # Iterator-backed random_coincidences mapping
    rand_coinc_iter = {"col1": iter([1.0, 1.0]), "col2": iter([1.0, 1.0])}
    res_boot_rc = bootstrap_ci(
        sample, random_coincidences=rand_coinc_iter, level_of_measurement="nominal", random_state=42
    )
    assert np.isclose(res_boot_rc.alpha, 1.0)


def test_table_with_record_iterator() -> None:
    # Copilot review finding:
    # Single-pass traversal in _extract_table_columns handles one-shot record iterators
    # in table_to_value_counts(), alpha_table(), and bootstrap_ci_table().
    records = [
        {"coder": "c1", "unit": "u1", "val": 1},
        {"coder": "c2", "unit": "u1", "val": 1},
        {"coder": "c1", "unit": "u2", "val": 2},
        {"coder": "c2", "unit": "u2", "val": 2},
    ]

    vc, dom = table_to_value_counts(iter(records), coder_col="coder", unit_col="unit", value_col="val")
    assert np.array_equal(vc, [[2, 0], [0, 2]])
    assert np.array_equal(dom, [1, 2])

    res_alpha = alpha_table(
        iter(records), coder_col="coder", unit_col="unit", value_col="val", level_of_measurement="nominal"
    )
    assert np.isclose(res_alpha, 1.0)

    res_ci = bootstrap_ci_table(
        iter(records),
        coder_col="coder",
        unit_col="unit",
        value_col="val",
        level_of_measurement="nominal",
        random_state=42,
    )
    assert np.isclose(res_ci.alpha, 1.0)


def test_mixed_dtype_dataframe_preserves_large_integers() -> None:
    # Copilot review finding (discussion_r4279110639):
    # Mixed-dtype DataFrames must not be coerced to float64, which would merge large integers >= 2**53.
    b = 2**53
    df = MockDataFrame({"u0": [0.0, 0.0], "u1": [b, b + 1], "u2": [b + 1, b]})
    assert np.isclose(alpha(df, level_of_measurement="nominal"), 1 / 6)

    res_ci = bootstrap_ci(df, level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_ci.alpha, 1 / 6)

    # Test column-oriented layout
    df_cols = MockDataFrame({"c0": [0.0, b, b + 1], "c1": [0.0, b + 1, b]})
    assert np.isclose(alpha(df_cols, coders_axis="columns", level_of_measurement="nominal"), 1 / 6)
    res_ci_cols = bootstrap_ci(df_cols, coders_axis="columns", level_of_measurement="nominal", random_state=42)
    assert np.isclose(res_ci_cols.alpha, 1 / 6)


def test_sentinel_supports_none_valued_column_labels() -> None:
    # Copilot review finding (Previously missed):
    # None is a valid hashable column label and must not be treated as an omitted selector.
    table = {None: ["c1", "c2", "c1", "c2"], "unit": ["u1", "u1", "u2", "u2"], "value": [1, 1, 2, 2]}

    vc, dom = table_to_value_counts(table, coder_col=None, unit_col="unit", value_col="value")
    assert np.array_equal(vc, [[2, 0], [0, 2]])
    assert np.array_equal(dom, [1, 2])

    assert np.isclose(
        alpha_table(table, coder_col=None, unit_col="unit", value_col="value", level_of_measurement="nominal"), 1.0
    )
    res_ci_table = bootstrap_ci_table(
        table, coder_col=None, unit_col="unit", value_col="value", level_of_measurement="nominal", random_state=42
    )
    assert np.isclose(res_ci_table.alpha, 1.0)

    assert np.isclose(
        alpha(table, coder_col=None, unit_col="unit", value_col="value", level_of_measurement="nominal"), 1.0
    )
    res_ci = bootstrap_ci(
        table, coder_col=None, unit_col="unit", value_col="value", level_of_measurement="nominal", random_state=42
    )
    assert np.isclose(res_ci.alpha, 1.0)

    # None as unit_col
    table_u = {"coder": ["c1", "c2", "c1", "c2"], None: ["u1", "u1", "u2", "u2"], "value": [1, 1, 2, 2]}
    assert np.isclose(
        alpha_table(table_u, coder_col="coder", unit_col=None, value_col="value", level_of_measurement="nominal"), 1.0
    )

    # None as value_col
    table_v = {"coder": ["c1", "c2", "c1", "c2"], "unit": ["u1", "u1", "u2", "u2"], None: [1, 1, 2, 2]}
    assert np.isclose(
        alpha_table(table_v, coder_col="coder", unit_col="unit", value_col=None, level_of_measurement="nominal"), 1.0
    )

    # Partial arguments with None selector must raise
    with pytest.raises(ValueError, match="All three of coder_col, unit_col, and value_col must be specified together"):
        alpha(table, coder_col=None)

    with pytest.raises(ValueError, match="All three of coder_col, unit_col, and value_col must be specified together"):
        alpha(table, coder_col=None, unit_col="unit")


class MockDataFrameWithILoc:
    """Mock DataFrame implementing .iloc column indexing."""

    def __init__(self, data: dict[str, list[Any]]) -> None:
        self.columns = list(data.keys())
        self._data = data
        self.shape = (len(next(iter(data.values()))), len(data.keys())) if data else (0, 0)

        class _ILoc:
            def __init__(self, parent: MockDataFrameWithILoc) -> None:
                self.parent = parent

            def __getitem__(self, item: Any) -> list[Any]:
                col_idx = item[1]
                col_name = self.parent.columns[col_idx]
                return self.parent._data[col_name]

        self.iloc = _ILoc(self)


class MockDataFrameFaultyExtraction:
    """Mock DataFrame whose column indexing raises an exception, falling back to to_numpy()."""

    def __init__(self, data: dict[str, list[Any]]) -> None:
        self.columns = list(data.keys())
        self._data = data

    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("Indexing failed")

    def to_numpy(self) -> np.ndarray:
        return np.array([self._data[k] for k in self.columns]).T


def test_to_numpy_array_dataframe_edge_cases() -> None:
    # Test DataFrame with .iloc
    df_iloc = MockDataFrameWithILoc({"u1": [1.0, 2.0], "u2": [1.0, 2.0]})
    assert np.isclose(alpha(df_iloc, level_of_measurement="interval"), -0.5)

    # Test empty DataFrame with 0 columns
    class MockDataFrameEmpty:
        def __init__(self) -> None:
            self.columns: list[str] = []

    with pytest.raises(ValueError, match="reliability_data must be a 2D array"):
        alpha(MockDataFrameEmpty())

    # Test fallback to to_numpy when column extraction raises
    df_faulty = MockDataFrameFaultyExtraction({"u1": [1.0, 2.0], "u2": [1.0, 2.0]})
    assert np.isclose(alpha(df_faulty, level_of_measurement="interval"), -0.5)

    # Test all_reliability_data object array
    all_obj = np.array([["1", "2"], ["1", "2"]], dtype=object)
    assert np.isclose(
        alpha([["1", "2"], ["1", "2"]], all_reliability_data=all_obj, level_of_measurement="nominal"), 1.0
    )


def test_normalize_integral_scalars_prevents_duplicate_label_counting() -> None:
    # b = 2**53 = 9007199254740992, where np.float64(b) compares equal to both b and b + 1
    # without integer scalar normalization.
    b = 2**53
    ratings = np.array([[0, np.float64(b), b + 1], [0, b + 1, np.float64(b)]], dtype=object)

    # Value counts should count each label exactly once per coder/unit
    domain = np.array([0, b, b + 1])
    counts = krippendorff.krippendorff._reliability_data_to_value_counts(ratings, domain)
    np.testing.assert_array_equal(counts, np.array([[2, 0, 0], [0, 1, 1], [0, 1, 1]]))

    # Nominal alpha should be exactly 1/6 (~0.166667), not 0.3
    res_alpha = alpha(ratings, level_of_measurement="nominal")
    assert np.isclose(res_alpha, 1.0 / 6.0)

    # bootstrap_ci should also yield nominal alpha = 1/6
    res_ci = bootstrap_ci(ratings, level_of_measurement="nominal", n_resamples=20, random_state=42)
    assert np.isclose(res_ci.alpha, 1.0 / 6.0)


def test_homogeneous_numeric_dataframe_fast_path() -> None:
    # Homogeneous float DataFrame
    df_float = MockDataFrame({"u1": [1.0, 2.0], "u2": [3.0, 4.0]})
    arr_float = krippendorff.krippendorff._to_numpy_array(df_float)
    assert np.issubdtype(arr_float.dtype, np.floating)
    assert np.isclose(alpha(df_float, level_of_measurement="interval"), 0.7)

    # Homogeneous integer DataFrame
    df_int = MockDataFrame({"u1": [1, 2], "u2": [3, 4]})
    arr_int = krippendorff.krippendorff._to_numpy_array(df_int)
    assert np.issubdtype(arr_int.dtype, np.integer)
    assert np.isclose(alpha(df_int, level_of_measurement="interval"), 0.7)

    # Homogeneous boolean DataFrame
    df_bool = MockDataFrame({"u1": [True, False], "u2": [False, True]})
    arr_bool = krippendorff.krippendorff._to_numpy_array(df_bool)
    assert arr_bool.dtype == bool
    assert np.isclose(alpha(df_bool, level_of_measurement="nominal"), -0.5)


def test_boolean_ratings_nominal_measurement() -> None:
    # Wide dict with boolean values
    wide_bool = {"c1": [False, True, False], "c2": [False, True, False]}
    assert np.isclose(alpha(wide_bool, coders_axis="columns", level_of_measurement="nominal"), 1.0)
    ci = bootstrap_ci(wide_bool, coders_axis="columns", level_of_measurement="nominal", n_resamples=10, random_state=42)
    assert np.isclose(ci.alpha, 1.0)

    # 2D boolean ndarray
    arr_bool = np.array([[True, False, True], [True, False, True]], dtype=bool)
    assert np.isclose(alpha(arr_bool, level_of_measurement="nominal"), 1.0)

    # Boolean with non-nominal measurement without explicit value_domain must raise
    with pytest.raises(
        ValueError, match="ordered value_domain is required for level_of_measurement other than 'nominal'"
    ):
        alpha(wide_bool, coders_axis="columns", level_of_measurement="ordinal")

    with pytest.raises(
        ValueError, match="ordered value_domain is required for level_of_measurement other than 'nominal'"
    ):
        alpha(arr_bool, level_of_measurement="interval")


def test_iterator_backed_columns_overlapping_selectors() -> None:
    # When coder_col == value_col and columns are backed by iterators
    def make_table() -> dict[str, Any]:
        return {"coder": iter([1, 2, 1, 2]), "unit": iter(["a", "a", "b", "b"])}

    # Direct table_to_value_counts
    counts, domain = table_to_value_counts(make_table(), coder_col="coder", unit_col="unit", value_col="coder")
    np.testing.assert_array_equal(counts, np.array([[1, 1], [1, 1]]))
    np.testing.assert_array_equal(domain, np.array([1, 2]))

    # alpha_table
    res_alpha = alpha_table(
        make_table(), coder_col="coder", unit_col="unit", value_col="coder", level_of_measurement="nominal"
    )
    assert np.isclose(res_alpha, -0.5)

    # bootstrap_ci_table
    res_ci = bootstrap_ci_table(
        make_table(),
        coder_col="coder",
        unit_col="unit",
        value_col="coder",
        level_of_measurement="nominal",
        n_resamples=10,
        random_state=42,
    )
    assert np.isclose(res_ci.alpha, -0.5)


def test_avoid_reference_sized_allocations_for_object_sample() -> None:
    # A small object sample against a large numeric reference dataset
    rng = np.random.default_rng(42)
    large_ref = rng.choice([1, 2, 3, 4, 5], size=(2, 5000))
    sample = np.array([["1", "2"], ["1", "2"]], dtype=object)

    res = alpha(sample, all_reliability_data=large_ref, level_of_measurement="nominal")
    assert isinstance(res, float)
    assert not np.isnan(res)


def test_prevent_overflow_when_normalizing_integral_floating_point_domains() -> None:
    # Copilot review finding (discussion_r4279405240):
    # Signed floating point domain case:
    # ratings [[-5e18, 0.0, 5e18], [-5e18, 5e18, 0.0]] with value_domain=[-5e18, 0.0, 5e18]
    # must not overflow int64 subtraction; expected alpha is 7/12.
    ratings_signed_f = [[-5e18, 0.0, 5e18], [-5e18, 5e18, 0.0]]
    domain_signed_f = [-5e18, 0.0, 5e18]
    alpha_signed_f = alpha(ratings_signed_f, value_domain=domain_signed_f, level_of_measurement="interval")
    assert np.isclose(alpha_signed_f, 7.0 / 12.0)

    ci_signed_f = bootstrap_ci(
        ratings_signed_f, value_domain=domain_signed_f, level_of_measurement="interval", n_resamples=10, random_state=42
    )
    assert np.isclose(ci_signed_f.alpha, 7.0 / 12.0)

    # Native int64 array case
    domain_i64 = np.array([-5_000_000_000_000_000_000, 0, 5_000_000_000_000_000_000], dtype=np.int64)
    ratings_i64 = np.array(
        [
            [-5_000_000_000_000_000_000, 0, 5_000_000_000_000_000_000],
            [-5_000_000_000_000_000_000, 5_000_000_000_000_000_000, 0],
        ],
        dtype=np.int64,
    )
    assert np.isclose(alpha(ratings_i64, value_domain=domain_i64, level_of_measurement="interval"), 7.0 / 12.0)

    # Unsigned floating point domain case:
    # ratings [[1e19, 1.25e19, 1.5e19], [1e19, 1.5e19, 1.25e19]] with value_domain=[1e19, 1.25e19, 1.5e19]
    # must not wrap in uint64; expected alpha is 7/12.
    ratings_unsigned_f = [[1e19, 1.25e19, 1.5e19], [1e19, 1.5e19, 1.25e19]]
    domain_unsigned_f = [1e19, 1.25e19, 1.5e19]
    alpha_unsigned_f = alpha(ratings_unsigned_f, value_domain=domain_unsigned_f, level_of_measurement="interval")
    assert np.isclose(alpha_unsigned_f, 7.0 / 12.0)

    ci_unsigned_f = bootstrap_ci(
        ratings_unsigned_f,
        value_domain=domain_unsigned_f,
        level_of_measurement="interval",
        n_resamples=10,
        random_state=42,
    )
    assert np.isclose(ci_unsigned_f.alpha, 7.0 / 12.0)

    # Native uint64 array case
    domain_u64 = np.array(
        [10_000_000_000_000_000_000, 12_500_000_000_000_000_000, 15_000_000_000_000_000_000], dtype=np.uint64
    )
    ratings_u64 = np.array(
        [
            [10_000_000_000_000_000_000, 12_500_000_000_000_000_000, 15_000_000_000_000_000_000],
            [10_000_000_000_000_000_000, 15_000_000_000_000_000_000, 12_500_000_000_000_000_000],
        ],
        dtype=np.uint64,
    )
    assert np.isclose(alpha(ratings_u64, value_domain=domain_u64, level_of_measurement="interval"), 7.0 / 12.0)


def test_stream_annotation_normalization_coder_mappings() -> None:
    # Copilot review finding (discussion_r4279405304):
    # Large coder mapping with few categories must stream normalization into dict.fromkeys.
    c1 = {f"u_{i}": 1 if i % 2 == 0 else 2 for i in range(2000)}
    c2 = {f"u_{i}": 1 if i % 2 == 0 else 2 for i in range(2000)}
    data = [c1, c2]

    res_alpha = alpha(data, level_of_measurement="nominal")
    assert np.isclose(res_alpha, 1.0)


def test_preserve_input_precision_interval_and_ratio_metrics() -> None:
    # Copilot review findings (discussion_r4279483025, discussion_r4279483052):
    # Preserving input precision when output dtype is narrower (e.g. complex64).
    a = 1.0
    b = 1.0 + 2**-30
    ratings = [[a, b], [a, b]]

    res_interval = alpha(ratings, level_of_measurement="interval", dtype=np.complex64)
    assert np.isclose(res_interval, 1.0)

    res_ratio = alpha(ratings, level_of_measurement="ratio", dtype=np.complex64)
    assert np.isclose(res_ratio, 1.0)

    # Extended-precision longdouble with default float64 output
    if np.finfo(np.longdouble).precision > np.finfo(np.float64).precision:
        a_ld = np.longdouble(1.0)
        eps = np.finfo(np.longdouble).eps
        b_ld = a_ld + eps
        ratings_ld = np.array([[a_ld, b_ld], [a_ld, b_ld]])

        res_interval_ld = alpha(ratings_ld, level_of_measurement="interval", dtype=np.float64)
        assert np.isclose(res_interval_ld, 1.0)

        res_ratio_ld = alpha(ratings_ld, level_of_measurement="ratio", dtype=np.float64)
        assert np.isclose(res_ratio_ld, 1.0)

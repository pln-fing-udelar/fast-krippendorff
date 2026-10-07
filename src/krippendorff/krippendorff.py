"""
This module provides a function to compute the Krippendorff's alpha statistical measure of the agreement achieved
when coding a set of units based on the values of a variable.

For more information, see: https://en.wikipedia.org/wiki/Krippendorff%27s_alpha

The module naming follows the one from the Wikipedia link.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

DEFAULT_DTYPE = np.float64

__all__ = [
    "DEFAULT_DTYPE",
    "DistanceMetric",
    "LevelOfMeasurement",
    "MetricResultScalarType",
    "ValueScalarType",
    "alpha",
]


ValueScalarType = TypeVar("ValueScalarType", bound=np.generic)
MetricResultScalarType = TypeVar("MetricResultScalarType", bound=np.inexact)


class DistanceMetric(Protocol):
    def __call__(
        self,
        v1: npt.NDArray[ValueScalarType],
        v2: npt.NDArray[ValueScalarType],
        i1: npt.NDArray[np.int_],
        i2: npt.NDArray[np.int_],
        n_v: npt.NDArray[MetricResultScalarType],
        dtype: np.dtype[MetricResultScalarType] = DEFAULT_DTYPE,  # ty:ignore[invalid-parameter-default]
    ) -> npt.NDArray[MetricResultScalarType]:
        """Computes the distance for two arrays element-wise.

        Parameters
        ----------
        v1 : ndarray
            First array of values.

        v2 : ndarray
            Second array of values. If `v1.shape != v2.shape`, they must be broadcastable to a common shape (which
            becomes the shape of the output).

        i1 : ndarray
            Ordinal indices of the first array (with the same shape).

        i2 : ndarray
            Ordinal indices of the second array (with the same shape).

        n_v : ndarray, with shape (V,)
            Number of pairable elements for each value.

        dtype : data-type
            Result and computation data-type.

        Returns
        -------
        d : ndarray, with shape (A, B)
            Distance between `v1` and `v2`, element-wise.
        """


LevelOfMeasurement = Literal["nominal", "ordinal", "interval", "ratio"] | DistanceMetric


def _nominal_metric(
    v1: npt.NDArray[ValueScalarType],
    v2: npt.NDArray[ValueScalarType],
    i1: npt.NDArray[np.int_],
    i2: npt.NDArray[np.int_],
    n_v: npt.NDArray[MetricResultScalarType],
    dtype: np.dtype[MetricResultScalarType] = DEFAULT_DTYPE,  # ty:ignore[invalid-parameter-default]
) -> npt.NDArray[MetricResultScalarType]:
    """Metric for nominal data."""
    return (v1 != v2).astype(dtype)


def _ordinal_metric(
    v1: npt.NDArray[ValueScalarType],
    v2: npt.NDArray[ValueScalarType],
    i1: npt.NDArray[np.int_],
    i2: npt.NDArray[np.int_],
    n_v: npt.NDArray[np.number],
    dtype: np.dtype[MetricResultScalarType] = DEFAULT_DTYPE,  # ty:ignore[invalid-parameter-default]
) -> npt.NDArray[MetricResultScalarType]:
    """Metric for ordinal data."""
    i1, i2 = np.minimum(i1, i2), np.maximum(i1, i2)

    ranges = np.dstack((i1, i2 + 1))
    sums_between_indices = np.add.reduceat(np.append(n_v, 0), ranges.reshape(-1))[::2].reshape(*i1.shape)

    return (sums_between_indices - np.divide(n_v[i1] + n_v[i2], 2, dtype=dtype)) ** 2


def _interval_metric(
    v1: npt.NDArray[ValueScalarType],
    v2: npt.NDArray[ValueScalarType],
    i1: npt.NDArray[np.int_],
    i2: npt.NDArray[np.int_],
    n_v: npt.NDArray[np.number],
    dtype: np.dtype[MetricResultScalarType] = DEFAULT_DTYPE,  # ty:ignore[invalid-parameter-default]
) -> npt.NDArray[MetricResultScalarType]:
    """Metric for interval data."""
    return (v1 - v2).astype(dtype) ** 2  # ty:ignore[unsupported-operator]


def _ratio_metric(
    v1: npt.NDArray[ValueScalarType],
    v2: npt.NDArray[ValueScalarType],
    i1: npt.NDArray[np.int_],
    i2: npt.NDArray[np.int_],
    n_v: npt.NDArray[np.number],
    dtype: np.dtype[MetricResultScalarType] = DEFAULT_DTYPE,  # ty:ignore[invalid-parameter-default]
) -> npt.NDArray[MetricResultScalarType]:
    """Metric for ratio data."""
    v1_plus_v2 = v1 + v2  # ty:ignore[unsupported-operator]
    return (
        np.divide(v1 - v2, v1_plus_v2, out=np.zeros(np.broadcast(v1, v2).shape), where=v1_plus_v2 != 0, dtype=dtype)  # ty:ignore[unsupported-operator]
        ** 2
    )


def _coincidences(
    value_counts: npt.NDArray[np.int_],
    dtype: np.dtype[MetricResultScalarType] = DEFAULT_DTYPE,  # ty:ignore[invalid-parameter-default]
) -> npt.NDArray[MetricResultScalarType]:
    """Coincidence matrix.

    Parameters
    ----------
    value_counts : ndarray, with shape (N, V)
        Number of coders that assigned a certain value to a determined unit, where N is the number of units
        and V is the value count.

    dtype : data-type
        Result and computation data-type.

    Returns
    -------
    o : ndarray, with shape (V, V)
        Coincidence matrix.
    """
    _, V = value_counts.shape  # noqa: N806
    pairable = np.maximum(value_counts.sum(axis=1), 2)
    diagonals = value_counts[:, np.newaxis, :] * np.eye(V)[np.newaxis, ...]
    unnormalized_coincidences = value_counts[..., np.newaxis] * value_counts[:, np.newaxis, :] - diagonals
    return np.divide(unnormalized_coincidences, (pairable - 1).reshape((-1, 1, 1)), dtype=dtype).sum(axis=0)


def _random_coincidences(
    n_v: npt.NDArray[MetricResultScalarType],
    dtype: np.dtype[MetricResultScalarType] = DEFAULT_DTYPE,  # ty:ignore[invalid-parameter-default]
) -> npt.NDArray[MetricResultScalarType]:
    """Random coincidence matrix.

    Parameters
    ----------
    n_v : ndarray, with shape (V,)
        Number of pairable elements for each value.

    dtype : data-type
        Result and computation data-type.

    Returns
    -------
    e : ndarray, with shape (V, V)
        Random coincidence matrix.
    """
    return np.divide(np.outer(n_v, n_v) - np.diagflat(n_v), n_v.sum() - 1, dtype=dtype)


def _distances(
    value_domain: npt.NDArray[ValueScalarType],
    distance_metric: DistanceMetric,
    n_v: npt.NDArray[np.int_],
    dtype: np.dtype[MetricResultScalarType] = DEFAULT_DTYPE,  # ty:ignore[invalid-parameter-default]
) -> npt.NDArray[MetricResultScalarType]:
    """Distances of the different possible values.

    Parameters
    ----------
    value_domain : ndarray, with shape (V,)
        Possible values V the units can take.
        If the level of measurement is not nominal, it must be ordered.

    distance_metric : callable
        Callable that returns the distance of two given values.

    n_v : ndarray, with shape (V,)
        Number of pairable elements for each value.

    dtype : data-type
        Result and computation data-type.

    Returns
    -------
    d : ndarray, with shape (V, V)
        Distance matrix for each value pair.
    """
    indices = np.arange(len(value_domain))
    return distance_metric(
        value_domain[:, np.newaxis],
        value_domain[np.newaxis, :],
        i1=indices[:, np.newaxis],
        i2=indices[np.newaxis, :],
        n_v=n_v,  # ty:ignore[invalid-argument-type]
        dtype=dtype,
    )


def _distance_metric(level_of_measurement: LevelOfMeasurement) -> DistanceMetric:
    """Distance metric callable of the level of measurement.

    Parameters
    ----------
    level_of_measurement : string or callable
        Steven's level of measurement of the variable.
        It must be one of "nominal", "ordinal", "interval", "ratio", or a callable.

    Returns
    -------
    metric : callable
        Distance callable.
    """
    return {
        "nominal": _nominal_metric,
        "ordinal": _ordinal_metric,
        "interval": _interval_metric,
        "ratio": _ratio_metric,
    }.get(level_of_measurement, level_of_measurement)  # ty:ignore[invalid-return-type]


def _reliability_data_to_value_counts(
    reliability_data: npt.NDArray[ValueScalarType], value_domain: npt.NDArray[ValueScalarType]
) -> npt.NDArray[np.int_]:
    """Return the value counts given the reliability data.

    Parameters
    ----------
    reliability_data : ndarray, with shape (M, N)
        Reliability data matrix which has the rate the i coder gave to the j unit, where M is the number of raters
        and N is the unit count.
        Missing rates are represented with `np.nan`.

    value_domain : ndarray, with shape (V,)
        Possible values the units can take.

    Returns
    -------
    value_counts : ndarray, with shape (N, V)
        Number of coders that assigned a certain value to a determined unit, where N is the number of units
        and V is the value count.
    """
    return (reliability_data.T[..., np.newaxis] == value_domain[np.newaxis, np.newaxis, :]).sum(axis=1)


def _is_dict_reliability_data(data: Any) -> bool:
    """Check if the data is structured as dictionary/mapping annotations."""
    if isinstance(data, Mapping):
        return True
    if (
        isinstance(data, Sequence)
        and not isinstance(data, (str, bytes))
        and len(data) > 0
        and any(isinstance(x, Mapping) for x in data)
    ):
        return True
    return False


def _is_missing(val: Any) -> bool:
    """Check if a value represents a missing annotation."""
    if val is None:
        return True
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return True
    return False


def _is_numeric_scalar(val: Any) -> bool:
    """Check if a value is a real numeric scalar type."""
    return isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool)


def _to_domain_array(domain_values: Any) -> npt.NDArray:
    """Convert domain values to a 1-D ndarray, preserving object dtype for heterogeneous types."""
    if isinstance(domain_values, np.ndarray):
        return domain_values
    domain_list = (
        list(domain_values)
        if hasattr(domain_values, "__iter__") and not isinstance(domain_values, (str, bytes))
        else list(np.asarray(domain_values))
    )
    if all(isinstance(v, str) for v in domain_list):
        return np.asarray(domain_values)
    if all(isinstance(v, bytes) for v in domain_list):
        return np.asarray(domain_values)
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in domain_list):
        return np.asarray(domain_values)
    if all(isinstance(v, (float, np.floating)) for v in domain_list):
        return np.asarray(domain_values)
    if all(_is_numeric_scalar(v) for v in domain_list):
        try:
            as_arr = np.asarray(domain_values)
            if (
                as_arr.ndim == 1
                and len(as_arr) == len(domain_list)
                and len(set(as_arr.tolist())) == len(set(domain_list))
                and all(v == arr_v.item() for v, arr_v in zip(domain_list, as_arr, strict=True))
            ):
                return as_arr
        except Exception:
            pass
    arr = np.empty(len(domain_list), dtype=object)
    for i, v in enumerate(domain_list):
        arr[i] = v
    return arr


def _extract_coder_dicts(
    data: Mapping[Any, Mapping[Any, Any]] | Sequence[Mapping[Any, Any]],
) -> list[Mapping[Any, Any]]:
    """Extract and validate coder mapping annotations."""
    if isinstance(data, Mapping):
        for coder, coder_dict in data.items():
            if not isinstance(coder_dict, Mapping):
                raise ValueError(
                    f"Expected a mapping of coders to unit annotations, but coder {coder!r} "
                    f"has non-mapping annotations of type {type(coder_dict).__name__}."
                )
        return list(data.values())
    for i, coder_dict in enumerate(data):
        if not isinstance(coder_dict, Mapping):
            raise ValueError(
                f"Expected a sequence of mappings for coder annotations, but element {i} "
                f"has non-mapping annotations of type {type(coder_dict).__name__}."
            )
    return list(data)


def _domain_from_raw_values(
    raw_values: list[Any],
    value_domain: npt.ArrayLike | None,
    level_of_measurement: LevelOfMeasurement,
) -> npt.NDArray:
    """Compute and validate the value domain from extracted mapping values."""
    unique_vals = list(dict.fromkeys(raw_values))
    try:
        unique_vals = sorted(unique_vals)
    except TypeError:
        pass

    all_numeric = all(_is_numeric_scalar(v) for v in unique_vals)
    all_str = all(isinstance(v, str) for v in unique_vals)
    all_bytes = all(isinstance(v, bytes) for v in unique_vals)
    has_mixed_types = not (all_numeric or all_str or all_bytes)
    computed_domain = _to_domain_array(unique_vals)

    if value_domain is None:
        if has_mixed_types and level_of_measurement != "nominal":
            raise ValueError(
                "When using mixed types, an ordered value_domain is required "
                "for level_of_measurement other than 'nominal'."
            )
        if len(raw_values) > 0 and isinstance(raw_values[0], (str, bytes)) and level_of_measurement != "nominal":
            raise ValueError(
                "When using strings, an ordered value_domain is required for level_of_measurement other than 'nominal'."
            )
        return computed_domain

    domain_arr = _to_domain_array(value_domain)
    domain_set = set(domain_arr)
    if any(v not in domain_set for v in unique_vals):
        raise ValueError("The reliability data contains out-of-domain values.")
    return domain_arr


def _dict_reliability_data_to_value_counts(
    data: Mapping[Any, Mapping[Any, Any]] | Sequence[Mapping[Any, Any]],
    value_domain: npt.ArrayLike | None,
    level_of_measurement: LevelOfMeasurement,
) -> tuple[npt.NDArray[np.int_], npt.NDArray]:
    """Convert dictionary-based reliability data directly into value counts."""
    coder_dicts = _extract_coder_dicts(data)

    units = list(dict.fromkeys(u for d in coder_dicts for u in d.keys()))
    raw_values = [v for d in coder_dicts for v in d.values() if not _is_missing(v)]
    domain_arr = _domain_from_raw_values(raw_values, value_domain, level_of_measurement)

    unit_to_idx = {u: i for i, u in enumerate(units)}
    val_to_idx = {v: i for i, v in enumerate(domain_arr)}

    value_counts = np.zeros((len(units), len(domain_arr)), dtype=np.int_)
    for d in coder_dicts:
        for u, v in d.items():
            if not _is_missing(v):
                idx = val_to_idx.get(v)
                if idx is None:
                    raise ValueError("The reliability data contains out-of-domain values.")
                value_counts[unit_to_idx[u], idx] += 1

    return value_counts, domain_arr


def alpha(  # noqa: C901
    reliability_data: npt.ArrayLike | Mapping[Any, Mapping[Any, Any]] | Sequence[Mapping[Any, Any]] | None = None,
    value_counts: npt.ArrayLike | None = None,
    value_domain: npt.ArrayLike | None = None,
    level_of_measurement: LevelOfMeasurement = "interval",
    dtype: npt.DTypeLike = DEFAULT_DTYPE,
) -> float:
    """Compute Krippendorff's alpha.

    See https://en.wikipedia.org/wiki/Krippendorff%27s_alpha for more information.

    Parameters
    ----------
    reliability_data : array_like or sequence of dicts or dict of dicts, optional
        Reliability data containing the ratings assigned by coders to units.
        Can be:
        - A 2D array_like of shape (M, N) where M is the number of coders and N is the unit count.
          Missing rates are represented with `np.nan`.
        - A sequence of dicts where each dict represents a coder mapping units to values:
          `[{unit1: val, unit2: val}, {unit1: val, ...}]`.
        - A dict of dicts where outer keys represent coders and inner dicts map units to values:
          `{coder1: {unit1: val, ...}, coder2: {unit1: val, ...}}`.
        If it's provided then `value_counts` must not be provided.

    value_counts : array_like, with shape (N, V)
        Number of coders that assigned a certain value to a determined unit, where N is the number of units
        and V is the value count.
        If it's provided then `reliability_data` must not be provided.

    value_domain : array_like, with shape (V,)
        Possible values the units can take.
        If the level of measurement is not nominal, it must be ordered.
        If `reliability_data` is provided, then the default value is the ordered list of unique rates that appear.
        Else, the default value is `list(range(V))`.

    level_of_measurement : string or callable
        Steven's level of measurement of the variable.
        It must be one of "nominal", "ordinal", "interval", "ratio", or a callable.

    dtype : data-type
        Result and computation data-type.

    Returns
    -------
    alpha : ndarray
        Scalar value of Krippendorff's alpha of type `dtype`.

    Examples
    --------
    >>> reliability_data = [[np.nan, np.nan, np.nan, np.nan, np.nan, 3, 4, 1, 2, 1, 1, 3, 3, np.nan, 3],
    ...                     [1, np.nan, 2, 1, 3, 3, 4, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ...                     [np.nan, np.nan, 2, 1, 3, 4, 4, np.nan, 2, 1, 1, 3, 3, np.nan, 4]]
    >>> print(round(alpha(reliability_data=reliability_data, level_of_measurement="nominal"), 6))
    0.691358
    >>> print(round(alpha(reliability_data=reliability_data, level_of_measurement="interval"), 6))
    0.810845
    >>> value_counts = np.array([[1, 0, 0, 0],
    ...                          [0, 0, 0, 0],
    ...                          [0, 2, 0, 0],
    ...                          [2, 0, 0, 0],
    ...                          [0, 0, 2, 0],
    ...                          [0, 0, 2, 1],
    ...                          [0, 0, 0, 3],
    ...                          [1, 0, 1, 0],
    ...                          [0, 2, 0, 0],
    ...                          [2, 0, 0, 0],
    ...                          [2, 0, 0, 0],
    ...                          [0, 0, 2, 0],
    ...                          [0, 0, 2, 0],
    ...                          [0, 0, 0, 0],
    ...                          [0, 0, 1, 1]])
    >>> print(round(alpha(value_counts=value_counts, level_of_measurement="nominal"), 6))
    0.691358
    >>> # The following examples were extracted from
    >>> # https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2016/07/fulltext.pdf, page 8.
    >>> reliability_data = [[1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
    ...                     [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
    ...                     [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
    ...                     [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan]]
    >>> print(round(alpha(reliability_data, level_of_measurement="ordinal"), 3))
    0.815
    >>> print(round(alpha(reliability_data, value_domain=[1,2,3,4,5], level_of_measurement="ordinal"), 3))
    0.815
    >>> print(round(alpha(reliability_data, level_of_measurement="ratio"), 3))
    0.797
    >>> reliability_data = [["very low", "low", "mid", "mid", "low", "very low", "high", "very low", "low", np.nan,
    ...                      np.nan, np.nan],
    ...                     ["very low", "low", "mid", "mid", "low", "low", "high", "very low", "low", "very high",
    ...                      np.nan, "mid"],
    ...                     [np.nan, "mid", "mid", "mid", "low", "mid", "high", "low", "low", "very high", "very low",
    ...                      np.nan],
    ...                     ["very low", "low", "mid", "mid", "low", "high", "high", "very low", "low", "very high",
    ...                      "very low", np.nan]]
    >>> print(round(alpha(reliability_data, level_of_measurement="ordinal",
    ...                   value_domain=["very low", "low", "mid", "high", "very high"]), 3))
    0.815
    >>> # Note that without an ordered value_domain, we can only calculate nominal distances on strings.
    >>> print(round(alpha(reliability_data, level_of_measurement="nominal"), 3))
    0.743
    >>> # Sequence of dicts example:
    >>> data_dicts = [
    ...     {"u1": 1, "u2": 2, "u3": 3},
    ...     {"u1": 1, "u2": 2, "u3": 4},
    ...     {"u2": 2, "u3": 3},
    ... ]
    >>> print(round(alpha(reliability_data=data_dicts, level_of_measurement="interval"), 3))
    0.883
    """
    if (reliability_data is None) == (value_counts is None):
        raise ValueError("Either reliability_data or value_counts must be provided, but not both.")

    # Don't know if it's a `list` or NumPy array. If it's the latter, the truth value is ambiguous. So, ask for `None`.
    if value_counts is None:
        if _is_dict_reliability_data(reliability_data):
            value_counts, value_domain = _dict_reliability_data_to_value_counts(
                reliability_data,  # ty:ignore[invalid-argument-type]
                value_domain,
                level_of_measurement,
            )
        else:
            reliability_data = np.asarray(reliability_data)

            kind = reliability_data.dtype.kind
            if kind in {"i", "u", "f"}:
                # `np.isnan` only operates on signed integers, unsigned integers, and floats, not strings.
                computed_value_domain = np.unique(reliability_data[~np.isnan(reliability_data)])
            elif kind in {"U", "S"}:  # Unicode or byte string.
                # `np.asarray` will coerce `np.nan` values to "nan".
                computed_value_domain = np.unique(reliability_data[reliability_data != "nan"])
            else:
                raise ValueError(f"Don't know how to construct value domain for dtype kind {kind}.")

            if value_domain is None:
                # Check if Unicode or byte string.
                if kind in {"U", "S"} and level_of_measurement != "nominal":
                    raise ValueError(
                        "When using strings, an ordered value_domain is required"
                        " for level_of_measurement other than 'nominal'."
                    )
                value_domain = computed_value_domain
            else:
                value_domain = _to_domain_array(value_domain)
                # Note: We do not need to test for `np.nan` in the input data.
                # `np.nan` indicates the absence of a domain value and is always allowed.
                if not np.isin(computed_value_domain, value_domain).all():
                    raise ValueError("The reliability data contains out-of-domain values.")

            value_counts = _reliability_data_to_value_counts(reliability_data, value_domain)
    else:
        value_counts = np.asarray(value_counts)

        if value_domain is None:
            value_domain = np.arange(value_counts.shape[1])
        else:
            value_domain = _to_domain_array(value_domain)
            if value_counts.shape[1] != len(value_domain):
                raise ValueError("The value domain should be equal to the number of columns of value_counts.")

    if len(value_domain) <= 1:
        raise ValueError("There has to be more than one value in the domain.")

    if (value_counts.sum(axis=-1) <= 1).all():
        raise ValueError("There has to be at least one unit with values assigned by at least two coders.")

    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.inexact):
        raise ValueError("`dtype` must be an inexact type.")

    distance_metric = _distance_metric(level_of_measurement)

    o = _coincidences(value_counts, dtype=dtype)
    n_v = o.sum(axis=0)
    e = _random_coincidences(n_v, dtype=dtype)
    d = _distances(value_domain, distance_metric, n_v, dtype=dtype)
    return 1 - (o * d).sum() / (e * d).sum()

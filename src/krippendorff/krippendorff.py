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
    "bipolar_metric",
    "circular_metric",
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


LevelOfMeasurement = Literal["nominal", "ordinal", "interval", "ratio", "circular", "bipolar"] | DistanceMetric


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


def _has_complex(arr: npt.NDArray) -> bool:
    """Check if an array has complex dtype or contains complex elements in object dtype."""
    return np.iscomplexobj(arr) or any(isinstance(x, (complex, np.complexfloating)) for x in arr.flat)


def _is_finite_array(arr: npt.NDArray) -> bool:
    """Check if an array contains exclusively finite numeric values."""
    try:
        if np.issubdtype(arr.dtype, np.number):
            return bool(np.isfinite(arr).all())
        return bool(np.isfinite(arr.astype(float)).all())
    except (TypeError, ValueError):
        return False


def _is_all_integer(arr: npt.NDArray) -> bool:
    """Check if all elements of an array represent integer values."""
    if np.issubdtype(arr.dtype, np.integer):
        return True
    return bool(np.all(arr.astype(float) % 1 == 0))


def _safe_diff(a: Any, b: Any) -> Any:
    """Compute difference safely, avoiding integer underflow/overflow and precision loss."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if np.issubdtype(a_arr.dtype, np.floating) or np.issubdtype(b_arr.dtype, np.floating):
        return a - b
    if np.issubdtype(a_arr.dtype, np.integer) or np.issubdtype(b_arr.dtype, np.integer):
        if a_arr.dtype == np.uint64 or b_arr.dtype == np.uint64:
            return np.asarray(a, dtype=object) - np.asarray(b, dtype=object)
        return np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64)
    return a - b


def circular_metric(
    circumference: float | None = None,
) -> DistanceMetric:
    """Distance metric factory for circular data.

    Parameters
    ----------
    circumference : float, optional
        Circumference U of the circular scale (period before values repeat).
        If None, U is calculated as `max(values) - min(values) + 1` for discrete integer categories.

    Returns
    -------
    metric : DistanceMetric
        Callable that computes circular distance between two arrays element-wise.
    """
    if circumference is not None and (not np.isfinite(circumference) or circumference <= 0):
        raise ValueError("Circumference must be a finite, positive number.")

    def _metric(
        v1: npt.NDArray[ValueScalarType],
        v2: npt.NDArray[ValueScalarType],
        i1: npt.NDArray[np.int_],
        i2: npt.NDArray[np.int_],
        n_v: npt.NDArray[MetricResultScalarType],
        dtype: np.dtype[MetricResultScalarType] = DEFAULT_DTYPE,  # ty:ignore[invalid-parameter-default]
    ) -> npt.NDArray[MetricResultScalarType]:
        if _has_complex(v1) or _has_complex(v2):
            raise ValueError("Circular metric does not support complex values.")
        if not _is_finite_array(v1) or not _is_finite_array(v2):
            raise ValueError("Circular metric requires finite values.")
        if circumference is None:
            if not (_is_all_integer(v1) and _is_all_integer(v2)):
                raise ValueError("An explicit circumference must be provided for non-integer circular data.")
            v_min = min(np.min(v1), np.min(v2))  # ty:ignore[invalid-argument-type]
            v_max = max(np.max(v1), np.max(v2))  # ty:ignore[invalid-argument-type]
            u = _safe_diff(v_max, v_min) + 1
        else:
            u = circumference
        diff = _safe_diff(v1, v2)
        real_dtype = np.empty((), dtype=dtype).real.dtype
        diff_scaled = np.asarray(diff, dtype=real_dtype) / float(u)
        return (np.sin(np.pi * diff_scaled) ** 2).astype(dtype)

    return _metric


def bipolar_metric(
    low: float | None = None,
    high: float | None = None,
) -> DistanceMetric:
    """Distance metric factory for bipolar data.

    Parameters
    ----------
    low : float, optional
        The lowest possible value on the bipolar scale. If None, defaults to `min(values)`.
    high : float, optional
        The highest possible value on the bipolar scale. If None, defaults to `max(values)`.

    Returns
    -------
    metric : DistanceMetric
        Callable that computes bipolar distance between two arrays element-wise.
    """
    if low is not None and not np.isfinite(low):
        raise ValueError("low must be a finite number.")
    if high is not None and not np.isfinite(high):
        raise ValueError("high must be a finite number.")
    if low is not None and high is not None and low >= high:
        raise ValueError("low must be strictly less than high.")

    def _metric(
        v1: npt.NDArray[ValueScalarType],
        v2: npt.NDArray[ValueScalarType],
        i1: npt.NDArray[np.int_],
        i2: npt.NDArray[np.int_],
        n_v: npt.NDArray[MetricResultScalarType],
        dtype: np.dtype[MetricResultScalarType] = DEFAULT_DTYPE,  # ty:ignore[invalid-parameter-default]
    ) -> npt.NDArray[MetricResultScalarType]:
        if _has_complex(v1) or _has_complex(v2):
            raise ValueError("Bipolar metric does not support complex values.")
        if not _is_finite_array(v1) or not _is_finite_array(v2):
            raise ValueError("Bipolar metric requires finite values.")
        v_min = low if low is not None else min(np.min(v1), np.min(v2))  # ty:ignore[invalid-argument-type]
        v_max = high if high is not None else max(np.max(v1), np.max(v2))  # ty:ignore[invalid-argument-type]
        if v_min >= v_max:
            raise ValueError("low must be strictly less than high.")
        if (v1 < v_min).any() or (v1 > v_max).any() or (v2 < v_min).any() or (v2 > v_max).any():  # ty:ignore[unsupported-operator]
            raise ValueError("The data contains out-of-bounds values for the specified bipolar endpoints.")

        diff = _safe_diff(v1, v2)
        term1 = _safe_diff(v1, v_min) + _safe_diff(v2, v_min)
        term2 = _safe_diff(v_max, v1) + _safe_diff(v_max, v2)
        nonzero = v1 != v2
        real_dtype = np.empty((), dtype=dtype).real.dtype
        diff_f = np.asarray(diff, dtype=real_dtype)
        term1_f = np.asarray(term1, dtype=real_dtype)
        term2_f = np.asarray(term2, dtype=real_dtype)
        num = diff_f**2
        denom = term1_f * term2_f
        res = np.divide(
            num,
            denom,
            out=np.zeros(np.broadcast(v1, v2).shape, dtype=real_dtype),
            where=nonzero,
        )
        return res.astype(dtype)

    return _metric


_circular_metric = circular_metric()
_bipolar_metric = bipolar_metric()


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
    pairable = np.maximum(value_counts.sum(axis=1), 2)
    weights = np.divide(value_counts, (pairable - 1)[:, np.newaxis], dtype=dtype)
    value_counts_float = value_counts.astype(dtype, copy=False)
    coincidences = np.dot(value_counts_float.T, weights)
    diagonal = np.sum((value_counts_float - 1) * weights, axis=0, dtype=dtype)
    np.fill_diagonal(coincidences, diagonal)
    return coincidences


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
        It must be one of "nominal", "ordinal", "interval", "ratio", "circular", "bipolar", or a callable.

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
        "circular": _circular_metric,
        "bipolar": _bipolar_metric,
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
    if isinstance(val, (complex, np.complexfloating)) and (np.isnan(val.real) or np.isnan(val.imag)):
        return True
    return False


def _is_numeric_scalar(val: Any) -> bool:
    """Check if a value is a real numeric scalar type."""
    return isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool)


def _to_domain_array(domain_values: Any) -> npt.NDArray:  # noqa: C901
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
        try:
            as_arr = np.asarray(domain_values)
            if np.issubdtype(as_arr.dtype, np.integer):
                return as_arr
        except Exception:
            pass
        arr = np.empty(len(domain_list), dtype=object)
        for i, v in enumerate(domain_list):
            arr[i] = v
        return arr
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
    distinct_values: Sequence[Any],
    value_domain: npt.ArrayLike | None,
    level_of_measurement: LevelOfMeasurement,
) -> npt.NDArray:
    """Compute and validate the value domain from extracted mapping values."""
    unique_vals = list(distinct_values)
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
        if (
            len(distinct_values) > 0
            and isinstance(distinct_values[0], (str, bytes))
            and level_of_measurement != "nominal"
        ):
            raise ValueError(
                "When using strings, an ordered value_domain is required for level_of_measurement other than 'nominal'."
            )
        return computed_domain

    domain_arr = _to_domain_array(value_domain)
    domain_set = set(domain_arr)
    if any(v not in domain_set for v in unique_vals):
        raise ValueError("The reliability data contains out-of-domain values.")
    if level_of_measurement in ("interval", "ratio", "circular", "bipolar") and (
        np.iscomplexobj(domain_arr) or any(isinstance(v, (complex, np.complexfloating)) for v in domain_arr)
    ):
        raise ValueError(f"Level of measurement {level_of_measurement!r} does not support complex values.")
    return domain_arr


def _dict_reliability_data_to_value_counts(
    data: Mapping[Any, Mapping[Any, Any]] | Sequence[Mapping[Any, Any]],
    value_domain: npt.ArrayLike | None,
    level_of_measurement: LevelOfMeasurement,
) -> tuple[npt.NDArray[np.int_], npt.NDArray]:
    """Convert dictionary-based reliability data directly into value counts."""
    coder_dicts = _extract_coder_dicts(data)

    units = list(dict.fromkeys(u for d in coder_dicts for u in d.keys()))
    distinct_values = list(dict.fromkeys(v for d in coder_dicts for v in d.values() if not _is_missing(v)))
    domain_arr = _domain_from_raw_values(distinct_values, value_domain, level_of_measurement)

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


def _domain_from_reliability_data(reliability_data: npt.NDArray) -> npt.NDArray:
    """Extract unique non-missing values from reliability data."""
    kind = reliability_data.dtype.kind
    if kind in {"i", "u", "f"}:
        return np.unique(reliability_data[~np.isnan(reliability_data)])
    elif kind in {"U", "S"}:
        return np.unique(reliability_data[reliability_data != "nan"])
    raise ValueError(f"Don't know how to construct value domain for dtype kind {kind}.")


def alpha(  # noqa: C901
    reliability_data: npt.ArrayLike | Mapping[Any, Mapping[Any, Any]] | Sequence[Mapping[Any, Any]] | None = None,
    value_counts: npt.ArrayLike | None = None,
    value_domain: npt.ArrayLike | None = None,
    level_of_measurement: LevelOfMeasurement = "interval",
    dtype: npt.DTypeLike = DEFAULT_DTYPE,
    *,
    all_reliability_data: npt.ArrayLike | None = None,
    all_value_counts: npt.ArrayLike | None = None,
    random_coincidences: npt.ArrayLike | None = None,
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
        It must be one of "nominal", "ordinal", "interval", "ratio", "circular", "bipolar", or a callable.

    dtype : data-type
        Result and computation data-type.

    all_reliability_data : array_like, with shape (M, N_all), optional
        Reference reliability data matrix representing the full population or reference dataset,
        used to compute expected random coincidences (and distances for ordinal measurement)
        when computing alpha for subsamples.
        At most one of `all_reliability_data`, `all_value_counts`, or `random_coincidences` can be provided.

    all_value_counts : array_like, with shape (N_all, V), optional
        Reference value counts representing the full population or reference dataset,
        used to compute expected random coincidences (and distances for ordinal measurement)
        when computing alpha for subsamples.
        At most one of `all_reliability_data`, `all_value_counts`, or `random_coincidences` can be provided.

    random_coincidences : array_like, with shape (V, V), optional
        Precomputed random coincidences matrix representing expected chance agreement.
        It must be symmetric, non-negative, finite, and have a positive sum.
        Note that if a custom distance metric callable depends on unnormalized pairable counts
        `n_v`, `random_coincidences` should provide unnormalized counts rather than proportions.
        At most one of `all_reliability_data`, `all_value_counts`, or `random_coincidences` can be provided.

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
    >>> # Subsample alpha calculation using all_reliability_data:
    >>> subsample = [row[:6] for row in reliability_data]
    >>> print(round(alpha(subsample, level_of_measurement="ordinal",
    ...                   value_domain=["very low", "low", "mid", "high", "very high"],
    ...                   all_reliability_data=reliability_data), 3))
    0.72
    >>> # Circular metric example (e.g. cyclic scale 0..3):
    >>> circ_data = [[0, 1, 2, 3], [0, 2, 2, 0]]
    >>> print(round(alpha(circ_data, level_of_measurement="circular"), 4))
    0.5625
    >>> # Bipolar metric example (scale from -1 to 1):
    >>> bip_data = [[-1, 0, 1], [-1, 1, 1]]
    >>> print(round(alpha(bip_data, level_of_measurement="bipolar"), 4))
    0.7826
    """
    if (reliability_data is None) == (value_counts is None):
        raise ValueError("Either reliability_data or value_counts must be provided, but not both.")

    subsample_params = [
        all_reliability_data is not None,
        all_value_counts is not None,
        random_coincidences is not None,
    ]
    if sum(subsample_params) > 1:
        raise ValueError(
            "At most one of all_reliability_data, all_value_counts, or random_coincidences can be provided."
        )

    if all_reliability_data is not None:
        all_rel_arr = np.asarray(all_reliability_data)
        if all_rel_arr.ndim != 2:
            raise ValueError("The all_reliability_data must be a 2D array.")

    if reliability_data is not None:
        if _is_dict_reliability_data(reliability_data):
            if all_reliability_data is not None:
                raise ValueError("Subsample with dict reliability_data is not supported.")
            value_counts, value_domain = _dict_reliability_data_to_value_counts(
                reliability_data,  # ty:ignore[invalid-argument-type]
                value_domain,
                level_of_measurement,
            )
        else:
            rel_arr = np.asarray(reliability_data)
            computed_value_domain = _domain_from_reliability_data(rel_arr)

            if all_reliability_data is not None:
                all_computed_domain = _domain_from_reliability_data(all_rel_arr)
                combined_computed_domain = np.unique(np.concatenate([computed_value_domain, all_computed_domain]))
            else:
                combined_computed_domain = computed_value_domain

            if value_domain is None:
                kind = rel_arr.dtype.kind
                all_kind = all_rel_arr.dtype.kind if all_reliability_data is not None else None
                if (kind in {"U", "S"} or all_kind in {"U", "S"}) and level_of_measurement != "nominal":
                    raise ValueError(
                        "When using strings, an ordered value_domain is required"
                        " for level_of_measurement other than 'nominal'."
                    )
                value_domain = combined_computed_domain
            else:
                value_domain = _to_domain_array(value_domain)
                if not np.isin(computed_value_domain, value_domain).all():
                    raise ValueError("The reliability data contains out-of-domain values.")
                if all_reliability_data is not None and not np.isin(all_computed_domain, value_domain).all():
                    raise ValueError("The reference reliability data contains out-of-domain values.")

            value_counts = _reliability_data_to_value_counts(rel_arr, value_domain)
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

    if level_of_measurement in ("interval", "ratio", "circular", "bipolar") and (
        np.iscomplexobj(value_domain) or any(isinstance(v, (complex, np.complexfloating)) for v in value_domain)
    ):
        raise ValueError(f"Level of measurement {level_of_measurement!r} does not support complex values.")

    distance_metric = _distance_metric(level_of_measurement)

    o = _coincidences(value_counts, dtype=dtype)
    n_v = o.sum(axis=0)

    if all_reliability_data is not None:
        all_rel = all_rel_arr
        all_computed_domain = _domain_from_reliability_data(all_rel)
        if not np.isin(all_computed_domain, value_domain).all():
            raise ValueError("The reference reliability data contains out-of-domain values.")
        all_counts = _reliability_data_to_value_counts(all_rel, value_domain)
        if (all_counts.sum(axis=-1) <= 1).all():
            raise ValueError(
                "There has to be at least one unit in all_reliability_data with values assigned by at least two coders."
            )
        all_o = _coincidences(all_counts, dtype=dtype)
        all_n_v = all_o.sum(axis=0)
        e = _random_coincidences(all_n_v, dtype=dtype)
        dist_n_v = all_n_v
    elif all_value_counts is not None:
        all_counts = np.asarray(all_value_counts)
        if all_counts.ndim != 2:
            raise ValueError("The all_value_counts must be a 2D array.")
        if all_counts.shape[1] != len(value_domain):
            raise ValueError("The number of columns of all_value_counts should be equal to the value domain length.")
        if (
            (
                not np.issubdtype(all_counts.dtype, np.integer)
                and not (np.issubdtype(all_counts.dtype, np.floating) and (all_counts % 1 == 0).all())
            )
            or not np.isfinite(all_counts).all()
            or (all_counts < 0).any()
        ):
            raise ValueError("The all_value_counts must contain finite, non-negative integer counts.")
        all_counts = all_counts.astype(np.int_, copy=False)
        if (all_counts.sum(axis=-1) <= 1).all():
            raise ValueError(
                "There has to be at least one unit in all_value_counts with values assigned by at least two coders."
            )
        all_o = _coincidences(all_counts, dtype=dtype)
        all_n_v = all_o.sum(axis=0)
        e = _random_coincidences(all_n_v, dtype=dtype)
        dist_n_v = all_n_v
    elif random_coincidences is not None:
        e = np.asarray(random_coincidences, dtype=dtype)
        if e.shape != (len(value_domain), len(value_domain)):
            raise ValueError(
                f"The random_coincidences shape {e.shape} must be equal to {(len(value_domain), len(value_domain))}."
            )
        if not np.isfinite(e).all() or (e < 0).any() or e.sum() <= 0:
            raise ValueError("The random_coincidences matrix must be non-negative, finite, and have a positive sum.")
        if not np.allclose(e, e.T):
            raise ValueError("The random_coincidences matrix must be symmetric.")
        dist_n_v = e.sum(axis=0)
    else:
        e = _random_coincidences(n_v, dtype=dtype)
        dist_n_v = n_v

    d = _distances(value_domain, distance_metric, dist_n_v, dtype=dtype)

    do = (o * d).sum() / o.sum()
    de = (e * d).sum() / e.sum()
    if de == 0:
        raise ValueError("Expected disagreement is zero, making Krippendorff's alpha undefined.")
    return 1 - do / de

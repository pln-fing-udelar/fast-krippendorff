"""
This module provides a function to compute the Krippendorff's alpha statistical measure of the agreement achieved
when coding a set of units based on the values of a variable.

For more information, see: https://en.wikipedia.org/wiki/Krippendorff%27s_alpha

The module naming follows the one from the Wikipedia link.
"""

from typing import Literal, Protocol, TypeVar

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


def _domain_from_reliability_data(reliability_data: npt.NDArray) -> npt.NDArray:
    """Extract unique non-missing values from reliability data."""
    kind = reliability_data.dtype.kind
    if kind in {"i", "u", "f"}:
        return np.unique(reliability_data[~np.isnan(reliability_data)])
    elif kind in {"U", "S"}:
        return np.unique(reliability_data[reliability_data != "nan"])
    raise ValueError(f"Don't know how to construct value domain for dtype kind {kind}.")


def alpha(  # noqa: C901
    reliability_data: npt.ArrayLike | None = None,
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
    reliability_data : array_like, with shape (M, N)
        Reliability data matrix which has the rate the i coder gave to the j unit, where M is the number of raters
        and N is the unit count.
        Missing rates are represented with `np.nan`.
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
    >>> # Subsample alpha calculation using all_reliability_data:
    >>> subsample = [row[:6] for row in reliability_data]
    >>> print(round(alpha(subsample, level_of_measurement="ordinal",
    ...                   value_domain=["very low", "low", "mid", "high", "very high"],
    ...                   all_reliability_data=reliability_data), 3))
    0.72
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

    if reliability_data is not None:
        rel_arr = np.asarray(reliability_data)
        computed_value_domain = _domain_from_reliability_data(rel_arr)

        if all_reliability_data is not None:
            all_rel_arr = np.asarray(all_reliability_data)
            all_computed_domain = _domain_from_reliability_data(all_rel_arr)
            combined_computed_domain = np.unique(np.concatenate([computed_value_domain, all_computed_domain]))
        else:
            combined_computed_domain = computed_value_domain

        if value_domain is None:
            kind = rel_arr.dtype.kind
            all_kind = np.asarray(all_reliability_data).dtype.kind if all_reliability_data is not None else None
            if (kind in {"U", "S"} or all_kind in {"U", "S"}) and level_of_measurement != "nominal":
                raise ValueError(
                    "When using strings, an ordered value_domain is required"
                    " for level_of_measurement other than 'nominal'."
                )
            value_domain = combined_computed_domain
        else:
            value_domain = np.asarray(value_domain)
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
            value_domain = np.asarray(value_domain)

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

    if all_reliability_data is not None:
        all_rel = np.asarray(all_reliability_data)
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
        if do == 0:
            return 1.0
        raise ValueError("Expected disagreement is zero, making Krippendorff's alpha undefined.")
    return 1 - do / de

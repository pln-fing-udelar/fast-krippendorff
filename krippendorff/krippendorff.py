"""
This module provides a function to compute the Krippendorff's alpha statistical measure of the agreement achieved
when coding a set of units based on the values of a variable.

For more information, see: https://en.wikipedia.org/wiki/Krippendorff%27s_alpha

The module naming follows the one from the Wikipedia link.
"""
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import numpy as np


def _nominal_metric(v1: np.ndarray, v2: np.ndarray, dtype: Any = np.float64, **kwargs) -> np.ndarray:  # noqa
    """Metric for nominal data."""
    return (v1 != v2).astype(dtype)


def _ordinal_metric(v1: np.ndarray, v2: np.ndarray, i1: np.ndarray, i2: np.ndarray,  # noqa
                    n_v: np.ndarray, dtype: Any = np.float64, **kwargs) -> np.ndarray:  # noqa
    """Metric for ordinal data."""
    i1, i2 = np.minimum(i1, i2), np.maximum(i1, i2)

    ranges = np.dstack((i1, i2 + 1))
    sums_between_indices = np.add.reduceat(np.append(n_v, 0), ranges.reshape(-1))[::2].reshape(*i1.shape)

    return (sums_between_indices - np.divide(n_v[i1] + n_v[i2], 2, dtype=dtype)) ** 2


def _interval_metric(v1: np.ndarray, v2: np.ndarray, dtype: Any = np.float64, **kwargs) -> np.ndarray:  # noqa
    """Metric for interval data."""
    return (v1 - v2).astype(dtype) ** 2


def _ratio_metric(v1: np.ndarray, v2: np.ndarray, dtype: Any = np.float64, **kwargs) -> np.ndarray:  # noqa
    """Metric for ratio data."""
    v1_plus_v2 = v1 + v2
    return np.divide(v1 - v2, v1_plus_v2, out=np.zeros(np.broadcast(v1, v2).shape), where=v1_plus_v2 != 0,
                     dtype=dtype) ** 2


def _coincidences(value_counts: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
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
    N, V = value_counts.shape
    pairable = np.maximum(value_counts.sum(axis=1), 2)
    diagonals = value_counts[:, np.newaxis, :] * np.eye(V)[np.newaxis, ...]
    unnormalized_coincidences = value_counts[..., np.newaxis] * value_counts[:, np.newaxis, :] - diagonals
    return np.divide(unnormalized_coincidences, (pairable - 1).reshape((-1, 1, 1)), dtype=dtype).sum(axis=0)


def _random_coincidences(n_v: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
    """Random coincidence matrix.

    Parameters
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


def _distances(value_domain: np.ndarray, distance_metric: Callable[..., np.ndarray], n_v: np.ndarray,
               dtype: Any = np.float64) -> np.ndarray:
    """Distances of the different possible values.

    Parameters
    ----------
    value_domain : ndarray, with shape (V,)
        Possible values V the units can take.
        If the level of measurement is not nominal, it must be ordered.

    distance_metric : callable
        Callable that return the distance of two given values.

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
    return distance_metric(value_domain[:, np.newaxis], value_domain[np.newaxis, :], i1=indices[:, np.newaxis],
                           i2=indices[np.newaxis, :], n_v=n_v, dtype=dtype)


def _distance_metric(level_of_measurement: Union[str, Callable[..., np.ndarray]]) -> Callable[..., np.ndarray]:
    """Distance metric callable of the level of measurement.

    Parameters
    ----------
    level_of_measurement : string or callable
        Steven's level of measurement of the variable.
        It must be one of 'nominal', 'ordinal', 'interval', 'ratio' or a callable.

    Returns
    -------
    metric : callable
        Distance callable.
    """
    return {
        'nominal': _nominal_metric,
        'ordinal': _ordinal_metric,
        'interval': _interval_metric,
        'ratio': _ratio_metric,
    }.get(level_of_measurement, level_of_measurement)


def _reliability_data_to_value_counts(reliability_data: np.ndarray, value_domain: np.ndarray) -> np.ndarray:
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
    return (reliability_data.T[..., np.newaxis] == value_domain[np.newaxis, np.newaxis, :]).sum(axis=1)  # noqa


def alpha(reliability_data: Optional[Iterable[Any]] = None, value_counts: Optional[np.ndarray] = None,
          value_domain: Optional[Sequence[Any]] = None,
          level_of_measurement: Union[str, Callable[..., Any]] = 'interval', dtype: Any = np.float64) -> float:
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
        It must be one of 'nominal', 'ordinal', 'interval', 'ratio' or a callable.

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
    >>> print(round(alpha(reliability_data=reliability_data, level_of_measurement='nominal'), 6))
    0.691358
    >>> print(round(alpha(reliability_data=reliability_data, level_of_measurement='interval'), 6))
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
    >>> print(round(alpha(value_counts=value_counts, level_of_measurement='nominal'), 6))
    0.691358
    >>> # The following examples were extracted from
    >>> # https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2016/07/fulltext.pdf, page 8.
    >>> reliability_data = [[1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
    ...                     [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
    ...                     [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
    ...                     [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan]]
    >>> print(round(alpha(reliability_data, level_of_measurement='ordinal'), 3))
    0.815
    >>> print(round(alpha(reliability_data, level_of_measurement='ratio'), 3))
    0.797
    >>> reliability_data = [["very low", "low", "mid", "mid", "low", "very low", "high", "very low", "low", np.nan,
    ...                      np.nan, np.nan],
    ...                     ["very low", "low", "mid", "mid", "low", "low", "high", "very low", "low", "very high",
    ...                      np.nan, "mid"],
    ...                     [np.nan, "mid", "mid", "mid", "low", "mid", "high", "low", "low", "very high", "very low",
    ...                      np.nan],
    ...                     ["very low", "low", "mid", "mid", "low", "high", "high", "very low", "low", "very high",
    ...                      "very low", np.nan]]
    >>> print(round(alpha(reliability_data, level_of_measurement='ordinal',
    ...                   value_domain=["very low", "low", "mid", "high", "very high"]), 3))
    0.815
    """
    if (reliability_data is None) == (value_counts is None):
        raise ValueError("Either reliability_data or value_counts must be provided, but not both.")

    # Don't know if it's a list or numpy array. If it's the latter, the truth value is ambiguous. So, ask for None.
    if value_counts is None:
        reliability_data = np.asarray(reliability_data)

        if value_domain is None:
            value_domain = np.unique(reliability_data[~np.isnan(reliability_data)])
        else:
            value_domain = np.asarray(value_domain)
            assert np.isin(reliability_data, np.append(value_domain, np.nan)).all(), \
                "The reliability data contains out-of-domain values."

        value_counts = _reliability_data_to_value_counts(reliability_data, value_domain)
    else:  # elif reliability_data is None
        value_counts = np.asarray(value_counts)

        if value_domain is None:
            value_domain = np.arange(value_counts.shape[1])
        else:
            value_domain = np.asarray(value_domain)
            assert value_counts.shape[1] == len(value_domain), \
                "The value domain should be equal to the number of columns of value_counts."

    assert len(value_domain) > 1, "There has to be more than one value in the domain."

    distance_metric = _distance_metric(level_of_measurement)

    o = _coincidences(value_counts, dtype=dtype)
    n_v = o.sum(axis=0)
    e = _random_coincidences(n_v, dtype=dtype)
    d = _distances(value_domain, distance_metric, n_v, dtype=dtype)
    return 1 - (o * d).sum() / (e * d).sum()

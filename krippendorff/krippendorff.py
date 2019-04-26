"""
This module provides a function to compute the Krippendorff's alpha statistical measure of the agreement achieved
when coding a set of units based on the values of a variable.

For more information, see: https://en.wikipedia.org/wiki/Krippendorff%27s_alpha

The module naming follows the one from the Wikipedia link.
"""

import numpy as np


def _nominal_metric(v1, v2, **_kwargs):
    """Metric for nominal data."""
    return v1 != v2


def _ordinal_metric(_v1, _v2, i1, i2, n_v):
    """Metric for ordinal data."""
    if i1 > i2:
        i1, i2 = i2, i1
    return (np.sum(n_v[i1:(i2 + 1)]) - (n_v[i1] + n_v[i2]) / 2) ** 2


def _interval_metric(v1, v2, **_kwargs):
    """Metric for interval data."""
    return (v1 - v2) ** 2


def _ratio_metric(v1, v2, **_kwargs):
    """Metric for ratio data."""
    return (((v1 - v2) / (v1 + v2)) ** 2) if v1 + v2 != 0 else 0


def _coincidences(value_counts, value_domain, dtype=np.float64):
    """Coincidence matrix.

    Parameters
    ----------
    value_counts : ndarray, with shape (N, V)
        Number of coders that assigned a certain value to a determined unit, where N is the number of units
        and V is the value count.

    value_domain : array_like, with shape (V,)
        Possible values V the units can take.
        If the level of measurement is not nominal, it must be ordered.

    dtype : data-type
        Result and computation data-type.

    Returns
    -------
    o : ndarray, with shape (V, V)
        Coincidence matrix.
    """
    value_counts_matrices = value_counts.reshape(value_counts.shape + (1,))
    pairable = np.maximum(np.sum(value_counts, axis=1), 2)
    diagonals = np.tile(np.eye(len(value_domain)), (len(value_counts), 1, 1)) \
        * value_counts.reshape((value_counts.shape[0], 1, value_counts.shape[1]))
    unnormalized_coincidences = value_counts_matrices * value_counts_matrices.transpose((0, 2, 1)) - diagonals
    return np.sum(np.divide(unnormalized_coincidences, (pairable - 1).reshape((-1, 1, 1)), dtype=dtype), axis=0)


def _random_coincidences(value_domain, n, n_v):
    """Random coincidence matrix.

    Parameters
    ----------
    value_domain : array_like, with shape (V,)
        Possible values V the units can take.
        If the level of measurement is not nominal, it must be ordered.

    n : scalar
        Number of pairable values.

    n_v : ndarray, with shape (V,)
        Number of pairable elements for each value.

    Returns
    -------
    e : ndarray, with shape (V, V)
        Random coincidence matrix.
    """
    n_v_column = n_v.reshape(-1, 1)
    return (n_v_column.dot(n_v_column.T) - np.eye(len(value_domain)) * n_v_column) / (n - 1)


def _distances(value_domain, distance_metric, n_v):
    """Distances of the different possible values.

    Parameters
    ----------
    value_domain : array_like, with shape (V,)
        Possible values V the units can take.
        If the level of measurement is not nominal, it must be ordered.

    distance_metric : callable
        Callable that return the distance of two given values.

    n_v : ndarray, with shape (V,)
        Number of pairable elements for each value.

    Returns
    -------
    d : ndarray, with shape (V, V)
        Distance matrix for each value pair.
    """
    return np.array([[distance_metric(v1, v2, i1=i1, i2=i2, n_v=n_v)
                      for i2, v2 in enumerate(value_domain)]
                     for i1, v1 in enumerate(value_domain)])


def _distance_metric(level_of_measurement):
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


def _transpose_list(list_of_lists):
    """Transpose a list of lists."""
    return list(map(list, zip(*list_of_lists)))


def _reliability_data_to_value_counts(reliability_data, value_domain):
    """Return the value counts given the reliability data.

    Parameters
    ----------
    reliability_data : ndarray, with shape (M, N)
        Reliability data matrix which has the rate the i coder gave to the j unit, where M is the number of raters
        and N is the unit count.
        Missing rates are represented with `np.nan`.

    value_domain : array_like, with shape (V,)
        Possible values the units can take.

    Returns
    -------
    value_counts : ndarray, with shape (N, V)
        Number of coders that assigned a certain value to a determined unit, where N is the number of units
        and V is the value count.
    """
    return np.array([[sum(1 for rate in unit if rate == v) for v in value_domain] for unit in reliability_data.T])


def alpha(reliability_data=None, value_counts=None, value_domain=None, level_of_measurement='interval',
          dtype=np.float64):
    """Compute Krippendorff's alpha.

    See https://en.wikipedia.org/wiki/Krippendorff%27s_alpha for more information.

    Parameters
    ----------
    reliability_data : array_like, with shape (M, N)
        Reliability data matrix which has the rate the i coder gave to the j unit, where M is the number of raters
        and N is the unit count.
        Missing rates are represented with `np.nan`.
        If it's provided then `value_counts` must not be provided.

    value_counts : ndarray, with shape (N, V)
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
    alpha : `dtype`
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
    ...                     [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3.],
    ...                     [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
    ...                     [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan]]
    >>> print(round(alpha(reliability_data, level_of_measurement='ordinal'), 3))
    0.815
    >>> print(round(alpha(reliability_data, level_of_measurement='ratio'), 3))
    0.797
    """
    if (reliability_data is None) == (value_counts is None):
        raise ValueError("Either reliability_data or value_counts must be provided, but not both.")

    # Don't know if it's a list or numpy array. If it's the latter, the truth value is ambiguous. So, ask for None.
    if value_counts is None:
        if type(reliability_data) is not np.ndarray:
            reliability_data = np.array(reliability_data)

        value_domain = value_domain or np.unique(reliability_data[~np.isnan(reliability_data)])

        value_counts = _reliability_data_to_value_counts(reliability_data, value_domain)
    else:  # elif reliability_data is None
        if value_domain:
            assert value_counts.shape[1] == len(value_domain), \
                "The value domain should be equal to the number of columns of value_counts."
        else:
            value_domain = tuple(range(value_counts.shape[1]))

    distance_metric = _distance_metric(level_of_measurement)

    o = _coincidences(value_counts, value_domain, dtype=dtype)
    n_v = np.sum(o, axis=0)
    n = np.sum(n_v)
    e = _random_coincidences(value_domain, n, n_v)
    d = _distances(value_domain, distance_metric, n_v)
    return 1 - np.sum(o * d) / np.sum(e * d)

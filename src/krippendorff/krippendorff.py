"""
This module provides a function to compute the Krippendorff's alpha statistical measure of the agreement achieved
when coding a set of units based on the values of a variable.

For more information, see: https://en.wikipedia.org/wiki/Krippendorff%27s_alpha

The module naming follows the one from the Wikipedia link.
"""

import math
import numbers
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


LevelOfMeasurement = Literal["bipolar", "circular", "interval", "nominal", "ordinal", "ratio"] | DistanceMetric


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
    """Check if an array contains exclusively finite real numeric values."""
    if np.issubdtype(arr.dtype, np.number):
        return bool(np.isfinite(arr).all())
    for x in arr.flat:
        if isinstance(x, (int, np.integer)):
            continue
        if not isinstance(x, (numbers.Real, np.floating)) or not math.isfinite(x):
            return False
    return True


def _is_all_pure_integer(arr: npt.NDArray) -> bool:
    """Check if array has integer dtype or is an object array containing pure integers."""
    if np.issubdtype(arr.dtype, np.integer):
        return True
    if arr.dtype == object:
        return all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in arr.flat)
    return False


def _is_all_integer(arr: npt.NDArray) -> bool:
    """Check if all elements of an array represent integer values."""
    if _is_all_pure_integer(arr):
        return True
    for x in arr.flat:
        if isinstance(x, bool):
            return False
        if isinstance(x, (int, np.integer)):
            continue
        if isinstance(x, (numbers.Real, np.floating)) and np.isfinite(x):
            try:
                if float(x).is_integer():
                    continue
            except (OverflowError, ValueError):
                pass
        return False
    return True


def _to_pure_integer_if_all_integral(arr: npt.NDArray) -> npt.NDArray:
    """Normalize object array containing integral values to pure integer object array."""
    if np.issubdtype(arr.dtype, np.integer):
        return arr
    if arr.dtype == object:
        for x in arr.flat:
            if isinstance(x, bool):
                return arr
            if isinstance(x, (int, np.integer)):
                continue
            if isinstance(x, (numbers.Real, np.floating)) and np.isfinite(x):
                try:
                    if float(x).is_integer():
                        continue
                except (OverflowError, ValueError):
                    pass
            return arr
        normalized = np.empty(arr.shape, dtype=object)
        for i, x in enumerate(arr.flat):
            normalized.flat[i] = int(x)
        return normalized
    return arr


def _to_object_int(x: Any) -> Any:
    """Convert an integer scalar or array to Python int objects to avoid NumPy overflow."""
    x_arr = np.asarray(x)
    if x_arr.ndim == 0:
        return int(x_arr.item())
    return np.array([int(item) for item in x_arr.flat], dtype=object).reshape(x_arr.shape)


def _to_int_if_integral(x: Any) -> Any:
    """Convert an integral scalar to Python int if it represents an integer value."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)) and np.isfinite(x) and x.is_integer():
        return int(x)
    return x


def _safe_diff(a: Any, b: Any) -> Any:
    """Compute difference safely, avoiding integer underflow/overflow and precision loss."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if np.issubdtype(a_arr.dtype, np.floating) or np.issubdtype(b_arr.dtype, np.floating):
        return a - b
    if np.issubdtype(a_arr.dtype, np.integer) and np.issubdtype(b_arr.dtype, np.integer):
        if a_arr.dtype in (np.int64, np.uint64) or b_arr.dtype in (np.int64, np.uint64):
            return _to_object_int(a) - _to_object_int(b)
        return np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64)
    return a - b


def _scale_val(val: Any, u: int) -> float:
    """Scale a single scalar val by huge circumference u."""
    if isinstance(val, (int, np.integer)):
        val_int = int(val)
        if val_int == 0:
            return 0.0
        shift = u.bit_length() - 53
        u_mantissa = float(u >> shift)
        abs_v = abs(val_int)
        v_bits = abs_v.bit_length()
        sign = -1.0 if val_int < 0 else 1.0
        if v_bits > 53:
            v_shift = v_bits - 53
            v_m = sign * float(abs_v >> v_shift)
            return math.ldexp(v_m / u_mantissa, v_shift - shift)
        return math.ldexp(sign * float(abs_v) / u_mantissa, -shift)
    shift = u.bit_length() - 53
    u_mantissa = float(u >> shift)
    return math.ldexp(float(val), -shift) / u_mantissa


def _scale_operand_by_u(arr: Any, u: int, dtype: np.dtype) -> npt.NDArray:
    """Scale an array by a huge integer circumference u without overflow."""
    shift = u.bit_length() - 53
    u_mantissa = float(u >> shift)
    arr_np = np.asarray(arr)
    if np.issubdtype(arr_np.dtype, np.floating):
        return np.asarray(np.ldexp(arr_np, -shift) / u_mantissa, dtype=dtype)
    out = np.zeros(arr_np.shape, dtype=dtype)
    for i, val in enumerate(arr_np.flat):
        out.flat[i] = _scale_val(val, u)
    return out


def _circular_scaled_diff(
    v1: npt.NDArray,
    v2: npt.NDArray,
    u: Any,
    real_dtype: np.dtype,
) -> npt.NDArray:
    """Compute normalized shortest circular difference |v1 - v2|_circ / u in [0, 0.5]."""
    is_pure_int = _is_all_pure_integer(v1) and _is_all_pure_integer(v2)
    if is_pure_int:
        diff = _safe_diff(v1, v2)
        if isinstance(u, (int, np.integer)) and int(u).bit_length() >= 1024:
            u_int = int(u)
            diff_mod = np.remainder(diff, u_int)
            diff_short = np.where(diff_mod < u_int - diff_mod, diff_mod, u_int - diff_mod)
            return _scale_operand_by_u(diff_short, u_int, real_dtype)
        u_val = u if isinstance(u, (int, np.integer)) else float(u)
        diff_mod = np.remainder(diff, u_val)
        diff_short = np.where(diff_mod < u_val - diff_mod, diff_mod, u_val - diff_mod)
        return np.asarray(diff_short / u_val, dtype=real_dtype)

    is_pure_float = np.issubdtype(v1.dtype, np.floating) and np.issubdtype(v2.dtype, np.floating)
    if is_pure_float:
        if isinstance(u, (int, np.integer)) and int(u).bit_length() >= 1024:
            u_int = int(u)
            s1 = _scale_operand_by_u(v1, u_int, real_dtype)
            s2 = _scale_operand_by_u(v2, u_int, real_dtype)
            diff_mod = np.abs(s1 - s2) % 1.0
            return np.where(diff_mod < 1.0 - diff_mod, diff_mod, 1.0 - diff_mod)
        calc_dtype = np.float64 if np.issubdtype(real_dtype, np.floating) and real_dtype.itemsize < 8 else real_dtype
        u_f = float(u)
        diff = np.remainder(np.asarray(v1, dtype=calc_dtype), u_f) - np.remainder(np.asarray(v2, dtype=calc_dtype), u_f)
        diff_mod = np.remainder(diff, u_f)
        diff_short = np.where(diff_mod < u_f - diff_mod, diff_mod, u_f - diff_mod)
        return np.asarray(diff_short / u_f, dtype=real_dtype)

    out_shape = np.broadcast(v1, v2).shape
    out = np.zeros(out_shape, dtype=real_dtype)
    b_v1, b_v2 = np.broadcast_arrays(v1, v2)
    for idx in np.ndindex(out_shape):
        a = b_v1[idx]
        b = b_v2[idx]
        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
            diff = int(a) - int(b)
            if isinstance(u, (int, np.integer)) and int(u).bit_length() >= 1024:
                u_int = int(u)
                diff_mod = diff % u_int
                diff_short = min(diff_mod, u_int - diff_mod)
                out[idx] = _scale_val(diff_short, u_int)
            else:
                diff_mod = diff % u
                diff_short = min(diff_mod, u - diff_mod)
                out[idx] = diff_short / u
        else:
            if isinstance(u, (int, np.integer)) and int(u).bit_length() >= 1024:
                u_int = int(u)
                s1 = _scale_val(a, u_int)
                s2 = _scale_val(b, u_int)
                diff_mod = abs(s1 - s2) % 1.0
                out[idx] = min(diff_mod, 1.0 - diff_mod)
            else:
                u_f = float(u)
                u_is_int = isinstance(u, (int, np.integer))
                a_val = float(int(a) % int(u)) if (isinstance(a, (int, np.integer)) and u_is_int) else float(a)
                b_val = float(int(b) % int(u)) if (isinstance(b, (int, np.integer)) and u_is_int) else float(b)
                diff = (a_val % u_f) - (b_val % u_f)
                diff_mod = diff % u_f
                diff_short = min(diff_mod, u_f - diff_mod)
                out[idx] = diff_short / u_f
    return out


def _normalize_circumference(circumference: float | None) -> Any:
    """Validate and normalize circumference parameter."""
    if circumference is None:
        return None
    if isinstance(circumference, (int, np.integer)):
        if circumference <= 0:
            raise ValueError("Circumference must be a finite, positive number.")
        return int(circumference)
    if (
        not isinstance(circumference, (numbers.Real, np.floating))
        or not np.isfinite(circumference)
        or circumference <= 0
    ):
        raise ValueError("Circumference must be a finite, positive number.")
    return _to_int_if_integral(circumference)


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
    circumference = _normalize_circumference(circumference)

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
        v1 = _to_pure_integer_if_all_integral(v1)
        v2 = _to_pure_integer_if_all_integral(v2)
        if circumference is None:
            if not (_is_all_integer(v1) and _is_all_integer(v2)):
                raise ValueError("An explicit circumference must be provided for non-integer circular data.")
            v_min = min(np.min(v1), np.min(v2))
            v_max = max(np.max(v1), np.max(v2))
            u = _safe_diff(v_max, v_min) + 1
        else:
            u = circumference
        u = _to_int_if_integral(u)
        real_dtype = np.empty((), dtype=dtype).real.dtype
        diff_scaled = _circular_scaled_diff(v1, v2, u, real_dtype)
        return (np.sin(np.pi * diff_scaled) ** 2).astype(dtype)

    return _metric


def _frexp_val(val: Any) -> tuple[float, int]:
    """Return (mantissa, exponent) such that val == mantissa * 2**exponent."""
    if isinstance(val, (int, np.integer)):
        val_int = int(val)
        if val_int == 0:
            return 0.0, 0
        bits = val_int.bit_length()
        if bits >= 1024:
            shift = bits - 53
            m = float(abs(val_int) >> shift)
            if val_int < 0:
                m = -m
            return math.ldexp(m, -53), bits
    val_f = float(val)
    return math.frexp(val_f)


def _scale_val_by_exp(val: Any, exp: int) -> float:
    """Scale a scalar val by 2**(-exp)."""
    m, e = _frexp_val(val)
    return math.ldexp(m, e - exp)


def _scale_bipolar_array(arr: Any, exp: int, dtype: np.dtype) -> npt.NDArray:
    """Scale an array by 2**(-exp) without overflow."""
    arr_np = np.asarray(arr)
    if np.issubdtype(arr_np.dtype, np.floating) or np.issubdtype(arr_np.dtype, np.integer):
        return np.asarray(np.ldexp(arr_np.astype(dtype), -exp), dtype=dtype)
    out = np.zeros(arr_np.shape, dtype=dtype)
    for i, val in enumerate(arr_np.flat):
        out.flat[i] = _scale_val_by_exp(val, exp)
    return out


def _max_exp_arr(arr: npt.NDArray) -> int:
    """Compute maximum binary exponent across array elements."""
    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
        max_v = float(np.max(np.abs(arr)))
        return math.frexp(max_v)[1] if max_v > 0 else 0
    return max((_frexp_val(x)[1] for x in arr.flat), default=0)


def _bipolar_terms(
    v1: npt.NDArray,
    v2: npt.NDArray,
    v_min: Any,
    v_max: Any,
    calc_dtype: npt.DTypeLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute bounded ratios for bipolar metric without overflow."""
    v1 = _to_pure_integer_if_all_integral(v1)
    v2 = _to_pure_integer_if_all_integral(v2)
    v_min_norm = _to_int_if_integral(v_min)
    v_max_norm = _to_int_if_integral(v_max)
    is_int_domain = (
        _is_all_pure_integer(v1)
        and _is_all_pure_integer(v2)
        and isinstance(v_min_norm, (int, np.integer))
        and isinstance(v_max_norm, (int, np.integer))
    )
    nonzero = v1 != v2
    out_shape = np.broadcast(v1, v2).shape
    if is_int_domain:
        diff = _safe_diff(v1, v2)
        term1 = _safe_diff(v1, v_min_norm) + _safe_diff(v2, v_min_norm)
        term2 = _safe_diff(v_max_norm, v1) + _safe_diff(v_max_norm, v2)
        ratio1 = np.divide(
            diff,
            term1,
            out=np.zeros(out_shape, dtype=object),
            where=nonzero & (term1 != 0),
        )
        ratio2 = np.divide(
            diff,
            term2,
            out=np.zeros(out_shape, dtype=object),
            where=nonzero & (term2 != 0),
        )
        return (
            np.asarray(ratio1, dtype=calc_dtype),
            np.asarray(ratio2, dtype=calc_dtype),
        )

    min_m, min_exp = _frexp_val(v_min)
    max_m, max_exp = _frexp_val(v_max)
    calc_dt = np.dtype(calc_dtype)
    max_v_exp = max(_max_exp_arr(v1), _max_exp_arr(v2))

    exp1 = max(min_exp, max_v_exp)
    v1_1 = _scale_bipolar_array(v1, exp1, calc_dt)
    v2_1 = _scale_bipolar_array(v2, exp1, calc_dt)
    min_1 = math.ldexp(min_m, min_exp - exp1)
    term1_f = (v1_1 - min_1) + (v2_1 - min_1)

    exp2 = max(max_exp, max_v_exp)
    v1_2 = _scale_bipolar_array(v1, exp2, calc_dt)
    v2_2 = _scale_bipolar_array(v2, exp2, calc_dt)
    max_2 = math.ldexp(max_m, max_exp - exp2)
    term2_f = (max_2 - v1_2) + (max_2 - v2_2)

    if not (np.issubdtype(np.asarray(v1).dtype, np.floating) and np.issubdtype(np.asarray(v2).dtype, np.floating)):
        diff = _safe_diff(v1, v2)
        diff_1 = _scale_bipolar_array(diff, exp1, calc_dt)
        diff_2 = _scale_bipolar_array(diff, exp2, calc_dt)
    else:
        diff_1 = v1_1 - v2_1
        diff_2 = v1_2 - v2_2

    ratio1 = np.divide(
        diff_1,
        term1_f,
        out=np.zeros(out_shape, dtype=calc_dtype),
        where=nonzero & (term1_f != 0),
    )
    ratio2 = np.divide(
        diff_2,
        term2_f,
        out=np.zeros(out_shape, dtype=calc_dtype),
        where=nonzero & (term2_f != 0),
    )
    return ratio1, ratio2


def _normalize_endpoint(val: Any, name: str) -> Any:
    """Validate and normalize a scale endpoint."""
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    if not (isinstance(val, (numbers.Real, np.floating)) and np.isfinite(val)):
        raise ValueError(f"{name} must be a finite number.")
    return _to_int_if_integral(val)


def _val_less_than(arr: npt.NDArray, bound: Any) -> bool:
    """Check if any element in arr is strictly less than bound, avoiding OverflowError."""
    if isinstance(bound, (int, np.integer)) and int(bound).bit_length() >= 1024:
        bound_int = int(bound)
        if bound_int > 0:
            if arr.dtype == object:
                return any(x < bound_int for x in arr.flat)
            return True
        else:
            if arr.dtype == object:
                return any(x < bound_int for x in arr.flat)
            return False
    try:
        return bool((arr < bound).any())
    except OverflowError:
        return any(x < bound for x in arr.flat)


def _val_greater_than(arr: npt.NDArray, bound: Any) -> bool:
    """Check if any element in arr is strictly greater than bound, avoiding OverflowError."""
    if isinstance(bound, (int, np.integer)) and int(bound).bit_length() >= 1024:
        bound_int = int(bound)
        if bound_int > 0:
            if arr.dtype == object:
                return any(x > bound_int for x in arr.flat)
            return False
        else:
            if arr.dtype == object:
                return any(x > bound_int for x in arr.flat)
            return True
    try:
        return bool((arr > bound).any())
    except OverflowError:
        return any(x > bound for x in arr.flat)


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
    low = _normalize_endpoint(low, "low")
    high = _normalize_endpoint(high, "high")
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
        v1 = _to_pure_integer_if_all_integral(v1)
        v2 = _to_pure_integer_if_all_integral(v2)
        v_min = low if low is not None else min(np.min(v1), np.min(v2))
        v_max = high if high is not None else max(np.max(v1), np.max(v2))
        v_min = _to_int_if_integral(v_min)
        v_max = _to_int_if_integral(v_max)
        if v_min >= v_max:
            raise ValueError("low must be strictly less than high.")
        if (
            _val_less_than(v1, v_min)
            or _val_greater_than(v1, v_max)
            or _val_less_than(v2, v_min)
            or _val_greater_than(v2, v_max)
        ):
            raise ValueError("The data contains out-of-bounds values for the specified bipolar endpoints.")

        real_dtype = np.empty((), dtype=dtype).real.dtype
        calc_dtype = np.float64 if np.issubdtype(real_dtype, np.floating) and real_dtype.itemsize < 8 else real_dtype
        ratio1, ratio2 = _bipolar_terms(v1, v2, v_min, v_max, calc_dtype)
        return (ratio1 * ratio2).astype(dtype)

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
        It must be one of "bipolar", "circular", "interval", "nominal", "ordinal", "ratio", or a callable.

    Returns
    -------
    metric : callable
        Distance callable.
    """
    return {
        "bipolar": _bipolar_metric,
        "circular": _circular_metric,
        "interval": _interval_metric,
        "nominal": _nominal_metric,
        "ordinal": _ordinal_metric,
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
    return _to_pure_integer_if_all_integral(arr)


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
        It must be one of "bipolar", "circular", "interval", "nominal", "ordinal", "ratio", or a callable.

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

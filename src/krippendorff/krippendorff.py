"""
This module provides a function to compute the Krippendorff's alpha statistical measure of the agreement achieved
when coding a set of units based on the values of a variable.

For more information, see: https://en.wikipedia.org/wiki/Krippendorff%27s_alpha

The module naming follows the one from the Wikipedia link.
"""

import math
import numbers
import warnings
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Generic, Literal, NamedTuple, Protocol, TypeVar, overload

import numpy as np
import numpy.typing as npt

DEFAULT_DTYPE = np.float64

__all__ = [
    "DEFAULT_DTYPE",
    "AlphaResult",
    "BootstrapResult",
    "CodersAxis",
    "DistanceMetric",
    "KrippendorffResult",
    "LevelOfMeasurement",
    "MetricResultScalarType",
    "ValueScalarType",
    "alpha",
    "alpha_table",
    "bipolar_metric",
    "bootstrap_ci",
    "bootstrap_ci_table",
    "circular_metric",
    "polar_metric",
    "table_to_value_counts",
]


ResultScalarType = TypeVar("ResultScalarType", bound=float | complex | np.inexact)


class AlphaResult(NamedTuple, Generic[ResultScalarType]):
    """Result of Krippendorff's alpha calculation when agreement components are requested.

    Attributes
    ----------
    alpha : float, complex, or np.inexact
        Krippendorff's alpha agreement measure.
    d_o : float, complex, or np.inexact
        Observed disagreement ($D_o$).
    d_e : float, complex, or np.inexact
        Expected disagreement ($D_e$) under chance agreement.
    """

    alpha: ResultScalarType
    d_o: ResultScalarType
    d_e: ResultScalarType

    @property
    def do(self) -> ResultScalarType:
        """Alias for `d_o` (observed disagreement)."""
        return self.d_o

    @property
    def de(self) -> ResultScalarType:
        """Alias for `d_e` (expected disagreement)."""
        return self.d_e


KrippendorffResult = AlphaResult


class BootstrapResult(NamedTuple):
    """Result of Krippendorff's alpha bootstrap confidence interval calculation.

    Attributes
    ----------
    alpha : float
        Observed point estimate of Krippendorff's alpha on the original sample.
    confidence_interval : tuple[float, float]
        Bootstrap confidence interval `(ci_low, ci_high)`.
    standard_error : float
        Bootstrap standard error of alpha (sample standard deviation of bootstrap replicates).
    q_value : float or None
        Krippendorff's q-value: empirical probability that the true reliability falls below
        `alternative_threshold` ($q = P(\\alpha^* < \\alpha_{\\text{min}})$). When degenerate resamples
        occur (omitted with a warning), $q$ is estimated conditionally over all valid resamples:
        $P(\\alpha^* < \\alpha_{\\text{min}} \\mid \\alpha^* \\text{ is defined})$.
        None if `alternative_threshold` was not provided.
    resamples : ndarray or None
        Array of bootstrap estimates of shape `(n_resamples,)` if `return_distribution` is True,
        otherwise None.
    """

    alpha: float
    confidence_interval: tuple[float, float]
    standard_error: float
    q_value: float | None = None
    resamples: npt.NDArray[np.floating] | None = None

    @property
    def ci_low(self) -> float:
        """Alias for lower confidence interval bound."""
        return self.confidence_interval[0]

    @property
    def ci_high(self) -> float:
        """Alias for upper confidence interval bound."""
        return self.confidence_interval[1]


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


LevelOfMeasurement = Literal["nominal", "ordinal", "interval", "ratio", "circular", "bipolar", "polar"] | DistanceMetric
CodersAxis = Literal[0, 1, "rows", "columns", "row", "col", "cols"]


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


setattr(_nominal_metric, "marginal_dependent", False)  # noqa: B010
setattr(_ordinal_metric, "marginal_dependent", True)  # noqa: B010
setattr(_interval_metric, "marginal_dependent", False)  # noqa: B010
setattr(_ratio_metric, "marginal_dependent", False)  # noqa: B010


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
        if not isinstance(x, (numbers.Real, np.floating)):
            return False
        if isinstance(x, np.floating):
            if not np.isfinite(x):
                return False
        elif not math.isfinite(x):
            return False
    return True


def _is_all_pure_integer(arr: npt.NDArray) -> bool:
    """Check if array has integer dtype or is an object array containing pure integers."""
    if np.issubdtype(arr.dtype, np.integer):
        return True
    if arr.dtype == object:
        return all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in arr.flat)
    return False


def _is_integral_scalar(x: Any) -> bool:
    """Check if scalar x represents an exact integer without precision loss."""
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, np.integer)):
        return True
    if isinstance(x, (float, np.floating)):
        return bool(np.isfinite(x) and x.is_integer())
    if isinstance(x, numbers.Real):
        try:
            return bool(int(x) == x)
        except (OverflowError, ValueError):
            return False
    return False


def _is_all_integer(arr: npt.NDArray) -> bool:
    """Check if all elements of an array represent integer values."""
    if _is_all_pure_integer(arr):
        return True
    return all(_is_integral_scalar(x) for x in arr.flat)


def _to_pure_integer_if_all_integral(arr: npt.NDArray) -> npt.NDArray:
    """Normalize object array containing integral values to pure integer object array."""
    if np.issubdtype(arr.dtype, np.integer):
        return arr
    if arr.dtype == object:
        if not all(_is_integral_scalar(x) for x in arr.flat):
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
    if _is_integral_scalar(x):
        return int(x)
    return x


def _normalize_object_integers(x: Any) -> Any:
    """Convert NumPy integer scalars (standalone or inside object arrays) to Python ints."""
    if isinstance(x, np.integer):
        return int(x)
    x_arr = np.asarray(x)
    if np.issubdtype(x_arr.dtype, np.integer):
        return _to_object_int(x_arr)
    if x_arr.dtype == object:
        out = np.empty(x_arr.shape, dtype=object)
        for i, val in enumerate(x_arr.flat):
            out.flat[i] = int(val) if isinstance(val, np.integer) else val
        return out
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
    return _normalize_object_integers(a) - _normalize_object_integers(b)


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


def _circular_residue(x: Any, u: Any) -> float:
    """Return x mod u in [0, u) as a float, reducing integers exactly before any float conversion."""
    if isinstance(x, (int, np.integer)):
        if isinstance(u, (int, np.integer)):
            return float(int(x) % int(u))
        return float(Fraction(int(x)) % Fraction(float(u)))
    return float(x) % float(u)


def _circular_scaled_diff_scalar(a: Any, b: Any, u: Any) -> float:
    """Compute normalized shortest circular difference for a single pair of (possibly mixed-type) values."""
    huge_u = isinstance(u, (int, np.integer)) and int(u).bit_length() >= 1024
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        diff = int(a) - int(b)
        if huge_u:
            u_int = int(u)
            diff_mod = diff % u_int
            return _scale_val(min(diff_mod, u_int - diff_mod), u_int)
        u_f = float(u)
        r = _circular_residue(diff, u)
        return min(r, u_f - r) / u_f
    if huge_u:
        u_int = int(u)
        a_red = int(a) % u_int if isinstance(a, (int, np.integer)) else a
        b_red = int(b) % u_int if isinstance(b, (int, np.integer)) else b
        diff_mod = abs(_scale_val(a_red, u_int) - _scale_val(b_red, u_int)) % 1.0
        return min(diff_mod, 1.0 - diff_mod)
    u_f = float(u)
    diff_mod = (_circular_residue(a, u) - _circular_residue(b, u)) % u_f
    return min(diff_mod, u_f - diff_mod) / u_f


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
        if isinstance(u, (int, np.integer)):
            diff_mod = np.remainder(diff, u)
            diff_short = np.where(diff_mod < u - diff_mod, diff_mod, u - diff_mod)
            return np.asarray(diff_short / u, dtype=real_dtype)
        u_f = float(u)
        diff_arr = np.asarray(diff)
        out = np.zeros(diff_arr.shape, dtype=real_dtype)
        for i, d in enumerate(diff_arr.flat):
            r = _circular_residue(d, u_f)
            out.flat[i] = min(r, u_f - r) / u_f
        return out

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
        out[idx] = _circular_scaled_diff_scalar(b_v1[idx], b_v2[idx], u)
    return out


def _normalize_circumference(circumference: float | np.floating | None) -> Any:
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
    circumference: float | np.floating | None = None,
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

    setattr(_metric, "marginal_dependent", False)  # noqa: B010
    return _metric


def _frexp_val(val: Any) -> tuple[Any, int]:
    """Return (mantissa, exponent) such that val == mantissa * 2**exponent."""
    if isinstance(val, (int, np.integer)):
        val_int = int(val)
        if val_int == 0:
            return 0.0, 0
        bits = val_int.bit_length()
        if bits >= 1024:
            shift = max(0, bits - 113)
            chunk = abs(val_int) >> shift
            ld = np.longdouble(chunk)
            if val_int < 0:
                ld = -ld
            return np.ldexp(ld, -(bits - shift)), bits
    if isinstance(val, Fraction):
        if val == 0:
            return 0.0, 0
        sign = -1.0 if val < 0 else 1.0
        abs_val = abs(val)
        num_m, num_e = _frexp_val(abs_val.numerator)
        den_m, den_e = _frexp_val(abs_val.denominator)
        m = (num_m / den_m) * sign
        e = num_e - den_e
        if isinstance(m, np.floating):
            norm_m, norm_e = np.frexp(m)
        else:
            norm_m, norm_e = math.frexp(float(m))
        return norm_m, e + int(norm_e)
    if isinstance(val, np.floating):
        m, e = np.frexp(val)
        return m, int(e)
    val_f = float(val)
    return math.frexp(val_f)


def _scale_val_by_exp(val: Any, exp: int, dtype: npt.DTypeLike = DEFAULT_DTYPE) -> Any:
    """Scale a scalar val by 2**(-exp)."""
    m, e = _frexp_val(val)
    dtype_obj = np.dtype(dtype)
    if np.issubdtype(dtype_obj, np.floating):
        return np.ldexp(dtype_obj.type(m), e - exp)
    return math.ldexp(m, e - exp)


def _scale_bipolar_array(arr: Any, exp: int, dtype: np.dtype) -> npt.NDArray:
    """Scale an array by 2**(-exp) without overflow."""
    arr_np = np.asarray(arr)
    if np.issubdtype(arr_np.dtype, np.floating) or np.issubdtype(arr_np.dtype, np.integer):
        return np.asarray(np.ldexp(arr_np.astype(dtype), -exp), dtype=dtype)
    out = np.zeros(arr_np.shape, dtype=dtype)
    for i, val in enumerate(arr_np.flat):
        out.flat[i] = _scale_val_by_exp(val, exp, dtype)
    return out


def _max_exp_arr(arr: npt.NDArray) -> int:
    """Compute maximum binary exponent across array elements."""
    arr_np = np.asarray(arr)
    if np.issubdtype(arr_np.dtype, np.floating):
        if arr_np.size == 0:
            return 0
        max_v = np.max(np.abs(arr_np))
        if max_v > 0:
            return int(np.frexp(max_v)[1])
        return 0
    return max((_frexp_val(x)[1] for x in arr_np.flat), default=0)


def _to_exact(x: Any) -> Any:
    """Convert scalar to int or Fraction for exact arithmetic."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, Fraction):
        return x
    return Fraction(float(x))


def _exact_sub(a: Any, b: Any) -> Any:
    """Subtract two scalars preserving exact precision when integers/fractions are involved."""
    res = _to_exact(a) - _to_exact(b)
    return res.numerator if getattr(res, "denominator", 1) == 1 else res


def _exact_add(a: Any, b: Any) -> Any:
    """Add two scalars preserving exact precision when integers/fractions are involved."""
    res = _to_exact(a) + _to_exact(b)
    return res.numerator if getattr(res, "denominator", 1) == 1 else res


def _safe_endpoint_gap(a: Any, b: Any) -> Any:
    """Compute a - b preserving exact precision for integer and mixed operands."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if np.issubdtype(a_arr.dtype, np.integer) and np.issubdtype(b_arr.dtype, np.integer):
        return _safe_diff(a_arr, b_arr)
    out_shape = np.broadcast(a_arr, b_arr).shape
    out = np.empty(out_shape, dtype=object)
    b_a, b_b = np.broadcast_arrays(a_arr, b_arr)
    for idx in np.ndindex(out_shape):
        out[idx] = _exact_sub(b_a[idx], b_b[idx])
    return out


def _safe_endpoint_sum(a: Any, b: Any) -> Any:
    """Compute a + b preserving exact precision for integer and mixed operands."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if np.issubdtype(a_arr.dtype, np.integer) and np.issubdtype(b_arr.dtype, np.integer):
        return a_arr + b_arr
    out_shape = np.broadcast(a_arr, b_arr).shape
    out = np.empty(out_shape, dtype=object)
    b_a, b_b = np.broadcast_arrays(a_arr, b_arr)
    for idx in np.ndindex(out_shape):
        out[idx] = _exact_add(b_a[idx], b_b[idx])
    return out


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
    nonzero = v1 != v2
    out_shape = np.broadcast(v1, v2).shape

    is_pure_float = np.issubdtype(np.asarray(v1).dtype, np.floating) and np.issubdtype(
        np.asarray(v2).dtype, np.floating
    )

    if is_pure_float:
        min_m, min_exp = _frexp_val(v_min)
        max_m, max_exp = _frexp_val(v_max)
        calc_dt = np.dtype(calc_dtype)
        max_v_exp = max(_max_exp_arr(v1), _max_exp_arr(v2))

        exp1 = max(min_exp, max_v_exp)
        v1_1 = _scale_bipolar_array(v1, exp1, calc_dt)
        v2_1 = _scale_bipolar_array(v2, exp1, calc_dt)
        if np.issubdtype(calc_dt, np.floating):
            min_1 = np.ldexp(calc_dt.type(min_m), min_exp - exp1)
        else:
            min_1 = math.ldexp(min_m, min_exp - exp1)
        term1_f = (v1_1 - min_1) + (v2_1 - min_1)

        exp2 = max(max_exp, max_v_exp)
        v1_2 = _scale_bipolar_array(v1, exp2, calc_dt)
        v2_2 = _scale_bipolar_array(v2, exp2, calc_dt)
        if np.issubdtype(calc_dt, np.floating):
            max_2 = np.ldexp(calc_dt.type(max_m), max_exp - exp2)
        else:
            max_2 = math.ldexp(max_m, max_exp - exp2)
        term2_f = (max_2 - v1_2) + (max_2 - v2_2)

        diff_1 = v1_1 - v2_1
        diff_2 = v1_2 - v2_2
    else:
        calc_dt = np.dtype(calc_dtype)
        gap1_1 = _safe_endpoint_gap(v1, v_min_norm)
        gap1_2 = _safe_endpoint_gap(v2, v_min_norm)
        term1 = _safe_endpoint_sum(gap1_1, gap1_2)

        gap2_1 = _safe_endpoint_gap(v_max_norm, v1)
        gap2_2 = _safe_endpoint_gap(v_max_norm, v2)
        term2 = _safe_endpoint_sum(gap2_1, gap2_2)

        diff = _safe_endpoint_gap(v1, v2)

        exp1 = max(_max_exp_arr(diff), _max_exp_arr(term1))
        diff_1 = _scale_bipolar_array(diff, exp1, calc_dt)
        term1_f = _scale_bipolar_array(term1, exp1, calc_dt)

        exp2 = max(_max_exp_arr(diff), _max_exp_arr(term2))
        diff_2 = _scale_bipolar_array(diff, exp2, calc_dt)
        term2_f = _scale_bipolar_array(term2, exp2, calc_dt)

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


def _scalar_less_than(x: Any, b: Any) -> bool:
    """Compare scalar x < b preserving arbitrary-precision integer comparisons."""
    if isinstance(x, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(x) < int(b)
    try:
        return bool(x < b)
    except (OverflowError, TypeError):
        return bool(float(x) < b)


def _scalar_greater_than(x: Any, b: Any) -> bool:
    """Compare scalar x > b preserving arbitrary-precision integer comparisons."""
    if isinstance(x, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(x) > int(b)
    try:
        return bool(x > b)
    except (OverflowError, TypeError):
        return bool(float(x) > b)


def _val_less_than(arr: npt.NDArray, bound: Any) -> bool:
    """Check if any element in arr is strictly less than bound, avoiding OverflowError."""
    if isinstance(bound, (int, np.integer)) and int(bound).bit_length() >= 1024:
        return any(_scalar_less_than(x, bound) for x in arr.flat)
    try:
        return bool((arr < bound).any())
    except (OverflowError, TypeError):
        return any(_scalar_less_than(x, bound) for x in arr.flat)


def _val_greater_than(arr: npt.NDArray, bound: Any) -> bool:
    """Check if any element in arr is strictly greater than bound, avoiding OverflowError."""
    if isinstance(bound, (int, np.integer)) and int(bound).bit_length() >= 1024:
        return any(_scalar_greater_than(x, bound) for x in arr.flat)
    try:
        return bool((arr > bound).any())
    except (OverflowError, TypeError):
        return any(_scalar_greater_than(x, bound) for x in arr.flat)


def bipolar_metric(
    low: float | np.floating | None = None,
    high: float | np.floating | None = None,
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

    setattr(_metric, "marginal_dependent", False)  # noqa: B010
    return _metric


_circular_metric = circular_metric()
_bipolar_metric = bipolar_metric()
polar_metric = bipolar_metric


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
        It must be one of "nominal", "ordinal", "interval", "ratio", "circular", "bipolar", "polar", or a callable.

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
        "polar": _bipolar_metric,
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
    if hasattr(data, "columns") or hasattr(data, "to_numpy"):
        return False
    if isinstance(data, Mapping):
        if len(data) > 0 and all(
            (isinstance(v, (list, tuple, np.ndarray)) or (hasattr(v, "__iter__") and not isinstance(v, (str, bytes))))
            and not isinstance(v, Mapping)
            for v in data.values()
        ):
            return False
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
    if type(val).__name__ in ("NAType", "NA", "NaTType", "MockNA"):
        return True
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return True
    if isinstance(val, (complex, np.complexfloating)) and (np.isnan(val.real) or np.isnan(val.imag)):
        return True
    return False


def _to_numpy_array(data: Any, dtype: Any = None) -> npt.NDArray:
    """Convert array-like, tensor, or dataframe to a numpy ndarray without forcing framework imports."""
    if hasattr(data, "detach") and callable(data.detach):
        data = data.detach()
    if hasattr(data, "cpu") and callable(data.cpu):
        data = data.cpu()
    if hasattr(data, "to_numpy") and callable(data.to_numpy):
        data = data.to_numpy()
    elif hasattr(data, "numpy") and callable(data.numpy):
        data = data.numpy()
    elif hasattr(data, "__array__") or hasattr(data, "__array_interface__"):
        data = np.asarray(data)
    elif hasattr(data, "values") and not callable(data.values):
        data = np.asarray(data.values)
    elif (
        isinstance(data, Mapping)
        and len(data) > 0
        and all(isinstance(v, (list, tuple, np.ndarray)) or hasattr(v, "__iter__") for v in data.values())
    ):
        data = np.asarray(list(data.values()))
    else:
        data = np.asarray(data)
    return np.asarray(data, dtype=dtype) if dtype is not None else np.asarray(data)


def _can_safely_cast_to_float(vals: Sequence[Any]) -> bool:
    """Check if all numeric values can be converted to float without losing integer precision."""
    for v in vals:
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            try:
                if int(float(v)) != v:
                    return False
            except (OverflowError, ValueError):
                return False
    return True


def _clean_object_array(arr: npt.NDArray) -> npt.NDArray:  # noqa: C901
    """Normalize object ndarray by replacing missing sentinels with NaN and downcasting numeric arrays when safe."""
    if arr.dtype != object:
        return arr
    out = np.empty(arr.shape, dtype=object)
    for idx, val in np.ndenumerate(arr):
        if _is_missing(val):
            out[idx] = np.nan
        else:
            out[idx] = val
    non_nan = [v for v in out.flat if not (isinstance(v, float) and np.isnan(v))]
    if len(non_nan) > 0:
        if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in non_nan):
            # If all are integers and there are no NaNs, try integer downcasting if it doesn't overflow or alter values
            if not any(isinstance(v, float) and np.isnan(v) for v in out.flat):
                try:
                    int_arr = out.astype(int)
                    if all(int_arr[idx] == out[idx] for idx, _ in np.ndenumerate(out)):
                        return int_arr
                except Exception:
                    pass
            # If there are NaNs or large integers, only downcast to float if every integer is preserved exactly
            if _can_safely_cast_to_float(non_nan):
                try:
                    return out.astype(float)
                except Exception:
                    pass
            return out
        elif all(_is_numeric_scalar(v) for v in non_nan):
            if _can_safely_cast_to_float(non_nan):
                try:
                    return out.astype(float)
                except Exception:
                    pass
            return out
        elif all(isinstance(v, str) for v in non_nan):
            str_out = np.empty(arr.shape, dtype=object)
            for idx, val in np.ndenumerate(out):
                str_out[idx] = "nan" if (isinstance(val, float) and np.isnan(val)) else str(val)
            try:
                return str_out.astype(str)
            except Exception:
                pass
    return out


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
    unique_vals = list(dict.fromkeys(distinct_values))
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
    if level_of_measurement in ("interval", "ratio", "circular", "bipolar", "polar") and (
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
    if kind in {"i", "u"}:
        return np.unique(reliability_data)
    elif kind == "f":
        return np.unique(reliability_data[~np.isnan(reliability_data)])
    elif kind in {"U", "S"}:
        return np.unique(reliability_data[reliability_data != "nan"])
    elif kind == "O":
        non_missing = [v for v in reliability_data.flat if not _is_missing(v)]
        if len(non_missing) > 0 and (
            all(_is_numeric_scalar(v) for v in non_missing) or all(isinstance(v, (str, bytes)) for v in non_missing)
        ):
            unique_vals = list(dict.fromkeys(non_missing))
            try:
                unique_vals = sorted(unique_vals)
            except TypeError:
                pass
            return _to_domain_array(unique_vals)
    raise ValueError(f"Don't know how to construct value domain for dtype kind {kind}.")


def _extract_table_columns(
    table: Any, coder_col: Any, unit_col: Any, value_col: Any
) -> tuple[list[Any], list[Any], list[Any]]:
    """Extract coder, unit, and value columns from tabular data."""
    is_column_accessible = (
        isinstance(table, Mapping)
        or hasattr(table, "columns")
        or (hasattr(table, "dtype") and getattr(table.dtype, "names", None) is not None)
    )
    if is_column_accessible:
        return list(table[coder_col]), list(table[unit_col]), list(table[value_col])
    return (
        [row[coder_col] for row in table],
        [row[unit_col] for row in table],
        [row[value_col] for row in table],
    )


def table_to_value_counts(  # noqa: C901
    table: Any,
    coder_col: Any,
    unit_col: Any,
    value_col: Any,
    value_domain: npt.ArrayLike | None = None,
    *,
    level_of_measurement: LevelOfMeasurement = "nominal",
) -> tuple[npt.NDArray[np.int_], npt.NDArray]:
    """Convert a long/tidy tabular dataset into a (value_counts, value_domain) pair.

    Parameters
    ----------
    table : DataFrame, mapping of columns, or sequence of records
        Tabular reliability data where each row/record represents a single rating assigned
        by a coder to a unit.
    coder_col : hashable
        Name or index of the column containing coder identifiers.
    unit_col : hashable
        Name or index of the column containing unit identifiers.
    value_col : hashable
        Name or index of the column containing assigned values/ratings.
    value_domain : array_like, optional
        Explicit domain of possible values. If not provided, it is inferred from unique
        non-missing values observed in `value_col`.
    level_of_measurement : string or callable, default "nominal"
        Level of measurement used to validate the inferred value domain.

    Returns
    -------
    value_counts : ndarray, shape (N, V)
        Number of coders that assigned each value in `value_domain` to each unit.
    value_domain : ndarray, shape (V,)
        The ordered domain of values corresponding to the columns of `value_counts`.
    """
    c_list, u_list, v_list = _extract_table_columns(table, coder_col, unit_col, value_col)

    if not (len(c_list) == len(u_list) == len(v_list)):
        raise ValueError("Columns coder_col, unit_col, and value_col must all have the same length.")
    if len(c_list) == 0:
        raise ValueError("The provided table is empty.")

    # Register each non-missing unit first so units with all-missing ratings are retained
    unit_map: dict[Any, int] = {}
    for u in u_list:
        if not _is_missing(u) and u not in unit_map:
            unit_map[u] = len(unit_map)

    if len(unit_map) == 0:
        raise ValueError("No valid units found in the table.")

    # Check for duplicate ratings per (coder, unit) pair
    seen_pairs: set[tuple[Any, Any]] = set()
    for c, u in zip(c_list, u_list, strict=True):
        if not _is_missing(c) and not _is_missing(u):
            pair = (c, u)
            if pair in seen_pairs:
                raise ValueError(f"Coder {c!r} has duplicate ratings for unit {u!r}.")
            seen_pairs.add(pair)

    # Collect valid non-missing ratings
    valid_u_indices: list[int] = []
    valid_v: list[Any] = []
    for c, u, v in zip(c_list, u_list, v_list, strict=True):
        if not _is_missing(c) and not _is_missing(u) and not _is_missing(v):
            valid_u_indices.append(unit_map[u])
            valid_v.append(v)

    if len(valid_v) == 0:
        raise ValueError("No valid non-missing ratings found in the table.")

    if value_domain is not None:
        domain_arr = _to_domain_array(value_domain)
        val_map = {val: i for i, val in enumerate(domain_arr)}
        for v in valid_v:
            if v not in val_map:
                raise ValueError("The reliability data contains out-of-domain values.")
        val_indices = np.array([val_map[v] for v in valid_v], dtype=int)
    else:
        distinct_v = list(dict.fromkeys(valid_v))
        domain_arr = _domain_from_raw_values(distinct_v, value_domain=None, level_of_measurement=level_of_measurement)
        val_map = {val: i for i, val in enumerate(domain_arr)}
        val_indices = np.array([val_map[v] for v in valid_v], dtype=int)

    unit_arr = np.array(valid_u_indices, dtype=np.int_)
    val_arr = np.array(val_indices, dtype=np.int_)
    value_counts = np.zeros((len(unit_map), len(domain_arr)), dtype=np.int_)
    np.add.at(value_counts, (unit_arr, val_arr), 1)

    return value_counts, domain_arr


def _prepare_data_and_domain(  # noqa: C901
    reliability_data: Any = None,
    value_counts: npt.ArrayLike | None = None,
    value_domain: npt.ArrayLike | None = None,
    level_of_measurement: LevelOfMeasurement = "interval",
    all_reliability_data: Any = None,
    all_value_counts: npt.ArrayLike | None = None,
    random_coincidences: npt.ArrayLike | None = None,
    coders_axis: CodersAxis = "rows",
    coder_col: Any = None,
    unit_col: Any = None,
    value_col: Any = None,
) -> tuple[npt.NDArray[np.int_], npt.NDArray, npt.NDArray | None]:
    """Validate and prepare value_counts and value_domain from raw data inputs."""
    subsample_params = [
        all_reliability_data is not None,
        all_value_counts is not None,
        random_coincidences is not None,
    ]
    if sum(subsample_params) > 1:
        raise ValueError(
            "At most one of all_reliability_data, all_value_counts, or random_coincidences can be provided."
        )

    all_rel_arr: npt.NDArray | None = None
    if all_reliability_data is not None:
        all_rel_arr = _to_numpy_array(all_reliability_data)
        if all_rel_arr.ndim != 2:
            raise ValueError("The all_reliability_data must be a 2D array.")
        if coders_axis in (1, "columns", "col", "cols"):
            all_rel_arr = all_rel_arr.T
        elif coders_axis not in (0, "rows", "row"):
            raise ValueError(f"Invalid coders_axis {coders_axis!r}. Expected 0, 1, 'rows', or 'columns'.")
        all_rel_arr = _clean_object_array(all_rel_arr)

    if coder_col is not None or unit_col is not None or value_col is not None:
        if not (coder_col is not None and unit_col is not None and value_col is not None):
            raise ValueError("All three of coder_col, unit_col, and value_col must be specified together.")
        if reliability_data is None:
            raise ValueError("A tabular dataset must be provided as reliability_data when specifying column names.")
        if value_counts is not None:
            raise ValueError("Cannot provide value_counts when coder_col, unit_col, and value_col are specified.")

        c_list, u_list, v_list = _extract_table_columns(reliability_data, coder_col, unit_col, value_col)
        if not (len(c_list) == len(u_list) == len(v_list)):
            raise ValueError("Columns coder_col, unit_col, and value_col must all have the same length.")
        if len(c_list) == 0:
            raise ValueError("The provided table is empty.")

        table_valid_v = [
            v
            for c, u, v in zip(c_list, u_list, v_list, strict=True)
            if not _is_missing(c) and not _is_missing(u) and not _is_missing(v)
        ]
        if len(table_valid_v) == 0:
            raise ValueError("No valid non-missing ratings found in the table.")

        if value_domain is not None:
            effective_domain = _to_domain_array(value_domain)
            domain_set = set(effective_domain)
            if any(v not in domain_set for v in table_valid_v):
                raise ValueError("The reliability data contains out-of-domain values.")
            if all_rel_arr is not None:
                all_computed_domain = _domain_from_reliability_data(all_rel_arr)
                if not np.isin(all_computed_domain, effective_domain).all():
                    raise ValueError("The reference reliability data contains out-of-domain values.")
        else:
            distinct_table_v = list(dict.fromkeys(table_valid_v))
            if all_rel_arr is not None:
                all_computed_domain = _domain_from_reliability_data(all_rel_arr)
                combined_vals = list(dict.fromkeys(distinct_table_v + list(all_computed_domain)))
                effective_domain = _domain_from_raw_values(
                    combined_vals, value_domain=None, level_of_measurement=level_of_measurement
                )
            else:
                effective_domain = _domain_from_raw_values(
                    distinct_table_v, value_domain=None, level_of_measurement=level_of_measurement
                )

        res_counts, res_domain = table_to_value_counts(
            reliability_data,
            coder_col,
            unit_col,
            value_col,
            value_domain=effective_domain,
            level_of_measurement=level_of_measurement,
        )
        if len(res_domain) <= 1:
            raise ValueError("There has to be more than one value in the domain.")
        if (res_counts.sum(axis=-1) <= 1).all():
            raise ValueError("There has to be at least one unit with values assigned by at least two coders.")
        return res_counts, res_domain, all_rel_arr

    if (reliability_data is None) == (value_counts is None):
        raise ValueError("Either reliability_data or value_counts must be provided, but not both.")

    if reliability_data is not None:
        if _is_dict_reliability_data(reliability_data):
            if all_reliability_data is not None:
                raise ValueError("Subsample with dict reliability_data is not supported.")
            res_counts, res_domain = _dict_reliability_data_to_value_counts(
                reliability_data,
                value_domain,
                level_of_measurement,
            )
        else:
            rel_arr = _to_numpy_array(reliability_data)
            if rel_arr.ndim != 2:
                raise ValueError("reliability_data must be a 2D array with shape (coders, units).")
            if coders_axis in (1, "columns", "col", "cols"):
                rel_arr = rel_arr.T
            elif coders_axis not in (0, "rows", "row"):
                raise ValueError(f"Invalid coders_axis {coders_axis!r}. Expected 0, 1, 'rows', or 'columns'.")
            rel_arr = _clean_object_array(rel_arr)
            computed_value_domain = _domain_from_reliability_data(rel_arr)

            if all_reliability_data is not None:
                assert all_rel_arr is not None
                all_computed_domain = _domain_from_reliability_data(all_rel_arr)
                combined_vals = list(dict.fromkeys(list(computed_value_domain) + list(all_computed_domain)))
                try:
                    combined_vals = sorted(combined_vals)
                except TypeError:
                    pass
                combined_computed_domain = _to_domain_array(combined_vals)
            else:
                combined_computed_domain = computed_value_domain

            if value_domain is None:
                kind = rel_arr.dtype.kind
                all_kind = all_rel_arr.dtype.kind if all_rel_arr is not None else None
                if (kind in {"U", "S", "O"} or all_kind in {"U", "S", "O"}) and level_of_measurement != "nominal":
                    has_str = any(isinstance(v, (str, bytes)) for v in combined_computed_domain)
                    if has_str:
                        raise ValueError(
                            "When using strings, an ordered value_domain is required"
                            " for level_of_measurement other than 'nominal'."
                        )
                res_domain = combined_computed_domain
            else:
                res_domain = _to_domain_array(value_domain)
                if not np.isin(computed_value_domain, res_domain).all():
                    raise ValueError("The reliability data contains out-of-domain values.")
                if all_rel_arr is not None and not np.isin(all_computed_domain, res_domain).all():
                    raise ValueError("The reference reliability data contains out-of-domain values.")

            res_counts = _reliability_data_to_value_counts(rel_arr, res_domain)
    else:
        res_counts = _to_numpy_array(value_counts)
        if res_counts.ndim != 2:
            raise ValueError("The value_counts must be a 2D array.")

        if value_domain is None:
            res_domain = np.arange(res_counts.shape[1])
        else:
            res_domain = _to_domain_array(value_domain)

        if res_counts.shape[1] != len(res_domain):
            raise ValueError("The value domain should be equal to the number of columns of value_counts.")

    if len(res_domain) <= 1:
        raise ValueError("There has to be more than one value in the domain.")

    if (res_counts.sum(axis=-1) <= 1).all():
        raise ValueError("There has to be at least one unit with values assigned by at least two coders.")

    return res_counts, res_domain, all_rel_arr


def _compute_expected_coincidences(  # noqa: C901
    value_domain: npt.NDArray,
    n_v: npt.NDArray,
    dtype: np.dtype,
    all_rel_arr: npt.NDArray | None,
    all_value_counts: npt.ArrayLike | None,
    random_coincidences: npt.ArrayLike | None,
) -> tuple[npt.NDArray, npt.NDArray, bool]:
    """Compute expected coincidence matrix e, marginal counts for distances dist_n_v, and reference flag."""
    if all_rel_arr is not None:
        all_computed_domain = _domain_from_reliability_data(all_rel_arr)
        if not np.isin(all_computed_domain, value_domain).all():
            raise ValueError("The reference reliability data contains out-of-domain values.")
        all_counts = _reliability_data_to_value_counts(all_rel_arr, value_domain)
        if (all_counts.sum(axis=-1) <= 1).all():
            raise ValueError(
                "There has to be at least one unit in all_reliability_data with values assigned by at least two coders."
            )
        all_o = _coincidences(all_counts, dtype=dtype)
        all_n_v = all_o.sum(axis=0)
        e = _random_coincidences(all_n_v, dtype=dtype)
        return e, all_n_v, True
    elif all_value_counts is not None:
        all_counts = _to_numpy_array(all_value_counts)
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
        return e, all_n_v, True
    elif random_coincidences is not None:
        e = _to_numpy_array(random_coincidences, dtype=dtype)
        if e.shape != (len(value_domain), len(value_domain)):
            raise ValueError(
                f"The random_coincidences shape {e.shape} must be equal to {(len(value_domain), len(value_domain))}."
            )
        if not np.isfinite(e).all() or (e < 0).any() or e.sum() <= 0:
            raise ValueError("The random_coincidences matrix must be non-negative, finite, and have a positive sum.")
        if not np.allclose(e, e.T):
            raise ValueError("The random_coincidences matrix must be symmetric.")
        return e, e.sum(axis=0), True
    else:
        e = _random_coincidences(n_v, dtype=dtype)
        return e, n_v, False


@overload
def alpha(
    reliability_data: Any = ...,
    value_counts: npt.ArrayLike | None = ...,
    value_domain: npt.ArrayLike | None = ...,
    level_of_measurement: LevelOfMeasurement = ...,
    dtype: type[float] | np.dtype[np.floating] = ...,
    *,
    coders_axis: CodersAxis = ...,
    coder_col: Any = ...,
    unit_col: Any = ...,
    value_col: Any = ...,
    all_reliability_data: Any = ...,
    all_value_counts: npt.ArrayLike | None = ...,
    random_coincidences: npt.ArrayLike | None = ...,
    return_components: Literal[False] = ...,
) -> float: ...


@overload
def alpha(
    reliability_data: Any = ...,
    value_counts: npt.ArrayLike | None = ...,
    value_domain: npt.ArrayLike | None = ...,
    level_of_measurement: LevelOfMeasurement = ...,
    dtype: npt.DTypeLike = ...,
    *,
    coders_axis: CodersAxis = ...,
    coder_col: Any = ...,
    unit_col: Any = ...,
    value_col: Any = ...,
    all_reliability_data: Any = ...,
    all_value_counts: npt.ArrayLike | None = ...,
    random_coincidences: npt.ArrayLike | None = ...,
    return_components: Literal[False] = ...,
) -> float | complex | np.inexact: ...


@overload
def alpha(
    reliability_data: Any = ...,
    value_counts: npt.ArrayLike | None = ...,
    value_domain: npt.ArrayLike | None = ...,
    level_of_measurement: LevelOfMeasurement = ...,
    dtype: npt.DTypeLike = ...,
    *,
    coders_axis: CodersAxis = ...,
    coder_col: Any = ...,
    unit_col: Any = ...,
    value_col: Any = ...,
    all_reliability_data: Any = ...,
    all_value_counts: npt.ArrayLike | None = ...,
    random_coincidences: npt.ArrayLike | None = ...,
    return_components: Literal[True],
) -> AlphaResult[Any]: ...


@overload
def alpha(
    reliability_data: Any = ...,
    value_counts: npt.ArrayLike | None = ...,
    value_domain: npt.ArrayLike | None = ...,
    level_of_measurement: LevelOfMeasurement = ...,
    dtype: npt.DTypeLike = ...,
    *,
    coders_axis: CodersAxis = ...,
    coder_col: Any = ...,
    unit_col: Any = ...,
    value_col: Any = ...,
    all_reliability_data: Any = ...,
    all_value_counts: npt.ArrayLike | None = ...,
    random_coincidences: npt.ArrayLike | None = ...,
    return_components: bool,
) -> float | complex | np.inexact | AlphaResult[Any]: ...


def alpha(
    reliability_data: Any = None,
    value_counts: npt.ArrayLike | None = None,
    value_domain: npt.ArrayLike | None = None,
    level_of_measurement: LevelOfMeasurement = "interval",
    dtype: npt.DTypeLike = DEFAULT_DTYPE,
    *,
    coders_axis: CodersAxis = "rows",
    coder_col: Any = None,
    unit_col: Any = None,
    value_col: Any = None,
    all_reliability_data: Any = None,
    all_value_counts: npt.ArrayLike | None = None,
    random_coincidences: npt.ArrayLike | None = None,
    return_components: bool = False,
) -> Any:
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
        It must be one of "nominal", "ordinal", "interval", "ratio", "circular", "bipolar", "polar", or a callable.

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

    return_components : bool, default=False
        If True, return an `AlphaResult` named tuple containing `(alpha, d_o, d_e)`
        (which also provides `.do` and `.de` attributes), where `d_o` is observed disagreement
        and `d_e` is expected disagreement.

    Returns
    -------
    alpha : float, complex, np.inexact, or AlphaResult
        Scalar value of Krippendorff's alpha of type `dtype`, or an `AlphaResult` named tuple
        containing `(alpha, d_o, d_e)` if `return_components=True`.

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
    value_counts, value_domain, all_rel_arr = _prepare_data_and_domain(
        reliability_data=reliability_data,
        value_counts=value_counts,
        value_domain=value_domain,
        level_of_measurement=level_of_measurement,
        all_reliability_data=all_reliability_data,
        all_value_counts=all_value_counts,
        random_coincidences=random_coincidences,
        coders_axis=coders_axis,
        coder_col=coder_col,
        unit_col=unit_col,
        value_col=value_col,
    )

    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.inexact):
        raise ValueError("`dtype` must be an inexact type.")

    if level_of_measurement in ("interval", "ratio", "circular", "bipolar", "polar") and (
        np.iscomplexobj(value_domain) or any(isinstance(v, (complex, np.complexfloating)) for v in value_domain)
    ):
        raise ValueError(f"Level of measurement {level_of_measurement!r} does not support complex values.")

    distance_metric = _distance_metric(level_of_measurement)

    o = _coincidences(value_counts, dtype=dtype)
    n_v = o.sum(axis=0)

    e, dist_n_v, _ = _compute_expected_coincidences(
        value_domain=value_domain,
        n_v=n_v,
        dtype=dtype,
        all_rel_arr=all_rel_arr,
        all_value_counts=all_value_counts,
        random_coincidences=random_coincidences,
    )

    d = _distances(value_domain, distance_metric, dist_n_v, dtype=dtype)

    do = (o * d).sum() / o.sum()
    de = (e * d).sum() / e.sum()
    if de == 0:
        raise ValueError("Expected disagreement is zero, making Krippendorff's alpha undefined.")
    alpha_val = 1 - do / de
    if return_components:
        return AlphaResult(alpha=alpha_val, d_o=do, d_e=de)
    return alpha_val


def alpha_table(
    table: Any,
    coder_col: Any,
    unit_col: Any,
    value_col: Any,
    level_of_measurement: LevelOfMeasurement = "interval",
    dtype: npt.DTypeLike = DEFAULT_DTYPE,
    *,
    value_domain: npt.ArrayLike | None = None,
    all_reliability_data: npt.ArrayLike | None = None,
    all_value_counts: npt.ArrayLike | None = None,
    random_coincidences: npt.ArrayLike | None = None,
    return_components: bool = False,
) -> Any:
    """Compute Krippendorff's alpha from a long/tidy tabular dataset.

    Parameters
    ----------
    table : DataFrame, mapping of columns, or sequence of records
        Tabular reliability data where each row/record represents a rating assigned
        by a coder to a unit.
    coder_col : hashable
        Name or index of the column containing coder identifiers.
    unit_col : hashable
        Name or index of the column containing unit identifiers.
    value_col : hashable
        Name or index of the column containing assigned values/ratings.
    level_of_measurement : string or callable, default "interval"
        Level of measurement.
    dtype : data-type, default np.float64
        Result and computation data-type.
    value_domain : array_like, optional
        Explicit domain of possible values.
    all_reliability_data : array_like, optional
        Reference reliability data for subsample agreement.
    all_value_counts : array_like, optional
        Reference value counts for subsample agreement.
    random_coincidences : array_like, optional
        Precomputed random coincidences matrix.
    return_components : bool, default False
        Whether to return an `AlphaResult` named tuple `(alpha, d_o, d_e)`.

    Returns
    -------
    alpha : float or AlphaResult
        Krippendorff's alpha point estimate, or `AlphaResult` if `return_components=True`.
    """
    return alpha(
        reliability_data=table,
        coder_col=coder_col,
        unit_col=unit_col,
        value_col=value_col,
        level_of_measurement=level_of_measurement,
        dtype=dtype,
        value_domain=value_domain,
        all_reliability_data=all_reliability_data,
        all_value_counts=all_value_counts,
        random_coincidences=random_coincidences,
        return_components=return_components,
    )


def _norm_cdf(x: float) -> float:
    """Cumulative distribution function of standard normal distribution."""
    if x < 0.0:
        return 0.5 * math.erfc(-x / math.sqrt(2.0))
    return 1.0 - 0.5 * math.erfc(x / math.sqrt(2.0))


def _norm_ppf(p: float, *, lower_tail: bool = True) -> float:
    """Quantile function (inverse CDF) of standard normal distribution.

    Uses Acklam's rational approximation refined with Halley's method via erfc,
    achieving full machine precision across extreme tails.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1).")

    if not lower_tail:
        return -_norm_ppf(p, lower_tail=True)
    if p > 0.5:
        return -_norm_ppf(1.0 - p, lower_tail=True)

    # Acklam's rational approximation coefficients for lower tail (p <= 0.5)
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    else:
        q = p - 0.5
        r = q * q
        x = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )

    # Halley's method refinement steps using erfc for extreme tail numerical stability
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    for _ in range(2):
        err = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
        pdf = inv_sqrt2pi * math.exp(-0.5 * x * x)
        if pdf == 0.0:
            break
        delta = err / pdf
        x = x - delta / (1.0 + 0.5 * x * delta)

    return x


def _get_rng(random_state: int | np.random.Generator | np.random.RandomState | None) -> np.random.Generator:
    """Convert seed, Generator, or RandomState into a Generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(int(random_state))
    if isinstance(random_state, np.random.RandomState):
        seed = int(random_state.randint(0, 2**31 - 1))
        return np.random.default_rng(seed)
    if random_state is None:
        return np.random.default_rng()
    raise ValueError(f"Invalid random_state: {random_state!r}. Expected int, Generator, RandomState, or None.")


def _is_metric_marginal_dependent(metric_spec: LevelOfMeasurement) -> bool:
    """Check whether the distance metric depends on marginal counts n_v."""
    if isinstance(metric_spec, str):
        return metric_spec == "ordinal"
    # If the callable explicitly declares independence, honor it
    if getattr(metric_spec, "marginal_dependent", None) is False:
        return False
    # Treat custom callables as marginal-dependent by default because n_v is in the metric protocol
    return True


def _compute_bca_interval(
    valid_alphas: npt.NDArray[np.floating],
    observed_alpha: float | np.floating,
    confidence_level: float,
    num_u: npt.NDArray | None,
    den_u: npt.NDArray,
    nv_matrix: npt.NDArray,
    d: npt.NDArray | None,
    de_fixed: float | np.floating | None,
    fallback_pct: tuple[float, float],
    domain: npt.NDArray | None = None,
    distance_metric: DistanceMetric | None = None,
    counts: npt.NDArray | None = None,
    weights: npt.NDArray | None = None,
    dtype: np.dtype = DEFAULT_DTYPE,  # ty:ignore[invalid-parameter-default]
    is_marginal_dependent: bool = False,
) -> tuple[float, float]:
    """Compute Bias-Corrected and Accelerated (BCa) bootstrap confidence interval."""
    n_units = len(den_u)
    if n_units < 3 or len(valid_alphas) < 3:
        warnings.warn(
            "BCa bootstrap interval requires at least 3 units and 3 valid replicates; "
            "falling back to percentile interval.",
            UserWarning,
            stacklevel=3,
        )
        return float(np.percentile(valid_alphas, fallback_pct[0])), float(np.percentile(valid_alphas, fallback_pct[1]))

    prop = float(np.mean(valid_alphas < observed_alpha))
    n_boot = len(valid_alphas)
    prop = min(max(prop, 1.0 / (2.0 * n_boot)), 1.0 - 1.0 / (2.0 * n_boot))
    z0 = _norm_ppf(prop)

    # Compute jackknife leave-one-out totals in linear memory without constructing an (N x N) matrix
    total_den = float(np.sum(den_u))
    n_j = total_den - den_u
    valid_j = n_j > 1
    safe_nj = np.maximum(n_j, 2)

    if is_marginal_dependent:
        assert counts is not None and weights is not None and domain is not None and distance_metric is not None
        v_len = counts.shape[1]
        diag_idx = np.arange(v_len)
        o_total = counts.T @ weights
        o_total[diag_idx, diag_idx] -= weights.sum(axis=0)
        total_nv = np.sum(nv_matrix, axis=0)
        alpha_jack = np.empty(n_units, dtype=dtype)
        for u in range(n_units):
            nu = n_j[u]
            if nu <= 1:
                alpha_jack[u] = np.nan
                continue
            nv_u = total_nv - nv_matrix[u]
            cu = counts[u]
            wu = weights[u]
            ou_unit = np.outer(cu, wu)
            ou_unit[diag_idx, diag_idx] -= wu
            ou = o_total - ou_unit
            eu = _random_coincidences(nv_u, dtype=dtype)
            du = _distances(domain, distance_metric, nv_u, dtype=dtype)
            deu = (eu * du).sum() / nu
            if deu == 0.0:
                alpha_jack[u] = np.nan
                continue
            dou = (ou * du).sum() / nu
            alpha_jack[u] = 1.0 - dou / deu
    elif de_fixed is not None:
        assert num_u is not None
        total_num = float(np.sum(num_u))
        num_j = total_num - num_u
        do_j = np.where(valid_j, num_j / safe_nj, np.nan)
        alpha_jack = np.where(valid_j, 1.0 - do_j / de_fixed, np.nan)
    else:
        assert num_u is not None and d is not None
        total_num = float(np.sum(num_u))
        num_j = total_num - num_u
        total_nv = np.sum(nv_matrix, axis=0)
        nv_j = total_nv - nv_matrix
        do_j = np.where(valid_j, num_j / safe_nj, np.nan)
        quad_j = np.sum((nv_j @ d) * nv_j, axis=1)
        diag_j = nv_j @ np.diag(d)
        de_j = np.where(valid_j, (quad_j - diag_j) / (safe_nj * (safe_nj - 1)), 0.0)
        valid_mask_j = valid_j & (de_j > 0.0)
        safe_de_min = float(np.finfo(dtype).tiny)
        alpha_jack = np.where(valid_mask_j, 1.0 - do_j / np.maximum(de_j, safe_de_min), np.nan)

    if not np.all(np.isfinite(alpha_jack)):
        warnings.warn(
            "BCa jackknife acceleration could not be computed (one or more leave-one-out estimates were undefined); "
            "falling back to percentile interval.",
            UserWarning,
            stacklevel=3,
        )
        return float(np.percentile(valid_alphas, fallback_pct[0])), float(np.percentile(valid_alphas, fallback_pct[1]))

    mean_j = float(np.mean(alpha_jack))
    diff_j = mean_j - alpha_jack
    denom_a = 6.0 * (float(np.sum(diff_j**2)) ** 1.5)
    a = float(np.sum(diff_j**3) / denom_a) if denom_a > 0.0 else 0.0

    alpha_tail = (1.0 - confidence_level) / 2.0
    z_low = _norm_ppf(alpha_tail)
    z_high = -z_low

    denom_low = 1.0 - a * (z0 + z_low)
    denom_high = 1.0 - a * (z0 + z_high)

    if denom_low == 0.0 or denom_high == 0.0:
        warnings.warn(
            "BCa acceleration denominator was zero; falling back to percentile interval.",
            UserWarning,
            stacklevel=3,
        )
        return float(np.percentile(valid_alphas, fallback_pct[0])), float(np.percentile(valid_alphas, fallback_pct[1]))

    p_low = _norm_cdf(z0 + (z0 + z_low) / denom_low)
    p_high = _norm_cdf(z0 + (z0 + z_high) / denom_high)

    p_low = min(max(p_low, 0.0), 1.0)
    p_high = min(max(p_high, 0.0), 1.0)

    if p_low >= p_high:
        warnings.warn(
            "BCa adjusted quantiles were degenerate; falling back to percentile interval.",
            UserWarning,
            stacklevel=3,
        )
        return float(np.percentile(valid_alphas, fallback_pct[0])), float(np.percentile(valid_alphas, fallback_pct[1]))

    return float(np.percentile(valid_alphas, 100.0 * p_low)), float(np.percentile(valid_alphas, 100.0 * p_high))


def bootstrap_ci(  # noqa: C901
    reliability_data: Any = None,
    value_counts: npt.ArrayLike | None = None,
    value_domain: npt.ArrayLike | None = None,
    level_of_measurement: LevelOfMeasurement = "interval",
    dtype: npt.DTypeLike = DEFAULT_DTYPE,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 2000,
    alternative_threshold: float | None = None,
    method: Literal["percentile", "bca", "basic", "normal"] = "percentile",
    random_state: int | np.random.Generator | np.random.RandomState | None = None,
    all_reliability_data: Any = None,
    all_value_counts: npt.ArrayLike | None = None,
    random_coincidences: npt.ArrayLike | None = None,
    return_distribution: bool = False,
    coders_axis: CodersAxis = "rows",
    coder_col: Any = None,
    unit_col: Any = None,
    value_col: Any = None,
) -> BootstrapResult:
    """Compute bootstrap confidence intervals and statistical significance test for Krippendorff's alpha.

    This function resamples units with replacement using a vectorized decomposition of observed coincidences,
    achieving extreme performance (thousands of resamples in milliseconds).

    Parameters
    ----------
    reliability_data : array_like or mapping or sequence of mappings, optional
        Reliability data matrix with shape `(M, N)` (where `M` is the number of coders/raters and `N` is the
        number of units/items), or dictionary representation. Missing values are represented by `np.nan`.
        Either `reliability_data` or `value_counts` must be provided, but not both.
    value_counts : array_like, with shape (N, V), optional
        Number of coders that assigned a certain value to a determined unit, where N is the number of units
        and V is the value domain length.
    value_domain : array_like, with shape (V,), optional
        Possible values the units can take. If None, it is inferred from data.
    level_of_measurement : string or callable, default "interval"
        Metric for difference measurement. One of "nominal", "ordinal", "interval", "ratio", "circular",
        "bipolar", "polar", or a custom DistanceMetric callable. For custom callable metrics, distances
        are recomputed per bootstrap resample by default (since DistanceMetric accepts marginal counts n_v).
        If a custom metric does not depend on marginals, setting `metric.marginal_dependent = False` enables
        the faster vectorized bootstrap path.
    dtype : data-type, default np.float64
        Computation data type (must be an inexact, non-complex float type).
    confidence_level : float, default 0.95
        Confidence level for the bootstrap confidence interval (strictly between 0 and 1).
    n_resamples : int, default 2000
        Number of bootstrap resamples to draw.
    alternative_threshold : float, optional
        Acceptable reliability benchmark $\\alpha_{\\text{min}}$ (e.g. 0.800 or 0.667).
        When specified, calculates Krippendorff's $q$-value: the empirical probability
        $q = P(\\alpha^* < \\alpha_{\\text{min}})$ that true reliability fails to reach the threshold.
        When degenerate resamples occur (omitted with a warning), $q$ is estimated conditionally
        over all valid resamples: $P(\\alpha^* < \\alpha_{\\text{min}} \\mid \\alpha^* \\text{ is defined})$.
    method : {"percentile", "bca", "basic", "normal"}, default "percentile"
        Bootstrap confidence interval method:
        - "percentile": Empirical percentiles of bootstrap distribution (standard in Krippendorff literature).
        - "bca": Bias-corrected and accelerated bootstrap interval.
        - "basic": Empirical pivotal bootstrap interval ($2\\hat{\\alpha} - \\alpha^*_{(1-p/2)}, 2\\hat{\\alpha} -
          \\alpha^*_{(p/2)}$).
        - "normal": Normal approximation interval using bootstrap standard error ($\\hat{\\alpha} \\pm z \\cdot
          \\text{SE}$).
    random_state : int, Generator, RandomState, or None, optional
        Seed or random number generator for reproducible bootstrap draws.
    all_reliability_data : array_like, optional
        Reference population reliability data for subsample bootstrapping.
    all_value_counts : array_like, optional
        Reference population value counts for subsample bootstrapping.
    random_coincidences : array_like, optional
        Precomputed expected coincidences matrix.
    return_distribution : bool, default False
        If True, the resulting `BootstrapResult.resamples` contains the array of all $B$ bootstrap estimates.

    Returns
    -------
    result : BootstrapResult
        Named tuple containing:
        - `alpha`: observed point estimate.
        - `confidence_interval`: `(ci_low, ci_high)` tuple.
        - `standard_error`: estimated bootstrap standard error.
        - `q_value`: Krippendorff $q$-value if `alternative_threshold` is provided, else None.
        - `resamples`: array of shape `(n_resamples,)` if `return_distribution` is True, else None.

    Examples
    --------
    >>> import numpy as np
    >>> data = [
    ...     [np.nan, np.nan, np.nan, np.nan, np.nan, 3, 4, 1, 2, 1, 1, 3, 3, np.nan, 3],
    ...     [1, np.nan, 2, 1, 3, 3, 4, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ...     [np.nan, np.nan, 2, 1, 3, 4, 4, np.nan, 2, 1, 1, 3, 3, np.nan, 4],
    ... ]
    >>> res = bootstrap_ci(data, level_of_measurement="nominal", random_state=42)
    >>> round(res.alpha, 3)
    0.691
    >>> round(res.ci_low, 3) <= round(res.alpha, 3) <= round(res.ci_high, 3)
    True
    """
    if not (isinstance(confidence_level, (int, float, np.floating)) and 0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be between 0 and 1.")

    if not (isinstance(n_resamples, (int, np.integer)) and n_resamples >= 2):
        raise ValueError("n_resamples must be an integer >= 2.")

    if method not in ("percentile", "bca", "basic", "normal"):
        raise ValueError(f"Invalid method {method!r}. Expected one of 'percentile', 'bca', 'basic', 'normal'.")

    if alternative_threshold is not None:
        if not (
            isinstance(alternative_threshold, (int, float, np.floating))
            and np.isfinite(alternative_threshold)
            and not isinstance(alternative_threshold, bool)
        ):
            raise ValueError("alternative_threshold must be a finite real number.")

    dtype_dt = np.dtype(dtype)
    if not np.issubdtype(dtype_dt, np.inexact):
        raise ValueError("`dtype` must be an inexact type.")
    if np.issubdtype(dtype_dt, np.complexfloating):
        raise ValueError("bootstrap_ci does not support complex dtypes.")

    counts, domain, all_rel_arr = _prepare_data_and_domain(
        reliability_data=reliability_data,
        value_counts=value_counts,
        value_domain=value_domain,
        level_of_measurement=level_of_measurement,
        all_reliability_data=all_reliability_data,
        all_value_counts=all_value_counts,
        random_coincidences=random_coincidences,
        coders_axis=coders_axis,
        coder_col=coder_col,
        unit_col=unit_col,
        value_col=value_col,
    )

    if level_of_measurement in ("interval", "ratio", "circular", "bipolar", "polar") and (
        np.iscomplexobj(domain) or any(isinstance(v, (complex, np.complexfloating)) for v in domain)
    ):
        raise ValueError(f"Level of measurement {level_of_measurement!r} does not support complex values.")

    obs_res = alpha(
        value_counts=counts,
        value_domain=domain,
        level_of_measurement=level_of_measurement,
        dtype=dtype,
        all_reliability_data=all_rel_arr,
        all_value_counts=all_value_counts,
        random_coincidences=random_coincidences,
        coders_axis="rows",
        return_components=True,
    )
    observed_alpha = obs_res.alpha

    has_ref = all_reliability_data is not None or all_value_counts is not None or random_coincidences is not None
    de_fixed = obs_res.d_e if has_ref else None

    distance_metric = _distance_metric(level_of_measurement)

    n_units, v_domain_len = counts.shape
    counts_float = counts.astype(dtype_dt, copy=False)
    pairable = counts.sum(axis=1)
    valid_units = pairable >= 2
    safe_denom = np.maximum(pairable - 1, 1)[:, np.newaxis].astype(dtype_dt, copy=False)
    weights = np.where(
        valid_units[:, np.newaxis],
        np.divide(counts_float, safe_denom, dtype=dtype_dt),
        dtype_dt.type(0.0),
    )
    den_u = np.where(valid_units, pairable, 0).astype(dtype_dt, copy=False)
    nv_matrix = np.where(valid_units[:, np.newaxis], counts_float, dtype_dt.type(0.0))

    is_marginal_dep = de_fixed is None and _is_metric_marginal_dependent(level_of_measurement)

    if not is_marginal_dep:
        n_v = counts_float[valid_units].sum(axis=0, dtype=dtype_dt)
        _, dist_n_v, _ = _compute_expected_coincidences(
            value_domain=domain,
            n_v=n_v,
            dtype=dtype_dt,
            all_rel_arr=all_rel_arr,
            all_value_counts=all_value_counts,
            random_coincidences=random_coincidences,
        )
        d: npt.NDArray | None = _distances(domain, distance_metric, dist_n_v, dtype=dtype_dt)
        wd = weights @ d.T
        num_u: npt.NDArray | None = np.sum(counts_float * wd, axis=1) - np.sum(weights * np.diag(d), axis=1)
    else:
        d = None
        num_u = None

    rng = _get_rng(random_state)
    p_units = [1.0 / n_units] * n_units
    per_resample = n_units + (v_domain_len * v_domain_len if is_marginal_dep else v_domain_len)
    batch_size = min(n_resamples, max(1, 1_000_000 // max(per_resample, 1)))

    diag_idx = np.arange(v_domain_len) if is_marginal_dep else None

    alpha_star = np.empty(n_resamples, dtype=dtype_dt)
    for start in range(0, n_resamples, batch_size):
        end = min(start + batch_size, n_resamples)
        b_count = end - start
        w_batch = rng.multinomial(n_units, p_units, size=b_count)

        if is_marginal_dep:
            assert diag_idx is not None
            unit_chunk_size = max(1, 100_000 // max(v_domain_len, 1))
            for i in range(b_count):
                w_i = w_batch[i].astype(dtype_dt, copy=False)
                nb = float(w_i @ den_u)
                if nb <= 1.0:
                    alpha_star[start + i] = np.nan
                    continue

                nz = np.flatnonzero(w_i)
                o_b = np.zeros((v_domain_len, v_domain_len), dtype=dtype_dt)
                diag_sub = np.zeros(v_domain_len, dtype=dtype_dt)
                for chunk_start in range(0, len(nz), unit_chunk_size):
                    chunk_nz = nz[chunk_start : chunk_start + unit_chunk_size]
                    w_chunk = w_i[chunk_nz]
                    c_chunk = counts_float[chunk_nz]
                    wgt_chunk = weights[chunk_nz]
                    c_w_chunk = w_chunk[:, np.newaxis] * c_chunk
                    o_b += c_w_chunk.T @ wgt_chunk
                    diag_sub += w_chunk @ wgt_chunk

                o_b[diag_idx, diag_idx] -= diag_sub

                nv_b = np.sum(o_b, axis=1)
                e_b = _random_coincidences(nv_b, dtype=dtype_dt)
                d_b = _distances(domain, distance_metric, nv_b, dtype=dtype_dt)
                deb = (e_b * d_b).sum() / nb
                if deb == 0.0:
                    alpha_star[start + i] = np.nan
                    continue
                dob = (o_b * d_b).sum() / nb
                alpha_star[start + i] = 1.0 - dob / deb
        else:
            w_batch_fl = w_batch.astype(dtype_dt, copy=False)
            assert num_u is not None
            num_star = w_batch_fl @ num_u
            n_star = w_batch_fl @ den_u
            valid_n = n_star > 1
            safe_n = np.maximum(n_star, 2)
            do_star = np.where(valid_n, num_star / safe_n, np.nan)

            if de_fixed is not None:
                de_star = np.full(b_count, de_fixed, dtype=dtype_dt)
            else:
                assert d is not None
                nv_star = w_batch_fl @ nv_matrix
                quad = np.sum((nv_star @ d) * nv_star, axis=1)
                diag_term = nv_star @ np.diag(d)
                de_denom = safe_n * (safe_n - 1)
                de_star = np.where(valid_n, (quad - diag_term) / de_denom, 0.0)

            valid_mask = valid_n & (de_star > 0.0)
            safe_de_min = dtype_dt.type(np.finfo(dtype_dt).tiny)
            alpha_star[start:end] = np.where(valid_mask, 1.0 - do_star / np.maximum(de_star, safe_de_min), np.nan)

    finite_mask = np.isfinite(alpha_star)
    valid_alphas = alpha_star[finite_mask]
    n_invalid = n_resamples - len(valid_alphas)
    if len(valid_alphas) < 2:
        raise ValueError(
            "Fewer than 2 valid bootstrap resamples could be estimated "
            "(resamples had zero expected disagreement or fewer than 2 pairable ratings)."
        )
    if n_invalid > 0:
        warnings.warn(
            f"{n_invalid} of {n_resamples} bootstrap resamples had zero expected disagreement "
            "or fewer than 2 pairable ratings, and were omitted from confidence interval estimation.",
            UserWarning,
            stacklevel=2,
        )

    se = float(np.std(valid_alphas, ddof=1))

    alpha_low_pct = 100.0 * (1.0 - confidence_level) / 2.0
    alpha_high_pct = 100.0 - alpha_low_pct

    if method == "percentile":
        ci_low = float(np.percentile(valid_alphas, alpha_low_pct))
        ci_high = float(np.percentile(valid_alphas, alpha_high_pct))
    elif method == "basic":
        ci_low = float(2.0 * observed_alpha - np.percentile(valid_alphas, alpha_high_pct))
        ci_high = float(2.0 * observed_alpha - np.percentile(valid_alphas, alpha_low_pct))
    elif method == "normal":
        alpha_tail = (1.0 - confidence_level) / 2.0
        z = -_norm_ppf(alpha_tail)
        ci_low = float(observed_alpha - z * se)
        ci_high = float(observed_alpha + z * se)
    elif method == "bca":
        ci_low, ci_high = _compute_bca_interval(
            valid_alphas=valid_alphas,
            observed_alpha=observed_alpha,
            confidence_level=confidence_level,
            num_u=num_u,
            den_u=den_u,
            nv_matrix=nv_matrix,
            d=d,
            de_fixed=de_fixed,
            fallback_pct=(alpha_low_pct, alpha_high_pct),
            domain=domain,
            distance_metric=distance_metric,
            counts=counts_float,
            weights=weights,
            dtype=dtype_dt,
            is_marginal_dependent=is_marginal_dep,
        )

    q_val = float(np.mean(valid_alphas < alternative_threshold)) if alternative_threshold is not None else None

    return BootstrapResult(
        alpha=float(observed_alpha),
        confidence_interval=(ci_low, ci_high),
        standard_error=se,
        q_value=q_val,
        resamples=alpha_star if return_distribution else None,
    )


def bootstrap_ci_table(
    table: Any,
    coder_col: Any,
    unit_col: Any,
    value_col: Any,
    level_of_measurement: LevelOfMeasurement = "interval",
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    method: Literal["percentile", "bca", "basic", "normal"] = "percentile",
    alternative_threshold: float | None = None,
    all_reliability_data: npt.ArrayLike | None = None,
    all_value_counts: npt.ArrayLike | None = None,
    random_state: int | np.random.Generator | np.random.RandomState | None = None,
    random_coincidences: npt.ArrayLike | None = None,
    return_distribution: bool = False,
    dtype: npt.DTypeLike = DEFAULT_DTYPE,
    *,
    value_domain: npt.ArrayLike | None = None,
) -> BootstrapResult:
    """Compute bootstrap confidence intervals from a long/tidy tabular dataset.

    Parameters
    ----------
    table : DataFrame, mapping of columns, or sequence of records
        Tabular reliability data where each row/record represents a rating assigned
        by a coder to a unit.
    coder_col : hashable
        Name or index of the column containing coder identifiers.
    unit_col : hashable
        Name or index of the column containing unit identifiers.
    value_col : hashable
        Name or index of the column containing assigned values/ratings.
    level_of_measurement : string or callable, default "interval"
        Level of measurement.
    n_resamples : int, default 2000
        Number of bootstrap resamples.
    confidence_level : float, default 0.95
        Confidence level strictly between 0 and 1.
    method : {"percentile", "bca", "basic", "normal"}, default "percentile"
        Bootstrap confidence interval method.
    alternative_threshold : float, optional
        Reliability threshold $\\alpha_{\\text{min}}$ for hypothesis testing ($q$-value).
    all_reliability_data : array_like, optional
        Reference reliability data for subsample agreement.
    all_value_counts : array_like, optional
        Reference value counts for subsample agreement.
    random_state : int, Generator, RandomState, or None, optional
        Random seed or generator.
    random_coincidences : array_like, optional
        Precomputed random coincidences matrix.
    return_distribution : bool, default False
        Whether to return the full bootstrap distribution.
    dtype : data-type, default np.float64
        Computation data type.
    value_domain : array_like, optional
        Explicit domain of possible values.

    Returns
    -------
    result : BootstrapResult
        Named tuple containing bootstrap results.
    """
    return bootstrap_ci(
        reliability_data=table,
        coder_col=coder_col,
        unit_col=unit_col,
        value_col=value_col,
        level_of_measurement=level_of_measurement,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        alternative_threshold=alternative_threshold,
        all_reliability_data=all_reliability_data,
        all_value_counts=all_value_counts,
        random_state=random_state,
        random_coincidences=random_coincidences,
        return_distribution=return_distribution,
        dtype=dtype,
        value_domain=value_domain,
    )

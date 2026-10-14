from typing import Literal

import numpy as np
import pytest

import krippendorff
from krippendorff import BootstrapResult, alpha, bipolar_metric, bootstrap_ci, circular_metric


@pytest.fixture
def nominal_data() -> list[list[float]]:
    return [
        [np.nan, np.nan, np.nan, np.nan, np.nan, 3, 4, 1, 2, 1, 1, 3, 3, np.nan, 3],
        [1, np.nan, 2, 1, 3, 3, 4, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, 2, 1, 3, 4, 4, np.nan, 2, 1, 1, 3, 3, np.nan, 4],
    ]


@pytest.fixture
def ordinal_data() -> list[list[float]]:
    return [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
        [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan],
    ]


@pytest.fixture
def interval_data() -> list[list[float]]:
    return [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 4, 4, 5, 6, 8, 8, 9, 10],
    ]


def test_bootstrap_ci_basic(nominal_data: list[list[float]]) -> None:
    res = bootstrap_ci(nominal_data, level_of_measurement="nominal", random_state=42)

    assert isinstance(res, BootstrapResult)
    point_alpha = alpha(nominal_data, level_of_measurement="nominal")
    assert np.isclose(res.alpha, point_alpha)
    assert len(res.confidence_interval) == 2
    assert res.ci_low == res.confidence_interval[0]
    assert res.ci_high == res.confidence_interval[1]
    assert res.ci_low < res.ci_high
    assert res.ci_low <= res.alpha <= res.ci_high
    assert res.standard_error > 0.0
    assert res.q_value is None
    assert res.resamples is None


def test_bootstrap_ci_return_distribution(nominal_data: list[list[float]]) -> None:
    res = bootstrap_ci(
        nominal_data,
        level_of_measurement="nominal",
        n_resamples=500,
        return_distribution=True,
        random_state=42,
    )
    assert res.resamples is not None
    assert res.resamples.shape == (500,)
    assert np.all(np.isfinite(res.resamples))
    assert np.isclose(np.std(res.resamples, ddof=1), res.standard_error)


def test_bootstrap_ci_reproducibility(nominal_data: list[list[float]]) -> None:
    res1 = bootstrap_ci(nominal_data, n_resamples=200, random_state=42, return_distribution=True)
    res2 = bootstrap_ci(nominal_data, n_resamples=200, random_state=42, return_distribution=True)
    res3 = bootstrap_ci(nominal_data, n_resamples=200, random_state=99, return_distribution=True)

    assert res1.confidence_interval == res2.confidence_interval
    assert res1.standard_error == res2.standard_error
    assert res1.resamples is not None
    assert res2.resamples is not None
    assert res3.resamples is not None
    assert np.array_equal(res1.resamples, res2.resamples)
    assert not np.array_equal(res1.resamples, res3.resamples)


def test_bootstrap_ci_random_state_types(nominal_data: list[list[float]]) -> None:
    # Generator
    rng = np.random.default_rng(123)
    res_gen = bootstrap_ci(nominal_data, n_resamples=100, random_state=rng)
    assert res_gen.standard_error > 0.0

    # RandomState
    legacy_rng = np.random.RandomState(123)
    res_legacy = bootstrap_ci(nominal_data, n_resamples=100, random_state=legacy_rng)
    assert res_legacy.standard_error > 0.0


def test_bootstrap_ci_alternative_threshold_q_value(interval_data: list[list[float]]) -> None:
    # Point alpha is high (~0.98)
    res_low = bootstrap_ci(interval_data, level_of_measurement="interval", alternative_threshold=0.8, random_state=42)
    assert res_low.q_value is not None
    assert res_low.q_value < 0.05

    res_high = bootstrap_ci(
        interval_data, level_of_measurement="interval", alternative_threshold=0.999, random_state=42
    )
    assert res_high.q_value is not None
    assert res_high.q_value > 0.90


@pytest.mark.parametrize("method", ["percentile", "bca", "basic", "normal"])
def test_bootstrap_ci_methods(
    nominal_data: list[list[float]], method: Literal["percentile", "bca", "basic", "normal"]
) -> None:
    confidence_level = 0.95
    res = bootstrap_ci(
        nominal_data,
        level_of_measurement="nominal",
        method=method,
        confidence_level=confidence_level,
        random_state=42,
        return_distribution=True,
    )
    low, high = res.confidence_interval
    assert low < high
    assert np.isfinite(low) and np.isfinite(high)
    assert res.resamples is not None

    alpha_low_pct = 100.0 * (1.0 - confidence_level) / 2.0
    alpha_high_pct = 100.0 - alpha_low_pct

    if method == "percentile":
        expected_low = float(np.percentile(res.resamples, alpha_low_pct))
        expected_high = float(np.percentile(res.resamples, alpha_high_pct))
        assert np.isclose(low, expected_low)
        assert np.isclose(high, expected_high)
        assert -1.0 <= low <= 1.0
        assert -1.0 <= high <= 1.0
    elif method == "basic":
        expected_low = float(2.0 * res.alpha - np.percentile(res.resamples, alpha_high_pct))
        expected_high = float(2.0 * res.alpha - np.percentile(res.resamples, alpha_low_pct))
        assert np.isclose(low, expected_low)
        assert np.isclose(high, expected_high)
    elif method == "normal":
        from krippendorff.krippendorff import _norm_ppf

        z = -_norm_ppf((1.0 - confidence_level) / 2.0)
        expected_low = float(res.alpha - z * res.standard_error)
        expected_high = float(res.alpha + z * res.standard_error)
        assert np.isclose(low, expected_low)
        assert np.isclose(high, expected_high)
    elif method == "bca":
        assert -1.0 <= low <= 1.0
        assert -1.0 <= high <= 1.0
        assert np.isclose(low, 0.2960557308776609)
        assert np.isclose(high, 1.0)
        res_pct = bootstrap_ci(
            nominal_data,
            level_of_measurement="nominal",
            method="percentile",
            confidence_level=confidence_level,
            random_state=42,
        )
        assert not np.isclose(low, res_pct.ci_low)


def test_bootstrap_ci_confidence_level_widths(nominal_data: list[list[float]]) -> None:
    res_90 = bootstrap_ci(nominal_data, confidence_level=0.90, random_state=42)
    res_95 = bootstrap_ci(nominal_data, confidence_level=0.95, random_state=42)
    res_99 = bootstrap_ci(nominal_data, confidence_level=0.99, random_state=42)

    width_90 = res_90.ci_high - res_90.ci_low
    width_95 = res_95.ci_high - res_95.ci_low
    width_99 = res_99.ci_high - res_99.ci_low

    assert width_90 < width_95 < width_99


def test_bootstrap_ci_ordinal(ordinal_data: list[list[float]]) -> None:
    res = bootstrap_ci(ordinal_data, level_of_measurement="ordinal", random_state=42)
    point = alpha(ordinal_data, level_of_measurement="ordinal")
    assert np.isclose(res.alpha, point)
    assert res.ci_low < res.alpha < res.ci_high

    # Test passing a custom callable metric that is marginal-dependent
    custom_metric = krippendorff.krippendorff._ordinal_metric
    res_custom = bootstrap_ci(ordinal_data, level_of_measurement=custom_metric, random_state=42)
    assert np.isclose(res_custom.alpha, point)
    assert np.isclose(res_custom.ci_low, res.ci_low)
    assert np.isclose(res_custom.ci_high, res.ci_high)

    # Test BCa method with ordinal metric
    res_bca = bootstrap_ci(ordinal_data, level_of_measurement="ordinal", method="bca", random_state=42)
    assert np.isclose(res_bca.alpha, point)
    assert res_bca.ci_low < res_bca.alpha < res_bca.ci_high


def test_bootstrap_ci_ordinal_replicates_exact_match(ordinal_data: list[list[float]]) -> None:
    data_arr = np.array(ordinal_data)
    n_units = data_arr.shape[1]
    n_resamples = 10

    res = bootstrap_ci(
        data_arr,
        level_of_measurement="ordinal",
        n_resamples=n_resamples,
        random_state=42,
        return_distribution=True,
    )
    assert res.resamples is not None

    rng = np.random.default_rng(42)
    w_draws = rng.multinomial(n_units, [1.0 / n_units] * n_units, size=n_resamples)

    for b in range(n_resamples):
        resampled_cols = np.repeat(np.arange(n_units), w_draws[b])
        resampled_data = data_arr[:, resampled_cols]
        expected_alpha = alpha(resampled_data, level_of_measurement="ordinal")
        assert np.isclose(res.resamples[b], expected_alpha)


def test_bootstrap_ci_interval(interval_data: list[list[float]]) -> None:
    res = bootstrap_ci(interval_data, level_of_measurement="interval", random_state=42)
    point = alpha(interval_data, level_of_measurement="interval")
    assert np.isclose(res.alpha, point)
    assert res.ci_low < res.alpha < res.ci_high


def test_bootstrap_ci_ratio() -> None:
    data = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.1, 2.0, 2.9, 4.2, 5.0],
    ]
    res = bootstrap_ci(data, level_of_measurement="ratio", random_state=42)
    point = alpha(data, level_of_measurement="ratio")
    assert np.isclose(res.alpha, point)
    assert res.ci_low < res.ci_high


def test_bootstrap_ci_circular() -> None:
    data = [[0, 1, 2, 3, 0, 1], [0, 2, 2, 0, 0, 1]]
    res = bootstrap_ci(data, level_of_measurement="circular", random_state=42)
    point = alpha(data, level_of_measurement="circular")
    assert np.isclose(res.alpha, point)
    assert res.ci_low < res.ci_high

    # Custom circular metric factory
    metric = circular_metric(circumference=4)
    res_custom = bootstrap_ci(data, level_of_measurement=metric, random_state=42)
    assert np.isclose(res_custom.alpha, point)


def test_bootstrap_ci_bipolar() -> None:
    data = [[-1, 0, 1, -1, 1], [-1, 1, 1, -1, 0]]
    res = bootstrap_ci(data, level_of_measurement="bipolar", random_state=42)
    point = alpha(data, level_of_measurement="bipolar")
    assert np.isclose(res.alpha, point)
    assert res.ci_low < res.ci_high

    metric = bipolar_metric(low=-1, high=1)
    res_custom = bootstrap_ci(data, level_of_measurement=metric, random_state=42)
    assert np.isclose(res_custom.alpha, point)


def test_bootstrap_ci_custom_callable_metric(interval_data: list[list[float]]) -> None:
    # 1. Custom callable without attribute: treated as marginal-dependent by default
    def custom_diff(
        v1: np.ndarray,
        v2: np.ndarray,
        i1: np.ndarray,
        i2: np.ndarray,
        n_v: np.ndarray,
        dtype: object = np.float64,
    ) -> np.ndarray:
        return (v1 - v2) ** 2

    res_default = bootstrap_ci(interval_data, level_of_measurement=custom_diff, random_state=42)
    point = alpha(interval_data, level_of_measurement=custom_diff)
    assert np.isclose(res_default.alpha, point)

    # 2. Custom callable declaring marginal_dependent = False: uses independent tensor path
    setattr(custom_diff, "marginal_dependent", False)  # noqa: B010
    res_indep = bootstrap_ci(interval_data, level_of_measurement=custom_diff, random_state=42)
    assert np.isclose(res_indep.alpha, point)
    assert np.isclose(res_default.ci_low, res_indep.ci_low)
    assert np.isclose(res_default.ci_high, res_indep.ci_high)


def test_bootstrap_ci_value_counts() -> None:
    value_counts = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 2, 1],
            [0, 0, 0, 3],
            [1, 0, 1, 0],
            [0, 2, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
        ]
    )
    res = bootstrap_ci(value_counts=value_counts, level_of_measurement="nominal", random_state=42)
    point = alpha(value_counts=value_counts, level_of_measurement="nominal")
    assert np.isclose(res.alpha, point)
    assert res.ci_low < res.alpha < res.ci_high


def test_bootstrap_ci_dict_data() -> None:
    dict_data = [
        {"u1": 1, "u2": 2, "u3": 3, "u4": 4, "u5": 5},
        {"u1": 1, "u2": 2, "u3": 3, "u4": 4, "u5": 5},
        {"u1": 1, "u2": 3, "u3": 3, "u4": 4, "u5": 4},
    ]
    res = bootstrap_ci(reliability_data=dict_data, level_of_measurement="interval", random_state=42)
    point = alpha(reliability_data=dict_data, level_of_measurement="interval")
    assert np.isclose(res.alpha, point)
    assert res.ci_low < res.alpha < res.ci_high


def test_bootstrap_ci_subsample(nominal_data: list[list[float]]) -> None:
    subsample = [row[:8] for row in nominal_data]
    sub_arr = np.array(subsample)
    n_units = sub_arr.shape[1]
    n_resamples = 5

    # With all_reliability_data
    res_rel = bootstrap_ci(
        sub_arr,
        level_of_measurement="nominal",
        all_reliability_data=nominal_data,
        n_resamples=n_resamples,
        random_state=42,
        return_distribution=True,
    )
    point_rel = alpha(
        sub_arr,
        level_of_measurement="nominal",
        all_reliability_data=nominal_data,
    )
    assert np.isclose(res_rel.alpha, point_rel)
    assert res_rel.ci_low < res_rel.ci_high
    assert res_rel.resamples is not None

    # Verify bootstrap resampling reproduced with all_reliability_data fixed reference
    rng = np.random.default_rng(42)
    w_draws = rng.multinomial(n_units, [1.0 / n_units] * n_units, size=n_resamples)

    for b in range(n_resamples):
        resampled_cols = np.repeat(np.arange(n_units), w_draws[b])
        resampled_data = sub_arr[:, resampled_cols]
        expected_alpha = alpha(
            resampled_data,
            level_of_measurement="nominal",
            all_reliability_data=nominal_data,
        )
        assert np.isclose(res_rel.resamples[b], expected_alpha)

    # With all_value_counts
    all_counts = np.array(
        [
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 2],
        ]
    )
    res_vc = bootstrap_ci(
        subsample,
        level_of_measurement="nominal",
        value_domain=[1, 2, 3, 4],
        all_value_counts=all_counts,
        random_state=42,
    )
    point_vc = alpha(
        subsample,
        level_of_measurement="nominal",
        value_domain=[1, 2, 3, 4],
        all_value_counts=all_counts,
    )
    assert np.isclose(res_vc.alpha, point_vc)

    # With random_coincidences
    rand_coinc = np.ones((4, 4))
    np.fill_diagonal(rand_coinc, 0)
    res_rc = bootstrap_ci(
        subsample,
        level_of_measurement="nominal",
        value_domain=[1, 2, 3, 4],
        random_coincidences=rand_coinc,
        random_state=42,
    )
    point_rc = alpha(
        subsample,
        level_of_measurement="nominal",
        value_domain=[1, 2, 3, 4],
        random_coincidences=rand_coinc,
    )
    assert np.isclose(res_rc.alpha, point_rc)


def test_bootstrap_ci_validation_errors(nominal_data: list[list[float]]) -> None:
    # Neither reliability_data nor value_counts
    with pytest.raises(ValueError, match="Either reliability_data or value_counts must be provided"):
        bootstrap_ci()

    # Both reliability_data and value_counts
    with pytest.raises(ValueError, match="Either reliability_data or value_counts must be provided"):
        bootstrap_ci(reliability_data=nominal_data, value_counts=[[1, 2]])

    # Invalid confidence_level
    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        bootstrap_ci(nominal_data, confidence_level=0.0)

    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        bootstrap_ci(nominal_data, confidence_level=1.0)

    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        bootstrap_ci(nominal_data, confidence_level=-0.1)

    # Invalid n_resamples
    with pytest.raises(ValueError, match="n_resamples must be an integer >= 2"):
        bootstrap_ci(nominal_data, n_resamples=1)
    with pytest.raises(ValueError, match="n_resamples must be an integer >= 2"):
        bootstrap_ci(nominal_data, n_resamples=0)

    # Invalid method
    with pytest.raises(ValueError, match="Invalid method 'invalid'"):
        bootstrap_ci(nominal_data, method="invalid")  # ty:ignore[invalid-argument-type]

    # Non-inexact dtype
    with pytest.raises(ValueError, match="`dtype` must be an inexact type"):
        bootstrap_ci(nominal_data, dtype=int)

    # Complex dtype
    with pytest.raises(ValueError, match="bootstrap_ci does not support complex dtypes"):
        bootstrap_ci(nominal_data, dtype=complex)

    # Multiple subsample options
    with pytest.raises(ValueError, match="At most one of all_reliability_data"):
        bootstrap_ci(
            nominal_data,
            all_reliability_data=nominal_data,
            all_value_counts=[[1, 1], [1, 1]],
        )

    # Invalid alternative_threshold
    with pytest.raises(ValueError, match="alternative_threshold must be a finite real number"):
        bootstrap_ci(nominal_data, alternative_threshold=np.nan)

    with pytest.raises(ValueError, match="alternative_threshold must be a finite real number"):
        bootstrap_ci(nominal_data, alternative_threshold="invalid")  # ty:ignore[invalid-argument-type]

    with pytest.raises(ValueError, match="alternative_threshold must be a finite real number"):
        bootstrap_ci(nominal_data, alternative_threshold=float("inf"))

    # Invalid random_state
    with pytest.raises(ValueError, match="Invalid random_state"):
        bootstrap_ci(nominal_data, random_state="invalid")  # ty:ignore[invalid-argument-type]

    # Complex value_domain with real level_of_measurement
    with pytest.raises(ValueError, match="does not support complex values"):
        bootstrap_ci(value_counts=[[2, 0], [0, 2]], value_domain=[1j, 2j], level_of_measurement="interval")

    # norm_ppf boundary checks
    from krippendorff.krippendorff import _norm_ppf

    with pytest.raises(ValueError, match="p must be in"):
        _norm_ppf(0.0)
    with pytest.raises(ValueError, match="p must be in"):
        _norm_ppf(1.0)


def test_bootstrap_ci_bca_fallback_warning() -> None:
    # 2-unit dataset where BCa falls back to percentile interval
    data = [
        [1, 2],
        [1, 2],
    ]
    with pytest.warns(UserWarning, match="BCa bootstrap interval requires at least 3 units"):
        res = bootstrap_ci(data, method="bca", random_state=42)
    assert res.ci_low <= res.alpha <= res.ci_high


def test_bootstrap_ci_fewer_than_two_valid_replicates_raises() -> None:
    data = [
        [1, 1, 1],
        [1, 1, 2],
    ]
    with pytest.raises(ValueError, match="Fewer than 2 valid bootstrap resamples could be estimated"):
        bootstrap_ci(data, n_resamples=2, random_state=1)


def test_bootstrap_ci_degenerate_resamples_warning() -> None:
    # Dataset where some resamples could draw identical values
    data = [
        [1, 1, 2],
        [1, 1, 2],
    ]
    with pytest.warns(UserWarning, match="resamples had zero expected disagreement"):
        res = bootstrap_ci(data, level_of_measurement="nominal", n_resamples=500, random_state=42)
    assert np.isfinite(res.standard_error)


def test_bootstrap_ci_default_measurement_level_is_interval(interval_data: list[list[float]]) -> None:
    res = bootstrap_ci(interval_data, random_state=42)
    point_interval = alpha(interval_data, level_of_measurement="interval")
    assert np.isclose(res.alpha, point_interval)


def test_norm_ppf_and_cdf_extreme_tails() -> None:
    from krippendorff.krippendorff import _norm_cdf, _norm_ppf

    # Extreme lower tail
    p = 1e-12
    z_low = _norm_ppf(p)
    assert np.isclose(z_low, -7.034483825301133)
    p_rec = _norm_cdf(z_low)
    assert np.isclose(p_rec, p, rtol=1e-6)

    # Upper tail via lower_tail=False
    z_high = _norm_ppf(p, lower_tail=False)
    assert np.isclose(z_high, 7.034483825301133)
    assert np.isclose(z_low, -z_high)

    # Extreme confidence level in normal method does not crash or raise
    data = [[1, 2, 3], [1, 2, 3]]
    res = bootstrap_ci(data, confidence_level=1.0 - 1e-10, method="normal", random_state=42)
    assert np.isfinite(res.ci_low) and np.isfinite(res.ci_high)


def test_bootstrap_ci_bca_jackknife_incomplete_fallback() -> None:
    # 3 units: dropping unit 2 leaves [1, 1] and [1, 1], where expected disagreement is zero
    data = [
        [1, 1, 1],
        [1, 1, 2],
    ]
    with pytest.warns(UserWarning, match="BCa jackknife acceleration could not be computed"):
        res = bootstrap_ci(data, level_of_measurement="nominal", method="bca", random_state=42)
    assert res.ci_low <= res.alpha <= res.ci_high


def test_bootstrap_ci_large_domain_budgeting() -> None:
    # Explicit large domain V=200 forces batch size = 24 (< 50) and exercises multi-batch path
    rng = np.random.default_rng(42)
    domain = np.arange(200)
    data = rng.choice(domain, size=(2, 10))
    res = bootstrap_ci(
        data,
        value_domain=domain,
        level_of_measurement="ordinal",
        n_resamples=100,
        random_state=42,
    )
    assert np.isfinite(res.alpha)
    assert res.ci_low <= res.ci_high

    # Explicit domain V=1000 with 250 units forces both sub-50 batch size and unit chunking
    domain_large = np.arange(1000)
    data_chunked = rng.choice(domain_large, size=(2, 250))
    res_chunked = bootstrap_ci(
        data_chunked,
        value_domain=domain_large,
        level_of_measurement="ordinal",
        n_resamples=5,
        random_state=42,
    )
    assert np.isfinite(res_chunked.alpha)
    assert res_chunked.ci_low <= res_chunked.ci_high


def test_bootstrap_ci_bca_subsample(nominal_data: list[list[float]]) -> None:
    subsample = [row[:8] for row in nominal_data]
    res = bootstrap_ci(
        subsample,
        all_reliability_data=nominal_data,
        level_of_measurement="nominal",
        method="bca",
        random_state=42,
    )
    assert res.ci_low <= res.alpha <= res.ci_high


def test_bootstrap_ci_ordinal_degenerate_resamples() -> None:
    data = [
        [1, 1, 1],
        [1, 1, 2],
    ]
    with pytest.warns(UserWarning, match="resamples had zero expected disagreement"):
        res = bootstrap_ci(data, level_of_measurement="ordinal", n_resamples=500, random_state=42)
    assert np.isfinite(res.standard_error)


def test_bootstrap_ci_ordinal_drop_unit_nu_le_1() -> None:
    data = [
        [1, 1, np.nan],
        [2, np.nan, 2],
    ]
    with pytest.warns(UserWarning):
        res = bootstrap_ci(data, level_of_measurement="ordinal", method="bca", random_state=42)
    assert res.ci_low <= res.alpha <= res.ci_high


def test_bootstrap_ci_bca_degenerate_quantiles_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import krippendorff.krippendorff as km

    data = [[1, 2, 3], [1, 2, 3]]
    monkeypatch.setattr(km, "_norm_cdf", lambda x: 0.1 if x > 0 else 0.9)
    with pytest.warns(UserWarning, match="BCa adjusted quantiles were degenerate"):
        res = bootstrap_ci(data, method="bca", random_state=42)
    assert res.ci_low <= res.alpha <= res.ci_high


def test_bootstrap_ci_random_state_none(nominal_data: list[list[float]]) -> None:
    res = bootstrap_ci(nominal_data, n_resamples=10, random_state=None)
    assert np.isfinite(res.alpha)
    assert res.ci_low <= res.ci_high


def test_norm_ppf_pdf_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    import krippendorff.krippendorff as km

    monkeypatch.setattr(km.math, "exp", lambda _: 0.0)
    x = km._norm_ppf(0.5)
    assert np.isclose(x, 0.0)


def test_bootstrap_ci_ordinal_jackknife_zero_deu() -> None:
    data = [
        [1, 1, 1, 1],
        [2, 1, 1, 1],
    ]
    with pytest.warns(UserWarning, match="BCa jackknife acceleration could not be computed"):
        res = bootstrap_ci(data, level_of_measurement="ordinal", method="bca", random_state=42)
    assert res.ci_low <= res.alpha <= res.ci_high


def test_bootstrap_ci_bca_zero_denominator_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    import krippendorff.krippendorff as km

    orig_ppf = km._norm_ppf

    def mock_ppf(p: float) -> float:
        frame = sys._getframe(1)
        if "a" in frame.f_locals and frame.f_code.co_name == "_compute_bca_interval":
            a = frame.f_locals["a"]
            z0 = frame.f_locals["z0"]
            if a != 0:
                return 1.0 / a - z0
            return 0.0
        return orig_ppf(p)

    monkeypatch.setattr(km, "_norm_ppf", mock_ppf)

    data = [[1, 2, 3], [1, 2, 3], [1, 2, 4]]
    with pytest.warns(UserWarning, match="BCa acceleration denominator was zero"):
        res = bootstrap_ci(data, method="bca", random_state=42)
    assert res.ci_low <= res.alpha <= res.ci_high


def test_package_exports() -> None:
    assert hasattr(krippendorff, "bootstrap_ci")
    assert hasattr(krippendorff, "BootstrapResult")
    assert "bootstrap_ci" in krippendorff.__all__
    assert "BootstrapResult" in krippendorff.__all__

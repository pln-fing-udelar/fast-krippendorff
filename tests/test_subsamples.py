from __future__ import annotations

import numpy as np
import pytest

import krippendorff
import krippendorff.krippendorff as krip


def test_subsample_identical_to_full_data() -> None:
    data = np.array(
        [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
            [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan],
        ]
    )

    for level in ("nominal", "ordinal", "interval", "ratio"):
        alpha_full = krippendorff.alpha(data, level_of_measurement=level)
        alpha_ref = krippendorff.alpha(data, level_of_measurement=level, all_reliability_data=data)
        assert np.isclose(alpha_full, alpha_ref)


def test_subsample_with_all_reliability_data() -> None:
    data = np.array(
        [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
            [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan],
        ]
    )
    domain = [1, 2, 3, 4, 5]
    subsample = data[:, :6]

    alpha_sub_standalone = krippendorff.alpha(subsample, level_of_measurement="ordinal", value_domain=domain)
    alpha_sub_with_all = krippendorff.alpha(
        subsample,
        level_of_measurement="ordinal",
        value_domain=domain,
        all_reliability_data=data,
    )
    # The standalone subsample alpha is ~0.550 while with full De it is ~0.720
    assert not np.isclose(alpha_sub_standalone, alpha_sub_with_all)
    assert round(alpha_sub_with_all, 3) == 0.720


def test_subsample_with_all_value_counts() -> None:
    data = np.array(
        [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
            [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan],
        ]
    )
    domain = np.array([1, 2, 3, 4, 5])
    subsample = data[:, :6]

    all_counts = krip._reliability_data_to_value_counts(data, domain)
    sub_counts = krip._reliability_data_to_value_counts(subsample, domain)

    alpha_counts = krippendorff.alpha(
        value_counts=sub_counts,
        value_domain=domain,
        level_of_measurement="interval",
        all_value_counts=all_counts,
    )
    alpha_direct = krippendorff.alpha(
        reliability_data=subsample,
        value_domain=domain,
        level_of_measurement="interval",
        all_reliability_data=data,
    )
    assert np.isclose(alpha_counts, alpha_direct)


def test_subsample_with_random_coincidences() -> None:
    data = np.array(
        [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
            [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan],
        ]
    )
    domain = np.array([1, 2, 3, 4, 5])
    subsample = data[:, :6]

    all_counts = krip._reliability_data_to_value_counts(data, domain)
    all_o = krip._coincidences(all_counts)
    all_nv = all_o.sum(axis=0)
    all_e = krip._random_coincidences(all_nv)

    alpha_from_all_data = krippendorff.alpha(
        reliability_data=subsample,
        value_domain=domain,
        level_of_measurement="nominal",
        all_reliability_data=data,
    )
    alpha_from_e = krippendorff.alpha(
        reliability_data=subsample,
        value_domain=domain,
        level_of_measurement="nominal",
        random_coincidences=all_e,
    )
    assert np.isclose(alpha_from_all_data, alpha_from_e)

    # Test that scaling / normalizing random_coincidences produces identical results
    normalized_e = all_e / all_e.sum()
    alpha_from_norm_e = krippendorff.alpha(
        reliability_data=subsample,
        value_domain=domain,
        level_of_measurement="nominal",
        random_coincidences=normalized_e,
    )
    assert np.isclose(alpha_from_e, alpha_from_norm_e)


def test_subsample_missing_domain_values_in_subsample() -> None:
    full_data = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ]
    )
    # Subsample only contains categories 1 and 2
    subsample = np.array(
        [
            [1, 2],
            [1, 2],
        ]
    )
    # Agreement is perfect in the subsample
    res = krippendorff.alpha(
        subsample,
        level_of_measurement="nominal",
        all_reliability_data=full_data,
    )
    assert np.isclose(res, 1.0)


def test_subsample_strings_with_all_reliability_data() -> None:
    all_data = np.array(
        [
            ["a", "b", "c"],
            ["a", "b", "c"],
        ]
    )
    subsample = all_data[:, :2]

    res = krippendorff.alpha(
        subsample,
        level_of_measurement="nominal",
        all_reliability_data=all_data,
    )
    assert np.isclose(res, 1.0)


def test_subsample_validation_errors() -> None:
    data = np.array([[1, 2], [1, 2]])

    # Multiple reference parameters
    with pytest.raises(ValueError, match="At most one of"):
        krippendorff.alpha(data, all_reliability_data=data, all_value_counts=[[1, 0], [0, 1]])

    with pytest.raises(ValueError, match="At most one of"):
        krippendorff.alpha(data, all_reliability_data=data, random_coincidences=np.eye(2))

    # Out of domain in all_reliability_data
    with pytest.raises(ValueError, match="reference reliability data contains out-of-domain values"):
        krippendorff.alpha(
            data,
            value_domain=[1, 2],
            all_reliability_data=np.array([[1, 3], [1, 3]]),
        )

    # random_coincidences shape mismatch
    with pytest.raises(ValueError, match="random_coincidences shape"):
        krippendorff.alpha(
            data,
            random_coincidences=np.eye(3),
        )

    # Reference data with no pairable units
    with pytest.raises(ValueError, match="at least one unit in all_reliability_data"):
        krippendorff.alpha(
            data,
            all_reliability_data=np.array([[1, 2], [np.nan, np.nan]]),
        )

    # all_value_counts column count mismatch
    with pytest.raises(ValueError, match="number of columns of all_value_counts"):
        krippendorff.alpha(
            value_counts=[[1, 1], [1, 1]],
            value_domain=[1, 2],
            all_value_counts=[[1, 1, 0], [0, 1, 1]],
        )

    # all_value_counts with no pairable units
    with pytest.raises(ValueError, match="at least one unit in all_value_counts"):
        krippendorff.alpha(
            value_counts=[[1, 1], [1, 1]],
            all_value_counts=[[1, 0], [0, 1]],
        )

    # String all_reliability_data without ordered domain when not nominal
    with pytest.raises(ValueError, match="ordered value_domain is required"):
        krippendorff.alpha(
            data,
            level_of_measurement="ordinal",
            all_reliability_data=[["a", "b"], ["a", "b"]],
        )


def test_subsample_zero_expected_and_observed_disagreement() -> None:
    # Coincidences where all raters agreed completely
    sub_counts = np.array([[2, 0], [0, 2]])
    ref_counts = np.array([[2, 0], [0, 2]])
    res = krippendorff.alpha(
        value_counts=sub_counts,
        level_of_measurement="nominal",
        all_value_counts=ref_counts,
    )
    assert np.isclose(res, 1.0)


def test_subsample_zero_expected_nonzero_observed() -> None:
    # If expected disagreement is 0 but observed is not, alpha is 0.0
    # Custom distance metric that returns 0 for expected coincidences
    e = np.array([[1.0, 0.0], [0.0, 1.0]])  # only diagonal coincidences
    sub_data = np.array([[0, 1], [1, 0]])  # complete disagreement
    res = krippendorff.alpha(
        reliability_data=sub_data,
        value_domain=[0, 1],
        random_coincidences=e,
        level_of_measurement="nominal",
    )
    assert res == 0.0


def test_subsample_out_of_domain_reference_with_value_counts() -> None:
    with pytest.raises(ValueError, match="reference reliability data contains out-of-domain values"):
        krippendorff.alpha(
            value_counts=[[2, 0], [0, 2]],
            value_domain=[1, 2],
            all_reliability_data=[[1, 3], [1, 3]],
        )


def test_subsample_invalid_random_coincidences() -> None:
    data = np.array([[1, 2], [1, 2]])

    with pytest.raises(ValueError, match="non-negative, finite, and have a positive sum"):
        krippendorff.alpha(data, random_coincidences=[[np.nan, 0], [0, 1]])

    with pytest.raises(ValueError, match="non-negative, finite, and have a positive sum"):
        krippendorff.alpha(data, random_coincidences=[[-1, 2], [2, 1]])

    with pytest.raises(ValueError, match="non-negative, finite, and have a positive sum"):
        krippendorff.alpha(data, random_coincidences=[[0, 0], [0, 0]])


def test_subsample_zero_disagreement_equal_total() -> None:
    # Sample and reference both only use category 0 in domain [0, 1]
    # o_sum == e_sum and Do == 0, De == 0 -> should return 1.0, not NaN
    data = np.array([[0, 0], [0, 0]])
    res = krippendorff.alpha(data, value_domain=[0, 1], level_of_measurement="nominal")
    assert res == 1.0

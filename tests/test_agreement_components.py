"""Tests for return_components parameter in krippendorff.alpha."""

import numpy as np
import pytest

import krippendorff
from krippendorff import AlphaResult, KrippendorffResult


def test_alpha_result_namedtuple_properties() -> None:
    """Verify AlphaResult named tuple attributes and backwards-compatible tuple behavior."""
    assert AlphaResult is KrippendorffResult

    res = AlphaResult(alpha=0.8, d_o=0.1, d_e=0.5)
    assert res.alpha == 0.8
    assert res.d_o == 0.1
    assert res.d_e == 0.5
    assert res.do == 0.1
    assert res.de == 0.5

    # Tuple indexing and unpacking
    assert len(res) == 3
    assert res[0] == 0.8
    assert res[1] == 0.1
    assert res[2] == 0.5

    a, do, de = res
    assert a == 0.8
    assert do == 0.1
    assert de == 0.5


def test_return_components_default_false() -> None:
    """Verify default behavior returns only a scalar alpha."""
    data = [[1, 2, 3], [1, 2, 4]]
    res_default = krippendorff.alpha(data, level_of_measurement="interval")
    assert isinstance(res_default, (float, np.floating))
    assert not isinstance(res_default, tuple)

    res_false = krippendorff.alpha(data, level_of_measurement="interval", return_components=False)
    assert res_default == res_false


def test_return_components_nominal() -> None:
    """Verify return_components with nominal metric."""
    reliability_data = [
        [np.nan, np.nan, np.nan, np.nan, np.nan, 3, 4, 1, 2, 1, 1, 3, 3, np.nan, 3],
        [1, np.nan, 2, 1, 3, 3, 4, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, 2, 1, 3, 4, 4, np.nan, 2, 1, 1, 3, 3, np.nan, 4],
    ]
    single_alpha = krippendorff.alpha(reliability_data, level_of_measurement="nominal")
    res = krippendorff.alpha(reliability_data, level_of_measurement="nominal", return_components=True)

    assert isinstance(res, AlphaResult)
    assert np.isclose(res.alpha, single_alpha)
    assert np.isclose(res.alpha, 0.691358024691358)
    assert res.d_o > 0
    assert res.d_e > 0
    assert np.isclose(res.alpha, 1.0 - res.d_o / res.d_e)
    assert res.do == res.d_o
    assert res.de == res.d_e


def test_return_components_interval() -> None:
    """Verify return_components with interval metric."""
    reliability_data = [
        [np.nan, np.nan, np.nan, np.nan, np.nan, 3, 4, 1, 2, 1, 1, 3, 3, np.nan, 3],
        [1, np.nan, 2, 1, 3, 3, 4, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, 2, 1, 3, 4, 4, np.nan, 2, 1, 1, 3, 3, np.nan, 4],
    ]
    single_alpha = krippendorff.alpha(reliability_data, level_of_measurement="interval")
    res = krippendorff.alpha(reliability_data, level_of_measurement="interval", return_components=True)

    assert isinstance(res, AlphaResult)
    assert np.isclose(res.alpha, single_alpha)
    assert np.isclose(res.alpha, 0.8108448928121059)
    assert np.isclose(res.alpha, 1.0 - res.d_o / res.d_e)


def test_return_components_perfect_agreement() -> None:
    """Verify return_components with perfect agreement (D_o == 0, alpha == 1.0)."""
    data = [[1, 2, 3], [1, 2, 3]]
    res = krippendorff.alpha(data, level_of_measurement="nominal", return_components=True)

    assert isinstance(res, AlphaResult)
    assert np.isclose(res.alpha, 1.0)
    assert np.isclose(res.d_o, 0.0)
    assert res.d_e > 0.0


def test_return_components_ordinal_and_ratio() -> None:
    """Verify return_components with ordinal and ratio levels of measurement."""
    data = [[1, 2, 3, 3, 2, 1, 4, 1, 2], [1, 2, 3, 3, 2, 2, 4, 1, 2]]
    res_ord = krippendorff.alpha(data, level_of_measurement="ordinal", return_components=True)
    assert isinstance(res_ord, AlphaResult)
    assert np.isclose(res_ord.alpha, 1.0 - res_ord.d_o / res_ord.d_e)

    res_ratio = krippendorff.alpha(data, level_of_measurement="ratio", return_components=True)
    assert isinstance(res_ratio, AlphaResult)
    assert np.isclose(res_ratio.alpha, 1.0 - res_ratio.d_o / res_ratio.d_e)


def test_return_components_circular_and_bipolar() -> None:
    """Verify return_components with circular and bipolar metrics."""
    circ_data = [[0, 1, 2, 3], [0, 2, 2, 0]]
    res_circ = krippendorff.alpha(circ_data, level_of_measurement="circular", return_components=True)
    assert isinstance(res_circ, AlphaResult)
    assert np.isclose(res_circ.alpha, 1.0 - res_circ.d_o / res_circ.d_e)

    bip_data = [[-1, 0, 1], [-1, 1, 1]]
    res_bip = krippendorff.alpha(bip_data, level_of_measurement="bipolar", return_components=True)
    assert isinstance(res_bip, AlphaResult)
    assert np.isclose(res_bip.alpha, 1.0 - res_bip.d_o / res_bip.d_e)


def test_return_components_with_value_counts() -> None:
    """Verify return_components when providing value_counts instead of reliability_data."""
    value_counts = np.array([
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 2, 1],
        [0, 0, 0, 3],
        [1, 0, 1, 0],
    ])
    single_alpha = krippendorff.alpha(value_counts=value_counts, level_of_measurement="nominal")
    res = krippendorff.alpha(value_counts=value_counts, level_of_measurement="nominal", return_components=True)

    assert isinstance(res, AlphaResult)
    assert np.isclose(res.alpha, single_alpha)
    assert np.isclose(res.alpha, 1.0 - res.d_o / res.d_e)


def test_return_components_with_dict_reliability_data() -> None:
    """Verify return_components with dictionary inputs."""
    data_dicts = [
        {"u1": 1, "u2": 2, "u3": 3},
        {"u1": 1, "u2": 2, "u3": 4},
        {"u2": 2, "u3": 3},
    ]
    single_alpha = krippendorff.alpha(reliability_data=data_dicts, level_of_measurement="interval")
    res = krippendorff.alpha(reliability_data=data_dicts, level_of_measurement="interval", return_components=True)

    assert isinstance(res, AlphaResult)
    assert np.isclose(res.alpha, single_alpha)
    assert np.isclose(res.alpha, 1.0 - res.d_o / res.d_e)


def test_return_components_with_subsamples() -> None:
    """Verify return_components with subsample reference datasets."""
    full_data = [
        [1, 2, 3, 4, 5, 1, 2],
        [1, 2, 3, 4, 5, 2, 2],
    ]
    subsample = [row[:5] for row in full_data]

    single_alpha = krippendorff.alpha(
        reliability_data=subsample,
        all_reliability_data=full_data,
        level_of_measurement="interval",
    )
    res = krippendorff.alpha(
        reliability_data=subsample,
        all_reliability_data=full_data,
        level_of_measurement="interval",
        return_components=True,
    )

    assert isinstance(res, AlphaResult)
    assert np.isclose(res.alpha, single_alpha)
    assert np.isclose(res.alpha, 1.0 - res.d_o / res.d_e)

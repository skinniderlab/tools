from pathlib import Path

import numpy as np
import pytest

from tools import Spectra
from tools.spectra import Spectrum


def _make_spectrum(mz, intensity):
    return Spectrum(
        spectrum_index=0,
        ms_level=2,
        rtime=1.0,
        scan_index=1,
        file=Path("test.mzML"),
        mz=np.array(mz, dtype=np.float64),
        intensity=np.array(intensity, dtype=np.float64),
        polarity=1,
        rtime_unit="minute",
    )


def test_spectra_len(data_dir, spectra):
    assert len(spectra) == 600
    s = Spectra(filepaths=[data_dir / "Blank1A.mzML"])
    assert len(s) == 100


def test_spectra_iter(spectra):
    collected = list(spectra)
    assert len(collected) == 600
    assert all(isinstance(sp, Spectrum) for sp in collected)


def test_spectra_rtime_unit(spectra):
    assert spectra.rtime_unit == "minute"
    # tests if rtime_unit propagates for all files
    assert all(sp.rtime_unit == spectra.rtime_unit for sp in spectra)

    s = Spectra(filepaths=[])
    assert s.rtime_unit == "seconds"
    s._configure_retention_time(1.0, "minute")
    assert s.rtime_unit == "seconds"
    for unit in ("seconds", "minute", "hour"):
        assert s._configure_retention_time_unit(unit) == unit

    with pytest.raises(ValueError, match="Unknown retention time unit"):
        s._configure_retention_time_unit("milliseconds")


def test_spectrum_file_paths(data_dir, spectra):
    expected_files = {
        "Blank1A.mzML",
        "GAS01.mzML",
        "GB01.mzML",
        "L01.mzML",
        "LB01.mzML",
        "T01A.mzML",
    }
    assert {sp.file for sp in spectra} == expected_files


def test_get_by(spectra):
    ms1 = spectra.get_by("ms_level", 1)
    ms2 = spectra.get_by("ms_level", 2)
    assert (ms1["ms_level"] == 1).all()
    assert (ms2["ms_level"] == 2).all()
    assert len(ms1) + len(ms2) == len(spectra)

    blanks = spectra.get_by("file", "Blank1A.mzML")
    assert len(blanks) == 100
    assert (blanks["file"] == "Blank1A.mzML").all()

    # Original row order/index is preserved in the returned frame.
    assert ms1.index.tolist() == sorted(ms1.index.tolist())

    # Cached index yields identical results on a repeat call.
    assert spectra.get_by("ms_level", 1).index.tolist() == ms1.index.tolist()

    # No spectrum matches -> empty frame.
    assert len(spectra.get_by("ms_level", 99)) == 0

    with pytest.raises(AttributeError, match="Unknown Spectrum attribute"):
        spectra.get_by("not_an_attribute", 1)

    # Array-valued attributes cannot be equality-indexed.
    with pytest.raises(TypeError, match="not hashable"):
        spectra.get_by("mz", np.array([1.0]))


def test_filter(spectra):
    # Range query on retention time.
    rts = sorted(sp.rtime for sp in spectra)
    lo, hi = rts[len(rts) // 4], rts[3 * len(rts) // 4]
    in_range = spectra.filter(lambda sp: lo <= sp.rtime <= hi)
    assert in_range["rtime"].between(lo, hi).all()
    assert len(in_range) == sum(1 for sp in spectra if lo <= sp.rtime <= hi)

    # Compound condition.
    ms2_pos = spectra.filter(lambda sp: sp.ms_level == 2 and sp.polarity == 1)
    assert ((ms2_pos["ms_level"] == 2) & (ms2_pos["polarity"] == 1)).all()

    # Filter and get_by agree on a plain equality condition.
    assert spectra.filter(lambda sp: sp.ms_level == 1).index.tolist() == (
        spectra.get_by("ms_level", 1).index.tolist()
    )

    # Always-false predicate -> empty frame.
    assert len(spectra.filter(lambda sp: False)) == 0


def test_configure_retention_time_same_unit_passthrough():
    s = Spectra(filepaths=[], rtime_unit="minute")
    result = s._configure_retention_time(10.0, "minute")
    assert result == pytest.approx(10.0)


def test_configure_retention_time_seconds_to_minute():
    s = Spectra(filepaths=[], rtime_unit="minute")
    result = s._configure_retention_time(120.0, "seconds")
    assert result == pytest.approx(2.0)


def test_configure_retention_time_hour_to_seconds():
    s = Spectra(filepaths=[])
    result = s._configure_retention_time(1.0, "hour")
    assert result == pytest.approx(3600.0)


def test_configure_retention_time_minute_to_hour():
    s = Spectra(filepaths=[])
    result = s._configure_retention_time(60.0, "minute")
    assert result == pytest.approx(3600.0)


def test_match_peaks_all_matched():
    """
    All self peaks have a counterpart in other_spectrum within the ppm window.
    No unmatched rows should appear on either side.
    """
    spec = _make_spectrum(mz=[100.0, 200.0], intensity=[0.5, 0.8])
    other = np.array([[100.0005, 0.6], [200.001, 0.9]])  # both within 20 ppm

    matches = spec._match_peaks(other, ppm_error=20)

    expected = np.array(
        [
            [100.0, 0.5, 100.0005, 0.6],
            [200.0, 0.8, 200.001, 0.9],
        ]
    )
    assert matches.shape == (2, 4)
    assert np.allclose(matches, expected)


def test_match_peaks_partial_match():
    spec = _make_spectrum(mz=[100.0, 200.0, 300.0], intensity=[0.5, 0.8, 0.3])
    other = np.array([[100.0, 0.6], [500.0, 0.7]])

    matches = spec._match_peaks(other, ppm_error=10)

    expected = np.array(
        [
            [100.0, 0.5, 100.0, 0.6],
            [200.0, 0.8, 0.0, 0.0],
            [300.0, 0.3, 0.0, 0.0],
            [0.0, 0.0, 500.0, 0.7],
        ]
    )
    assert matches.shape == (4, 4)
    assert np.allclose(matches, expected)


def test_match_peaks_no_match():
    spec = _make_spectrum(mz=[100.0], intensity=[0.5])
    other = np.array([[300.0, 0.8]])

    matches = spec._match_peaks(other, ppm_error=10)

    expected = np.array(
        [
            [100.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 300.0, 0.8],
        ]
    )
    assert matches.shape == (2, 4)
    assert np.allclose(matches, expected)


def test_match_peaks_closest_chosen():
    spec = _make_spectrum(mz=[100.0], intensity=[0.5])
    other = np.array([[99.999, 0.4], [100.0005, 0.6]])

    matches = spec._match_peaks(other, ppm_error=20)

    # 99.999 is closer to 100.0 than 100.0005
    expected = np.array(
        [
            [100.0, 0.5, 100.0005, 0.6],
            [0.0, 0.0, 99.999, 0.4],
        ]
    )
    assert matches.shape == (2, 4)
    assert np.allclose(matches, expected)


def test_match_peaks_abs_tol():
    spec = _make_spectrum(mz=[100.0], intensity=[0.5])
    other = np.array([[100.005, 0.6]])

    without_abs_tol = spec._match_peaks(other, ppm_error=10, abs_tol=0)
    with_abs_tol = spec._match_peaks(other, ppm_error=10, abs_tol=0.005)

    assert np.allclose(without_abs_tol, [[100.0, 0.5, 0.0, 0.0], [0.0, 0.0, 100.005, 0.6]])
    assert np.allclose(with_abs_tol, [[100.0, 0.5, 100.005, 0.6]])


def test_compare_spectra_empty_other():
    spec = _make_spectrum(mz=[100.0], intensity=[0.5])
    result = spec.compare_spectra(np.empty((0, 2)), ppm_error=10, function=np.dot)
    assert result == 0.0


def test_compare_spectra_matched_peaks():
    spec = _make_spectrum(mz=[100.0, 200.0], intensity=[0.5, 0.8])
    other = np.array([[100.0, 0.6], [200.0, 0.9]])

    result = spec.compare_spectra(other, ppm_error=10, function=np.dot)
    assert result == pytest.approx(1.02)


def test_compare_spectra_no_match():
    spec = _make_spectrum(mz=[100.0], intensity=[0.5])
    other = np.array([[300.0, 0.8]])

    result = spec.compare_spectra(other, ppm_error=10, function=np.dot)
    assert result == pytest.approx(0.0)


def test_combine_peaks():
    spec = _make_spectrum(mz=[100.0, 100.0005, 200.0], intensity=[0.5, 0.6, 0.8])
    combined = spec.combine_peaks(ppm_error=10)
    assert np.allclose(combined, [[100.00025, 0.55], [200.0, 0.8]])

    combined = spec.combine_peaks(ppm_error=1)
    assert np.allclose(combined, [[100.0, 0.5], [100.0005, 0.6], [200.0, 0.8]])

    spec = _make_spectrum(mz=[100.0, 100.0008, 100.0016], intensity=[0.3, 0.6, 0.9])
    combined = spec.combine_peaks(ppm_error=10)
    assert np.allclose(combined, [[100.0008, 0.6]])

    spec = _make_spectrum(mz=[100.0, 0.0, 200.0], intensity=[0.5, 0.9, 0.0])
    combined = spec.combine_peaks(ppm_error=10)
    assert np.allclose(combined, [[100.0, 0.5]])

    # No positive peaks yields an empty (0, 2) array.
    spec = _make_spectrum(mz=[], intensity=[])
    combined = spec.combine_peaks(ppm_error=10)
    assert combined.shape == (0, 2)

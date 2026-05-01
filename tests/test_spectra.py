import pytest

from tools import Spectra
from tools.spectra import Spectrum


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
    assert s.rtime_unit == "unknown"
    s._configure_retention_time(1.0, "minute")
    assert s.rtime_unit == "minute"
    for unit in ("seconds", "minute", "hour"):
        assert s._configure_retention_time_unit(unit) == unit

    with pytest.raises(ValueError, match="Unknown retention time unit"):
        s._configure_retention_time_unit("milliseconds")


def test_spectrum_file_paths(data_dir, spectra):
    expected_files = {
        data_dir / "Blank1A.mzML",
        data_dir / "GAS01.mzML",
        data_dir / "GB01.mzML",
        data_dir / "L01.mzML",
        data_dir / "LB01.mzML",
        data_dir / "T01A.mzML",
    }
    assert {sp.file for sp in spectra} == expected_files


def test_configure_retention_time_same_unit_passthrough():
    s = Spectra(filepaths=[])
    s._configure_retention_time(5.0, "minute")
    result = s._configure_retention_time(10.0, "minute")
    assert result == pytest.approx(10.0)


def test_configure_retention_time_seconds_to_minute():
    s = Spectra(filepaths=[])
    s._configure_retention_time(0.0, "minute")
    result = s._configure_retention_time(120.0, "seconds")
    assert result == pytest.approx(2.0)


def test_configure_retention_time_hour_to_seconds():
    s = Spectra(filepaths=[])
    s._configure_retention_time(0.0, "seconds")
    result = s._configure_retention_time(1.0, "hour")
    assert result == pytest.approx(3600.0)


def test_configure_retention_time_minute_to_hour():
    s = Spectra(filepaths=[])
    s._configure_retention_time(0.0, "hour")
    result = s._configure_retention_time(60.0, "minute")
    assert result == pytest.approx(1.0)

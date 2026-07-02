from dataclasses import fields

import pytest

from tools import Compound, Peak


def test_peaks_object(isotope_db, peaks):
    peak = Peak(
        peak_id=1,
        mz=288.908623547669,
        rt=64.8,
    )

    assert peak in peaks, "Peaks object returned inaccurate membership result. "
    assert 1 in peaks, "Peak object returned inaccurate membership result. "


def test_peak_instantiation(isotope_db, peaks):
    peak_1 = Peak(
        peak_id=28,
        mz=256.961023547669,
        rt=109.2,
        level=4.0,
        annotation="HMDB0060649 Ascorbic acid 2-sulfate C6H8O9S1 + O1 -> C6H8O10S1 - C1H2 -> C5H6O10S1",
        formula=Compound.from_str("C5H6O10S1", isotope_db),
    )
    peak_2 = Peak(
        peak_id=1089,
        mz=387.096923547669,
        rt=159.0,
        annotation="Peak 2807 C9H25N4O2P1 + H2K1O4P1 -> C9H27K1N4O6P2",
        formula=Compound.from_str("C9H27O6N4P2K", isotope_db),
    )

    assert peaks.get_peak(28) == peak_1
    assert peaks.get_peak(1089) == peak_2


def test_getitem_column(peaks):
    mz = peaks["mz"]
    assert len(mz) == len(peaks)
    assert peaks.get_peak(28).mz in set(mz)

    subset = peaks[["mz", "rt"]]
    assert list(subset.columns) == ["mz", "rt"]

    with pytest.raises(KeyError):
        peaks["not_a_column"]


def test_peak_2(peaks_2, isotope_db):
    peak = Peak(peak_id="131.0462_11.0590", mz=131.046249, rt=11.059, smiles="C(CNC(=O)N)C(=O)O")
    assert peaks_2.get_peak("131.0462_11.0590") == peak
    assert peak in peaks_2
    assert "131.0462_11.0590" in peaks_2


def test_mz(peaks):
    assert list(peaks.match_mz(288.908623547669, ppm_error=5)["peak_id"]) == [1]
    assert 1 in set(peaks.match_mz(288.908623547669 * (1 + 3 / 1e6), ppm_error=5)["peak_id"])
    assert {3, 4}.issubset(set(peaks.match_mz(360.934623547669, ppm_error=5)["peak_id"]))
    assert peaks.match_mz(1.0, ppm_error=5).empty


def test_mzs(peaks):
    result = peaks.match_mzs([288.908623547669, 360.934623547669], ppm_error=5)
    assert {1, 3, 4}.issubset(set(result["peak_id"]))

    deduped = peaks.match_mzs([360.934623547669, 360.934523547669], ppm_error=5)
    assert deduped["peak_id"].is_unique

    empty = peaks.match_mzs([1.0], ppm_error=5)
    assert empty.empty
    assert list(empty.columns) == [field.name for field in fields(Peak)]

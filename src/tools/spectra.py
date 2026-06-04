from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pymzml
from tools.utils import get_ppm_range


class Spectra:
    """A collection of mass spectra parsed from one or more mzML files."""

    VALID_RT_UNITS = {"seconds", "minute", "hour"}

    CONVERSIONS: dict[tuple[str, str], Callable[[float], float]] = {
        ("seconds", "minute"): lambda x: x / 60,
        ("seconds", "hour"): lambda x: x / 3600,
        ("minute", "seconds"): lambda x: x * 60,
        ("minute", "hour"): lambda x: x / 60,
        ("hour", "seconds"): lambda x: x * 3600,
        ("hour", "minute"): lambda x: x * 60,
    }

    def __init__(self, filepaths: list[str | Path]):
        """
        Initialize from a list of mzML file paths.

        Parameters
        ----------
        filepaths : list[str or Path]
            Paths to mzML files to parse.
        """
        self.rtime_unit: str = "unknown"
        self.spectra = self._read_mzml_files(filepaths)

    def __len__(self) -> int:
        return len(self.spectra)

    def __iter__(self) -> Iterator["Spectrum"]:
        return iter(self.spectra)

    def _configure_retention_time_unit(self, unit: str) -> str:
        """
        Validate and return a retention time unit string.

        Parameters
        ----------
        unit : str
            Retention time unit to validate. Must be one of 'seconds', 'minute', or 'hour'.

        Returns
        -------
        str
            The validated unit string.

        Raises
        ------
        ValueError
            If the unit is not one of the accepted values.
        """
        if unit not in self.VALID_RT_UNITS:
            raise ValueError(
                f"Unknown retention time unit. Expected one of:"
                f" seconds, minute, or hour. Received {unit}"
            )
        return unit

    def _configure_retention_time(self, rtime: float, unit: str) -> float:
        """
        Convert a retention time value to the collection's established unit.

        Sets the collection's unit from the first spectrum encountered, then converts
        all subsequent values to match.

        Parameters
        ----------
        rtime : float
            Retention time value to convert.
        unit : str
            Unit of the provided retention time value.

        Returns
        -------
        float
            Retention time converted to the collection's established unit.
        """
        unit = self._configure_retention_time_unit(unit)

        # establish the target unit
        if self.rtime_unit == "unknown":
            self.rtime_unit = unit
            return float(rtime)

        if self.rtime_unit == unit:
            return float(rtime)

        return self.CONVERSIONS[(unit, self.rtime_unit)](float(rtime))

    def _read_mzml_files(self, filepaths: list[str | Path]) -> "list[Spectrum]":
        """
        Parse mzML files and return a list of Spectrum objects.

        Parameters
        ----------
        filepaths : list[str or Path]
            Paths to mzML files to parse.

        Returns
        -------
        list[Spectrum]
            Parsed spectra from all provided files.
        """
        spectra: list[Spectrum] = []
        for file in filepaths:
            run = pymzml.run.Reader(file)
            for i, spec in enumerate(run):
                rtime = self._configure_retention_time(spec.scan_time[0], spec.scan_time[1])

                polarity: Literal[0, 1, -1]
                try:
                    polarity = 0 if spec["negative scan"] else 1
                except KeyError:
                    polarity = -1

                spectra.append(
                    Spectrum(
                        spectrum_index=spec.index,
                        ms_level=spec.ms_level,
                        rtime=rtime,
                        scan_index=spec.ID,
                        file=Path(run.path_or_file),
                        mz=spec.mz,
                        intensity=spec.i,
                        polarity=polarity,
                        rtime_unit=self.rtime_unit,
                    )
                )
        return spectra


@dataclass()
class Spectrum:
    """A single mass spectrum with associated metadata."""

    spectrum_index: int
    ms_level: int
    rtime: float
    scan_index: int
    file: Path
    mz: npt.NDArray[np.float64]
    intensity: npt.NDArray[np.float64]
    polarity: Literal[0, 1, -1]
    rtime_unit: str

    def _match_peaks(
        self, other_spectrum: npt.NDArray[np.float64], ppm_error: float, abs_tol: float = 0
    ) -> npt.NDArray[np.float64]:
        """
        Match peaks from this spectrum against ``exp`` within a ppm tolerance.

        Each unmatched peak from ``self`` contributes a row with zero exp values;
        each unmatched peak from ``exp`` contributes a row with zero self values.

        Returns an (n, 4) array with columns [self_mz, self_int, exp_mz, exp_int].
        """
        spec1 = np.column_stack([self.mz, self.intensity])
        spec1 = spec1[np.argsort(spec1[:, 0])]
        spec2 = other_spectrum[np.argsort(other_spectrum[:, 0])]
        matches = []

        for i, spec in enumerate(spec1):
            lower_bound, upper_bound = get_ppm_range(spec[0], ppm_error, abs_tol)
            mask = (spec2[:, 0] >= lower_bound) & (spec2[:, 0] <= upper_bound)
            if sum(mask) > 0:
                matches.append(
                    np.hstack([spec, spec2[mask][np.argsort(np.abs(spec2[mask, 0] - spec[0]))[0]]])
                )
            else:
                matches.append(np.r_[spec, [0, 0]])

        matches = np.array(matches)
        for spec in other_spectrum:
            if spec[0] not in matches[:, 2]:
                matches = np.vstack((matches, np.r_[[0, 0], spec]))

        return matches

    def compare_spectra(
        self,
        other_spectrum: npt.NDArray[np.float64],
        ppm_error: float,
        function: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float],
    ) -> float:
        """
        Score this spectrum against ``other_spectrum``.

        Parameters
        ----------
        other_spectrum:
            Array of shape ``(n, 2)`` with columns ``[mz, intensity]``.

        ppm_error:
            Mass tolerance in parts-per-million for peak matching.

        function:
            Scoring function applied to ``(self_intensities, other_intensities)``
            extracted from matched peak rows.

        Returns
        -------
        float
            Score returned by ``function``, or 0 if ``other_spectrum`` is empty.
        """
        if other_spectrum.size == 0:
            return 0.0
        matches = self._match_peaks(other_spectrum, ppm_error)
        return float(function(matches[:, 1], matches[:, 3]))

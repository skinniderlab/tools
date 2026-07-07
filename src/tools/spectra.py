from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
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

    def __init__(
        self,
        filepaths: list[str | Path],
        rtime_unit: Literal["seconds", "minute", "hour"] = "seconds",
    ):
        """
        Initialize from a list of mzML file paths.

        Parameters
        ----------
        filepaths : list[str or Path]
            Paths to mzML files to parse.

        rtime_unit : {'seconds', 'minute', 'hour'}
            Target retention time unit. All parsed spectra are converted to
            this unit. Defaults to 'seconds'.
        """
        self.rtime_unit: str = self._configure_retention_time_unit(rtime_unit)
        self._df = self._read_mzml_files(filepaths)
        # hash indexes mapping attribute's distinct values to the row positions
        self._equality_indexes: dict[str, dict[object, list[int]]] = {}

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterator["Spectrum"]:
        return (Spectrum(**row._asdict()) for row in self._df.itertuples(index=False))

    def __getitem__(self, item: str | list[str]) -> "pd.Series | pd.DataFrame":
        """
        Access spectrum metadata column(s) by name.

        ``spectra["ms_level"]`` returns that column as a Series;
        ``spectra[["ms_level", "rtime"]]`` returns a DataFrame of those columns.

        Parameters
        ----------
        item : str or list of str
            Column name, or list of column names, to select.

        Returns
        -------
        pd.Series or pd.DataFrame
            The selected column (Series) or columns (DataFrame).

        Raises
        ------
        KeyError
            If any requested column does not exist.
        """
        return self._df[item]

    def get_by(self, attribute: str, value: object) -> pd.DataFrame:
        """
        Return the spectra whose ``attribute`` equals ``value``.

        Backed by a cached hash index built once per attribute on first use, so
        repeated equality lookups are O(1) average (plus the cost of returning
        the matched rows) rather than an O(N) scan. Use :meth:`filter` for range
        queries or any condition more complex than exact equality.

        Parameters
        ----------
        attribute : str
            Name of a scalar :class:`Spectrum` attribute to filter on, e.g.
            ``"ms_level"``, ``"polarity"``, or ``"file"``.
        value : object
            Value the attribute must equal for a spectrum to be included.

        Returns
        -------
        pd.DataFrame
            The subset of the metadata frame whose rows match, with the
            collection's original index preserved. Empty when nothing matches.

        Raises
        ------
        AttributeError
            If ``attribute`` is not a valid :class:`Spectrum` attribute.
        TypeError
            If ``attribute`` holds unhashable (array-valued) data such as
            ``"mz"`` or ``"intensity"``; use :meth:`filter` for those.
        """
        positions = self._equality_index(attribute).get(value, [])
        return self._df.iloc[positions].copy()

    def filter(self, predicate: Callable[["Spectrum"], bool]) -> pd.DataFrame:
        """
        Return the spectra for which ``predicate`` is truthy.

        The general-purpose companion to :meth:`get_by`: use it for range
        queries or any condition more complex than exact equality. Each spectrum
        is materialized and tested, so this is an O(N) scan.

        Parameters
        ----------
        predicate : Callable[[Spectrum], bool]
            Function applied to each :class:`Spectrum`; a spectrum is included
            when the returned value is truthy. For example::

                spectra.filter(lambda sp: 1.0 <= sp.rtime <= 5.0)
                spectra.filter(lambda sp: sp.ms_level == 2 and sp.polarity == 1)

        Returns
        -------
        pd.DataFrame
            The subset of the metadata frame whose rows match, with the
            collection's original index preserved. Empty when nothing matches.
        """
        mask = np.array([predicate(spectrum) for spectrum in self], dtype=bool)
        return self._df[mask].copy()

    def _equality_index(self, attribute: str) -> dict[object, list[int]]:
        """
        Return a cached ``value -> row positions`` index for ``attribute``.

        The index is built once on first request and reused thereafter. Building
        it is a single O(N) pass; subsequent lookups against it are O(1) average.

        Parameters
        ----------
        attribute : str
            Name of a scalar :class:`Spectrum` attribute to index.

        Returns
        -------
        dict[object, list[int]]
            Mapping from each distinct attribute value to the ascending list of
            row positions holding it.

        Raises
        ------
        AttributeError
            If ``attribute`` is not a valid :class:`Spectrum` attribute.
        TypeError
            If the attribute's values are not hashable.
        """
        valid_attributes = {f.name for f in fields(Spectrum)}
        if attribute not in valid_attributes:
            raise AttributeError(
                f"Unknown Spectrum attribute {attribute!r}."
                f" Expected one of: {', '.join(sorted(valid_attributes))}."
            )

        if attribute not in self._equality_indexes:
            index: dict[object, list[int]] = {}
            try:
                for position, value in enumerate(self._df[attribute].tolist()):
                    index.setdefault(value, []).append(position)
            except TypeError as error:
                raise TypeError(
                    f"Cannot build an equality index on {attribute!r}: its values"
                    f" are not hashable. Use filter() for array-valued attributes"
                    f" such as 'mz' and 'intensity'."
                ) from error
            self._equality_indexes[attribute] = index

        return self._equality_indexes[attribute]


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
        Convert a retention time value to the collection's target unit.

        Parameters
        ----------
        rtime : float
            Retention time value to convert.
        unit : str
            Unit of the provided retention time value.

        Returns
        -------
        float
            Retention time converted to ``self.rtime_unit``.
        """
        unit = self._configure_retention_time_unit(unit)

        if self.rtime_unit == unit:
            return float(rtime)

        return self.CONVERSIONS[(unit, self.rtime_unit)](float(rtime))

    def _read_mzml_files(self, filepaths: list[str | Path]) -> pd.DataFrame:
        """
        Parse mzML files into the canonical spectrum metadata DataFrame.

        The returned frame has exactly one column per :class:`Spectrum` field,
        in field order, and is the single source of truth from which Spectrum
        objects are materialized and which :meth:`to_df` returns.

        Parameters
        ----------
        filepaths : list[str or Path]
            Paths to mzML files to parse.

        Returns
        -------
        pd.DataFrame
            Canonical spectrum metadata frame with columns matching the
            Spectrum fields.
        """
        records: list[dict] = []
        for file in filepaths:
            run = pymzml.run.Reader(file)
            for spec in run:
                rtime = self._configure_retention_time(spec.scan_time[0], spec.scan_time[1])

                polarity: Literal[0, 1, -1]
                try:
                    polarity = 0 if spec["negative scan"] else 1
                except KeyError:
                    polarity = -1

                records.append(
                    {
                        "spectrum_index": spec.index,
                        "ms_level": spec.ms_level,
                        "rtime": rtime,
                        "scan_index": spec.ID,
                        "file": Path(run.path_or_file).name,
                        "mz": spec.mz,
                        "intensity": spec.i,
                        "polarity": polarity,
                        "rtime_unit": self.rtime_unit,
                    }
                )

        columns = [field.name for field in fields(Spectrum)]
        return pd.DataFrame(records, columns=columns).reset_index(drop=True)


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

    def combine_peaks(self, ppm_error: float) -> np.ndarray:
        """
        Merge peaks whose m/z values lie within a ppm tolerance of one another.

        Peaks with a non-positive m/z or intensity are discarded and the
        remaining peaks are sorted by ascending m/z. Consecutive peaks whose
        m/z difference is no larger than ``mz * ppm_error * 1e-6`` are collapsed
        into a single peak. Grouping is transitive: a run of adjacent peaks that
        each fall within tolerance of their neighbour are all merged together.
        Each merged peak's m/z and intensity are the means of its members.

        Parameters
        ----------
        ppm_error : float
            Mass tolerance in parts-per-million used to decide whether two
            adjacent peaks belong to the same group.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n, 2)`` with columns ``[mz, intensity]``, one row
            per merged peak, sorted by ascending m/z. An empty ``(0, 2)`` array
            is returned when the spectrum contains no positive peaks.
        """
        peak = np.column_stack([self.mz, self.intensity])
        spectrum = peak[np.all(peak > 0, axis=1)]
        spectrum = spectrum[np.argsort(spectrum[:, 0])]

        if spectrum.shape[0] == 0:
            return spectrum.reshape(0, 2)

        mzs = spectrum[:, 0]
        mz_diffs = np.round(mzs[1:] - mzs[:-1], 9)
        within_window = mz_diffs <= mzs[:-1] * ppm_error * 1e-6

        grouped_mzs = []
        grouped_intensities = []
        group = [0]
        for i, within in enumerate(within_window):
            if within:
                group.append(i + 1)
            else:
                grouped_mzs.append(np.mean(mzs[group]))
                grouped_intensities.append(np.mean(spectrum[group, 1]))
                group = [i + 1]
        grouped_mzs.append(np.mean(mzs[group]))
        grouped_intensities.append(np.mean(spectrum[group, 1]))

        return np.stack([grouped_mzs, grouped_intensities], axis=1)


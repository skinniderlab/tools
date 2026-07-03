import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

from tools import Compound, IsotopeDB
from tools.utils import SortedValueIndex, get_ppm_range


class Peaks:
    """A collection of chromatographic peaks parsed from a peak list file."""

    def __init__(self, filepath: str | Path, isotope_filepath: str | Path | None = None):
        """
        Initialize from a peak list file.

        Parameters
        ----------
        filepath : str or Path
            Path to the CSV peak list file containing m/z and retention time values.
        isotope_filepath : str or Path, optional
            Path to an isotope database file. If None, the default database is used.
        """
        self.isotope_db = IsotopeDB(isotope_filepath)
        self._df = self._build_dataframe(filepath)
        self._row_by_id = {peak_id: pos for pos, peak_id in enumerate(self._df["peak_id"])}
        # m/z sorted once so match queries can binary-search their window
        # (O(log N) per query) instead of scanning every peak.
        self._mz_index = SortedValueIndex(self._df["mz"].to_numpy(), get_ppm_range)

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterator["Peak"]:
        return (self._to_peak(row) for row in self._df.itertuples(index=False))

    def __getitem__(self, item: str | list[str]) -> "pd.Series | pd.DataFrame":
        """
        Access peak column(s) by name.

        ``peaks["mz"]`` returns that column as a Series; ``peaks[["mz", "rt"]]``
        returns a DataFrame of those columns.

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

    def get_peak(self, peak_id: int | str) -> "Peak":
        """
        Retrieve a single peak by its peak ID.

        Parameters
        ----------
        peak_id : int or str
            Peak ID to look up.

        Returns
        -------
        Peak
            The matching Peak object.

        Raises
        ------
        KeyError
            If no peak with the given ID exists in the collection.
        """
        if (pos := self._row_by_id.get(peak_id)) is None:
            raise KeyError(
                f"Peak with peak id {peak_id} is not present in the provided peaks file."
            )

        return self._to_peak(next(self._df.iloc[[pos]].itertuples(index=False)))

    def __contains__(self, item: object) -> bool:
        if isinstance(item, Peak):
            return any(item == peak for peak in self)
        return item in self._row_by_id

    @staticmethod
    def _read_and_validate(filepath: str | Path) -> pd.DataFrame:
        """
        Read and validate a peak list CSV, normalizing column names.

        Accepts files with either 'mz'/'RT' or 'medMz'/'medRt' column naming conventions.
        Auto-generates peak IDs from m/z and RT if not present, and ensures optional
        metadata columns exist.

        Parameters
        ----------
        filepath : str or Path
            Path to the CSV peak list file.

        Returns
        -------
        pd.DataFrame
            Validated DataFrame with normalized columns: 'mz', 'rt', 'peak_id', and
            optional columns 'formula', 'level', 'accession', 'annotation'.

        Raises
        ------
        ValueError
            If required m/z or retention time columns are missing, or if peak IDs are not unique.
        """
        df = pd.read_csv(filepath)

        if "mz" in df.columns:
            mz_col = "mz"
        elif "medMz" in df.columns:
            mz_col = "medMz"
        else:
            raise ValueError(
                f"Invalid peak file. {filepath} does not contain m/z values for "
                f" peaks. Please ensure your peak file contains m/z and rt values."
            )

        if "RT" in df.columns:
            rt_col = "RT"
        elif "medRt" in df.columns:
            rt_col = "medRt"
        else:
            raise ValueError(
                f"Invalid peak file. {filepath} does not contain retention time for "
                f" peaks. Please ensure your peak file contains m/z and rt values. "
            )

        if "peak_id" not in df.columns:
            df["peak_id"] = df[mz_col].map("{:.4f}".format) + "_" + df[rt_col].map("{:.4f}".format)

        if len(df) != df.peak_id.nunique():
            raise ValueError(
                f"Duplicate peak IDs found in the {Path(filepath).name}. "
                "Please ensure all peak IDs are unique."
            )

        # normalizing column names
        df = df.rename(columns={rt_col: "rt", mz_col: "mz"})

        # ensuring optional columns exist
        for col in ("formula", "level", "accession", "annotation"):
            if col not in df.columns:
                df[col] = None

        return df.replace({"Unknown": None, float("nan"): None})

    def _build_dataframe(self, filepath: str | Path) -> pd.DataFrame:
        """
        Read and validate the peak file into the canonical peak DataFrame.

        The returned frame has exactly one column per Peak field, in field order.
        Formula strings are canonicalized (invalid ones become None); a ``smiles``
        column is derived from ``true_SMILES`` when present. This frame is the
        single source of truth from which Peak objects are built and which
        :meth:`to_df` returns.

        Parameters
        ----------
        filepath : str or Path
            Path to the CSV peak list file.

        Returns
        -------
        pd.DataFrame
            Canonical peak DataFrame with columns matching the Peak fields.
        """
        df = self._read_and_validate(filepath)

        df["smiles"] = df["true_SMILES"] if "true_SMILES" in df.columns else None
        df["formula"] = self._resolve_formulas(df["formula"])

        columns = [field.name for field in fields(Peak)]
        return df[columns].reset_index(drop=True)

    def _resolve_formulas(self, formulas: pd.Series) -> pd.Series:
        """
        Canonicalize a column of formula strings, memoizing repeated values.

        Each distinct formula is parsed at most once; unparseable formulas are
        logged and mapped to None.

        Parameters
        ----------
        formulas : pd.Series
            Raw formula strings (or None) from the peak file.

        Returns
        -------
        pd.Series
            Canonical formula strings, with None where the input was None or unparseable.
        """
        cache: dict[str, str | None] = {}

        def resolve(formula: str | None) -> str | None:
            if formula is None:
                return None
            if formula not in cache:
                try:
                    cache[formula] = Compound.from_str(
                        formula=formula, isotope_db=self.isotope_db
                    ).formula
                except ValueError:
                    logging.warning(f"Unable to parse formula: {formula}")
                    cache[formula] = None
            return cache[formula]

        # dtype=object keeps None as None (a plain Series would infer a string
        # dtype and coerce None to NaN).
        return pd.Series([resolve(f) for f in formulas], index=formulas.index, dtype=object)

    @staticmethod
    def _to_peak(row: "tuple") -> "Peak":
        """
        Build a Peak from a row of the canonical DataFrame.

        Parameters
        ----------
        row : namedtuple
            A row yielded by ``self._df.itertuples(index=False)``; its fields
            match the Peak fields one-to-one.

        Returns
        -------
        Peak
            The Peak object for that row.
        """
        return Peak(**row._asdict())

    def match_mz(self, mz: float, ppm_error: float) -> pd.DataFrame:
        """
        Match a single m/z value against the peaks in this collection.

        Find every peak whose m/z falls within ``ppm_error`` parts-per-million
        of the query value.

        Parameters
        ----------
        mz : float
            m/z value to match against the collection's peaks.
        ppm_error : float
            Mass tolerance in parts-per-million used to define the match window.

        Returns
        -------
        pd.DataFrame
            The subset of the peak DataFrame whose rows match, with the
            collection's original index preserved. Empty when no peak matches.
        """
        return self.match_mzs([mz], ppm_error)

    def match_mzs(self, mzs: Iterable[float], ppm_error: float) -> pd.DataFrame:
        """
        Match a sequence of m/z values against the peaks in this collection.

        Return the subset of peaks whose m/z falls within ``ppm_error``
        parts-per-million of *any* of the query values. Each matching peak
        appears once, even if it matches several query values, in the
        collection's original order.

        Parameters
        ----------
        mzs : Iterable[float]
            m/z values to match against the collection's peaks.
        ppm_error : float
            Mass tolerance in parts-per-million used to define the match window.

        Returns
        -------
        pd.DataFrame
            The subset of the peak DataFrame (see :meth:`to_df`) whose rows match,
            with the collection's original index preserved. Empty when nothing matches.
        """
        positions = self._mz_index.search_many(mzs, ppm_error)
        return self._df.iloc[positions].copy()


@dataclass(frozen=True, slots=True)
class Peak:
    """A single chromatographic peak with optional compound annotation."""

    peak_id: int | str
    mz: float
    rt: float
    formula: str | None = None
    accession: str | None = None
    level: int | None = None
    annotation: str | None = None
    smiles: str | None = None

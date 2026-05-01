import logging
from collections.abc import Iterator
from dataclasses import InitVar, dataclass, field
from pathlib import Path

import pandas as pd

from tools import Compound, IsotopeDB


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
        self.peaks = self._process_peaks(filepath)

    def __len__(self) -> int:
        return len(self.peaks)

    def __iter__(self) -> Iterator["Peak"]:
        return iter(self.peaks.values())

    def __getitem__(self, item: int | str) -> "Peak":
        """
        Retrieve a peak by its peak ID.

        Parameters
        ----------
        item : int or str
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
        if (query_peak := self.peaks.get(item, None)) is None:
            raise KeyError(f"Peak with peak id {item} is not present in the provided peaks file.")

        return query_peak

    def __contains__(self, item: int | str) -> bool:
        return item in self.peaks or item in self.peaks.values()

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

    def _process_peaks(self, filepath: str | Path) -> "dict[int | str, Peak]":
        """
        Read, validate, and convert peak rows into Peak objects.

        Parameters
        ----------
        filepath : str or Path
            Path to the CSV peak list file.

        Returns
        -------
        dict[str or int, Peak]
            Dictionary mapping peak IDs to their corresponding Peak objects.
        """
        df = self._read_and_validate(filepath)

        has_smiles = True if "true_SMILES" in df.columns else False

        peak_info = {}
        for row in df.itertuples():
            computed_formula = None
            try:
                if row.formula is not None:
                    computed_formula = Compound.from_str(
                        formula=row.formula, isotope_db=self.isotope_db
                    )
            except ValueError:
                # ignore invalid formulas
                logging.warning(f"Unable to parse formula: {row.formula}")

            peak_info[row.peak_id] = Peak(
                peak_id=row.peak_id,
                mz=row.mz,
                rt=row.rt,
                formula=computed_formula,
                level=row.level,
                accession=row.accession,
                annotation=row.annotation,
                smiles=row.true_SMILES if has_smiles else None,
            )
        return peak_info


@dataclass(frozen=True)
class Peak:
    """A single chromatographic peak with optional compound annotation."""

    peak_id: int | str
    mz: float
    rt: float
    formula: Compound | None = None
    accession: str | None = None
    level: int | None = None
    annotation: str | None = None
    smiles: str | None = None

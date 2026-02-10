from dataclasses import InitVar, dataclass, field

import pandas as pd

from tools import Compound, IsotopeDB
import logging


class Peaks:
    def __init__(self, filepath, isotope_filepath: str = None):
        self.isotope_db = IsotopeDB(isotope_filepath)
        self.peaks = self._process_peaks(filepath)

    def __len__(self):
        return len(self.peaks)

    def __iter__(self):
        return iter(self.peaks.values())

    def __getitem__(self, item):
        if (query_peak := self.peaks.get(item, None)) is None:
            raise KeyError(f"Peak with peak id {item} is not present in the provided peaks file.")

        return query_peak

    def __contains__(self, item: int):
        return item in self.peaks or item in self.peaks.values()

    @staticmethod
    def _read_and_validate(filepath):
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
            df["peak_id"] = df.apply(lambda x: f"{x[mz_col]}_{x[rt_col]}", axis=1)

        if len(df) != df.peak_id.nunique():
            raise ValueError(
                f"Duplicate peak IDs found in the {filepath.name}. "
                "Please ensure all peak IDs are unique."
            )

        # normalizing column names
        df = df.rename(columns={rt_col: "rt", mz_col: "mz"})

        # ensuring optional columns exist
        for col in ("formula", "level", "accession", "annotation"):
            if col not in df.columns:
                df[col] = None

        return df.replace({"Unknown": None, float("nan"): None})

    def _process_peaks(self, filepath):
        df = self._read_and_validate(filepath)

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
            )
        return peak_info


@dataclass(frozen=True)
class Peak:
    peak_id: int | str
    mz: float
    rt: float
    formula: Compound | None = None
    accession: str | None = None
    level: int | None = None
    annotation: str | None = None

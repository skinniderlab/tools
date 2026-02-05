from dataclasses import InitVar, dataclass, field

import pandas as pd

from tools import Compound, IsotopeDB


class Peaks:
    def __init__(self, filepath, isotope_db):
        self.isotope_db = isotope_db
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

        return {
            row.peak_id: Peak(
                id=row.peak_id,
                mz=row.mz,
                rt=row.rt,
                formula_str=row.formula,
                level=row.level,
                accession=row.accession,
                isotope_db=self.isotope_db,
                annotation=row.annotation,
            )
            for row in df.itertuples()
        }


@dataclass(frozen=True)
class Peak:
    id: int | str
    mz: float
    rt: float
    isotope_db: InitVar[IsotopeDB]
    formula_str: InitVar[str | None] = None
    accession: str | None = None
    level: int | None = None
    annotation: str | None = None
    formula: Compound | None = field(init=False, default=None)

    def __post_init__(self, isotope_db, formula_str):
        computed_formula = None
        try:
            if formula_str is not None:
                computed_formula = Compound.from_str(formula=formula_str, isotope_db=isotope_db)
        except ValueError:
            # ignore invalid formulas
            pass

        object.__setattr__(self, "formula", computed_formula)

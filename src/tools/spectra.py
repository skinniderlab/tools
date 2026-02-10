from pathlib import Path
from typing import Literal

import pymzml
import pandas as pd
from dataclasses import dataclass
import numpy.typing as npt

import numpy as np


class Spectra:
    VALID_RT_UNITS = {"seconds", "minute", "hour"}

    CONVERSIONS = {
        ("seconds", "minute"): lambda x: x / 60,
        ("seconds", "hour"): lambda x: x / 3600,
        ("minute", "seconds"): lambda x: x * 60,
        ("minute", "hour"): lambda x: x / 60,
        ("hour", "seconds"): lambda x: x * 3600,
        ("hour", "minute"): lambda x: x * 60,
    }

    def __init__(self, filepaths: list[str | Path]):
        self.rtime_unit = None
        self.spectra = self._read_mzml_files(filepaths)

    def __len__(self) -> int:
        return len(self.spectra)

    def __iter__(self):
        return iter(self.spectra)

    def _configure_retention_time_unit(self, unit):
        if unit not in self.VALID_RT_UNITS:
            raise ValueError(
                f"Unknown retention time unit. Expected one of:"
                f" seconds, minute, or hour. Received {unit}"
            )
        return unit

    def _configure_retention_time(self, rtime, unit):
        unit = self._configure_retention_time_unit(unit)

        # establish the target unit
        if self.rtime_unit is None:
            self.rtime_unit = unit
            return float(rtime)

        if self.rtime_unit == unit:
            return float(rtime)

        return self.CONVERSIONS[(unit, self.rtime_unit)](float(rtime))

    def _read_mzml_files(self, filepaths: list[str | Path]):
        spectra = []
        for file in filepaths:
            run = pymzml.run.Reader(file)
            for i, spec in enumerate(run):
                rtime = self._configure_retention_time(spec.scan_time[0], spec.scan_time[1])

                try:
                    polarity = 0 if spec["negative scan"] else 1
                except KeyError:
                    polarity = -1

                spectra.append(Spectrum(
                    spectrum_index=spec.index,
                    ms_level=spec.ms_level,
                    rtime=rtime,
                    scan_index=spec.ID,
                    file=Path(run.path_or_file),
                    mz=spec.mz,
                    intensity=spec.i,
                    polarity=polarity,
                    rtime_unit=self.rtime_unit,
                ))
        return spectra


@dataclass()
class Spectrum:
    spectrum_index: int
    ms_level: int
    rtime: float
    scan_index: int
    file: Path
    mz: npt.NDArray[np.float64]
    intensity: npt.NDArray[np.float64]
    polarity: Literal[0, 1, -1]
    rtime_unit: str

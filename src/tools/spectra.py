from pathlib import Path
from typing import Literal

import pymzml
import pandas as pd
from dataclasses import dataclass
import numpy.typing as npt

import numpy as np


class Spectra:
    def __init__(self, filepaths: list[str | Path]):
        self.spectra = self._read_mzml_files(filepaths)

    def _read_mzml_files(self, filepaths: list[str | Path]):
        spectra = []
        for file in filepaths:
            run = pymzml.run.Reader(file)
            for i, spec in enumerate(run):
                match spec.scan_time[1]:
                    case "seconds":
                        rtime_unit = "seconds"
                    case "minute":
                        rtime_unit = "minute"
                    case "hour":
                        rtime_unit = "hour"
                    case _:
                        raise ValueError(
                            f"Unknown retention time unit. Expected one of:"
                            f" seconds, minute, or hour. Received {spec.scan_time[1]}"
                        )

                try:
                    polarity = 0 if spec["negative scan"] else 1
                except KeyError:
                    polarity = -1

                spectra.append(
                    Spectrum(
                        spectrum_index=spec.index,
                        ms_level=spec.ms_level,
                        rtime=spec.scan_time[0],
                        scan_index=spec.ID,
                        file=Path(run.path_or_file),
                        mz=spec.mz,
                        intensity=spec.i,
                        polarity=polarity,
                        rtime_unit=rtime_unit,
                    )
                )

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
    polarity: Literal[0, 1]
    rtime_unit: str

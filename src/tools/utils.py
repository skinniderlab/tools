import gzip
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def get_file_delimiter(filepath: Path) -> str:
    """
    Identify the delimiter used in a delimited text file.

    Parameters
    ----------
    filepath : Path
        Path to the file to inspect. Supports gzip-compressed files.

    Returns
    -------
    str
        The detected delimiter character: comma (','), tab ('\\t'), or space (' ').
    """
    f = gzip.open(filepath, "rt") if filepath.suffix == ".gz" else Path.open(filepath)  # noqa: SIM115
    first_line = next(f).strip()
    f.close()

    for sep in (",", "\t", " "):
        header = first_line.split(sep)
        if len(header) > 1:
            break

    return sep


def get_file_info(filepath: Path) -> dict[str, Any]:
    """
    Load metadata from a delimited file containing m/z and/or intensity data.

    Parameters
    ----------
    filepath : Path
        Path to the file to inspect. Supports gzip-compressed files.

    Returns
    -------
    dict
        Dictionary with keys 'delim', 'open_fn', 'mode', 'n_rows', 'n_columns',
        and 'has_header'.
    """

    def _get_open_method(_filepath: Path) -> tuple[Callable[..., Any], str]:
        return (gzip.open, "rt") if _filepath.suffix == ".gz" else (open, "r")

    delim = get_file_delimiter(filepath)

    open_fn, mode = _get_open_method(filepath)
    with open_fn(filepath, mode) as f:
        file_content = f.readlines()
        first_line = file_content[0].strip().split(delim)

    return {
        "delim": delim,
        "open_fn": open_fn,
        "mode": mode,
        "n_rows": len(file_content),
        "n_columns": len(first_line),
        "has_header": "mz" in first_line,
    }


def get_ppm_range(
    lower_bound: np.ndarray, upper_bound: np.ndarray, ppm_error: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand an m/z range by a given ppm tolerance.

    Parameters
    ----------
    lower_bound : np.ndarray
        Lower m/z boundary values.
    upper_bound : np.ndarray
        Upper m/z boundary values.
    ppm_error : float
        Parts-per-million tolerance to apply.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated (lower_bound, upper_bound) after applying the ppm expansion.
    """
    lower_bound += -ppm_error / 1e6 * lower_bound
    upper_bound += ppm_error / 1e6 * upper_bound
    return lower_bound, upper_bound


def calculate_ppm_error(
    observed_mz: float | np.ndarray, theoretical_mz: float | np.ndarray
) -> float | np.ndarray:
    """
    Calculate the absolute ppm error between observed and theoretical m/z values.

    Parameters
    ----------
    observed_mz : float or np.ndarray
        Observed m/z value(s).
    theoretical_mz : float or np.ndarray
        Theoretical m/z value(s).

    Returns
    -------
    float or np.ndarray
        Absolute ppm error(s).
    """
    return np.abs((observed_mz - theoretical_mz) / theoretical_mz) * 1e6


def aggregate_dict_values(dict1: dict[str, int], dict2: dict[str, int]) -> dict[str, int]:
    """
    Merge two dictionaries by summing values for matching keys.

    Keys present in dict1 but not dict2 are added to dict2 with their value from dict1.

    Parameters
    ----------
    dict1 : dict
        Source dictionary whose values are added into dict2.
    dict2 : dict
        Target dictionary that is updated in place.

    Returns
    -------
    dict
        Updated dict2 containing merged values from both dictionaries.
    """
    for k, v in dict1.items():
        dict2[k] = dict2.get(k, 0) + v
    return dict2


def get_element_count(formula: str) -> dict[str, int]:
    """
    Parse a chemical formula string into a dictionary of element counts.

    Supports common chemical abbreviations (e.g. 'ACN', 'DMSO') and formulas with
    parenthetical groups (e.g. '(CH3)2SO').

    Parameters
    ----------
    formula : str
        Chemical formula or abbreviated compound name.

    Returns
    -------
    dict[str, int]
        Dictionary mapping element symbols to their counts.

    Examples
    --------
    >>> get_element_count('H2O')
    {'H': 2, 'O': 1}
    >>> get_element_count('C6H12O6')
    {'C': 6, 'H': 12, 'O': 6}
    >>> get_element_count('ACN')
    {'C': 2, 'H': 3, 'N': 1}
    """
    # Dictionary that maps common abbreviations of chemicals to their molecular formulas
    compound_abbreviations = {
        "ACN": "CH3CN",
        "DMSO": "(CH3)2SO",
        "FA": "CH2O2",
        "HAc": "CH3COOH",
        "TFA": "CF3CO2H",
        "IsoProp": "CH3CHOHCH3",
    }

    def _get_element_count(element: str, multiplier: int = 1) -> dict[str, int]:
        """
        Given a chemical element (e.g., "H2", "O", "C12")
        with an optional count and a multiplier, return
        a dictionary mapping the element to its final count.
        """
        characters = "".join(filter(str.isalpha, element))
        integer = "".join(filter(str.isdigit, element))

        return {characters: int(integer or 1) * multiplier}

    element_count: dict[str, int] = {}

    # Extract the leading number from the formula (if present)
    # and set it as the multiplier for the atom count, defaulting to 1 if no number is found.
    match = re.match(r"^(\d+)(.*)", formula)

    if match:
        atom_count_multiplier, formula = int(match.group(1)), match.group(2)
    else:
        atom_count_multiplier = 1

    # If the given compound is abbreviated, get its molecular formula
    if _formula := compound_abbreviations.get(formula):
        formula = _formula

    grouped_elements = re.findall(r"\(([^)]+)\)(\d+)", formula)
    standalone_elements = re.findall(r"\([^()]*\)|([A-Za-z][A-Za-z0-9]*)", formula)
    grouped_elements.extend(zip(standalone_elements, [1] * len(standalone_elements), strict=False))

    for string, digit in grouped_elements:
        elements = re.findall("[A-Z][^A-Z]*", string)
        for elem in elements:
            element_count = aggregate_dict_values(
                _get_element_count(elem, multiplier=int(digit) * atom_count_multiplier),
                element_count,
            )

    return element_count


def modify_formula_dict(formula_dict: dict[str, int], adduct: str) -> dict[str, int] | None:
    """
    Apply adduct additions and subtractions to an element count dictionary.

    Parameters
    ----------
    formula_dict : dict
        Dictionary mapping element symbols to their counts in the base formula.
    adduct : str
        Adduct string specifying modifications, e.g. '[M+Na]+' or '[M-H]-'.

    Returns
    -------
    dict or None
        Updated element count dictionary after applying the adduct, or None if
        any element count becomes negative (invalid formula).
    """
    updated_formula = formula_dict.copy()

    adduct_components = re.split(r"([+-])", adduct)
    atom_count_multiplier = re.findall(r"^\[(\d+)M", adduct_components[0])

    if len(atom_count_multiplier):
        updated_formula = {k: int(atom_count_multiplier[0]) * v for k, v in updated_formula.items()}

    for i in range(1, len(adduct_components), 2):
        element_count = get_element_count(adduct_components[i + 1])
        element_count = {k: int(f"{adduct_components[i]}{v}") for k, v in element_count.items()}
        updated_formula = aggregate_dict_values(updated_formula, element_count)

    for v in updated_formula.values():
        if v < 0:
            return None

    return updated_formula


def modify_charge(charge: int, adduct: str, adduct_db: pd.DataFrame) -> int:
    """
    Compute the net charge of a molecule after applying an adduct.

    Parameters
    ----------
    charge : int
        Initial charge of the molecule.
    adduct : str
        Adduct ion name to look up in the database.
    adduct_db : pd.DataFrame
        DataFrame containing adduct definitions with 'Ion name' and 'Charge' columns.

    Returns
    -------
    int
        Net charge after applying the adduct.
    """
    adduct_info = adduct_db[adduct_db["Ion name"] == adduct]
    return int(adduct_info["Charge"].values[0]) + charge


def get_decoy_info(decoy: str) -> tuple[str, int, int]:
    """
    Extract element, multiplier, and sign from a decoy string.

    Parameters
    ----------
    decoy : str
        Decoy string in the format '[+/-][n][Element]', e.g. '+2C' or '-H'.

    Returns
    -------
    tuple[str, int, int]
        Tuple of (element_symbol, multiplier, sign), where sign is +1 or -1.

    Raises
    ------
    ValueError
        If the decoy string does not follow '[+/-][n][Element]' format.
    """
    decoy_matches = re.match(r"([+-])(\d+)?(.*)", decoy)

    if decoy_matches:
        sign = int(f"{decoy_matches.group(1)}1")
        multiple = int(decoy_matches.group(2)) if decoy_matches.group(2) else 1
        return decoy_matches.group(3), multiple, sign

    raise ValueError(
        "Invalid decoy format. Please make sure the format follows '[+/-][d][Element]'"
    )


def get_adducts(header: Sequence[str]) -> list[str]:
    """
    Extract valid adduct strings from a list of column headers.

    Parameters
    ----------
    header : Sequence
        List of strings to filter, typically column names from a DataFrame.

    Returns
    -------
    list[str]
        List of strings that match the adduct pattern (e.g. '[M+H]+').
    """
    return [item for item in header if re.match("^[M[+-].*](\d+)?[+-]", item)]


def normalize_scores(dist: float, dist_range: list[float] | None) -> float:
    """
    Normalize a distance or score value to the [0, 1] range.

    Parameters
    ----------
    dist : float
        Raw distance or score value.
    dist_range : list[float] or None
        Expected range [min, max] of the score. Use [0, np.inf] for strictly non-negative
        distances, [-np.inf, 0] for non-positive, or None to default to [0, 1].

    Returns
    -------
    float
        Normalized score clipped to [0, 1].
    """
    if dist_range is None:
        dist_range = [0, 1]

    if dist_range == [0, np.inf]:
        result = dist / (1 + dist)
    elif dist_range == [-np.inf, 0]:
        result = 1 / (1 - dist)
    else:
        result = (dist - dist_range[0]) / (dist_range[1] - dist_range[0])

    if result < 0:
        result = 0
    elif result > 1:
        result = 1

    return result


def remove_noise(spectra: np.ndarray | list, noise: float | None) -> np.ndarray:
    """
    Zero out intensities below a relative noise threshold.

    Parameters
    ----------
    spectra : np.ndarray or list
        2D array of shape (n, 2) with columns [m/z, intensity].
    noise : float or None
        Fraction of the maximum intensity below which values are set to zero.

    Returns
    -------
    np.ndarray
        Spectrum array with sub-threshold intensities replaced by zero.
    """
    spectra = np.array(spectra)
    intensities = np.where(spectra[:, 1] >= np.max(spectra[:, 1]) * noise, spectra[:, 1], 0)
    return np.stack([spectra[:, 0], intensities], axis=1)


def normalize_intensity(spectrum: np.ndarray) -> np.ndarray:
    """
    Normalize spectrum intensities using total-sum normalization.

    Parameters
    ----------
    spectrum : np.ndarray
        2D array of shape (n, 2) with columns [m/z, intensity].

    Returns
    -------
    np.ndarray
        Spectrum with intensities rescaled so that they sum to 1.
    """

    if len(spectrum) > 0 and (_sum := np.sum(spectrum[:, 1])) > 0:
        spectrum[:, 1] = spectrum[:, 1] / _sum
    return spectrum


def str_to_dict(formula: str) -> dict[str, int]:
    """
    Parse a chemical formula string into a dictionary of element counts.

    Parameters
    ----------
    formula : str
        Chemical formula string, optionally including a charge suffix,
        e.g. 'C6H12O6' or 'H2O+'.

    Returns
    -------
    dict[str, int]
        Dictionary mapping element symbols (or isotope-prefixed symbols) to their counts.

    Raises
    ------
    ValueError
        If the formula contains invalid characters or no valid elements are found.
    """
    formula = re.sub(r"[+-](\d+)?$", "", formula)
    pattern = r"(\[\d+\])?([A-Z][a-z]?)(\d*)"

    result: dict[str, int] = {}
    position = 0
    for match in re.finditer(pattern, formula):
        # check for invalid characters
        if match.start() != position:
            raise ValueError(
                f"Invalid characters in formula: '{formula[position : match.start()]}' "
                f"at position {position}"
            )
        isotope, element, count = match.groups()

        if not element:
            continue

        key = f"{isotope.strip('[/]')}{element}" if isotope else element
        count = int(count) if count else 1

        result[key] = result.get(key, 0) + count
        position = match.end()

    # check for trailing invalid characters
    if position != len(formula):
        raise ValueError(f"Invalid trailing characters: '{formula[position:]}'")

    if not result:
        raise ValueError(f"No valid elements found in formula: '{formula}'")

    return result


def get_charge(formula: str) -> int:
    """
    Parse the ionic charge from a chemical formula string.

    Parameters
    ----------
    formula : str
        Chemical formula string with an optional trailing charge suffix,
        e.g. 'C6H12O6+2' or 'H2O-'.

    Returns
    -------
    int
        Ionic charge. Returns 0 if no charge suffix is present.
    """
    pattern = re.compile(r"([+-])(\d*)$")
    match = pattern.search(formula)

    if match is None:
        return 0

    sign, magnitude = match.groups()

    magnitude = int(magnitude) if magnitude else 1
    return magnitude if sign == "+" else -magnitude


def get_formula(element_count: dict[str, int], charge: int) -> str:
    """
    Construct a chemical formula string from element counts and ionic charge.

    Parameters
    ----------
    element_count : dict[str, int]
        Dictionary mapping element symbols to their counts.
    charge : int
        Ionic charge of the compound. Use 0 for neutral compounds.

    Returns
    -------
    str
        Chemical formula string, e.g. 'C6H12O6' or 'C2H4O2+2'.
    """
    formula = "".join(f"{k}{'' if v == 1 else v}" for k, v in element_count.items())
    if charge:
        sign = "+" if charge > 0 else "-"
        magnitude = "" if abs(charge) == 1 else str(abs(int(charge)))
        formula += sign + magnitude

    return formula

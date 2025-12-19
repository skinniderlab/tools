import gzip
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def get_file_delimiter(filepath: Path) -> str:
    """
    Identifies and returns the delimiter (comma, tab, or space) used to
    separate values in the specified file.
    """
    f = gzip.open(filepath, "rt") if filepath.suffix == ".gz" else Path.open(filepath)  # noqa: SIM115
    first_line = next(f).strip()
    f.close()

    for sep in (",", "\t", " "):
        header = first_line.split(sep)
        if len(header) > 1:
            break

    return sep


def get_file_info(filepath: Path) -> dict:
    """
    Loads the experiment file from the specified filepath, containing
    mass-to-charge ratios (m/z) and/or intensity data.
    """

    def _get_open_method(_filepath: Path):
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
    Returns the m/z range around a specified peak, accounting for the
    potential error due to the given ppm tolerance.
    """
    lower_bound += -ppm_error / 1e6 * lower_bound
    upper_bound += ppm_error / 1e6 * upper_bound
    return lower_bound, upper_bound


def calculate_ppm_error(observed_mz, theoretical_mz):
    """Calculate the ppm error between two m/z values."""
    return np.abs((observed_mz - theoretical_mz) / theoretical_mz) * 1e6


def aggregate_dict_values(dict1, dict2):
    """
    Merges two dictionaries by adding the values of matching keys.
    """
    for k, v in dict1.items():
        dict2[k] = dict2.get(k, 0) + v
    return dict2


def get_element_count(formula: str) -> dict:
    """
    Given a chemical formula in string format or an abbreviated form,
    returns adictionary mapping element to respective amount.

    Example:
        'H2O' -> {'H': 2, 'O': 1}
        'C6H12O6' -> {'C': 6, 'H': 12, 'O': 6}
        'ACN' -> {'C': 2, 'H':3, 'N':1}
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

    def _get_element_count(element, multiplier=1):
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


def modify_formula_dict(formula_dict: dict, adduct: str) -> dict | None:
    """
    Given a dictionary that maps elements to their respective counts in a
    chemical formula, and an adduct represented as a string, updates the
    element counts by applying the additions and/or subtractions indicated
    by the adduct and returns the resulting dictionary.
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


def modify_charge(charge, adduct, adduct_db):
    """
    Computes the net charge of a molecule based on its initial charge,
    a database of adducts, and a specified adduct.
    """
    adduct_info = adduct_db[adduct_db["Ion name"] == adduct]
    return adduct_info["Charge"].values[0] + charge


def get_decoy_info(decoy: str) -> tuple[str, int, int]:
    """
    Extracts and returns the element, multiple, and sign in the form of +1, -1
    (indicating addition or subtraction) from a decoy string.

    :raises ValueError: if decoy string doesn't follow '[+/-][d][Element]'.
    """
    decoy_matches = re.match(r"([+-])(\d+)?(.*)", decoy)

    if decoy_matches:
        sign = int(f"{decoy_matches.group(1)}1")
        multiple = int(decoy_matches.group(2)) if decoy_matches.group(2) else 1
        return decoy_matches.group(3), multiple, sign

    raise ValueError(
        "Invalid decoy format. Please make sure the format follows '[+/-][d][Element]'"
    )


def get_adducts(header: Sequence):
    """Extracts vaild adduct strings from a list."""
    return [item for item in header if re.match("^[M[+-].*](\d+)?[+-]", item)]


def normalize_scores(dist, dist_range):
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
    Given a spectrum and a noise threshold, sets all intensity values
    below (noise threshold x maximum intensity) to 0.
    """
    spectra = np.array(spectra)
    intensities = np.where(spectra[:, 1] >= np.max(spectra[:, 1]) * noise, spectra[:, 1], 0)
    return np.stack([spectra[:, 0], intensities], axis=1)


def normalize_intensity(spectrum: np.ndarray) -> np.ndarray:
    """
    Normalize the intensities in a 2D array of m/z and intensity pairs using
    total sum normalization, such that the sum of all intensity values equals 1.
    """

    if len(spectrum) > 0 and (_sum := np.sum(spectrum[:, 1])) > 0:
        spectrum[:, 1] = spectrum[:, 1] / _sum
    return spectrum


def str_to_dict(formula: str) -> dict:
    """
    Given a chemical formula in string format, this methodr returns
    a dictionary mapping elements to respective amounts.
    """
    stripped = re.sub(r"[+-](\d+)?$", "", formula)
    split = re.split(r"([A-Z][a-z]?)", stripped)

    return {
        split[i]: int(amt if (amt := split[i + 1]) != "" else 1)
        for i in range(1, len(split) - 1, 2)
    }

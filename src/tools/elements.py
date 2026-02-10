from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from tools.utils import get_decoy_info, modify_formula_dict, str_to_dict


class IsotopeDB:
    """A list of elements from the isotopes database."""

    FILEPATH = Path(__file__).parent.parent.parent / "data/iso_list.csv"

    def __init__(self, filepath: Path = None):
        """
        Initialize a list of elements.

        Parameters.
        -----------
        filepath: Path
            Path to the isotope file.

        """
        filepath = self.FILEPATH if filepath is None else filepath
        self.filename = filepath.stem
        self.elements: list[Element] = []
        self._parse_file(filepath)

    def _parse_file(self, filepath: Path) -> None:
        """
        Parses isotopes from the specified file path, creates individual element objects,
        and stores them in the elements list.

        Parameters
        ----------
        filepath: Path
            Path to the isotope file.

        Returns
        -------
        None
        """
        isotope_df = pd.read_csv(filepath)

        for elem, df in isotope_df.groupby("element"):
            isotopes = {Isotope(row.isotope, row.mass, row.abundance) for _, row in df.iterrows()}
            self.elements.append(Element(symbol=str(elem), isotopes=isotopes))

    def get_mass_update(self, element):
        """
        Returns the mass adjustment required for a given decoy string.

        :raises ValueError: if decoy string doesn't follow '[+/-][d][Element]'.
        """

        element, multiple, sign = get_decoy_info(element)
        return self[element].monoisotope.mass * multiple * sign

    def __getitem__(self, item):
        """
        Retrieve an element from the list with either its string representation
        or an isotope object.

        :raises ValueError: If the specified element is not present in the provided isotope file.
        """
        for element in self.elements:
            if element == item or item in element:
                return element

        raise KeyError(f"Element {item} not present in the provided isotope file.")

    def __contains__(self, item):
        return item in self.elements


class Compound:
    """A chemical compound composed of multiple elements."""

    VALID_ELEMENTS = ("C", "H", "O", "N", "P", "S", "F", "Cl", "Br", "I")

    def __init__(self, formula: dict, isotope_db: IsotopeDB):
        """
        Initializes a compound.

        Parameters
        ----------
        formula: dict
            A dictionary mapping each element to its corresponding atom count.
        isotope_db: Elements
            Elements object consting a list of elements from the isotopes database.
        """
        self.isotope_db = isotope_db
        self.element_count: dict[Element, int] = {}
        self.formula: str = ""
        self.monomass = 0
        self.monoabund = 1
        self.monoisos: list[Isotope] = []
        self.nonmonoisos: list[Isotope] = []

        self.order_elements(formula)
        self._compute_mass_and_abundance()
        self._parse_isotopes()

    @classmethod
    def from_str(cls, formula: str, isotope_db: IsotopeDB):
        return cls(str_to_dict(formula), isotope_db)

    def __hash__(self):
        return hash((self.formula, self.monomass, tuple(self.monoisos)))

    def __repr__(self):
        return self.formula

    def __contains__(self, item):
        return item in self.element_count

    def __iter__(self):
        return iter(self.element_count)

    def __eq__(self, other):
        if isinstance(other, Compound):
            return self.formula == other.formula
        if isinstance(other, str):
            return self.formula == other
        return False

    def __getitem__(self, item):
        return self.element_count[item]

    def order_elements(self, formula: dict) -> None:
        """
        Parses the dictionary mapping elements (in string representation)
        to its atom counts (including 0) in a compound, and sorts elements according to
        the Hill System. The resulting dictionary maps Element objects to atom counts
        containing only those elements with a non-zero count (i.e.,
        those actually present in the compound).

        Parameters
        ----------
        formula: dict
            Dictionry mapping elements (in string representation) to the number
            of times it appears in the compound.

        Returns
        -------
        None
        """

        order = dict.fromkeys(self.VALID_ELEMENTS, 0)

        # Keys in self.dict are sorted in the order described by VALID_ELEMENTS tuple
        self.element_count = {self.isotope_db[k]: v for k, v in (order | formula).items() if v != 0}
        self.formula = "".join(f"{k}{'' if v == 1 else v}" for k, v in self.element_count.items())

    def _parse_isotopes(self):
        """
        Stores both the most abundant and non-abundant isotopes for every element in a compound.
        """
        for elem in self.element_count:
            self.monoisos.append(elem.monoisotope)
            self.nonmonoisos.extend(elem.other_isotopes)

    def _compute_mass_and_abundance(self):
        """
        Stores monoisotopic mass and abundance of the compound.
        """
        for elem in self:
            self.monomass += elem.monoisotope.mass * self[elem]
            self.monoabund *= elem.monoisotope.abundance ** self[elem]

        if self.monomass == 0:
            raise Exception(f"Monoisotopic mass could not be calculated for {self}!")

    def get_updated_compound(self, adduct):
        """
        Given an adduct repressented as a string, returns a new `Compound` object
        after updating the element counts of the original compound by applying
        the additions and/or subtractions indicated by the adduct.
        """
        return Compound(modify_formula_dict(self.element_count, adduct), self.isotope_db)

    def isopattern(
        self,
        charge: int,
        abundance_limit: float,
        max_iter: int,
        get_details: bool = False,
        scale: str = "abs",
    ) -> np.ndarray:
        """
        Calculate the theoretical isotopic distribution for a given chemical compound from
        its molecular formula, returning the expected masses and relative abundances.

        Parameters
        ----------
        charge: int
            Net charge of the specified compound.
        abundance_limit: float
            Minimum relative abundance threshold for including isotopic peaks in the pattern.
        max_iter: int
            Maximum number of iteration allowed for generating the isotopic pattern.

        Returns
        -------
        peaks: np.ndarray
            Theoretical isotopic distribution of the compound, including the expected masses
            and relative abundances.
        """

        def isotope_permutation(row: pd.Series, limit: float) -> pd.Series:
            """
            Modifies an isotopic pattern by transferring mass and abundance from the
            monoisotopic peak to a specified non-monoisotopic peak.
            """

            if row.stop == 1:
                return row

            if not pd.isna(row.noniso):
                nonmono = row.noniso
                monoiso = self.isotope_db[nonmono].monoisotope

                if row[monoiso] == 0:
                    return pd.Series(None, index=row.index)

                row.mass = row.mass - monoiso.mass + nonmono.mass

                monoiso_abund_proportion = row[monoiso] / monoiso.abundance
                row.abundance = row.abundance * (
                    nonmono.abundance / (row[nonmono] + 1) * monoiso_abund_proportion
                )

                row[monoiso] -= 1
                row[nonmono] += 1
                row.generation += 1

                if row.abundance < limit:
                    row.stop = 1

            return row

        peaks = pd.DataFrame(
            np.zeros((1 + len(self.nonmonoisos), len(self.nonmonoisos))),
            columns=self.nonmonoisos,
        )

        peaks["mass"] = self.monomass
        peaks["abundance"] = self.monoabund
        peaks["generation"] = 0
        peaks["stop"] = 0

        peaks.loc[:, self.monoisos] = list(self.element_count.values())
        peaks.loc[1 : len(self.nonmonoisos), "noniso"] = self.nonmonoisos

        # If this compound is made up of single monoisotopic element
        if len(self.monoisos) == 1 and self.monoisos[0].abundance == 1:
            peaks.loc[0, "stop"] = 1

        iteration, n_tries = 0, 0
        while any(peaks[peaks.generation == iteration].stop == 0) and n_tries < max_iter:
            peaks = peaks.apply(isotope_permutation, limit=abundance_limit, axis=1)
            peaks = peaks.dropna(how="all", ignore_index=True)

            # Remove duplicated isotope combination
            if iteration:
                _peaks = peaks[peaks.generation == iteration + 1].round({"abundance": 3, "mass": 9})
                _peaks = _peaks.loc[:, ~_peaks.columns.isin(["generation", "stop", "noniso"])]
                peaks = peaks.drop(_peaks[_peaks.duplicated()].index)

            iteration += 1
            n_tries = len(peaks)

            # Isotope permutations from current iteration where abundance is still above limit
            over_limit = peaks[(peaks.generation == iteration) & (peaks.stop == 0)]

            # Combine other permutations of non abundant isotopes with the filtered one
            if not over_limit.empty:
                for row in over_limit.itertuples():
                    over_limit.at[row.Index, "noniso"] = self.nonmonoisos

                over_limit = over_limit.explode(["noniso"])
                peaks = pd.concat([peaks.assign(stop=1), over_limit], ignore_index=True)

        peaks = peaks[peaks.abundance != 0]
        peaks["charge"] = charge
        if charge != 0:
            peaks["mass"] = (peaks["mass"] - (5.486 * (10 ** (-4)) * charge)) / abs(charge)

        if scale == "rel":
            peaks["abundance"] = (
                100
                * (peaks.abundance - peaks.abundance.min())
                / (peaks.abundance.max() - peaks.abundance.min())
            )
            peaks = peaks.sort_values("abundance", ascending=False).reset_index(drop=True)
        else:
            peaks = peaks.sort_values("mass").reset_index(drop=True)

        if get_details:
            return peaks.loc[:, ~peaks.columns.isin(["generation", "stop", "noniso"])]

        return peaks[["mass", "abundance"]].values


@dataclass(frozen=True)
class Isotope:
    """An individual isotope of an element."""

    symbol: str
    mass: float
    abundance: float

    def __repr__(self):
        return self.symbol

    def __hash__(self):
        return hash((self.symbol, self.mass, self.abundance))

    def __eq__(self, other):
        if isinstance(other, str):
            return self.symbol == other
        return (
            self.abundance == other.abundance
            and self.symbol == other.symbol
            and self.mass == other.mass
        )

    def __lt__(self, other):
        return self.abundance < other.abundance


@dataclass(frozen=True)
class Element:
    """A chemical element."""

    symbol: str
    isotopes: set[Isotope]

    def __repr__(self):
        return self.symbol

    def __hash__(self):
        return hash(self.symbol)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.symbol == other
        return self.symbol == other.symbol and self.isotopes == other.isotopes

    def __contains__(self, item):
        return any(item == isotope for isotope in self.isotopes)

    def __getitem__(self, item):
        if item == self.symbol:
            return self.monoisotope
        if isotope := self._isotope_lookup[item]:
            return isotope
        raise KeyError(f"Isotope {item} not present in the provided isotope file.")

    @property
    def n_isotopes(self):
        """Returns the number of isotopes of an element."""
        return len(self.isotopes)

    @cached_property
    def monoisotope(self):
        """Returns the most abundant isotope of an element."""
        return sorted(self.isotopes)[-1]

    @cached_property
    def other_isotopes(self):
        """Returns all but the most abundant isotope of an element."""
        return sorted(self.isotopes)[:-1]

    @cached_property
    def _isotope_lookup(self):
        # O(1) lookup by symbol
        return {iso.symbol: iso for iso in self.isotopes}

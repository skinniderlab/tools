from dataclasses import dataclass
from functools import cached_property
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from tools.utils import get_decoy_info, modify_formula_dict, str_to_dict, get_formula, get_charge

ELECTRON_MASS = 5.486e-4

class IsotopeDB:
    """A list of elements from the isotopes database."""

    DATA_FILE = files("tools.data").joinpath("iso_list.csv")

    def __init__(self, filepath: Path = None):
        """
        Initialize a list of elements.

        Parameters.
        -----------
        filepath: Path
            Path to the isotope file.

        """
        filepath = self.DATA_FILE if filepath is None else filepath
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
        Return the mass adjustment for a given decoy string.

        Parameters
        ----------
        element : str
            Decoy string in the format '[+/-][n][Element]', e.g. '+2C' or '-H'.

        Returns
        -------
        float
            Mass adjustment value for the specified element and multiplier.

        Raises
        ------
        ValueError
            If the decoy string does not follow '[+/-][n][Element]' format.
        """

        element, multiple, sign = get_decoy_info(element)
        return self[element].monoisotope.mass * multiple * sign

    def __getitem__(self, item):
        """
        Retrieve an element by its string symbol or Isotope object.

        Parameters
        ----------
        item : str or Isotope
            Element symbol or Isotope to look up.

        Returns
        -------
        Element
            The matching Element from the isotope database.

        Raises
        ------
        KeyError
            If the specified element is not present in the provided isotope file.
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

    def __init__(self, formula: dict, isotope_db: IsotopeDB, charge: int = 0):
        """
        Initializes a compound.

        Parameters
        ----------
        formula: dict
            A dictionary mapping each element to its corresponding atom count.

        isotope_db: Elements
            Elements object consting a list of elements from the isotopes database.

        charge: int or None, optional
            The net charge of the compound. Use 0 if the compound is neutral or if
            no charge information is available.
        """
        self.isotope_db = isotope_db
        self.charge = charge
        self.element_count = self._order_elements(formula)
        self.formula: str = get_formula(self.element_count, self.charge)
        self.monomass = 0
        self.monoabund = 1
        self.monoisos: list[Isotope] = []
        self.nonmonoisos: list[Isotope] = []

        self._compute_mass_and_abundance()
        self._parse_isotopes()

    @classmethod
    def from_str(cls, formula: str, isotope_db: IsotopeDB):
        """
        Construct a Compound from a formula string.

        Parameters
        ----------
        formula : str
            Chemical formula string, optionally including a charge suffix,
            e.g. 'C6H12O6' or 'C2H4O2+'.
        isotope_db : IsotopeDB
            Isotope database used to look up element properties.

        Returns
        -------
        Compound
            A new Compound instance corresponding to the given formula.
        """
        return cls(str_to_dict(formula), isotope_db, get_charge(formula))

    def __lt__(self, other):
        return self.formula < other.formula

    def __hash__(self):
        return hash(self.formula)

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

    def _order_elements(self, formula: dict) -> dict["Element", int]:
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
        dict[Element, int]
            Dictionary mapping elements (in string representation) to atom counts
            ordered according to the Hill System.
        """

        order = dict.fromkeys(self.VALID_ELEMENTS, 0)

        # Keys in self.dict are sorted in the order described by VALID_ELEMENTS tuple
        return {self.isotope_db[k]: v for k, v in (order | formula).items() if v != 0}

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
        abundance_limit: float,
        max_iter: int,
        apply_charges: bool = True,
        get_details: bool = False,
        scale: str = "abs",
    ) -> np.ndarray:
        """
        Calculate the theoretical isotopic distribution for a given chemical compound from
        its molecular formula, returning the expected masses and relative abundances.

        Parameters
        ----------
        abundance_limit : float
            Minimum relative abundance threshold for including isotopic peaks in the pattern.
        max_iter : int
            Maximum number of iterations allowed for generating the isotopic pattern.
        apply_charges : bool, optional
            Whether to adjust m/z values by the compound charge. Default is True.
        get_details : bool, optional
            If True, return a DataFrame with full isotope composition details instead of
            just mass and abundance columns. Default is False.
        scale : str, optional
            Abundance scaling mode. Use 'rel' for relative scaling (0-100) sorted by
            descending abundance, or 'abs' for absolute values sorted by mass.
            Default is 'abs'.

        Returns
        -------
        np.ndarray
            Theoretical isotopic distribution as an array of shape (n, 2) with columns
            [mass, abundance]. If get_details is True, returns a pd.DataFrame instead
            with full isotope composition columns.
        """

        def isotope_permutation(row: pd.Series, limit: float) -> pd.Series:
            """
            Modifies an isotopic pattern by transferring mass and abundance from the
            monoisotopic peak to a specified non-monoisotopic peak.
            """

            if row.stop or pd.isna(row.noniso):
                return row

            row = row.copy()
            nonmono = row.noniso
            monoiso = self.isotope_db[nonmono].monoisotope

            if row[monoiso] == 0:
                return pd.Series(None, index=row.index)

            row.mass = row.mass - monoiso.mass + nonmono.mass

            mono_proportion = row[monoiso] / monoiso.abundance
            row.abundance *= (nonmono.abundance / (row[nonmono] + 1) * mono_proportion)

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
                over_limit = over_limit.assign(noniso=[self.nonmonoisos] * len(over_limit))
                over_limit = over_limit.explode(["noniso"])
                peaks = pd.concat([peaks.assign(stop=1), over_limit], ignore_index=True)

        peaks["charge"] = self.charge
        if self.charge != 0 and apply_charges:
            peaks["mass"] = (peaks["mass"] - ELECTRON_MASS * self.charge) / abs(self.charge)


        if scale == "rel":
            peaks["abundance"] = peaks.abundance.pipe(
                lambda x: 100 * (x - x.min()) / (x.max() - x.min())
            )
            peaks = peaks.sort_values("abundance", ascending=False).reset_index(drop=True)
        else:
            peaks = peaks.sort_values("mass").reset_index(drop=True)


        peaks.columns = peaks.columns.astype(str)
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
        """
        Retrieve an isotope of this element by its symbol.

        Parameters
        ----------
        item : str
            Symbol of the isotope or element to look up.

        Returns
        -------
        Isotope
            The matching Isotope object.

        Raises
        ------
        KeyError
            If the isotope symbol is not present in this element.
        """
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

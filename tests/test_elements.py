from tools import Compound, Element, Isotope


def test_elements_object(isotope_db):
    hydrogen = Element(
        symbol="H",
        isotopes={Isotope("1H", 1.007825032, 0.999855), Isotope("2H", 2.014101778, 0.000145)},
    )

    assert hydrogen in isotope_db, "Elements object returned inaccurate membership result."
    assert "Cl" in isotope_db, "Elements object returned inaccurate membership result."
    assert "S" in isotope_db, "Elements object returned inaccurate membership result."
    assert "not an element" not in isotope_db, (
        "Elements object returned inaccurate membership result."
    )


def test_element_carbon(isotope_db):
    element = isotope_db["C"]

    assert element == "C", "Elements object failed to retrieve the correct element."
    assert element.n_isotopes == 2, "Element object's isotope count is inaccurate."
    assert element.monoisotope == "12C", "Element object's most abundant isotope is inaccurate."
    assert element.other_isotopes == ["13C"], (
        "Element object's list of none abundant isotopes is inaccurate."
    )


def test_element_iron(isotope_db):
    element = isotope_db["Fe"]

    assert element == "Fe", "Element object's string representation is inaccurate."
    assert element.n_isotopes == 4, "Element object's isotope count is inaccurate."
    assert element.monoisotope == "56Fe", "Element object's most abundant isotope is inaccurate."
    assert element.other_isotopes == [
        "58Fe",
        "57Fe",
        "54Fe",
    ], "Element object's list of none abundant isotopes is inaccurate."


def test_element_from_isotope(isotope_db):
    calcium_isotope = Isotope("42Ca", 41.95861778, 0.00647)
    element = isotope_db[calcium_isotope]

    assert element == "Ca", "Element object's string representation is inaccurate."
    assert element.n_isotopes == 6, "Element object's isotope count is inaccurate."
    assert element.monoisotope == "40Ca", "Element object's most abundant isotope is inaccurate."
    assert element.other_isotopes == [
        "46Ca",
        "43Ca",
        "48Ca",
        "42Ca",
        "44Ca",
    ], "Element object's list of none abundant isotopes is inaccurate."


def test_element_instantiation():
    hydrogen_isotope_1 = Isotope("1H", 1.007825, 0.999885)
    hydrogen_isotope_2 = Isotope("2H", 2.014102, 0.000115)

    element = Element(symbol="H", isotopes={hydrogen_isotope_1, hydrogen_isotope_2})

    assert element == "H", "Element object's string representation is inaccurate."
    assert element.n_isotopes == 2, "Element object's isotope count is inaccurate."
    assert element.monoisotope == "1H", "Element object's most abundant isotope is inaccurate."
    assert element.other_isotopes == ["2H"], (
        "Element object's list of none abundant isotopes is inaccurate."
    )
    # Check if isotope object is hashable
    assert {element: 5}


def test_isotope_instantiation():
    potassium_isotope = Isotope("40K", 39.963999, 0.000117)
    same_isotope = Isotope("40K", 39.963999, 0.000117)
    diff = Isotope("41K", 40.961826, 0.067302)

    assert potassium_isotope == "40K", "Isotope object's string representation is inaccurate."
    assert potassium_isotope == same_isotope, "Isotope object's equality check is inaccurate."
    assert potassium_isotope != diff, "Isotope object's equality check is inaccurate."

    # Check if isotope object is hashable
    assert {potassium_isotope: 5}


def test_compound_instantiation(isotope_db):
    water = Compound({"H": 2, "O": 1}, isotope_db)
    water_2 = Compound({"H": 2, "O": 1}, isotope_db)
    carbon_dioxide = Compound({"C": 1, "O": 2}, isotope_db)
    hydrochloric_acid = Compound({"H": 1, "Cl": 1}, isotope_db)

    assert water == "H2O", "Compound object's string representation is inaccurate."
    assert carbon_dioxide == "CO2", "Compound object's string representation is inaccurate."
    assert hydrochloric_acid == "HCl", "Compound object's string representation is inaccurate."

    assert water.monomass == 18.010564684, "Compound object's monomass is inaccurate."
    assert water.monoabund == 0.9972807256739092, "Compound object's mono abundance is inaccurate."
    assert carbon_dioxide.monomass == 43.98982924, "Compound object's monomass is inaccurate."
    assert carbon_dioxide.monoabund == 0.9845973583080598, "Compound object's mono  is inaccur"

    assert "H" in water, "Compound object returned inaccurate membership result."
    assert "O" in water, "Compound object returned inaccurate membership result."

    assert water == water_2, "Compound object's equality check is inaccurate."
    assert water != carbon_dioxide, "Compound object's equality check is inaccurate."

    assert list(water) == ["H", "O"], "Compound iteration returned inaccurate elements."
    assert list(carbon_dioxide) == ["C", "O"], "Compound iteration returned inaccurate elements."
    assert list(hydrochloric_acid) == ["H", "Cl"], (
        "Compound iteration returned inaccurate elements."
    )

    assert water["H"] == 2, "Compound object retrieved an incorrect atom count."
    assert water["O"] == 1, "Compound object retrieved an incorrect atom count."
    assert carbon_dioxide["C"] == 1, "Compound object retrieved an incorrect atom count."
    assert carbon_dioxide["O"] == 2, "Compound object retrieved an incorrect atom count."
    assert hydrochloric_acid["H"] == 1, "Compound object retrieved an incorrect atom count."
    assert hydrochloric_acid["Cl"] == 1, "Compound object retrieved an incorrect atom count."

    assert water.monoisos == [
        "1H",
        "16O",
    ], "Compound object's list of abundant isotope is inaccurate."
    assert water.nonmonoisos == [
        "2H",
        "17O",
        "18O",
    ], "Compound object's list of non-abundant isotope is inaccurate."
    assert carbon_dioxide.monoisos == [
        "12C",
        "16O",
    ], "Compound object's list of abundant isotope is inaccurate."
    assert carbon_dioxide.nonmonoisos == [
        "13C",
        "17O",
        "18O",
    ], "Compound object's list of non-abundant isotope is inaccurate."


def test_compound_modification(isotope_db):
    fragment_ion_1 = Compound({"C": 1, "O": 1, "H": 1}, isotope_db)
    fragment_ion_2 = Compound({"N": 1, "O": 1}, isotope_db)
    fragment_ion_3 = Compound({"C": 1, "H": 3}, isotope_db)

    adduct_ion_1 = "[M+H]+"
    adduct_ion_2 = "[M+Na]+"
    adduct_ion_3 = "[M-H]-"

    updated_compound_1 = fragment_ion_1.get_updated_compound(adduct_ion_1)
    updated_compound_2 = fragment_ion_2.get_updated_compound(adduct_ion_2)
    updated_compound_3 = fragment_ion_3.get_updated_compound(adduct_ion_3)

    assert updated_compound_1 == "CH2O", (
        "Updated compound object's string representation is inaccurate."
    )
    assert updated_compound_1.element_count == {"C": 1, "O": 1, "H": 2}, (
        "Update compound object's element count is inaccurate."
    )
    assert updated_compound_1.monomass == 30.010564684, (
        "Updated compound object's monomass is inaccurate."
    )
    assert updated_compound_1.monoisos == ["12C", "1H", "16O"], (
        "Updated compound object's list of abundant isotope is inaccurate."
    )
    assert updated_compound_1.nonmonoisos == ["13C", "2H", "17O", "18O"], (
        "Updated compound object's list of non-abundant isotope is inaccurate."
    )
    assert updated_compound_1.monoabund == 0.9867095499817659, (
        "Updated compound object's mono abundance is inaccurate."
    )
    assert list(updated_compound_1) == ["C", "H", "O"], (
        "Updated compound iteration returned inaccurate elements."
    )
    assert updated_compound_1["C"] == 1, (
        "Updated compound object retrieved an incorrect atom count."
    )
    assert updated_compound_1["O"] == 1, (
        "Updated compound object retrieved an incorrect atom count."
    )
    assert updated_compound_1["H"] == 2, (
        "Updated compound object retrieved an incorrect atom count."
    )

    assert updated_compound_2 == "ONNa", (
        "Updated compound object's string representation is inaccurate."
    )
    assert updated_compound_2.element_count == {"O": 1, "N": 1, "Na": 1}, (
        "Update compound object's element count is inaccurate."
    )
    assert updated_compound_2.monomass == 52.987757900000005, (
        "Updated compound object's monomass is inaccurate."
    )
    assert updated_compound_2.monoisos == ["16O", "14N", "23Na"], (
        "Updated compound object's list of abundant isotope is inaccurate."
    )
    assert updated_compound_2.nonmonoisos == ["17O", "18O", "15N"], (
        "Updated compound object's list of non-abundant isotope is inaccurate."
    )
    assert updated_compound_2.monoabund == 0.9937842218499999, (
        "Updated compound object's mono abundance is inaccurate."
    )
    assert list(updated_compound_2) == ["O", "N", "Na"], (
        "Updated compound iteration returned inaccurate elements."
    )
    assert updated_compound_2["O"] == 1, (
        "Updated compound object retrieved an incorrect atom count."
    )
    assert updated_compound_2["N"] == 1, (
        "Updated compound object retrieved an incorrect atom count."
    )
    assert updated_compound_2["Na"] == 1, (
        "Updated compound object retrieved an incorrect atom count."
    )

    assert updated_compound_3 == "CH2", (
        "Updated compound object's string representation is inaccurate."
    )
    assert updated_compound_3.element_count == {"C": 1, "H": 2}, (
        "Update compound object's element count is inaccurate."
    )
    assert updated_compound_3.monomass == 14.015650063999999, (
        "Updated compound object's monomass is inaccurate."
    )
    assert updated_compound_3.monoisos == ["12C", "1H"], (
        "Updated compound object's list of abundant isotope is inaccurate."
    )
    assert updated_compound_3.nonmonoisos == ["13C", "2H"], (
        "Updated compound object's list of non-abundant isotope is inaccurate."
    )
    assert updated_compound_3.monoabund == 0.989113094802135, (
        "Updated compound object's mono abundance is inaccurate."
    )
    assert list(updated_compound_3) == ["C", "H"], (
        "Updated compound iteration returned inaccurate elements."
    )
    assert updated_compound_3["C"] == 1, (
        "Updated compound object retrieved an incorrect atom count."
    )
    assert updated_compound_3["H"] == 2, (
        "Updated compound object retrieved an incorrect atom count."
    )


def test_compound_from_str(isotope_db):
    formulas_to_convert = {
        "C4H9NO2+": "C4H9O2N",
        "C11H19NO9": "C11H19O9N",
        "C22H36O2": "C22H36O2",
        "C20H38O2": "C20H38O2",
        "C22H32O2": "C22H32O2",
        "C15H33N5O3": "C15H33O3N5",
        "C22H34O2N5": "C22H34O2N5",
        "C20H34O2": "C20H34O2",
        "C13H20N4O8": "C13H20O8N4",
    }

    for formula, expected in formulas_to_convert.items():
        result = Compound.from_str(formula, isotope_db)
        assert result == expected, f"Expected {expected}, got {result}."

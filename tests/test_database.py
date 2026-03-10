import re

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tools import Compound
from tools.database import Database
from tools.utils import get_element_count


def test_load_database(data_dir, database_obj, isotope_db):
    """
    Checks whether the database object correctly loads and
    parses the specified file.

    Note: The loading method is called in the constructor,
    so testing the conftest fixture is sufficient.
    """
    df = (
        pd.concat([i for i in database_obj.df.values()])
        .sort_values(by="mass")
        .reset_index(drop=True)
    )
    expected_db = pd.read_pickle(data_dir / "database_load_result.pkl")
    expected_db = expected_db.sort_values(by="mass").reset_index(drop=True)
    expected_db["compound"] = expected_db.apply(
        lambda x: Compound(get_element_count(x.compound.formula), isotope_db, x.charge), axis=1
    )
    df.columns = df.columns.astype("object")
    assert_frame_equal(df, expected_db, check_dtype=False)


def test_load_database_error(data_dir, isotope_db):
    """
    Checks whether value error is raised if the formula database
    has elements other than ("C", "H", "O", "N", "P", "S", "F", "Cl", "Br", "I").

    Note: The loading method is called in the constructor,
    just initializing the class is sufficient.
    """
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Elements in the formula database must not be anything other than: ('C', 'H', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I')"
        ),
    ):
        Database(
            db_filepath=data_dir / "database_load_error.csv",
            adducts_filepath=data_dir / "ESI_MS_adducts_2020.csv",
            isotope_db=isotope_db,
        )


def test_get_adducts(data_dir, database_obj, isotope_db):
    """Checks whether the `get_adducts` function correctly computes and populates m/z values."""

    df_adducts = database_obj.get_adducts(
        adducts_to_consider=[
            "[M+H]+",
            "[M+Na]+",
            "[M+K]+",
            "[M-H]-",
            "[M+Cl]-",
            "[M]+",
            "[M+Na-H]+",
            "[M+K-H]+",
        ],
    )

    df_adducts = df_adducts.sort_values(by=["mass"]).reset_index(drop=True)
    # The resulting dataframe is too large so we're only testing 100 randomly selected columns
    result = df_adducts.sample(n=100, random_state=0, ignore_index=True)
    expected = pd.read_pickle(data_dir / "database_get_adducts_result.pkl")

    # Just updating column names of expected dataframe to make sure the values are matching
    expected.columns = result.columns.astype(str)
    expected["compound"] = expected.apply(
        lambda x: Compound(get_element_count(x.compound.formula), isotope_db, x.charge), axis=1
    )

    assert_frame_equal(result, expected, check_dtype=False)


def test_make_decoy_formula_single_decoy(data_dir, database_obj):
    """
    Checks whether the `make_decoy_formula` function accurately modifies m/z values
    and chemical formulas based on a given decoy mode.
    """
    input_matches = pd.read_pickle(data_dir / "experiment_make_decoy_formula_input.pkl")

    result = database_obj.get_decoy_formulas(input_matches, modes=["+H"])
    expected_result = pd.read_pickle(data_dir / "experiment_make_decoy_formula_result.pkl")

    assert_frame_equal(result, expected_result)


def test_make_decoy_formula_multiple_decoys(data_dir, database_obj):
    """
    Checks whether the `make_decoy_formula` function accurately modifies m/z values
    and chemical formulas based on a given decoy mode.
    """

    input_matches = pd.read_pickle(data_dir / "experiment_make_decoy_formula_input.pkl")

    result = database_obj.get_decoy_formulas(input_matches, modes=["+H", "+3Cl"])
    expected_result = pd.read_pickle(
        data_dir / "experiment_make_decoy_formula_result_multiple_decoys.pkl"
    )

    assert_frame_equal(result, expected_result)


def test_get_mass_update(isotope_db):
    """
    Checks whether the `get_decoy_mass` function correctly returns the
    mass adjustment required to account for a specific decoy formation.
    """
    decoy_to_mass_change = {
        "+H": 1.007825032,
        "+2H": 2.015650064,
        "-H": -1.007825032,
        "+Fe": 55.93493554,
        "+5Fe": 279.67467769999996,
        "-2Fe": -111.86987108,
    }

    for decoy, mass_change in decoy_to_mass_change.items():
        assert mass_change == isotope_db.get_mass_update(decoy)

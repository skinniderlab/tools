from pathlib import Path

import numpy as np
import pytest

from tools.database import Database
from tools.elements import IsotopeDB

np.set_printoptions(legacy="1.25")


@pytest.fixture(scope="session")
def data_dir():
    """Filepath to the test data directory used for unit tests."""
    return Path(__file__).parent.parent / "tests/data"


@pytest.fixture
def isotope_db(data_dir):
    """IsotopeDB instance initialized with test data for validating class methods."""
    return IsotopeDB(filepath=data_dir / "iso_list.csv")


@pytest.fixture
def database_obj(data_dir, isotope_db):
    """Database instance initialized with test data for validating class methods."""
    return Database(
        db_filepath=data_dir / "formula-database-truncated.tsv.gz",
        adducts_filepath=data_dir / "ESI_MS_adducts_2020.csv",
        isotope_db=isotope_db,
    )


@pytest.fixture(autouse=True)
def seed():
    """
    Set the random seed to 0 for all tests.
    This prevents individual tests that use set_seed from affecting each other.
    """
    np.random.seed(0)

import gzip

import numpy as np

from tools.utils import (
    SortedValueIndex,
    get_adducts,
    get_charge,
    get_decoy_info,
    get_element_count,
    get_file_delimiter,
    get_file_info,
    get_formula,
    modify_charge,
    modify_formula_dict,
    remove_noise,
)


def test_get_file_delimiter(data_dir):
    """
    Checks whether the `get_file_delimiter` function returns accurate delimiter
    for a given file.
    """

    filepath_to_delimiter = {
        "iso_list.csv": ",",
        "formula-database-truncated.tsv.gz": "\t",
    }

    for filepath, expected_delimiter in filepath_to_delimiter.items():
        delimiter = get_file_delimiter(data_dir / filepath)
        assert delimiter == expected_delimiter


def test_get_file_info(data_dir, tmp_path):
    """
    Checks whether `get_file_info` returns accurate metadata for plain and
    gzip-compressed delimited files, including delimiter, open function, mode,
    row count, column count, and whether 'mz' appears as a column header.
    """
    result = get_file_info(data_dir / "iso_list.csv")
    assert result["delim"] == ","
    assert result["open_fn"] is open
    assert result["mode"] == "r"
    assert result["n_rows"] == 296
    assert result["n_columns"] == 4
    assert result["has_header"] is False

    result = get_file_info(data_dir / "formula-database-truncated.tsv.gz")
    assert result["delim"] == "\t"
    assert result["open_fn"] is gzip.open
    assert result["mode"] == "rt"
    assert result["n_rows"] == 2000
    assert result["n_columns"] == 15
    assert result["has_header"] is False

    mz_csv = tmp_path / "mz_data.csv"
    mz_csv.write_text("mz,intensity\n100.0,0.5\n200.0,1.0\n")
    result = get_file_info(mz_csv)
    assert result["delim"] == ","
    assert result["open_fn"] is open
    assert result["mode"] == "r"
    assert result["n_rows"] == 3
    assert result["n_columns"] == 2
    assert result["has_header"] is True


def test_modify_formula_dict():
    """
    Checks whether the `modify_formula_dict` function accurately updates a dictionary
    mapping elements to their respective atom counts based on a given adduct.
    """

    formula_dict = {"N": 0, "C": 5, "O": 2, "H": 8, "P": 0, "S": 0, "Cl": 0, "I": 0}

    adducts_to_element_count_updates = {
        "-2C": {"N": 0, "C": 3, "O": 2, "H": 8, "P": 0, "S": 0, "Cl": 0, "I": 0},
        "-H2O": {"N": 0, "C": 5, "O": 1, "H": 6, "P": 0, "S": 0, "Cl": 0, "I": 0},
        "+H": {"N": 0, "C": 5, "O": 2, "H": 9, "P": 0, "S": 0, "Cl": 0, "I": 0},
        "[M+H-NH4]": None,
        "[M+H+Na]2+": {"N": 0, "C": 5, "O": 2, "H": 9, "P": 0, "S": 0, "Cl": 0, "I": 0, "Na": 1},
        "[M-H]-": {"N": 0, "C": 5, "O": 2, "H": 7, "P": 0, "S": 0, "Cl": 0, "I": 0},
        "[M+CH3OH+H]+": {"N": 0, "C": 6, "O": 3, "H": 13, "P": 0, "S": 0, "Cl": 0, "I": 0},
        "[M+2Na-H]+": {"N": 0, "C": 5, "O": 2, "H": 7, "P": 0, "S": 0, "Cl": 0, "I": 0, "Na": 2},
        "[M+2Na]2+": {"N": 0, "C": 5, "O": 2, "H": 8, "P": 0, "S": 0, "Cl": 0, "I": 0, "Na": 2},
        "[M+DMSO+2H]2+": {"N": 0, "C": 7, "O": 3, "H": 16, "P": 0, "S": 1, "Cl": 0, "I": 0},
        "[M+2ACN+2H]2+": {"N": 2, "C": 9, "O": 2, "H": 16, "P": 0, "S": 0, "Cl": 0, "I": 0},
        "[2M+NH4]+": {"N": 1, "C": 10, "O": 4, "H": 20, "P": 0, "S": 0, "Cl": 0, "I": 0},
        "[3M-H]+": {"N": 0, "C": 15, "O": 6, "H": 23, "P": 0, "S": 0, "Cl": 0, "I": 0},
    }

    for adduct, expected_element_count in adducts_to_element_count_updates.items():
        assert expected_element_count == modify_formula_dict(formula_dict, adduct)


def test_get_element_count():
    """
    Checks whether the `get_element_count` function accurately returns the element count
    from a given chemical formula, provided as a string or in an abbreviated form.
    """

    formulas_to_element_counts = {
        "H2O": {"H": 2, "O": 1},
        "CH3CHOHCH3": {"C": 3, "H": 8, "O": 1},
        "(CH3)2SO": {"C": 2, "H": 6, "S": 1, "O": 1},
        "TFA": {"C": 2, "F": 3, "O": 2, "H": 1},
        "Pb(NO3)2": {"Pb": 1, "N": 2, "O": 6},
        "2DMSO": {"C": 4, "H": 12, "S": 2, "O": 2},
    }

    for formula, expected_element_count in formulas_to_element_counts.items():
        assert expected_element_count == get_element_count(formula)


def test_modify_charge(database_obj):
    """
    Checks whether the `modify_charge` function accurately updates the charge
    of a molecule based on a given adduct.
    """

    charges = [1, 2, 0, -1, -3, 4]

    adducts_to_charge_updates = {
        "[M+H+NH4]2+": [3, 4, 2, 1, -1, 6],
        "[M-H]-": [0, 1, -1, -2, -4, 3],
        "[M+CH3OH+H]+": [2, 3, 1, 0, -2, 5],
        "[M+2Na-H]+": [2, 3, 1, 0, -2, 5],
        "[M+2Na]2+": [3, 4, 2, 1, -1, 6],
        "[M+DMSO+H]+": [2, 3, 1, 0, -2, 5],
        "[M+2ACN+H]+": [2, 3, 1, 0, -2, 5],
        "[2M+ACN+H]+": [2, 3, 1, 0, -2, 5],
    }

    for adduct, expected_charges in adducts_to_charge_updates.items():
        assert expected_charges == [
            modify_charge(charge, adduct, database_obj.adducts_db) for charge in charges
        ]


def test_get_decoy_info():
    """
    Checks whether the `get_decoy_info` function correctly returns the element,
    number of elements, and the sign for a given decoy string.
    """
    decoy_to_mass_change = {
        "+H": ("H", 1, 1),
        "+2H": ("H", 2, 1),
        "-H": ("H", 1, -1),
        "+Fe": ("Fe", 1, 1),
        "+5Fe": ("Fe", 5, 1),
        "-2Fe": ("Fe", 2, -1),
    }

    for decoy, mass_change in decoy_to_mass_change.items():
        assert mass_change == get_decoy_info(decoy)


def test_get_adduct_from_list():
    """
    Checks whether the `get_adduct_from_list` function correctly identifies
    and returns adducts from a provided list of string values.
    """
    list_to_test = [
        "-2C",
        "-H2O",
        "+H",
        "[M+H-NH4]",
        "[M+H+Na]2+",
        "[M-H]-",
        "[M+CH3OH+H]+",
        "[M+2Na-H]+",
        "[M+2Na]2+",
        "[M+DMSO+2H]2+",
        "[M+2ACN+2H]2+",
        "[2M+NH4]+",
        "[3M-H]+",
    ]
    adducts = get_adducts(list_to_test)
    assert adducts == [
        "[M+H+Na]2+",
        "[M-H]-",
        "[M+CH3OH+H]+",
        "[M+2Na-H]+",
        "[M+2Na]2+",
        "[M+DMSO+2H]2+",
        "[M+2ACN+2H]2+",
        "[2M+NH4]+",
        "[3M-H]+",
    ]


def test_remove_noise():
    """
    Checks whether the `remove_noise` function accurately identifies
    and replaces intensity values below the specified noise threshold with 0.
    """

    spectra = [
        [[123.123, 0.93], [123.112, 0.0233], [123.03, 0.005], [122.93, 0.0417]],
        [
            [128.71589044, 0.34495419],
            [127.59401114, 0.00493238],
            [130.67092703, 0.22699608],
            [128.92881232, 0.24195734],
            [130.35453086, 0.18116001],
        ],
        [
            [155.26911885, 0.91971806],
            [155.53720916, 0.07885463],
            [155.0632367, 0.00096916],
            [155.74505433, 0.00045815],
        ],
    ]
    noises = [1e-4, 0.1, 1e-2]
    expected_results = [
        [[123.123, 0.93], [123.112, 0.0233], [123.03, 0.005], [122.93, 0.0417]],
        [
            [128.71589044, 0.34495419],
            [127.59401114, 0],
            [130.67092703, 0.22699608],
            [128.92881232, 0.24195734],
            [130.35453086, 0.18116001],
        ],
        [
            [155.26911885, 0.91971806],
            [155.53720916, 0.07885463],
            [155.0632367, 0],
            [155.74505433, 0],
        ],
    ]

    for i, spec in enumerate(spectra):
        result = remove_noise(spec, noise=noises[i])
        assert np.allclose(expected_results[i], result)


def test_get_formula():
    """
    Checks whether the `get_formula` correctly formats the
    given element-count dictionary and charge into a valid
    chemical formula string.
    """
    element_counts_and_charges = [
        ({"Pb": 1, "N": 2, "O": 6}, 2),
        ({"C": 4, "H": 12, "S": 2, "O": 2}, 0),
        ({"C": 4, "H": 12, "S": 2, "O": 2}, -1),
        ({"C": 4, "H": 12, "S": 2, "O": 2}, -1),
    ]
    expected = ["PbN2O6+2", "C4H12S2O2", "C4H12S2O2-", "C4H12S2O2-"]
    for i, (element_count, charge) in enumerate(element_counts_and_charges):
        result = get_formula(element_count, charge)
        assert result == expected[i]


def test_get_charge():
    """
    Checks whether the `get_charge` function accurately parses
    and ionic charge values from a provided list of compound in
    string format.
    """
    formulas_to_charges = {
        "C4H9NO2+": 1,
        "C11H19NO9": 0,
        "C22H36O2": 0,
        "C20H38O2-": -1,
        "C22H32O2-2": -2,
        "C15H33N5O3+3": 3,
    }

    for k, v in formulas_to_charges.items():
        result = get_charge(k)
        assert result == v


def test_sorted_value_index_search_many():
    values = np.array([10, 30, 20, 40, 25])
    index = SortedValueIndex(values)

    np.testing.assert_array_equal(index.search_many([10, 40], tolerance=1), [0, 3])

    np.testing.assert_array_equal(
        index.search_many([20, 24, 26], tolerance=2), [2, 4]
    )

    assert index.search_many([], tolerance=1).size == 0

    queries = [20, 30, 41]
    expected = np.unique(
        np.concatenate([index.search(q, tolerance=2) for q in queries])
    )
    np.testing.assert_array_equal(index.search_many(queries, tolerance=2), expected)


def test_sorted_value_index_search_many_pairs():
    values = np.array([10, 30, 20, 40, 25])
    index = SortedValueIndex(values)

    query_pos, ref_pos = index.search_many_pairs([10, 40], tolerance=1)
    np.testing.assert_array_equal(query_pos, [0, 1])
    np.testing.assert_array_equal(ref_pos, [0, 3])

    query_pos, ref_pos = index.search_many_pairs([20, 24, 26], tolerance=2)
    np.testing.assert_array_equal(query_pos, [0, 1, 2])
    np.testing.assert_array_equal(ref_pos, [2, 4, 4])

    query_pos, ref_pos = index.search_many_pairs([25], tolerance=6)
    np.testing.assert_array_equal(query_pos, [0, 0, 0])
    np.testing.assert_array_equal(ref_pos, [1, 2, 4])


    for empty in (index.search_many_pairs([], tolerance=1),
                  index.search_many_pairs([100], tolerance=1)):
        query_pos, ref_pos = empty
        assert query_pos.size == 0 and ref_pos.size == 0

    # The unique references across all pairs equal the search_many union.
    queries = [20, 24, 26]
    _, ref_pos = index.search_many_pairs(queries, tolerance=2)
    np.testing.assert_array_equal(
        np.unique(ref_pos), index.search_many(queries, tolerance=2)
    )

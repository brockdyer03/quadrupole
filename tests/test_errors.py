import re
from pathlib import Path

import numpy as np
import pytest

from quadrupole import (
    Atom,
    Element,
    Geometry,
    Quadrupole,
)
from quadrupole.geometry import FileFormatError, LatticeError


def test_invalid_symbol():
    with pytest.raises(
        ValueError,
        match="'bean' is not a valid Element",
    ):
        Element("bean")


def test_invalid_format():
    with pytest.raises(
        ValueError,
        match="Invalid format specifier 'bean' for object of type 'Element'",
    ):
        format(Element.Hydrogen, "bean")


def test_geometry_element_setter_too_many_elements():
    initial_elements = [
        Element.Hydrogen,
        Element.Ruthenium,
        Element.Bromine,
    ]

    xyzs = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)

    geometry = Geometry(list(map(Atom, initial_elements, xyzs)))

    assert(geometry.elements == initial_elements)
    np.testing.assert_array_equal(geometry.coordinates, xyzs)

    new_elements = [
        Element.Carbon,
        Element.Titanium,
        Element.Francium,
        Element.Francium,
    ]
    with pytest.raises(
        ValueError,
        match="Can not use list of length 4 for a geometry of 3 atoms!",
    ):
        geometry.elements = new_elements


def test_geometry_coordinate_setter_too_many_coordinates():
    elements = [
        Element.Hydrogen,
        Element.Ruthenium,
        Element.Bromine,
    ]

    initial_xyzs = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)

    geometry = Geometry(list(map(Atom, elements, initial_xyzs)))

    assert(geometry.elements == elements)
    np.testing.assert_array_equal(geometry.coordinates, initial_xyzs)

    new_xyzs = np.array([
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0],
        [19.0, 20.0, 21.0],
    ], dtype=np.float64)

    with pytest.raises(
        ValueError,
        match=re.escape("Can not set coordinates with shape (4, 3) for geometry with 3 atoms!"),
    ):
        geometry.coordinates = new_xyzs


def test_xyz_too_many_atoms(tmp_path):
    xyz = (
        "10\n"
        "comment\n"
        "H    1.0    2.0    3.0\n"
        "Ru   4.0    5.0    6.0\n"
        "Br   7.0    8.0    9.0\n"
    )

    temp_dir = tmp_path / Path("test_files")
    temp_dir.mkdir(exist_ok=True)

    xyz_path = temp_dir / Path("test.xyz")
    xyz_path.write_text(xyz, encoding="utf-8")
    with pytest.raises(
        FileFormatError,
        match=f"File {xyz_path} contains less atoms than expected!",
    ):
        Geometry.from_xyz(xyz_path)


def test_xyz_not_a_number(tmp_path):
    xyz = (
        "bean\n"
        "comment\n"
        "H    1.0    2.0    3.0\n"
        "Ru   4.0    5.0    6.0\n"
        "Br   7.0    8.0    9.0\n"
    )

    temp_dir = tmp_path / Path("test_files")
    temp_dir.mkdir(exist_ok=True)

    xyz_path = temp_dir / Path("test.xyz")
    xyz_path.write_text(xyz, encoding="utf-8")
    with pytest.raises(
        FileFormatError,
        match=(
            f"File {xyz_path} is improperly formatted at line 1,\n"
            "expected number of atoms, got 'bean\\n' instead!"
        ),
    ):
        Geometry.from_xyz(xyz_path)


def test_xyz_improper_format(tmp_path):
    xyz = (
        "3\n"
        "comment\n"
        "H    1.0    2.0    3.0\n"
        "Ru   4.0    5.0    6.0\n"
        "\n"
        "Br   7.0    8.0    9.0\n"
    )

    temp_dir = tmp_path / Path("test_files")
    temp_dir.mkdir(exist_ok=True)

    xyz_path = temp_dir / Path("test.xyz")
    xyz_path.write_text(xyz, encoding="utf-8")
    with pytest.raises(
        FileFormatError,
        match=f"File {xyz_path} is improperly formatted!",
    ):
        Geometry.from_xyz(xyz_path)


def test_list_length_mismatch():
    elements = [
        Element.Hydrogen,
        Element.Ruthenium,
    ]

    xyzs = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
    with pytest.raises(
        ValueError,
        match=(
            "The list of elements and coordinates must be of the same size!\n"
            "Number of elements:    2\n"
            "Number of coordinates: 3"
        ),
    ):
        Geometry.from_list(elements, xyzs)


def test_orca_no_input_block(tmp_path):
    fake_orca = (
        "bean\n"
        "bean bean\n"
        "bean bean bean\n"
        "bean bean bean bean\n"
        "bean bean bean bean bean\n"
    )

    temp_dir = tmp_path / Path("test_files")
    temp_dir.mkdir(exist_ok=True)

    orca_path = temp_dir / Path("test.out")
    orca_path.write_text(fake_orca, encoding="utf-8")
    with pytest.raises(
        FileFormatError,
        match=f"Error reading file '{orca_path}', did not find end of input!",
    ):
        Geometry.from_orca(orca_path)


def test_orca_no_calc_type():

    orca_output_path = Path(
        __file__ + "/../files/water_scf_improper.out"
    ).resolve()

    with pytest.raises(
        FileFormatError,
        match=f"Error reading file '{orca_output_path}' at line 14592!",
    ):
        Geometry.from_orca(orca_output_path)


def test_orca_no_final_geom():

    orca_output_path = Path(
        __file__ + "/../files/water_opt_improper.out"
    ).resolve()

    with pytest.raises(
        FileFormatError,
        match=f"Error reading file '{orca_output_path}', can not find final geometry",
    ):
        Geometry.from_orca(orca_output_path)


def test_invalid_bravais_index():
    cell_params = np.zeros(6)

    with pytest.raises(
        ValueError,
        match=(
            "Bravais lattice index 42 not supported!\n"
            "Please select from a supported index!"
        ),
    ):
        Geometry.generate_lattice(42, cell_params)


def test_lattice_mismatch_cubic():
    cell_params = np.array([42, 42, 20, np.pi/2, np.pi/2, np.pi/2])

    with pytest.raises(
        LatticeError,
        match=re.escape(
            "Can not generate Simple Cubic lattice with\n"
            "a=42.00000 b=42.00000 c=20.00000 α=1.57080 β=1.57080 γ=1.57080"
        ),
    ):
        Geometry.generate_lattice(1, cell_params)


def test_lattice_mismatch_tetragonal():
    cell_params = np.array([42, 20, 42, np.pi/2, np.pi/2, np.pi/2])

    with pytest.raises(
        LatticeError,
        match=re.escape(
            "Can not generate Simple Tetragonal lattice with\n"
            "a=42.00000 b=20.00000 c=42.00000 α=1.57080 β=1.57080 γ=1.57080"
        ),
    ):
        Geometry.generate_lattice(6, cell_params)


def test_lattice_mismatch_orthorhombic():
    cell_params = np.array([42, 20, 12, np.pi/3, np.pi/2, np.pi/2])

    with pytest.raises(
        LatticeError,
        match=re.escape(
            "Can not generate Face-Centered Orthorhombic lattice with\n"
            "a=42.00000 b=20.00000 c=12.00000 α=1.04720 β=1.57080 γ=1.57080"
        ),
    ):
        Geometry.generate_lattice(10, cell_params)


def test_lattice_mismatch_rhombohedral():
    cell_params = np.array([42, 42, 20, np.pi/2, np.pi/2, np.pi/2])
    with pytest.raises(
        LatticeError,
        match=re.escape(
            "Can not generate Rhombohedral lattice with\n"
            "a=42.00000 b=42.00000 c=20.00000 α=1.57080 β=1.57080 γ=1.57080"
        ),
    ):
        Geometry.generate_lattice(5, cell_params)


def test_lattice_mismatch_hexagonal():
    cell_params = np.array([42, 42, 20, np.pi/15, np.pi/2, 2*np.pi/3])
    with pytest.raises(
        LatticeError,
        match=re.escape(
            "Can not generate Simple Hexagonal lattice with\n"
            "a=42.00000 b=42.00000 c=20.00000 α=0.20944 β=1.57080 γ=2.09440"
        ),
    ):
        Geometry.generate_lattice(4, cell_params)


def test_lattice_mismatch_monoclinic_alpha_gamma():
    cell_params = np.array([42, 12, 20, np.pi/3, np.pi/2, np.pi/2])
    with pytest.raises(
        LatticeError,
        match=re.escape(
            "Can not generate Base-Centered Monoclinic (Unique axis b) lattice with\n"
            "a=42.00000 b=12.00000 c=20.00000 α=1.04720 β=1.57080 γ=1.57080"
        ),
    ):
        Geometry.generate_lattice(-13, cell_params)


def test_lattice_mismatch_monoclinic_beta_gamma():
    cell_params = np.array([42, 12, 20, np.pi/2, np.pi/3, np.pi/2])
    with pytest.raises(
        LatticeError,
        match=re.escape(
            "Can not generate Base-Centered Monoclinic lattice with\n"
            "a=42.00000 b=12.00000 c=20.00000 α=1.57080 β=1.04720 γ=1.57080"
        ),
    ):
        Geometry.generate_lattice(13, cell_params)


def test_prim_lattice_invalid_bravais_index():
    # This shouldn't actually be accessible by users, but I want 100% test coverage.
    cell_params = np.array([42, 12, 20, np.pi/2, np.pi/3, np.pi/2])

    with pytest.raises(
        ValueError,
        match="Invalid lattice type: 42",
    ):
        Geometry._gen_prim_lattice(42, cell_params)


def test_request_non_primitive_lattice():
    cell_params = np.array([20, 20, 20, np.pi/2, np.pi/2, np.pi/2])

    with pytest.raises(
        NotImplementedError,
        match="Only primitive cells are currently supported!",
    ):
        Geometry.generate_lattice(1, cell_params)


def test_cjson_no_atoms():

    cjson_path = Path(
        __file__ + "/../files/missing_atoms.cjson"
    ).resolve()

    with pytest.raises(
        FileFormatError,
        match=re.escape(
            "Expected 'atoms' field in CJSON file, but did not find any!\n"
            f"({cjson_path})"
        ),
    ):
        Geometry.from_cjson(cjson_path)


def test_cjson_unknown_version():
    cjson_path = Path(
        __file__ + "/../files/wrong_version.cjson"
    ).resolve()

    with pytest.warns(
        UserWarning,
        match=re.escape(
            f"This file ({cjson_path}) is CJSON version 8675309 however we only guarantee support for version 1."  # noqa: E501
        ),
    ):
        Geometry.from_cjson(cjson_path)


def test_invalid_quadrupole_shape():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot cast array of shape (4,) to a quadrupole!\n"
            "Supply either shape (3, 3) or (3,) or (6,)!"
        ),
    ):
        Quadrupole([1.0, 2.0, 3.0, 4.0])


def test_invalid_quadrupole_units():
    with pytest.raises(
        ValueError,
        match=re.escape("Invalid units, please select from ( 'au', 'buckingham', 'cm2', 'esu' )"),
    ):
        Quadrupole([1.0, 2.0, 3.0], units="bananas")


def test_invalid_quadrupole_as_unit():
    quadrupole = Quadrupole([1.0, 2.0, 3.0], units="buckingham")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unit bananas not recognized, please pick from ('au', 'buckingham', 'cm2', 'esu')"
        ),
    ):
        quadrupole.as_unit("bananas")


def test_quadrupole_from_orca_no_quadrupole():
    orca_output_path = Path(
        __file__ + "/../files/water_scf_improper.out"
    ).resolve()

    with pytest.raises(
        FileFormatError,
        match=f"Could not locate a quadrupole moment in output {orca_output_path}",
    ):
        Quadrupole.from_orca(orca_output_path)

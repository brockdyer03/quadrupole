import pytest
from pathlib import Path
import numpy as np
from quadrupole import (
    Element,
    Geometry,
    Quadrupole,
    FileFormatError,
    LatticeError,
)


@pytest.mark.xfail(
    reason="Too many atoms specified at top of file",
    raises=FileFormatError,
)
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

    Geometry.from_xyz(xyz_path)


@pytest.mark.xfail(
    reason="Did not get integer for number of atoms",
    raises=FileFormatError,
)
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

    Geometry.from_xyz(xyz_path)


@pytest.mark.xfail(
    reason="Erroneous newline in file",
    raises=FileFormatError,
)
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

    Geometry.from_xyz(xyz_path)


@pytest.mark.xfail(
    reason="List of elements is not the same length as the list of coordinates",
    raises=ValueError,
)
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

    Geometry.from_list(elements, xyzs)


@pytest.mark.xfail(
    reason="No input block in ORCA file",
    raises=FileFormatError,
)
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

    Geometry.from_orca(orca_path)


@pytest.mark.xfail(
    reason="No specification of calculation type in ORCA output",
    raises=FileFormatError,
)
def test_orca_no_calc_type():

    orca_output_path = Path(
        __file__ + "/../files/water_scf_improper.out"
    ).resolve()

    Geometry.from_orca(orca_output_path)


@pytest.mark.xfail(
    reason="No final geometry in ORCA output",
    raises=FileFormatError,
)
def test_orca_no_final_geom():

    orca_output_path = Path(
        __file__ + "/../files/water_opt_improper.out"
    ).resolve()

    Geometry.from_orca(orca_output_path)


@pytest.mark.xfail(
    reason="Bravais lattice index outside of supported values",
    raises=ValueError,
)
def test_invalid_bravais_index():
    cell_params = np.zeros(6)
    Geometry.generate_lattice(42, cell_params)


@pytest.mark.xfail(
    reason="Request simple cubic lattice with non-simple cubic values",
    raises=LatticeError,
)
def test_lattice_mismatch_cubic():
    cell_params = np.array([42, 42, 20, np.pi/2, np.pi/2, np.pi/2])
    Geometry.generate_lattice(1, cell_params)


@pytest.mark.xfail(
    reason="Request tetragonal lattice with non-tetragonal values",
    raises=LatticeError,
)
def test_lattice_mismatch_tetragonal():
    cell_params = np.array([42, 20, 42, np.pi/2, np.pi/2, np.pi/2])
    Geometry.generate_lattice(6, cell_params)


@pytest.mark.xfail(
    reason="Request orthorhombic lattice with non-orthorhombic values",
    raises=LatticeError,
)
def test_lattice_mismatch_orthorhombic():
    cell_params = np.array([42, 20, 12, np.pi/3, np.pi/2, np.pi/2])
    Geometry.generate_lattice(10, cell_params)


@pytest.mark.xfail(
    reason="Request rhombohedral lattice with non-rhombohedral values",
    raises=LatticeError,
)
def test_lattice_mismatch_rhombohedral():
    cell_params = np.array([42, 42, 20, np.pi/2, np.pi/2, np.pi/2])
    Geometry.generate_lattice(5, cell_params)


@pytest.mark.xfail(
    reason="Request hexagonal lattice with non-hexagonal values",
    raises=LatticeError,
)
def test_lattice_mismatch_hexagonal():
    cell_params = np.array([42, 42, 20, np.pi/15, np.pi/2, 2*np.pi/3])
    Geometry.generate_lattice(4, cell_params)


@pytest.mark.xfail(
    reason="Request monoclinic lattice with non-monoclinic values",
    raises=LatticeError,
)
def test_lattice_mismatch_monoclinic_alpha_gamma():
    cell_params = np.array([42, 12, 20, np.pi/3, np.pi/2, np.pi/2])
    Geometry.generate_lattice(-13, cell_params)


@pytest.mark.xfail(
    reason="Request monoclinic lattice with non-monoclinic values",
    raises=LatticeError,
)
def test_lattice_mismatch_monoclinic_beta_gamma():
    cell_params = np.array([42, 12, 20, np.pi/2, np.pi/3, np.pi/2])
    Geometry.generate_lattice(13, cell_params)


@pytest.mark.xfail(
    reason="Request lattice with invalid bravais index",
    raises=ValueError,
)
def test_prim_lattice_invalid_bravais_index():
    """This shouldn't actually be accessible by users, but
    I want 100% test coverage.
    """
    cell_params = np.array([42, 12, 20, np.pi/2, np.pi/3, np.pi/2])
    Geometry._gen_prim_lattice(42, cell_params)


@pytest.mark.xfail(
    reason="Request non-primitive lattice",
    raises=ValueError,
)
def test_request_non_primitive_lattice():
    cell_params = np.array([20, 20, 20, np.pi/2, np.pi/2, np.pi/2])
    Geometry.generate_lattice(1, cell_params)


@pytest.mark.xfail(
    reason="Can't convert array of incorrect shape to Quadrupole object",
    raises=ValueError,
)
def test_invalid_quadrupole_shape():
    Quadrupole([1.0, 2.0, 3.0, 4.0])


@pytest.mark.xfail(
    reason="Incorrect units specified",
    raises=ValueError,
)
def test_invalid_quadrupole_units():
    Quadrupole([1.0, 2.0, 3.0], units="bananas")


@pytest.mark.xfail(
    reason="Invalid unit conversion request",
    raises=ValueError,
)
def test_invalid_quadrupole_as_unit():
    quadrupole = Quadrupole([1.0, 2.0, 3.0], units="buckingham")
    quadrupole.as_unit("bananas")

import pytest
from pathlib import Path
import numpy as np
from quadrupole import Element, Atom, Geometry


def test_element():
    hydrogen = Element.Hydrogen
    assert(hydrogen == Element.H)
    assert(hydrogen == Element("H"))
    assert(hydrogen == Element(1))
    assert(hydrogen.name == "Hydrogen")
    assert(hydrogen.symbol == "H")
    assert(hydrogen.number == 1)
    assert(hydrogen.mass == 1.0080)
    assert(str(hydrogen) == "H")

    ruthenium = Element.Ruthenium
    assert(ruthenium == Element.Ru)
    assert(ruthenium == Element("Ru"))
    assert(ruthenium == Element(44))
    assert(ruthenium.name == "Ruthenium")
    assert(ruthenium.symbol == "Ru")
    assert(ruthenium.number == 44)
    assert(ruthenium.mass == 101.07)
    assert(str(ruthenium) == "Ru")


def test_atom():
    element = Element.Hydrogen
    xyz = np.array([3.14, 42.0, 137.0], dtype=np.float64)
    atom = Atom(element, xyz)

    assert(atom == Atom(Element.H, xyz))
    assert(atom == Atom(Element(1), xyz))
    assert(atom == Atom(Element("H"), xyz))
    assert(atom == Atom(1, xyz))
    assert(atom == Atom("H", xyz))

    assert(atom == Atom(element, [3.14, 42.0, 137.0]))
    assert(atom == Atom(element, [3.14, 42, 137]))

    assert(atom.element == Element.Hydrogen)
    assert(atom.element == Element.H)

    assert(np.all(atom.xyz == np.array([3.14, 42.0, 137.0], dtype=np.float64)))


def test_inertia():
    elements = [
        Element.Hydrogen,
        Element.Ruthenium,
        Element.Bromine,
    ]

    xyzs = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)

    atoms = [Atom(element, xyz) for element, xyz in zip(elements, xyzs)]

    geometry = Geometry(
        atoms=atoms,
        lat_vec=None,
        alat=None,
    )

    assert(geometry.atoms == atoms)
    assert(geometry.get_elements() == elements)
    np.testing.assert_array_equal(geometry.get_coords(), xyzs)

    ref_eigenvalues = np.array(
        [1261.1061354199865, -1.1368683772161603e-13, 1261.1061354199867],
        dtype=np.float64
    )

    ref_eigenvectors = np.array(
        [
            [ 0.816496580928, -0.57735026919,  0.027979205835],
            [-0.408248290464, -0.57735026919, -0.720681100695],
            [-0.408248290464, -0.57735026919,  0.692701894860],
        ], dtype=np.float64
    )

    eigenvalues, eigenvectors = geometry.calc_principal_moments()

    print(eigenvalues)
    print(eigenvectors)

    np.testing.assert_allclose(eigenvalues,  ref_eigenvalues,  rtol=1e-8)
    np.testing.assert_allclose(eigenvectors, ref_eigenvectors, rtol=1e-8)


def test_from_list():
    elements = [
        Element.Hydrogen,
        Element.Ruthenium,
        Element.Bromine,
    ]

    xyzs = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)

    geometry = Geometry.from_list(elements, xyzs)

    assert(geometry.get_elements() == elements)
    assert(np.all(geometry.get_coords() == xyzs))


def test_from_xyz(tmp_path):
    elements = [
        Element.Hydrogen,
        Element.Ruthenium,
        Element.Bromine,
    ]

    xyzs = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)

    xyz = (
        "3\n"
        "comment\n"
        "H    1.0    2.0    3.0\n"
        "Ru   4.0    5.0    6.0\n"
        "Br   7.0    8.0    9.0\n"
    )

    temp_dir = tmp_path / Path("test_files")
    temp_dir.mkdir()

    xyz_path = temp_dir / Path("test.xyz")
    xyz_path.write_text(xyz, encoding="utf-8")

    geometry = Geometry.from_xyz(xyz_path)

    assert(geometry.get_elements() == elements)
    np.testing.assert_array_equal(geometry.get_coords(), xyzs)


def test_from_xsf(tmp_path):
    elements = [
        Element.Oxygen,
        Element.Hydrogen,
        Element.Hydrogen,
    ]

    xyzs = np.array([
        [8.770451552, 9.174907782, 8.771938274],
        [9.544851428, 8.573414302, 8.771938274],
        [8.000511848, 8.567492748, 8.771938274],
    ], dtype=np.float64)

    lat_vec = np.array([
        [17.543876553,  0.000000000,  0.000000000],
        [ 0.000000000, 17.543876553,  0.000000000],
        [ 0.000000000,  0.000000000, 17.543876553],
    ], dtype=np.float64)

    alat = 17.543876553

    xsf = (
        "CRYSTAL\n"
        "PRIMVEC\n"
        "  17.543876553    0.000000000    0.000000000\n"
        "   0.000000000   17.543876553    0.000000000\n"
        "   0.000000000    0.000000000   17.543876553\n"
        "PRIMCOORD\n"
        "        3           1\n"
        "O         8.770451552    9.174907782    8.771938274\n"
        "H         9.544851428    8.573414302    8.771938274\n"
        "H         8.000511848    8.567492748    8.771938274\n"
    )

    temp_dir = tmp_path / Path("test_files")
    temp_dir.mkdir()

    xsf_path = temp_dir / Path("test.xsf")

    xsf_path.write_text(xsf, encoding="utf-8")

    geometry = Geometry.from_xsf(xsf_path)

    assert(geometry.get_elements() == elements)
    np.testing.assert_array_equal(geometry.get_coords(), xyzs)
    np.testing.assert_array_equal(geometry.lat_vec, lat_vec)
    assert(geometry.alat == alat)


def test_from_orca():
    elements = [
        Element.Oxygen,
        Element.Hydrogen,
        Element.Hydrogen,
    ]

    final_xyzs = np.array([
        [5.753633, 3.280382, 2.728205],
        [4.931827, 3.412032, 2.252440],
        [5.738252, 3.930645, 3.432457],
    ], dtype=np.float64)

    orca_output_path = Path(
        __file__ + "../../../notebook/example_outputs/water_random_rotation.out"
    ).resolve()

    geometry = Geometry.from_orca(orca_output_path)

    assert(geometry.get_elements() == elements)
    np.testing.assert_array_equal(geometry.get_coords(), final_xyzs)


def test_from_cube(tmp_path):
    elements = [
        Element.Oxygen,
        Element.Hydrogen,
        Element.Hydrogen,
    ]

    xyzs = np.array([
        [12.75798576, 13.05368522, 12.76021095],
        [13.52042164, 12.47252958, 12.76021095],
        [12.00000026, 12.46673668, 12.76021095],
    ], dtype=np.float64)

    lat_vec = np.array([
        [25.52039649,  0.00000000,  0.00000000],
        [ 0.00000000, 25.52039649,  0.00000000],
        [ 0.00000000,  0.00000000, 25.52039649],
    ], dtype=np.float64)

    alat = 25.52039649

    cube = (
        " Cubefile created from PWScf calculation\n"
        "Contains the selected quantity on a FFT grid\n"
        "    3    0.000000    0.000000    0.000000\n"
        "  320    0.150708    0.000000    0.000000\n"
        "  320    0.000000    0.150708    0.000000\n"
        "  320    0.000000    0.000000    0.150708\n"
        "    8    8.000000   24.109099   24.667890   24.113304\n"
        "    1    1.000000   25.549894   23.569665   24.113304\n"
        "    1    1.000000   22.676714   23.558718   24.113304\n"
        " -0.22991E-06  0.15171E-06  0.49928E-07 -0.95286E-07 -0.28930E-07  0.81718E-08\n"
        "  0.17149E-06 -0.27311E-06  0.11460E-06  0.74206E-07 -0.12380E-06 -0.50463E-07\n"
        " -0.56805E-07  0.25518E-06 -0.13170E-06 -0.15147E-07  0.13074E-06  0.23688E-07\n"
        " -0.11941E-06  0.43801E-07  0.69832E-07 -0.10074E-06  0.95414E-07 -0.43724E-07\n"
        "  0.16973E-08 -0.29553E-07  0.98935E-07 -0.51880E-07 -0.10265E-06  0.19040E-06\n"
        " -0.46587E-07 -0.49705E-07  0.41958E-07  0.68452E-09 -0.13268E-06  0.44203E-07\n"
    )

    temp_dir = tmp_path / Path("test_files")
    temp_dir.mkdir()

    cube_path = temp_dir / Path("test.cube")

    cube_path.write_text(cube, encoding="utf-8")

    geometry = Geometry.from_cube(cube_path)

    assert(geometry.get_elements() == elements)
    np.testing.assert_allclose(geometry.get_coords(), xyzs, rtol=1e-8)
    np.testing.assert_allclose(geometry.lat_vec, lat_vec, rtol=1e-7)
    np.testing.assert_almost_equal(geometry.alat,  alat, decimal=8)

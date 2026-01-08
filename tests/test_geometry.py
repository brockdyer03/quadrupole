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


def test_geometry():
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
    assert(np.all(geometry.get_coords() == xyzs))
    assert(geometry.get_elements() == elements)

    ref_eigenvalues = np.array([1261.1061354199865, -1.1368683772161603e-13, 1261.1061354199867], dtype=np.float64)

    ref_eigenvectors = np.array(
        [
            [ 0.81649658, -0.57735027,  0.02797921],
            [-0.40824829, -0.57735027, -0.72068110],
            [-0.40824829, -0.57735027,  0.69270189],
        ]
    )

    eigenvalues, eigenvectors = geometry.calc_principal_moments()

    assert(np.all(eigenvalues.round(8) == ref_eigenvalues.round(8)))
    assert(np.all(eigenvectors.round(8) == ref_eigenvectors.round(8)))


def test_geometry_from_list():
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


def test_geometry_from_xyz(tmp_path):
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
    assert(np.all(geometry.get_coords() == xyzs))


def test_geometry_from_xsf(tmp_path):
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
    assert(np.all(geometry.get_coords() == xyzs))
    assert(np.all(geometry.lat_vec == lat_vec))
    assert(geometry.alat == alat)

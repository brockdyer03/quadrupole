import pytest
from pathlib import Path
import numpy as np
from quadrupole import Element, Atom, Geometry


def test_element():
    hydrogen = Element.Hydrogen
    assert(hydrogen == Element.H)
    assert(hydrogen == Element("H"))
    assert(hydrogen == Element("h"))
    assert(hydrogen == Element(1))
    assert(hydrogen.name == "Hydrogen")
    assert(hydrogen.symbol == "H")
    assert(hydrogen.number == 1)
    assert(hydrogen.mass == 1.0080)
    assert(str(hydrogen) == "H")

    ruthenium = Element.Ruthenium
    assert(ruthenium == Element.Ru)
    assert(ruthenium == Element("Ru"))
    assert(ruthenium == Element("ru"))
    assert(ruthenium == Element("RU"))
    assert(ruthenium == Element("rU"))
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

    atom_repr = (
        "Element     X          Y          Z          \n"
        "H           3.140000  42.000000 137.000000\n"
    )

    assert(atom.__repr__() == atom_repr)


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

    assert(len(geometry) == 3)
    assert(geometry.atoms == atoms)
    assert(geometry[0] == atoms[0])
    assert(geometry[1] == atoms[1])
    assert(geometry[2] == atoms[2])
    assert(geometry.get_elements() == elements)
    np.testing.assert_array_equal(geometry.get_coords(), xyzs)

    ref_eigenvalues = np.array(
        [1261.1061354199865, -1.1368683772161603e-13, 1261.1061354199867],
        dtype=np.float64
    )

    # ref_eigenvectors = np.array([
    #     [ 0.816496580928, -0.57735026919,  0.027979205835],
    #     [-0.408248290464, -0.57735026919, -0.720681100695],
    #     [-0.408248290464, -0.57735026919,  0.692701894860],
    # ], dtype=np.float64)

    ref_diagonal_inertia_matrix = np.array([
        [ 1.26110614e+03, -1.26746377e-13, -1.04221575e-14],
        [-3.53654456e-14, -1.00000169e-08,  1.00177653e-13],
        [-1.39411949e-14,  2.90059484e-14,  1.26110614e+03],
    ], dtype=np.float64)

    # Turns out the eigenvectors and eigenvalues will not necessarily be
    # consistent across platforms, so instead of comparing them directly,
    # we are going to diagonalize the inertia matrix of the system and 
    # ensure that it is properly diagonalized to within tolerance.
    inertia_matrix = np.array([
        [ 840.73742361, -420.36871181, -420.36871181],
        [-420.36871181,  840.73742361, -420.36871181],
        [-420.36871181, -420.36871181,  840.73742361],
    ], dtype=np.float64)

    eigenvalues, eigenvectors = geometry.calc_principal_moments()

    diagonal_inertia_matrix = np.linalg.inv(eigenvectors) @ inertia_matrix @ eigenvectors

    np.testing.assert_allclose(eigenvalues,  ref_eigenvalues,  atol=1e-8)
    np.testing.assert_allclose(diagonal_inertia_matrix,  ref_diagonal_inertia_matrix,  atol=1e-8)

    #np.testing.assert_allclose(eigenvectors, ref_eigenvectors, atol=1e-8)


def test_simple_cubic():
    # Polonium
    ref_cell = np.array([
        [3.34000, 0.00000, 0.00000],
        [0.00000, 3.34000, 0.00000],
        [0.00000, 0.00000, 3.34000],
    ], dtype=np.float64)

    cell_params = np.array(
        [3.34000, 3.34000, 3.34000, np.pi/2, np.pi/2, np.pi/2]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=1,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_equal(gen_cell, ref_cell)


def test_face_centered_cubic():
    ref_cell = np.array([
        [-21.12,     0, 21.12],
        [     0, 21.12, 21.12],
        [-21.12, 21.12,     0],
    ], dtype=np.float64)

    cell_params = np.array(
        [42.24, 42.24, 42.24, np.pi/2, np.pi/2, np.pi/2]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=2,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_equal(gen_cell, ref_cell)


def test_body_centered_cubic_low_sym():
    ref_cell = np.array([
        [ 21.12,  21.12, 21.12],
        [-21.12,  21.12, 21.12],
        [-21.12, -21.12, 21.12],
    ], dtype=np.float64)

    cell_params = np.array(
        [42.24, 42.24, 42.24, np.pi/2, np.pi/2, np.pi/2]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=3,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_equal(gen_cell, ref_cell)


def test_body_centered_cubic_high_sym():
    ref_cell = np.array([
        [-21.12,  21.12,  21.12],
        [ 21.12, -21.12,  21.12],
        [ 21.12,  21.12, -21.12],
    ], dtype=np.float64)

    cell_params = np.array(
        [42.24, 42.24, 42.24, np.pi/2, np.pi/2, np.pi/2]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=-3,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_equal(gen_cell, ref_cell)


def test_hexagonal():
    # Elemental Cobalt
    ref_cell = np.array([
        [ 2.50710,       0,       0],
        [-1.25355, 2.17121,       0],
        [       0,       0, 4.06860],
    ], dtype=np.float64)

    cell_params = np.array(
        [2.50710, 2.50710, 4.06860, np.pi/2, np.pi/2, 2*np.pi/3]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=4,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_rhombohedral_z_symmetry():
    ref_cell = np.array([
        [ 12.000000, -6.928200, 19.595928],
        [  0.000000, 13.856400, 19.595928],
        [-12.000000, -6.928200, 19.595928],
    ], dtype=np.float64)

    cell_params = np.array(
        [24.000, 24.000, 24.000, np.pi/3, np.pi/3, np.pi/3]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=5,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_rhombohedral_111_symmetry():
    ref_cell = np.array([
        [ 0.000000, 16.970568, 16.970568],
        [16.970568,  0.000000, 16.970568],
        [16.970568, 16.970568,  0.000000],
    ], dtype=np.float64)

    cell_params = np.array(
        [24.000, 24.000, 24.000, np.pi/3, np.pi/3, np.pi/3]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=-5,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_simple_tetragonal():
    # β-tin
    ref_cell = np.array([
        [5.81970, 0.00000, 0.00000],
        [0.00000, 5.81970, 0.00000],
        [0.00000, 0.00000, 3.17488],
    ], dtype=np.float64)

    cell_params = np.array(
        [5.81970, 5.81970, 3.17488, np.pi/2, np.pi/2, np.pi/2]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=6,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_body_centered_tetragonal():
    ref_cell = np.array([
        [ 1.62615, -1.62615, 2.47295],
        [ 1.62615,  1.62615, 2.47295],
        [-1.62615, -1.62615, 2.47295],
    ], dtype=np.float64)

    cell_params = np.array(
        [3.25230, 3.25230, 4.94590, np.pi/2, np.pi/2, np.pi/2]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=7,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_simple_orthorhombic():
    # Cementite, Fe₃C
    ref_cell = np.array([
        [5.09230, 0.00000, 0.00000],
        [0.00000, 6.78960, 0.00000],
        [0.00000, 0.00000, 4.52920],
    ], dtype=np.float64)

    cell_params = np.array(
        [5.09230, 6.78960, 4.52920, np.pi/2, np.pi/2, np.pi/2]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=8,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_legacy_base_centered_orthorhombic_c_type():
    # OsCl₄
    ref_cell = np.array([
        [ 3.96450, 4.16300, 0.00000],
        [-3.96450, 4.16300, 0.00000],
        [ 0.00000, 0.00000, 3.56000],
    ], dtype=np.float64)

    cell_params = np.array(
        [7.92900, 8.32600, 3.56000, np.pi/2, np.pi/2, np.pi/2]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=9,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_base_centered_orthorhombic_c_type():
    # OsCl₄
    ref_cell = np.array([
        [3.96450, -4.16300, 0.00000],
        [3.96450,  4.16300, 0.00000],
        [0.00000,  0.00000, 3.56000],
    ], dtype=np.float64)

    cell_params = np.array(
        [7.92900, 8.32600, 3.56000, np.pi/2, np.pi/2, np.pi/2]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=-9,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_base_centered_orthorhombic_a_type():
    ref_cell = np.array([
        [24.000000,  0.000000,  0.000000],
        [ 0.000000,  7.200000, -4.800000],
        [ 0.000000,  7.200000,  4.800000],
    ], dtype=np.float64)

    cell_params = np.array(
        [24.000000, 14.400000, 9.600000, np.pi/2, np.pi/2, np.pi/2]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=91,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_face_centered_orthorhombic():
    # Cs₂P₃
    ref_cell = np.array([
        [4.71900, 0.00000, 7.49950],
        [4.71900, 4.98600, 0.00000],
        [0.00000, 4.98600, 7.49950],
    ], dtype=np.float64)

    cell_params = np.array(
        [9.43800, 9.97200, 14.99900, np.pi/2, np.pi/2, np.pi/2]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=10,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_body_centered_orthorhombic():
    # I (Pearson Symbol oI2)
    ref_cell = np.array([
        [ 1.45200,  1.51550, 2.62600],
        [-1.45200,  1.51550, 2.62600],
        [-1.45200, -1.51550, 2.62600],
    ], dtype=np.float64)

    cell_params = np.array(
        [2.90400, 3.03100, 5.25200, np.pi/2, np.pi/2, np.pi/2]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=11,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_simple_monoclinic_unique_c():
    ref_cell = np.array([
        [24.00000,  0.00000, 0.00000],
        [-6.00000, 10.39231, 0.00000],
        [ 0.00000,  0.00000, 6.00000],
    ], dtype=np.float64)

    cell_params = np.array(
        [24.00000, 12.00000, 6.00000, np.pi/2, np.pi/2, 2*np.pi/3]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=12,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_simple_monoclinic_unique_b():
    # S₈
    ref_cell = np.array([
        [10.66300,  0.00000,  0.00000],
        [ 0.00000, 10.68400,  0.00000],
        [-1.07443,  0.00000, 10.74542],
    ], dtype=np.float64)

    cell_params = np.array(
        [10.66300, 10.68400, 10.799, np.pi/2, (95.71*np.pi/180), np.pi/2]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=-12,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_base_centered_monoclinic_unique_c():
    # CCl₄, modified
    ref_cell = np.array([
        [ 10.09050,  0.00000, -9.88050],
        [ -4.15242, 10.56314,  0.00000],
        [ 10.09050,  0.00000,  9.88050],
    ], dtype=np.float64)

    cell_params = np.array(
        [20.18100, 11.35000, 19.76100, np.pi/2, np.pi/2, (111.46*np.pi/180)]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=13,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_base_centered_monoclinic_unique_b():
    # CCl₄
    ref_cell = np.array([
        [ 10.09050, 5.67500,  0.00000],
        [-10.09050, 5.67500,  0.00000],
        [ -7.22959, 0.00000, 18.39103],
    ], dtype=np.float64)

    cell_params = np.array(
        [20.18100, 11.35000, 19.76100, np.pi/2, (111.46*np.pi/180), np.pi/2]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=-13,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_triclinic():
    # Montbrayite, Au₂Te₃
    ref_cell = np.array([
        [10.80000,  0.00000,  0.00000],
        [-1.59507, 11.99441,  0.00000],
        [-3.38831, -4.62223, 12.17892],
    ], dtype=np.float64)

    cell_params = np.array(
        [10.80000, 12.10000, 13.46000, (107.892*np.pi/180), (104.58*np.pi/180), (97.575*np.pi/180)]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=14,
        cell_params=cell_params,
        primitive=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_qe_format_params_hexagonal():
    # Elemental Cobalt
    ref_cell = np.array([
        [ 2.50710,       0,       0],
        [-1.25355, 2.17121,       0],
        [       0,       0, 4.06860],
    ], dtype=np.float64)

    a = 2.50710
    c = 4.06860 / a

    cell_params = np.array(
        [a, 0, c, 0, 0, 0]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=4,
        cell_params=cell_params,
        primitive=True,
        espresso_like=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_qe_format_params_rhombohedral():
    ref_cell = np.array([
        [ 12.000000, -6.928200, 19.595928],
        [  0.000000, 13.856400, 19.595928],
        [-12.000000, -6.928200, 19.595928],
    ], dtype=np.float64)

    a = 24.00000
    gamma = np.cos(np.pi/3)

    cell_params = np.array(
        [a, 0, 0, gamma, 0, 0]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=5,
        cell_params=cell_params,
        primitive=True,
        espresso_like=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_qe_format_params_tetragonal():
    # β-tin
    ref_cell = np.array([
        [5.81970, 0.00000, 0.00000],
        [0.00000, 5.81970, 0.00000],
        [0.00000, 0.00000, 3.17488],
    ], dtype=np.float64)

    a = 5.81970
    c = 3.17488 / a

    cell_params = np.array(
        [a, 0, c, 0, 0, 0]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=6,
        cell_params=cell_params,
        primitive=True,
        espresso_like=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_qe_format_params_orthorhombic():
    # Cementite, Fe₃C
    ref_cell = np.array([
        [5.09230, 0.00000, 0.00000],
        [0.00000, 6.78960, 0.00000],
        [0.00000, 0.00000, 4.52920],
    ], dtype=np.float64)

    a = 5.09230
    b = 6.78960 / a
    c = 4.52920 / a

    cell_params = np.array(
        [a, b, c, 0, 0, 0]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=8,
        cell_params=cell_params,
        primitive=True,
        espresso_like=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_qe_format_simple_monoclinic_unique_b():
    # S₈
    ref_cell = np.array([
        [10.66300,  0.00000,  0.00000],
        [ 0.00000, 10.68400,  0.00000],
        [-1.07443,  0.00000, 10.74542],
    ], dtype=np.float64)

    a = 10.66300
    b = 10.68400 / a
    c = 10.79900 / a
    beta = np.cos(95.71*np.pi/180)

    cell_params = np.array(
        [a, b, c, 0, beta, 0]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=-12,
        cell_params=cell_params,
        primitive=True,
        espresso_like=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_qe_format_monoclinic_unique_c():
    ref_cell = np.array([
        [24.00000,  0.00000, 0.00000],
        [-6.00000, 10.39231, 0.00000],
        [ 0.00000,  0.00000, 6.00000],
    ], dtype=np.float64)

    a = 24.00000
    b = 12.00000 / a
    c =  6.00000 / a
    gamma = np.cos(2*np.pi/3)

    cell_params = np.array(
        [a, b, c, gamma, 0, 0]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=12,
        cell_params=cell_params,
        primitive=True,
        espresso_like=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_qe_format_base_centered_monoclinic_unique_c():
    # CCl₄, modified
    ref_cell = np.array([
        [ 10.09050,  0.00000, -9.88050],
        [ -4.15242, 10.56314,  0.00000],
        [ 10.09050,  0.00000,  9.88050],
    ], dtype=np.float64)

    a = 20.18100
    b = 11.35000 / a
    c = 19.76100 / a
    gamma = np.cos(111.46*np.pi/180)

    cell_params = np.array(
        [a, b, c, gamma, 0, 0]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=13,
        cell_params=cell_params,
        primitive=True,
        espresso_like=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_qe_format_base_centered_monoclinic_unique_b():
    # CCl₄
    ref_cell = np.array([
        [ 10.09050, 5.67500,  0.00000],
        [-10.09050, 5.67500,  0.00000],
        [ -7.22959, 0.00000, 18.39103],
    ], dtype=np.float64)

    a = 20.18100
    b = 11.35000 / a
    c = 19.76100 / a
    beta = np.cos(111.46*np.pi/180)

    cell_params = np.array(
        [a, b, c, 0, beta, 0]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=-13,
        cell_params=cell_params,
        primitive=True,
        espresso_like=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


def test_qe_format_triclinic():
    # Montbrayite, Au₂Te₃
    ref_cell = np.array([
        [10.80000,  0.00000,  0.00000],
        [-1.59507, 11.99441,  0.00000],
        [-3.38831, -4.62223, 12.17892],
    ], dtype=np.float64)

    a = 10.80000
    b = 12.10000 / a
    c = 13.46000 / a
    alpha = np.cos(107.892*np.pi/180)
    beta  = np.cos(104.58*np.pi/180)
    gamma = np.cos(97.575*np.pi/180)

    cell_params = np.array(
        [a, b, c, alpha, beta, gamma]
    )
    gen_cell = Geometry.generate_lattice(
        bravais_index=14,
        cell_params=cell_params,
        primitive=True,
        espresso_like=True,
    )

    np.testing.assert_array_almost_equal(gen_cell, ref_cell, decimal=5)


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
    temp_dir.mkdir(exist_ok=True)

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
    temp_dir.mkdir(exist_ok=True)

    xsf_path = temp_dir / Path("test.xsf")

    xsf_path.write_text(xsf, encoding="utf-8")

    geometry = Geometry.from_xsf(xsf_path)

    assert(geometry.get_elements() == elements)
    np.testing.assert_array_equal(geometry.get_coords(), xyzs)
    np.testing.assert_array_equal(geometry.lat_vec, lat_vec)
    assert(geometry.alat == alat)


def test_from_orca_opt():
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
        __file__ + "/../../notebook/example_outputs/water_random_rotation.out"
    ).resolve()

    geometry = Geometry.from_orca(orca_output_path)

    assert(geometry.get_elements() == elements)
    np.testing.assert_array_equal(geometry.get_coords(), final_xyzs)


def test_from_orca_scf():
    elements = [
        Element.Oxygen,
        Element.Hydrogen,
        Element.Hydrogen,
    ]

    final_xyzs = np.array([
        [-0.001484,  0.389368, 0.000000],
        [ 0.760953, -0.191788, 0.000000],
        [-0.759469, -0.197581, 0.000000],
    ], dtype=np.float64)

    orca_output_path = Path(
        __file__ + "/../files/water_scf.out"
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
    temp_dir.mkdir(exist_ok=True)

    cube_path = temp_dir / Path("test.cube")

    cube_path.write_text(cube, encoding="utf-8")

    geometry = Geometry.from_cube(cube_path)

    assert(geometry.get_elements() == elements)
    np.testing.assert_allclose(geometry.get_coords(), xyzs, atol=1e-8)
    np.testing.assert_allclose(geometry.lat_vec, lat_vec, atol=1e-7)
    np.testing.assert_almost_equal(geometry.alat,  alat, decimal=8)


def test_from_qe_pp_ibrav(tmp_path):
    """Test read from a QE post-processing output with ibrav!=0"""
    elements = [
        Element.Oxygen,
        Element.Hydrogen,
        Element.Hydrogen,
    ]

    xyzs = np.array([
        [12.75798555, 13.05368543, 12.76021096],
        [13.52042186, 12.47252963, 12.76021096],
        [12.00000006, 12.46673650, 12.76021096],
    ], dtype=np.float64)

    lat_vec = np.array([
        [25.52042192,  0.00000000,  0.00000000],
        [ 0.00000000, 25.52042192,  0.00000000],
        [ 0.00000000,  0.00000000, 25.52042192],
    ], dtype=np.float64)

    alat = 25.520421921897817

    ibrav_pp = (
        "                                                                           \n"
        "     320     320     320     320     320     320       3       2\n"
        "     1       48.22660805      0.00000000      0.00000000      0.00000000      0.00000000      0.00000000\n"
        "    23565.3388878648        4.0000000000      100.0000000000     0\n"
        "   1   H     1.00\n"
        "   2   O     6.00\n"
        "   1       0.499912799    0.511499593    0.500000000    2\n"
        "   2       0.529788336    0.488727407    0.500000000    1\n"
        "   3       0.470211664    0.488500407    0.500000000    1\n"
        " -2.299083350E-07 -7.838054251E-08  2.006116960E-07  1.688000881E-08 -4.123357949E-08\n"
        "  1.884541410E-07 -6.697910057E-09 -1.293898826E-07 -5.921296723E-08  2.149459995E-07\n"
        "  1.274520306E-07 -7.357167995E-08  2.090725496E-08 -3.161760662E-09 -1.262719220E-07\n"
        " -1.857340482E-07  1.293451047E-07  7.212858794E-08 -1.506511456E-07 -6.644561901E-08\n"
        "  7.335879733E-08  7.680247865E-08 -3.131195191E-08 -5.796952834E-09 -2.667106558E-08\n"
    )

    temp_dir = tmp_path / Path("test_files")
    temp_dir.mkdir(exist_ok=True)

    ibrav_path = temp_dir / Path("test_ibrav.pp")

    ibrav_path.write_text(ibrav_pp, encoding="utf-8")

    ibrav_geometry = Geometry.from_qe_pp(ibrav_path)

    assert(ibrav_geometry.get_elements() == elements)
    np.testing.assert_allclose(ibrav_geometry.get_coords(), xyzs, atol=1e-8)
    np.testing.assert_allclose(ibrav_geometry.lat_vec, lat_vec, atol=1e-7)
    np.testing.assert_almost_equal(ibrav_geometry.alat,  alat, decimal=8)


def test_from_qe_pp_cell(tmp_path):
    """Test read from a QE post-processing output with ibrav=0"""
    elements = [
        Element.Oxygen,
        Element.Hydrogen,
        Element.Hydrogen,
    ]

    xyzs = np.array([
        [12.75798555, 13.05368543, 12.76021096],
        [13.52042186, 12.47252963, 12.76021096],
        [12.00000006, 12.46673650, 12.76021096],
    ], dtype=np.float64)

    lat_vec = np.array([
        [25.52042192,  0.00000000,  0.00000000],
        [ 0.00000000, 25.52042192,  0.00000000],
        [ 0.00000000,  0.00000000, 25.52042192],
    ], dtype=np.float64)

    alat = 25.520421921897817

    cell_pp = (
        "                                                                           \n"
        "     320     320     320     320     320     320       3       2\n"
        "     0       48.22660805      0.00000000      0.00000000      0.00000000      0.00000000      0.00000000\n"
        "   1.0000000000000000        0.0000000000000000        0.0000000000000000     \n"
        "   0.0000000000000000        1.0000000000000000        0.0000000000000000     \n"
        "   0.0000000000000000        0.0000000000000000        1.0000000000000000     \n"
        "    23565.3388878648        4.0000000000      100.0000000000     0\n"
        "   1   H     1.00\n"
        "   2   O     6.00\n"
        "   1       0.499912799    0.511499593    0.500000000    2\n"
        "   2       0.529788336    0.488727407    0.500000000    1\n"
        "   3       0.470211664    0.488500407    0.500000000    1\n"
        " -2.299083350E-07 -7.838054251E-08  2.006116960E-07  1.688000881E-08 -4.123357949E-08\n"
        "  1.884541410E-07 -6.697910057E-09 -1.293898826E-07 -5.921296723E-08  2.149459995E-07\n"
        "  1.274520306E-07 -7.357167994E-08  2.090725496E-08 -3.161760660E-09 -1.262719220E-07\n"
        " -1.857340482E-07  1.293451047E-07  7.212858794E-08 -1.506511456E-07 -6.644561900E-08\n"
        "  7.335879733E-08  7.680247866E-08 -3.131195191E-08 -5.796952831E-09 -2.667106558E-08\n"
    )

    temp_dir = tmp_path / Path("test_files")
    temp_dir.mkdir(exist_ok=True)

    cell_path = temp_dir / Path("test_cell.pp")

    cell_path.write_text(cell_pp, encoding="utf-8")

    cell_geometry = Geometry.from_qe_pp(cell_path)

    assert(cell_geometry.get_elements() == elements)
    np.testing.assert_allclose(cell_geometry.get_coords(), xyzs, atol=1e-8)
    np.testing.assert_allclose(cell_geometry.lat_vec, lat_vec, atol=1e-7)
    np.testing.assert_almost_equal(cell_geometry.alat,  alat, decimal=8)


def test_repr():
    ref_repr = (
        "Element     X          Y          Z          \n"
        "\n"
        "H           1.000000   2.000000   3.000000\n"
        "Ru          4.000000   5.000000   6.000000\n"
        "Br          7.000000   8.000000   9.000000\n"
    )

    ref_repr_crystal = (
        "Lattice     X          Y          Z          \n"
        "Vectors    \n"
        "           10.000000   0.000000   0.000000\n"
        "            0.000000  20.000000   0.000000\n"
        "            0.000000   0.000000  30.000000\n"
        "\n"
        "Element     X          Y          Z          \n"
        "\n"
        "H           1.000000   2.000000   3.000000\n"
        "Ru          4.000000   5.000000   6.000000\n"
        "Br          7.000000   8.000000   9.000000\n"
    )

    atoms = [
        Atom(Element.Hydrogen, [1.0, 2.0, 3.0]),
        Atom(Element.Ruthenium, [4.0, 5.0, 6.0]),
        Atom(Element.Bromine, [7.0, 8.0, 9.0]),
    ]

    lat_vec = np.array([
        [10.0,  0.0,  0.0],
        [ 0.0, 20.0,  0.0],
        [ 0.0,  0.0, 30.0],
    ], dtype=np.float64)

    alat = 10.0

    geometry = Geometry(atoms)
    
    assert(geometry.__repr__() == ref_repr)

    geometry.lat_vec = lat_vec
    geometry.alat = alat

    assert(geometry.__repr__() == ref_repr_crystal)


def test_eq():
    atoms = [
        Atom(Element.Hydrogen, [1.0, 2.0, 3.0]),
        Atom(Element.Ruthenium, [4.0, 5.0, 6.0]),
        Atom(Element.Bromine, [7.0, 8.0, 9.0]),
    ]

    lat_vec = np.array([
        [10.0,  0.0,  0.0],
        [ 0.0, 20.0,  0.0],
        [ 0.0,  0.0, 30.0],
    ], dtype=np.float64)

    alat = 10.0

    geom_1 = Geometry(atoms, lat_vec, alat)
    geom_2 = Geometry(atoms, lat_vec, alat)

    assert(geom_1 == geom_2)


def test_neq():
    atoms = [
        Atom(Element.Hydrogen, [1.0, 2.0, 3.0]),
        Atom(Element.Ruthenium, [4.0, 5.0, 6.0]),
        Atom(Element.Bromine, [7.0, 8.0, 9.0]),
    ]
    alt_atoms = [
        Atom(Element.Ruthenium, [1.0, 2.0, 3.0]),
        Atom(Element.Hydrogen, [4.0, 5.0, 6.0]),
        Atom(Element.Bromine, [7.0, 8.0, 9.0]),
    ]

    lat_vec = np.array([
        [10.0,  0.0,  0.0],
        [ 0.0, 20.0,  0.0],
        [ 0.0,  0.0, 30.0],
    ], dtype=np.float64)

    alat = 10.0

    geom_1 = Geometry(atoms, lat_vec, alat)
    geom_2 = Geometry(alt_atoms, lat_vec, alat)

    assert not (geom_1 == geom_2)

    geom_1 = Geometry(atoms, lat_vec, alat)
    geom_2 = Geometry(atoms, lat_vec+1, alat)

    assert not (geom_1 == geom_2)

    geom_1 = Geometry(atoms, lat_vec, alat)
    geom_2 = Geometry(atoms, lat_vec, alat+1)

    assert not (geom_1 == geom_2)

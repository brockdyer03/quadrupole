import pytest
from pathlib import Path
import numpy as np
from quadrupole import Quadrupole, Geometry


def test_create():
    ref_quad_3x1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ], dtype=np.float64)

    ref_quad_6x1 = np.array([
        [1.0, 4.0, 5.0],
        [4.0, 2.0, 6.0],
        [5.0, 6.0, 3.0],
    ], dtype=np.float64)

    quad_3x1 = Quadrupole([1.0, 2.0, 3.0])
    arr_6x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    quad_6x1 = Quadrupole(arr_6x1)

    np.testing.assert_array_equal(quad_3x1.quadrupole, ref_quad_3x1)
    np.testing.assert_array_equal(quad_6x1.quadrupole, ref_quad_6x1)

    assert(quad_6x1[0,0] == arr_6x1[0])
    assert(quad_6x1[1,1] == arr_6x1[1])
    assert(quad_6x1[2,2] == arr_6x1[2])
    
    assert(quad_6x1[0,1] == arr_6x1[3])
    assert(quad_6x1[0,2] == arr_6x1[4])
    assert(quad_6x1[1,2] == arr_6x1[5])

    assert(quad_6x1[1,0] == arr_6x1[3])
    assert(quad_6x1[2,0] == arr_6x1[4])
    assert(quad_6x1[2,1] == arr_6x1[5])


def test_inertialize():
    geometry = Geometry.from_list(
        elements=["O", "H", "H"],
        xyzs = np.array([
            [5.753633, 3.280382, 2.728205],
            [4.931827, 3.412032, 2.252440],
            [5.738252, 3.930645, 3.432457],
        ], dtype=np.float64)
    )

    ref_quadrupole = np.array([
        [-6.101031186, -0.123733580,  1.166262020],
        [-0.123733580, -6.753731413,  1.086471116],
        [ 1.166262020,  1.086471116, -5.748576943],
    ], dtype=np.float64)

    ref_inertial_quadrupole = np.array([
        [-7.825521987729e+00,  6.786133097222e-07, -1.320025557922e-06],
        [ 6.786133112167e-07, -6.303766572670e+00, -9.855315930344e-07],
        [-1.320025558800e-06, -9.855315923792e-07, -4.474050981601e+00],
    ], dtype=np.float64)

    quadrupole = Quadrupole(
        np.array([
            [-6.101031186, -0.123733580,  1.166262020],
            [-0.123733580, -6.753731413,  1.086471116],
            [ 1.166262020,  1.086471116, -5.748576943],
        ], dtype=np.float64)
    )
    inertial_quadrupole = quadrupole.inertialize(geometry)

    np.testing.assert_array_equal(quadrupole.quadrupole, ref_quadrupole)
    np.testing.assert_allclose(inertial_quadrupole.quadrupole, ref_inertial_quadrupole, atol=1e-10)


def test_detrace():
    ref_traceless_quadrupole = np.array([
        [-1.5,  0.0,  0.0],
        [ 0.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.5],
    ], dtype=np.float64)

    quadrupole = Quadrupole([1.0, 2.0, 3.0])

    traceless_quadrupole = quadrupole.detrace()

    np.testing.assert_allclose(
        traceless_quadrupole.quadrupole, ref_traceless_quadrupole, atol=1e-10
    )


def test_unit_convert():
    tol = 1e-11

    ref_quad_buckingham = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ], dtype=np.float64)
    ref_quad_au = np.array([
        [0.74347545954, 0.00000000000, 0.00000000000],
        [0.00000000000, 1.48695091909, 0.00000000000],
        [0.00000000000, 0.00000000000, 2.23042637863],
    ], dtype=np.float64)
    ref_quad_cm2 = np.array([
        [3.33564095198e-40, 0.00000000000e+00, 0.00000000000e+00],
        [0.00000000000e+00, 6.67128190396e-40, 0.00000000000e+00],
        [0.00000000000e+00, 0.00000000000e+00, 1.00069228559e-39],
    ], dtype=np.float64)
    ref_quad_esu = np.array([
        [1.0e-26, 0.0e+00, 0.0e+00],
        [0.0e+00, 2.0e-26, 0.0e+00],
        [0.0e+00, 0.0e+00, 3.0e-26],
    ], dtype=np.float64)

    quad_buckingham = Quadrupole(
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], dtype=np.float64),
        units="buckingham"
    )
    quad_au = Quadrupole(
        np.array([
            [0.74347545954, 0.00000000000, 0.00000000000],
            [0.00000000000, 1.48695091909, 0.00000000000],
            [0.00000000000, 0.00000000000, 2.23042637863],
        ], dtype=np.float64),
        units="au"
    )
    quad_cm2 = Quadrupole(
        np.array([
            [3.33564095198e-40, 0.00000000000e+00, 0.00000000000e+00],
            [0.00000000000e+00, 6.67128190396e-40, 0.00000000000e+00],
            [0.00000000000e+00, 0.00000000000e+00, 1.00069228559e-39],
        ], dtype=np.float64),
        units="cm2"
    )
    quad_esu = Quadrupole(
        np.array([
            [1.0e-26, 0.0e+00, 0.0e+00],
            [0.0e+00, 2.0e-26, 0.0e+00],
            [0.0e+00, 0.0e+00, 3.0e-26],
        ], dtype=np.float64),
        units="esu"
    )

    np.testing.assert_array_equal(quad_buckingham.quadrupole, ref_quad_buckingham)
    assert(quad_buckingham.units == "buckingham")
    np.testing.assert_array_equal(quad_au.quadrupole, ref_quad_au)
    assert(quad_au.units == "au")
    np.testing.assert_array_equal(quad_cm2.quadrupole, ref_quad_cm2)
    assert(quad_cm2.units == "cm2")
    np.testing.assert_array_equal(quad_esu.quadrupole, ref_quad_esu)
    assert(quad_esu.units == "esu")

    # From buckingham
    quad_au_from_buckingham = quad_buckingham.buck_to_au()
    quad_cm2_from_buckingham = quad_buckingham.buck_to_cm2()
    quad_esu_from_buckingham = quad_buckingham.buck_to_esu()

    assert(quad_au_from_buckingham.units == "au")
    assert(quad_cm2_from_buckingham.units == "cm2")
    assert(quad_esu_from_buckingham.units == "esu")
    np.testing.assert_allclose(quad_au_from_buckingham.quadrupole, ref_quad_au, atol=tol)
    np.testing.assert_allclose(quad_cm2_from_buckingham.quadrupole, ref_quad_cm2, atol=tol)
    np.testing.assert_allclose(quad_esu_from_buckingham.quadrupole, ref_quad_esu, atol=tol)

    # From esu
    quad_buckingham_from_esu = quad_esu.esu_to_buck()
    quad_au_from_esu = quad_esu.esu_to_au()
    quad_cm2_from_esu = quad_esu.esu_to_cm2()

    assert(quad_buckingham_from_esu.units == "buckingham")
    assert(quad_au_from_esu.units == "au")
    assert(quad_cm2_from_esu.units == "cm2")
    np.testing.assert_allclose(quad_buckingham_from_esu.quadrupole, ref_quad_buckingham, atol=tol)
    np.testing.assert_allclose(quad_au_from_esu.quadrupole, ref_quad_au, atol=tol)
    np.testing.assert_allclose(quad_cm2_from_esu.quadrupole, ref_quad_cm2, atol=tol)

    # From au
    quad_buckingham_from_au = quad_au.au_to_buck()
    quad_cm2_from_au = quad_au.au_to_cm2()
    quad_esu_from_au = quad_au.au_to_esu()

    assert(quad_buckingham_from_au.units == "buckingham")
    assert(quad_cm2_from_au.units == "cm2")
    assert(quad_esu_from_au.units == "esu")
    np.testing.assert_allclose(quad_buckingham_from_au.quadrupole, ref_quad_buckingham, atol=tol)
    np.testing.assert_allclose(quad_cm2_from_au.quadrupole, ref_quad_cm2, atol=tol)
    np.testing.assert_allclose(quad_esu_from_au.quadrupole, ref_quad_esu, atol=tol)

    # From cm2
    quad_buckingham_from_cm2 = quad_cm2.cm2_to_buck()
    quad_au_from_cm2 = quad_cm2.cm2_to_au()
    quad_esu_from_cm2 = quad_cm2.cm2_to_esu()

    assert(quad_buckingham_from_cm2.units == "buckingham")
    assert(quad_au_from_cm2.units == "au")
    assert(quad_esu_from_cm2.units == "esu")
    np.testing.assert_allclose(quad_buckingham_from_cm2.quadrupole, ref_quad_buckingham, atol=tol)
    np.testing.assert_allclose(quad_au_from_cm2.quadrupole, ref_quad_au, atol=tol)
    np.testing.assert_allclose(quad_esu_from_cm2.quadrupole, ref_quad_esu, atol=tol)


def test_as_unit():
    tol = 1e-11

    ref_quad_buckingham = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ], dtype=np.float64)
    ref_quad_au = np.array([
        [0.74347545954, 0.00000000000, 0.00000000000],
        [0.00000000000, 1.48695091909, 0.00000000000],
        [0.00000000000, 0.00000000000, 2.23042637863],
    ], dtype=np.float64)
    ref_quad_cm2 = np.array([
        [3.33564095198e-40, 0.00000000000e+00, 0.00000000000e+00],
        [0.00000000000e+00, 6.67128190396e-40, 0.00000000000e+00],
        [0.00000000000e+00, 0.00000000000e+00, 1.00069228559e-39],
    ], dtype=np.float64)
    ref_quad_esu = np.array([
        [1.0e-26, 0.0e+00, 0.0e+00],
        [0.0e+00, 2.0e-26, 0.0e+00],
        [0.0e+00, 0.0e+00, 3.0e-26],
    ], dtype=np.float64)

    quad_buckingham = Quadrupole(
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], dtype=np.float64),
        units="buckingham"
    )
    quad_au = Quadrupole(
        np.array([
            [0.74347545954, 0.00000000000, 0.00000000000],
            [0.00000000000, 1.48695091909, 0.00000000000],
            [0.00000000000, 0.00000000000, 2.23042637863],
        ], dtype=np.float64),
        units="au"
    )
    quad_cm2 = Quadrupole(
        np.array([
            [3.33564095198e-40, 0.00000000000e+00, 0.00000000000e+00],
            [0.00000000000e+00, 6.67128190396e-40, 0.00000000000e+00],
            [0.00000000000e+00, 0.00000000000e+00, 1.00069228559e-39],
        ], dtype=np.float64),
        units="cm2"
    )
    quad_esu = Quadrupole(
        np.array([
            [1.0e-26, 0.0e+00, 0.0e+00],
            [0.0e+00, 2.0e-26, 0.0e+00],
            [0.0e+00, 0.0e+00, 3.0e-26],
        ], dtype=np.float64),
        units="esu"
    )

    np.testing.assert_array_equal(quad_buckingham.quadrupole, ref_quad_buckingham)
    assert(quad_buckingham.units == "buckingham")
    np.testing.assert_array_equal(quad_au.quadrupole, ref_quad_au)
    assert(quad_au.units == "au")
    np.testing.assert_array_equal(quad_cm2.quadrupole, ref_quad_cm2)
    assert(quad_cm2.units == "cm2")
    np.testing.assert_array_equal(quad_esu.quadrupole, ref_quad_esu)
    assert(quad_esu.units == "esu")

    # From buckingham
    quad_au_from_buckingham = quad_buckingham.as_unit("au")
    quad_cm2_from_buckingham = quad_buckingham.as_unit("cm2")
    quad_esu_from_buckingham = quad_buckingham.as_unit("esu")

    assert(quad_au_from_buckingham.units == "au")
    assert(quad_cm2_from_buckingham.units == "cm2")
    assert(quad_esu_from_buckingham.units == "esu")
    np.testing.assert_allclose(quad_au_from_buckingham.quadrupole, ref_quad_au, atol=tol)
    np.testing.assert_allclose(quad_cm2_from_buckingham.quadrupole, ref_quad_cm2, atol=tol)
    np.testing.assert_allclose(quad_esu_from_buckingham.quadrupole, ref_quad_esu, atol=tol)

    # From esu
    quad_buckingham_from_esu = quad_esu.as_unit("buckingham")
    quad_au_from_esu = quad_esu.as_unit("au")
    quad_cm2_from_esu = quad_esu.as_unit("cm2")

    assert(quad_buckingham_from_esu.units == "buckingham")
    assert(quad_au_from_esu.units == "au")
    assert(quad_cm2_from_esu.units == "cm2")
    np.testing.assert_allclose(quad_buckingham_from_esu.quadrupole, ref_quad_buckingham, atol=tol)
    np.testing.assert_allclose(quad_au_from_esu.quadrupole, ref_quad_au, atol=tol)
    np.testing.assert_allclose(quad_cm2_from_esu.quadrupole, ref_quad_cm2, atol=tol)

    # From au
    quad_buckingham_from_au = quad_au.as_unit("buckingham")
    quad_cm2_from_au = quad_au.as_unit("cm2")
    quad_esu_from_au = quad_au.as_unit("esu")

    assert(quad_buckingham_from_au.units == "buckingham")
    assert(quad_cm2_from_au.units == "cm2")
    assert(quad_esu_from_au.units == "esu")
    np.testing.assert_allclose(quad_buckingham_from_au.quadrupole, ref_quad_buckingham, atol=tol)
    np.testing.assert_allclose(quad_cm2_from_au.quadrupole, ref_quad_cm2, atol=tol)
    np.testing.assert_allclose(quad_esu_from_au.quadrupole, ref_quad_esu, atol=tol)

    # From cm2
    quad_buckingham_from_cm2 = quad_cm2.as_unit("buckingham")
    quad_au_from_cm2 = quad_cm2.as_unit("au")
    quad_esu_from_cm2 = quad_cm2.as_unit("esu")

    assert(quad_buckingham_from_cm2.units == "buckingham")
    assert(quad_au_from_cm2.units == "au")
    assert(quad_esu_from_cm2.units == "esu")
    np.testing.assert_allclose(quad_buckingham_from_cm2.quadrupole, ref_quad_buckingham, atol=tol)
    np.testing.assert_allclose(quad_au_from_cm2.quadrupole, ref_quad_au, atol=tol)
    np.testing.assert_allclose(quad_esu_from_cm2.quadrupole, ref_quad_esu, atol=tol)


def test_unit_cycle():
    """Ensure that there is not an accumulation of floating
    point errors or anything tricksy
    """
    tol = 1e-11

    ref_quad_buckingham = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ], dtype=np.float64)

    quadrupole = Quadrupole(
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], dtype=np.float64),
        units="buckingham"
    )

    for i in range(5000):
        quadrupole = quadrupole.as_unit("cm2")
        quadrupole = quadrupole.as_unit("au")
        quadrupole = quadrupole.as_unit("esu")
        quadrupole = quadrupole.as_unit("buckingham")

    np.testing.assert_allclose(quadrupole.quadrupole, ref_quad_buckingham, atol=tol)
    assert(quadrupole.units == "buckingham")


def test_arithmetic():
    tol = 1e-11

    ref_one_minus_two = np.array([
        [-5.0,  0.0,  0.0],
        [ 0.0, -3.0,  0.0],
        [ 0.0,  0.0, -1.0],
    ], dtype=np.float64)
    ref_one_plus_two = np.array([
        [7.0, 0.0, 0.0],
        [0.0, 7.0, 0.0],
        [0.0, 0.0, 7.0],
    ], dtype=np.float64)

    quad_one = Quadrupole(
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], dtype=np.float64),
        units="buckingham"
    )
    quad_two = Quadrupole(
        np.array([
            [6.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 4.0],
        ], dtype=np.float64),
        units="buckingham"
    )

    one_minus_two = quad_one - quad_two
    one_plus_two = quad_one + quad_two

    np.testing.assert_allclose(one_minus_two.quadrupole, ref_one_minus_two, atol=tol)
    np.testing.assert_allclose(one_plus_two.quadrupole, ref_one_plus_two, atol=tol)


def test_from_orca():
    ref_quadrupole = np.array([
        [-6.101031186, -0.123733580,  1.166262020],
        [-0.123733580, -6.753731413,  1.086471116],
        [ 1.166262020,  1.086471116, -5.748576943],
    ], dtype=np.float64)

    orca_output_path = Path(
        __file__ + "/../files/water_random_rotation.out"
    ).resolve()

    quadrupole, = Quadrupole.from_orca(orca_output_path)

    np.testing.assert_array_equal(quadrupole.quadrupole, ref_quadrupole)
    assert(quadrupole.units == "buckingham")


def test_compare():
    tol = 1e-11

    ref_compared_quad = np.array([
        [-0.15398,  0.00000,  0.00000],
        [ 0.00000,  2.59059,  0.00000],
        [ 0.00000,  0.00000, -2.43661],
    ], dtype=np.float64)

    ref_diff = np.array([
        [-0.02398,  0.00000,  0.00000],
        [ 0.00000, -0.03941,  0.00000],
        [ 0.00000,  0.00000,  0.06339],
    ], dtype=np.float64)

    # We aren't making this a Quadrupole object so that we can ensure
    # that the compare function will properly convert it to a Quadrupole
    expt_quad = np.array([
        [-0.13,  0.00,  0.00],
        [ 0.00,  2.63,  0.00],
        [ 0.00,  0.00, -2.50],
    ], dtype=np.float64)

    calc_quad = Quadrupole(
        np.array([
            [-2.43661,  0.00000,  0.00000],
            [ 0.00000, -0.15398,  0.00000],
            [ 0.00000,  0.00000,  2.59059],
        ], dtype=np.float64),
        units="buckingham"
    )

    compared_quad = calc_quad.compare(expt=expt_quad)
    diff = compared_quad - Quadrupole(expt_quad)

    np.testing.assert_allclose(compared_quad.quadrupole, ref_compared_quad, atol=tol)
    np.testing.assert_allclose(diff.quadrupole, ref_diff, atol=tol)


def test_compare_mismatched_signs():
    tol = 1e-11

    ref_compared_quad = np.array([
        [ 0.15398,  0.00000,  0.00000],
        [ 0.00000, -2.59059,  0.00000],
        [ 0.00000,  0.00000,  2.43661],
    ], dtype=np.float64)

    ref_diff = np.array([
        [ 0.28398,  0.00000,  0.00000],
        [ 0.00000, -5.22059,  0.00000],
        [ 0.00000,  0.00000,  4.93661],
    ], dtype=np.float64)

    expt_quad = Quadrupole(
        np.array([
            [-0.13,  0.00,  0.00],
            [ 0.00,  2.63,  0.00],
            [ 0.00,  0.00, -2.50],
        ], dtype=np.float64),
        units="buckingham"
    )

    calc_quad = Quadrupole(
        np.array([
            [ 2.43661,  0.00000,  0.00000],
            [ 0.00000,  0.15398,  0.00000],
            [ 0.00000,  0.00000, -2.59059],
        ], dtype=np.float64),
        units="buckingham"
    )

    compared_quad = calc_quad.compare(expt=expt_quad)
    diff = compared_quad - expt_quad

    np.testing.assert_allclose(compared_quad.quadrupole, ref_compared_quad, atol=tol)
    np.testing.assert_allclose(diff.quadrupole, ref_diff, atol=tol)


def test_repr():
    buck_repr = (
        "Quadrupole (buckingham):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      \n"
        "                  Total:    1.00000    2.00000    3.00000    0.00000    0.00000    0.00000\n"
    )
    au_repr = (
        "Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      \n"
        "          Total:    0.74348    1.48695    2.23043    0.00000    0.00000    0.00000\n"
    )
    cm2_repr = (
        "Quadrupole (cm2):      (xx)          (yy)          (zz)          (xy)          (xz)          (yz)         \n"
        "           Total:   3.33564e-40   6.67128e-40   1.00069e-39   0.00000e+00   0.00000e+00   0.00000e+00\n"
    )
    esu_repr = (
        "Quadrupole (esu):      (xx)          (yy)          (zz)          (xy)          (xz)          (yz)         \n"
        "           Total:   1.00000e-26   2.00000e-26   3.00000e-26   0.00000e+00   0.00000e+00   0.00000e+00\n"
    )

    quad_buckingham = Quadrupole(
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], dtype=np.float64),
        units="buckingham"
    )
    quad_au = Quadrupole(
        np.array([
            [0.74347545954, 0.00000000000, 0.00000000000],
            [0.00000000000, 1.48695091909, 0.00000000000],
            [0.00000000000, 0.00000000000, 2.23042637863],
        ], dtype=np.float64),
        units="au"
    )
    quad_cm2 = Quadrupole(
        np.array([
            [3.33564095198e-40, 0.00000000000e+00, 0.00000000000e+00],
            [0.00000000000e+00, 6.67128190396e-40, 0.00000000000e+00],
            [0.00000000000e+00, 0.00000000000e+00, 1.00069228559e-39],
        ], dtype=np.float64),
        units="cm2"
    )
    quad_esu = Quadrupole(
        np.array([
            [1.0e-26, 0.0e+00, 0.0e+00],
            [0.0e+00, 2.0e-26, 0.0e+00],
            [0.0e+00, 0.0e+00, 3.0e-26],
        ], dtype=np.float64),
        units="esu"
    )

    assert(quad_buckingham.__repr__() == buck_repr)
    assert(quad_au.__repr__() == au_repr)
    assert(quad_cm2.__repr__() == cm2_repr)
    assert(quad_esu.__repr__() == esu_repr)

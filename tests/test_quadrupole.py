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
    quad_6x1 = Quadrupole([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    np.testing.assert_array_equal(quad_3x1.quadrupole, ref_quad_3x1)
    np.testing.assert_array_equal(quad_6x1.quadrupole, ref_quad_6x1)


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
    np.testing.assert_allclose(inertial_quadrupole.quadrupole, ref_inertial_quadrupole, rtol=1e-10)


def test_detrace():
    ref_traceless_quadrupole = np.array([
        [-1.5,  0.0,  0.0],
        [ 0.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.5],
    ], dtype=np.float64)

    quadrupole = Quadrupole([1.0, 2.0, 3.0])

    traceless_quadrupole = quadrupole.detrace()

    np.testing.assert_allclose(
        traceless_quadrupole.quadrupole, ref_traceless_quadrupole, rtol=1e-10
    )

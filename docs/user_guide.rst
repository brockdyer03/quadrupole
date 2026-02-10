Quadrupole User Guide
=====================

Installation
------------

This package can be installed with any package manager that has access to the `Python Package Index <https://pypi.org/>`__. We recommend using `uv` to install, either in a virtual environment:

.. code-block:: bash

    > uv venv
    ...
    > source .venv/bin/activate
    > uv pip install quadrupole

or adding to a project:

.. code-block:: bash

    > uv init
    ...
    > uv add quadrupole


Getting Started
---------------

Once you have everything installed, you can start using Quadrupole!

The primary focus of this package is to enable the efficient manipulation and comparison of molecular quadrupole moments. Often, however, when you are working with molecular quadrupoles you will find yourself in need of the molecule's geometry, for operations such as rotating the quadrupole into the inertial frame of the molecule. We provide a number of methods in the ``Geometry`` class to read in a range of file formats, including

* XYZ (``.xyz``)
* XCrySDen Structure File (``.xsf``)
* Gaussian Cube (``.cube``)
* Quantum ESPRESSO Post-Processing Format (``.pp``)
* ORCA Outputs (``.out``)

We additionally support creating ``Geometry`` objects from a list of `elements <element>` and an array of coordinates.

The ``Geometry`` class is primarily built to support reading geometries from files, however it has one valuable method outside of file IO operations, ``calc_principal_moments``. This function will calculate the eigenvalues and eigenvectors of the inertia tensor of a given geometry. We use the following definition of the inertia tensor:

.. math::

    \textbf{I} = \sum_j \begin{bmatrix}
        m_j \left( y^2_j+z^2_j \right) & -m_j x_j y_j                   & -m_j x_j z_j \\
        -m_j y_j x_j                   & m_j \left( x^2_j+z^2_j \right) & -m_j y_j z_j \\
        -m_j z_j x_j                   & -m_j z_j y_j                   & m_j \left( x^2_j+y^2_j \right)
    \end{bmatrix}


For the eigenvalues and eigenvectors, we use NumPy's `np.linalg.eig <https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html>`__ function
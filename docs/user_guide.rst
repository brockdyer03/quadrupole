.. _user-guide:

User Guide
==========

.. toctree::
    :maxdepth: 1


.. _installation:

Installation
------------

This package can be installed with any package manager that has access to the `Python Package Index <https://pypi.org/>`__. We recommend using `uv <https://docs.astral.sh/uv/>`__ to install, either in a virtual environment:

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


.. _getting-started:

Getting Started
---------------

Once you have everything installed, you can start using Quadrupole!

The primary focus of this package is to enable the efficient manipulation and comparison of molecular quadrupole moments. Often, however, when you are working with molecular quadrupoles you will find yourself in need of the molecule's geometry, for operations such as rotating the quadrupole into the inertial frame of the molecule. We provide a number of methods in the :py:class:`Geometry` class to read in a range of file formats, including

* XYZ (``.xyz``)
* XCrySDen Structure File (``.xsf``)
* Gaussian Cube (``.cube``)
* Quantum ESPRESSO Post-Processing Format (``.pp``)
* ORCA Outputs (``.out``)

We additionally support creating :py:class:`Geometry` objects from a list of :ref:`elements <element-like>` and an array of coordinates.

A new :py:class:`Quadrupole` can be created in two ways. The first, and most direct, is to provide the class with a matrix or vector that represents a quadrupole moment:

.. code-block:: python

    >>> from quadrupole import Quadrupole
    >>> quad = Quadrupole([1, 2, 3], units="au")
    >>> print(quad)
    Quadrupole Moment (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
                     Total:    1.00000    2.00000    3.00000    0.00000    0.00000    0.00000
    >>> print(quad.quadrupole)
    [[1. 0. 0.]
     [0. 2. 0.]
     [0. 0. 3.]]

Here you can see that by supplying a sequence with length 3 the :py:class:`Quadrupole` class has used it to populate the diagonal elements of the quadrupole tensor. You can also create a :py:class:`Quadrupole` by supplying the 6 independent elements of the matrix in the order that the class would print them:

.. code-block:: python

    >>> from quadrupole import Quadrupole
    >>> quad = Quadrupole([1, 2, 3, 4, 5, 6], units="au")
    >>> print(quad)
    Quadrupole Moment (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
                     Total:    1.00000    2.00000    3.00000    4.00000    5.00000    6.00000
    >>> print(quad.quadrupole)
    [[1. 4. 5.]
     [4. 2. 6.]
     [5. 6. 3.]]

Here you will notice that not only has the function populated the upper right triangular portion of the matrix, but also the lower left portion. This is because the quadrupole tensor has, by definition, guaranteed transposition symmetry, therefore the :math:`xy` component will always be equal to the :math:`yx` component.
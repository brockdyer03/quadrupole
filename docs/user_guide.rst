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

Geometry Creation and IO
^^^^^^^^^^^^^^^^^^^^^^^^

The primary focus of this package is to enable the efficient manipulation and comparison of molecular quadrupole moments. Often, however, when you are working with molecular quadrupoles you will find yourself in need of the molecule's geometry, for operations such as rotating the quadrupole into the inertial frame of the molecule. We provide a number of methods in the :py:class:`Geometry` class to read in a range of file formats, including

* XYZ (``.xyz``)
* XCrySDen Structure File (``.xsf``)
* Gaussian Cube (``.cube``)
* Quantum ESPRESSO Post-Processing Format (``.pp``)
* ORCA Outputs (``.out``)

We additionally support creating :py:class:`Geometry` objects from a list of :ref:`elements <element-like>` and an array of coordinates. As an example, here we will look at reading in a molecular geometry, both with and without a unit cell.

The simplest file format that we support is the `XYZ format <https://openbabel.org/docs/FileFormats/XYZ_cartesian_coordinates_format.html>`__:

.. code-block:: python

    >>> from pathlib import Path
    >>> from quadrupole import Geometry
    >>> xyz_path = Path("tests/files/water_random.xyz")
    >>> geom = Geometry.from_xyz(xyz_path)
    >>> print(geom)
    Element     X          Y          Z          

    O           5.753633   3.280382   2.728205
    H           4.931827   3.412032   2.252440
    H           5.738252   3.930645   3.432457

In the event that an XYZ file is improperly formatted, a :py:class:`FileFormatError` will be raised that prints the file path and in most cases a description of what the formatting error was.

.. note::

    All file IO operations in the :py:class:`Geometry` class are designed to convert the units of the file to Ångstrom if their original format is not Ångstrom, however formats like the XYZ format are specified to always be in Ångstrom, so there are no conversions done. If your file is in another distance unit, you are responsible for ensuring the units are correct elsewhere!

For file formats that contain unit cell information, such as ``.xsf`` files, the :py:class:`Geometry` class will store the lattice vectors in addition to the atomic coordinates. Additionally, the :py:meth:`Geometry.__repr__` function will print the lattice vectors if they exist:

.. code-block:: python

    >>> from pathlib import Path
    >>> from quadrupole import Geometry
    >>> xsf_path = Path("tests/files/water.xsf")
    >>> geom = Geometry.from_xsf(xsf_path)
    >>> print(geom)
    Lattice     X          Y          Z          
    Vectors    
               17.543877   0.000000   0.000000
                0.000000  17.543877   0.000000
                0.000000   0.000000  17.543877

    Element     X          Y          Z          

    O           8.770452   9.174908   8.771938
    H           9.544851   8.573414   8.771938
    H           8.000512   8.567493   8.771938

Currently the functions for reading XSF files are limited to the most simple due to lack of available examples, but this will be updated in the future to include a wider range of options.

Quadrupole Creation and IO
^^^^^^^^^^^^^^^^^^^^^^^^^^

A new :py:class:`Quadrupole` can be created in two ways. The first, and most direct, is to provide the class with a matrix or vector that represents a quadrupole moment:

.. code-block:: python

    >>> from quadrupole import Quadrupole
    >>> quad = Quadrupole([1, 2, 3], units="au")
    >>> print(quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
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
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:    1.00000    2.00000    3.00000    4.00000    5.00000    6.00000

    >>> print(quad.quadrupole)
    [[1. 4. 5.]
     [4. 2. 6.]
     [5. 6. 3.]]

Here you will notice that not only has the function populated the upper right triangular portion of the matrix, but also the lower left portion. This is because the quadrupole tensor has, by definition, guaranteed transposition symmetry, therefore the :math:`xy` component will always be equal to the :math:`yx` component.

.. note::

    In contrast to supplying the 6 unique components of the tensor, supplying only the diagonal components makes no assumption about the remaining, unspecified components.

Though it is often redundant, for completeness you can also create a quadrupole moment by providing a complete 3x3 matrix:

.. code-block:: python

    >>> quad = Quadrupole(
    ...     quadrupole=[
    ...         [1, 4, 5],
    ...         [4, 2, 6],
    ...         [5, 6, 3],
    ...     ],
    ...     units="au"
    ... )
    >>> print(quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:    1.00000    2.00000    3.00000    4.00000    5.00000    6.00000
    
    >>> print(quad.quadrupole)
    [[1. 4. 5.]
     [4. 2. 6.]
     [5. 6. 3.]]

.. warning::

    Supplying the full 3x3 matrix can leave you open to accidentally creating a quadrupole that does not have transposition symmetry. This is an unphysical result, and can be avoided by only providing the non-redundant components.

In addition to directly supplying the components of the quadrupole matrix, we also offer support for reading in quadrupoles from `ORCA output files <https://orca-manual.mpi-muelheim.mpg.de/contents/spectroscopyproperties/electric.html>`__. These can be read in by providing a ``Path`` to an ORCA output, like so:

.. code-block:: python

    >>> from pathlib import Path
    >>> from quadrupole import Quadrupole
    >>> output_path = Path("tests/files/water_scf.out")
    >>> quad = Quadrupole.from_orca(output_path)[0]
    >>> print(quad)
    Quadrupole (buckingham):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
                      Total:   -4.54041   -6.36728   -7.87200    0.00696    0.00000    0.00000

You may notice that we have accessed the item at index zero after reading the quadrupole moment from the ORCA output. This is because the function :py:meth:`Quadrupole.from_orca()` always has the return type ``tuple[Quadrupole]``, which was implemented for consistency as ORCA outputs can contain several quadrupole moments, each corresponding to a different level of theory.

.. note::

    When reading ORCA outputs, we always pull the quadrupole moment in units of buckingham. This is primarily for historical reasons as the majority of available experimental data is provided only in buckingham.

It is also worth mentioning here that we always pull the total, non-diagonalized quadrupole tensor(s) from an ORCA output. This is because the current method by which ORCA diagonalizes the quadrupole matrix is by transforming it into the coordinate system of its eigenvectors, guaranteeing that it will be diagonal. This, however, is not always reflective of how one should obtain a quadrupole moment to compare to experimental values.

An important feature in the Quadrupole package is the ability to easily switch between the 4 main units of molecular quadrupole moments

* Atomic Units (a.u., :math:`\text{e}\text{a}_0`)
* Buckingham (B, :math:`10^{-26}\ \text{statC}\times\text{cm}^2`)
* Coulomb-meters squared (:math:`\text{C}\times\text{m}^2`)
* statCoulomb-centimeters squared (e.s.u., :math:`\text{statC}\times\text{cm}^2`)

This is easily accomplished by calling the :py:meth:`Quadrupole.as_unit()` method, and providing a unit that you would like to switch to. The accepted names for each unit are

* ``"au"``
* ``"buckingham"``
* ``"cm2"``
* ``"esu"``

The :py:meth:`Quadrupole.as_unit()` method will automatically determine the correct conversions to apply, no matter what your starting unit is, and return the quadrupole in the new units.

Due to the greatly disparate scales for some units, we provide two possible :py:meth:`Quadrupole.__repr__()` output formats for the quadrupole moment. If your units are buckingham or atomic units, we simply print the quadrupole moments as floats:

.. code-block:: python

    >>> print(quad)
    Quadrupole (buckingham):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
                      Total:   -4.54041   -6.36728   -7.87200    0.00696    0.00000    0.00000
    
    >>> print(quad.as_unit("au"))
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -3.37568   -4.73391   -5.85264    0.00518    0.00000    0.00000

For :math:`\text{C}\times\text{m}^2` and :math:`\text{statC}\times\text{cm}^2` (e.s.u.), we print the quadrupole moment in scientific notation:

.. code-block:: python

    >>> print(quad.as_unit("cm2"))
    Quadrupole (cm2):      (xx)          (yy)          (zz)          (xy)          (xz)          (yz)         
               Total:  -1.51452e-39  -2.12390e-39  -2.62582e-39   2.32188e-42   0.00000e+00   0.00000e+00

    >>> print(quad.as_unit("esu"))
    Quadrupole (esu):      (xx)          (yy)          (zz)          (xy)          (xz)          (yz)         
               Total:  -4.54041e-26  -6.36728e-26  -7.87200e-26   6.96082e-29   0.00000e+00   0.00000e+00


Quadrupole Analysis
^^^^^^^^^^^^^^^^^^^

When analyzing calculated molecular quadrupole moments, there are several useful methods provided in the Quadrupole package. Perhaps the most useful of which is the ability to automatically transform the quadrupole matrix into the inertial frame of the molecule. This can be accomplished in three steps, first create or read in the quadrupole moment, then create or read in the corresponding geometry, and finally call :py:meth:`Quadrupole.inertialize()`.

.. code-block:: python

    >>> from pathlib import Path
    >>> from quadrupole import Quadrupole, Geometry
    >>> orca_output = Path("tests/files/water_random_rotation.out")
    >>> quad = Quadrupole.from_orca(orca_output)[0]
    >>> quad = quad.as_unit("au") # atomic units fit better on-screen
    >>> print(quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -4.53597   -5.02123   -4.27393   -0.09199    0.86709    0.80776

    >>> geom = Geometry.from_orca(orca_output)
    >>> print(geom)
    Element     X          Y          Z          

    O           5.753633   3.280382   2.728205
    H           4.931827   3.412032   2.252440
    H           5.738252   3.930645   3.432457

    >>> quad = quad.inertialize(geom)
    >>> print(quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -5.81808   -4.68670   -3.32635    0.00000   -0.00000   -0.00000

Perhaps the most important feature of transforming the quadrupole into the inertial frame of the molecule is that, for this molecule, the tensor became diagonal. This is due to the higher-than-average symmetry of water (point group :math:`\text{C}_\text{2v}`), and thus will not necessarily be observed for all molecules.

This quadrupole moment is, however, not able to be compared directly to experimental values quite yet. For that, we need to detrace the quadrupole tensor. Again, we provide a simple method for this, :py:meth:`Quadrupole.detrace()`, which returns the traceless form of the quadrupole moment. Reusing the quadrupole from above,

.. code-block:: python

    >>> quad = quad.detrace()
    >>> print(quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -1.81156   -0.11448    1.92604    0.00000   -0.00000   -0.00000

    >>> print(quad[0,0] + quad[1,1] + quad[2,2]) # Trace of the matrix
    0.0

.. warning::

    The detracing operation is not reversible, while all other provided operations can be reversed. If you are going to serialize quadrupole data, it is strongly recommended to not detrace your quadrupoles before serializing the data, as you can not recover the original matrix!

Now that the quadrupole moment is both inertialized and traceless, it is suitable for comparison to literature. Using the value from `Flygare and Benson (1971) <https://doi.org/10.1080/00268977100100221>`__, we get a rather startling discrepancy:

.. code-block:: python

    >>> print(quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -1.81156   -0.11448    1.92604    0.00000   -0.00000   -0.00000

    >>> expt_quad = Quadrupole([-0.13, 2.63, -2.50])
    >>> expt_quad = expt_quad.as_unit("au")
    >>> print(expt_quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -0.09665    1.95534   -1.85869    0.00000    0.00000    0.00000

    >>> print(quad - expt_quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -1.71491   -2.06982    3.78473    0.00000   -0.00000   -0.00000

It looks like our quadrupole is nowhere near the experimental value, but how is this possible?

Some of you may have noticed that while the quadrupoles indeed do provide incredibly high deviations on direct subtraction, if we simply permuted the quadrupole such that :math:`xx \rightarrow yy \rightarrow zz`, we would be in a much better situation. Indeed, if we perform this permutation and attempt our subtraction again, we get a much more reasonable deviation:

.. code-block:: python

    >>> print(quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -1.81156   -0.11448    1.92604    0.00000   -0.00000   -0.00000

    >>> permuted_quad = Quadrupole([quad[1,1], quad[2,2], quad[0,0]], units="au")
    >>> print(permuted_quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -0.11448    1.92604   -1.81156    0.00000    0.00000    0.00000

    >>> print(permuted_quad - expt_quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -0.01783   -0.02930    0.04713    0.00000    0.00000    0.00000

While this operation is trivial to recognize for a single molecule and a single quadrupole moment, if one wished to perform this operation for a dataset of even just a few dozen quadrupole moments the process could take hours. It is for this reason that we provide an alternate route through semi-empirical statistical analysis. The function :py:meth:`Quadrupole.compare()` accepts one argument (other than ``self``), ``expt_quad``, and compares 6 permutations of the calculated quadrupole matrix to the experimental matrix, then selects the permutation with the both the lowest overall deviation from the experimental quadrupole and with the lowest standard deviation. Additionally, if the signs of the quadrupole moments differ (e.g. one has the signs [+,-,-] and the other is [+,+,-]), the function will temporarily negative the calculated quadrupole for the comparison, then return it to normal before returning the best match.

Using this function with the above example, we get the following output:

.. code-block:: python

    >>> print(quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -1.81156   -0.11448    1.92604    0.00000   -0.00000   -0.00000

    >>> print(expt_quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -0.09665    1.95534   -1.85869    0.00000    0.00000    0.00000

    >>> matched_quad = quad.compare(expt_quad)
    >>> print(matched_quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -0.11448    1.92604   -1.81156    0.00000    0.00000    0.00000

    >>> print(matched_quad - expt_quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -0.01783   -0.02930    0.04713    0.00000    0.00000    0.00000

As expected, we get the same result as our by-hand comparison. In the case the two quadrupoles differ by a sign flip, we get the result which most closely matches the magnitudes of the quadrupole moments:

.. code-block:: python

    >>> print(quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -1.81156   -0.11448    1.92604    0.00000   -0.00000   -0.00000

    >>> quad.quadrupole *= -1
    >>> print(quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:    1.81156    0.11448   -1.92604   -0.00000    0.00000    0.00000

    >>> print(expt_quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -0.09665    1.95534   -1.85869    0.00000    0.00000    0.00000

    >>> matched_quad = quad.compare(expt_quad)
    >>> print(matched_quad)
    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:    0.11448   -1.92604    1.81156    0.00000    0.00000    0.00000

.. caution::

    The :py:meth:`Quadrupole.compare()` function **does not** guarantee the correct permutation of the quadrupole matrix. It simply provides the form which an outside observer would most likely say matches the experimental quadrupole moment.

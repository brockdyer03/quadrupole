.. _quadrupole-theory:

######
Theory
######

The theory is here-ey.

.. _multipole-expansion:

Multipole Expansion
===================

The multipole expansion is a powerful tool for representing arbitrary charge distributions :math:`\rho(\textbf{r})` as the sum of discrete moments (the multipoles), similar to how one may represent a transcendental function such as :math:`e^x` as the infinite series :math:`\sum_{\ell=0}^{\infty}x^\ell / \ell!` [exp]_.

These multipoles are commonly referred to by the value of :math:`\ell`, with prefixes given by :math:`2^\ell`. This means the :math:`2^{\ell=0} = 1` term is called the monopole moment, :math:`\ell = 1` the dipole moment, :math:`\ell = 2` the quadrupole moment. There are two higher order moments with names, the octupole and hexadecapole, but after this the multipoles are named by their binary prefix, e.g. the 32-pole, 64-pole, 128-pole, and so on. In literature you are unlikely to encounter multipoles beyond the quadrupole, and even more unlikely to encounter those beyond the octupole. For this reason we will largely limit our discussion to the monopole, dipole, and quadrupole, with only a brief mention of the octupole.

.. _monopole-dipole:

Monopole and Dipole Moment
--------------------------

The monopole moment :math:`q` is the simplest of the multipole moments, and is calculated by integration of the charge density:

.. math::

    q = \int \rho(\textbf{r}) d^3\textbf{r}

This moment is simply the total charge of the system. The dipole moment is the first moment that includes a geometric component, and is calculated by the distance-weighted integration of the charge density:

.. math::

    \mu_{\alpha} = \int r_{\alpha}\rho(\textbf{r}) d^3\textbf{r}\quad (\alpha = x,y,z)

This multipole produces a vector :math:`\mu = (\mu_x, \mu_y, \mu_z)` that points from concentrations of positive charge towards concentrations of negative charge. Experimental measurements of the dipole moment are often limited to the magnitude of the dipole, given by the Euclidean norm of the dipole vector:

.. math::

    \mu = \sqrt{\mu_{x}^2 + \mu_{y}^2 + \mu_{z}^2}




.. _inertia-tensor:

Inertia Tensor and Eigenvectors
===============================

The inertia tensor :math:`\textbf{I}` of a molecule with its center of mass at the origin is given as

.. math::

    \textbf{I} = \sum_j \begin{bmatrix}
        m_j \left( y^2_j+z^2_j \right) & -m_j x_j y_j                   & -m_j x_j z_j \\
        -m_j y_j x_j                   & m_j \left( x^2_j+z^2_j \right) & -m_j y_j z_j \\
        -m_j z_j x_j                   & -m_j z_j y_j                   & m_j \left( x^2_j+y^2_j \right)
    \end{bmatrix}


with the index :math:`j` running over all atoms and :math:`m` being their mass. For samples with standard isotopic distributions the masses are the average atomic masses and can be accessed by the function :py:meth:`Geometry.get_atomic_mass()`. Due to the transposition symmetry of the inertia tensor (:math:`\textbf{I}_{\alpha\beta} = \textbf{I}_{\beta\alpha}`), one need only calculate the upper right (or lower left) triangular portion of the tensor, simplifying the calculations to

.. math::

    \textbf{I}_{xx} &= \sum_j^N m_j \left( y'^2_j+z'^2_j \right) \\
    \textbf{I}_{yy} &= \sum_j^N m_j \left( x'^2_j+z'^2_j \right) \\
    \textbf{I}_{zz} &= \sum_j^N m_j \left( x'^2_j+y'^2_j \right) \\
    \textbf{I}_{xy} &= \textbf{I}_{yx} = -\sum_j^N m_j x'_j y'_j \\
    \textbf{I}_{xz} &= \textbf{I}_{zx} = -\sum_j^N m_j x'_j z'_j \\
    \textbf{I}_{yz} &= \textbf{I}_{zy} = -\sum_j^N m_j y'_j z'_j \\

and for a system with a center of mass :math:`\textbf{R}_\alpha = (\textbf{R}_x,\quad \textbf{R}_y,\quad \textbf{R}_z)` given by

.. math::

    \textbf{R}_\alpha = \frac{1}{M}\sum_{j} m_j * \textbf{r}_j;\quad M = \sum_{j} m_j

that is not at the origin, set :math:`(x'_j,\quad y'_j,\quad z'_j) = (x_j-\textbf{R}_x,\quad y_j-\textbf{R}_y,\quad z_j-\textbf{R}_z)`.

.. _detracing-operator:

Detracing Operation
-------------------

The quadrupole tensor is a 3x3 matrix with transposition symmetry. These can be calculated as

.. math::

    \Theta_{\alpha\beta} = \sum_i e_i\textbf{r}_{i_\alpha}\textbf{r}_{i_\beta}

.. math::

    \mathbb{A}_{traceless} = \frac{3}{2}\left( \mathbb{A} - \mathbb{I}\frac{tr(\mathbb{A})}{3} \right)



.. [exp] https://en.wikipedia.org/wiki/Exponential_function
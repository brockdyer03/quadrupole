.. _quadrupole-theory:

######
Theory
######

In this document I describe some aspects of the theory of multipole moments and the math behind some of the functions in this package.

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

    q = \int \rho(\textbf{r})\, d^3\textbf{r}

This moment is simply the total charge of the system. The dipole moment is the first moment that includes a geometric component, and is calculated by the distance-weighted integration of the charge density:

.. math::

    \mu_{\alpha} = \int r_{\alpha}\rho(\textbf{r})\, d^3\textbf{r}\quad (\alpha = x,y,z)

This multipole produces a vector :math:`\mu = (\mu_x, \mu_y, \mu_z)` that points from concentrations of positive charge towards concentrations of negative charge. Experimental measurements of the dipole moment are often limited to the magnitude of the dipole, given by the Euclidean norm of the dipole vector:

.. math::

    \mu = \sqrt{\mu_{x}^2 + \mu_{y}^2 + \mu_{z}^2}

The dipole moment vanishes for any molecule whose point group contains a center of inversion and/or belongs to a cubic point group (:math:`T`, :math:`O`, and :math:`I`) [GeThWe95]_. Point groups that contain a center of inversion are listed here (with :math:`\text{N}\in \mathbb{Z}`):

- :math:`C_i`
- :math:`D_{\infty h}`
- :math:`C_{(2\text{N})h}`
- :math:`D_{(2\text{N})h}`
- :math:`D_{(2\text{N}+1)d}`
- :math:`S_{(4\text{N}+2)}`
- :math:`T_h`
- :math:`T_d`
- :math:`O_h`
- :math:`I_h`


.. _quadrupole-moment:

Quadrupole Moment
-----------------

The quadrupole moment :math:`Q_{\alpha\beta}` is a rank-2 tensor with transposition symmetry (i.e. swapping the rows and columns has no effect) [Buck59]_. It is calculated as

.. math::

    Q_{\alpha\beta} = \int \rho(\textbf{r})r_\alpha r_\beta\, d^3\textbf{r}\quad (\alpha, \beta = x,y,z)

This produces a quadrupole tensor of the form

.. math::

    Q = \begin{pmatrix}
        Q_{xx} & Q_{xy} & Q_{xz} \\
        Q_{yx} & Q_{yy} & Q_{yz} \\
        Q_{zx} & Q_{zy} & Q_{zz}
    \end{pmatrix}

However, as the coordinate axes of the molecule are not always clearly defined, we will often refer to the :math:`x`, :math:`y`, and :math:`z` components as :math:`a`, :math:`b`, and :math:`c` to avoid implying a known directionality. In this notation, our quadrupole tensor appears

.. math::

    Q = \begin{pmatrix}
        Q_{aa} & Q_{ab} & Q_{ac} \\
        Q_{ba} & Q_{bb} & Q_{bc} \\
        Q_{ca} & Q_{cb} & Q_{cc}
    \end{pmatrix}

The quadrupole tensor is often shown in its traceless form, which we distinguish from the typical quadrupole tensor with the symbol :math:`\Theta_{\alpha\beta}`. The tracless quadrupole can be calculated in two ways, either directly from the original charge density :math:`\rho(\textbf{r})`:

.. math::

    \Theta_{\alpha\beta} = \frac{1}{2}\int \left( 3r_\alpha r_\beta - r^2 \delta_{\alpha\beta} \right) \rho_\text{diff}(\textbf{r})\,d^3\textbf{r}

or alternately, using a detracing operator:

.. math::

    \mathbb{A}_{traceless} = \frac{3}{2}\left( \mathbb{A} - \mathbb{I}\frac{tr(\mathbb{A})}{3} \right)

to get

.. math::

    \Theta_{\alpha\beta} = \frac{3}{2}\left(Q_{\alpha\beta} - \mathbb{I}\frac{tr(Q_{\alpha\beta})}{3}\right)

where the notation :math:`Q_{\gamma\gamma}` implies only the terms where :math:`\alpha=\beta`; in other words, only the diagonal terms.

.. _quadrupole-symmetry:

Symmetry Dependence
^^^^^^^^^^^^^^^^^^^

The symmetry of a molecule largely dictates the complexity of the quadrupole tensor [GeThWe95]_. The trivial case is a molecule belonging to a cubic point group (:math:`T`, :math:`O`, and :math:`I`) that, by definition, contains at least four threefold rotation axes and thus has no quadrupole moment. Some molecules with these point groups would be methane (:math:`T_d`), hexafluorophosphate (:math:`O_h`), and buckminsterfullerene (:math:`I_h`). Molecules with a rotation operation :math:`C_n` with :math:`n>2` will contain a quadrupole moment that is diagonal in the inertial frame of the molecule (see :ref:`Inertia Tensor <inertial-frame>` for an explanation of this) and has the components :math:`[Q_{\parallel}, Q_{\perp}, Q_{\perp}]`. For the traceless moment, this means :math:`\Theta_{\perp} = -\frac{1}{2}\Theta_{\parallel}`.

Any point group with a point group with more symmetry operations than :math:`C_s` have a diagonal tensor (i.e. :math:`Q_{\alpha\beta}=0` if :math:`\alpha \neq \beta`), but with :math:`Q_{aa} \neq Q_{bb} \neq Q_{cc}`. Those with :math:`C_s` or lower symmetry can contain at least one off-diagonal component :math:`Q_{ab}`.

.. _octupole-moment:

Octupole Moment
---------------

The octupole moment is a rank-3 tensor and is calculated as

.. math::

    R_{\alpha\beta\gamma} = \int \rho(\textbf{r}) r_\alpha r_\beta r_\gamma\, d^3\textbf{r}

The octupole moment also has a traceless form, given as

.. math::

    \Omega_{\alpha\beta\gamma} = \frac{1}{2}(5 R_{\alpha\beta\gamma} - R_{\alpha\epsilon\epsilon}\delta_{\beta\gamma} - R_{\beta\epsilon\epsilon}\delta_{\gamma\alpha} - R_{\gamma\epsilon\epsilon}\delta_{\alpha\beta})

where we use a subscript :math:`\epsilon` to denote that the remaining two indices are the same, either :math:`\alpha\alpha`, :math:`\beta\beta`, or :math:`\gamma\gamma`. This tensor, like the quadrupole, is symmetric with respect to the swap of subscripts [Buck59]_.


.. _traceless:

Traceless Tensors
-----------------

The use of traceless tensors is occasionally debated in literature, but its origin is seldom mentioned [Raab75]_. The equation that this is used in is that of the interaction energy :math:`U` of a charge distribution with an external potential :math:`\phi`, which has some value at the origin :math:`\phi_O` [Buck59]_. The equation yielding this energy in terms of the multipoles is

.. math::

    U = q\phi_O - \mu_\alpha F_\alpha - \frac{1}{2} Q_{\alpha\beta} F'_{\alpha\beta} - \frac{1}{6} R_{\alpha\beta\gamma} F''_{\alpha\beta\gamma} - ...

where

.. math::

    F_\alpha = -\left(\frac{\partial \phi}{\partial r_\alpha} \right)_O ,\quad F'_{\alpha\beta} = -\left(\frac{\partial^2 \phi}{\partial r_\alpha \partial r_\beta} \right)_O ,\quad F''_{\alpha\beta\gamma} = -\left(\frac{\partial^3 \phi}{\partial r_\alpha \partial r_\beta \partial r_\gamma} \right)_O

In essence, :math:`F_\alpha` is the :math:`\alpha`-component of the external field at the origin :math:`O`, :math:`F'_{\alpha\beta}` is the :math:`\alpha\beta`-component of the field *gradient* at :math:`O`, :math:`F''_{\alpha\beta\gamma}` is the :math:`\alpha\beta\gamma`-component of the field *Hessian* at :math:`O`, and so on for the higher order multipoles.

The traceless tensor is a consequence of using Laplace's equation, :math:`F'_{\alpha\alpha} = F'_{xx} + F'_{yy} + F'_{zz} = 0`, which changes the equation to

.. math::

    U = q\phi_O - \mu_\alpha F_\alpha - \frac{1}{3} \Theta_{\alpha\beta} F'_{\alpha\beta} - \frac{1}{15} \Omega_{\alpha\beta\gamma} F''_{\alpha\beta\gamma} - ...

The switch from the normal tensors to their traceless forms also alter their meaning. While the nromal tensors show their complete moments, the traceless tensors describe only the deviation of the moment from one of perfectly spherical symmetry.

The traceless form comes with some significant downsides, most notably being that they have limited ability to describe multipole-field interactions in electrostatic fields without full symmetry of the field gradient tensors [Raab75]_.

Additionally, from a data perspective, the traceless tensors are not preferable for representing the quadrupole and/or octupole moments as the detracing operation is not reversible. This means that if one has a normal quadrupole/octupole and applies a detracing operator, then stores the now-traceless tensor, the information about the normal tensor is unrecoverable should the original source not be supplied. Meanwhile, one can always extract the traceless tensor from the normal tensor. Thus if you are able to, you should report the quadrupole/octupole moments in their normal forms.


.. _inertial-frame:

Inertia Tensor and Eigenvectors
===============================

A common phrase in the preceding sections was "in the inertial frame of the molecule." When I refer to the inertial frame of a molecule, I mean a frame in which the eigenvectors of the inertia tensor are aligned on the principle coordinate axes, making the inertia tensor diagonal. Often, the inclusion of this condition in literature is implicit, rather than explicit, but I have tried to be very deliberate about where I include it in this documentation. The properties which are affected by this are those that deal with defining which components of the quadrupole tensor are zero (see :ref:`Symmetry Dependence <quadrupole-symmetry>`).

Most computational chemistry calculations are done on molecules that are not aligned with the principle coordinate axes; as such, I have included functions in this program that can take a quadrupole moment from being in an arbitrary frame and rotate it into the inertial frame of a molecule (see :py:meth:`Quadrupole.inertialize()`).

The inertia tensor :math:`\textbf{I}` of a molecule with its center of mass at the origin is given as

.. math::

    \textbf{I} = \sum_j \begin{bmatrix}
        m_j \left( y^2_j+z^2_j \right) & -m_j x_j y_j                   & -m_j x_j z_j \\
        -m_j y_j x_j                   & m_j \left( x^2_j+z^2_j \right) & -m_j y_j z_j \\
        -m_j z_j x_j                   & -m_j z_j y_j                   & m_j \left( x^2_j+y^2_j \right)
    \end{bmatrix}

with the index :math:`j` running over all atoms and :math:`m` being their mass. For samples with standard isotopic distributions the masses are the average atomic masses and can be accessed through the :py:class:`Element` class. Due to the transposition symmetry of the inertia tensor (:math:`\textbf{I}_{\alpha\beta} = \textbf{I}_{\beta\alpha}`), one need only calculate the upper right (or lower left) triangular portion of the tensor, simplifying the calculations to

.. math::

    \textbf{I}_{xx} &= \sum_j^N m_j \left( y'^2_j+z'^2_j \right) \\
    \textbf{I}_{yy} &= \sum_j^N m_j \left( x'^2_j+z'^2_j \right) \\
    \textbf{I}_{zz} &= \sum_j^N m_j \left( x'^2_j+y'^2_j \right) \\
    \textbf{I}_{xy} &= \textbf{I}_{yx} = -\sum_j^N m_j x'_j y'_j \\
    \textbf{I}_{xz} &= \textbf{I}_{zx} = -\sum_j^N m_j x'_j z'_j \\
    \textbf{I}_{yz} &= \textbf{I}_{zy} = -\sum_j^N m_j y'_j z'_j \\

and for a system with a center of mass :math:`\textbf{R}_\alpha = (\textbf{R}_x,\quad \textbf{R}_y,\quad \textbf{R}_z)` given by

.. math::

    \textbf{R}_\alpha = \frac{1}{M}\sum_{j} m_j \textbf{r}_j\, ,\quad M = \sum_{j} m_j

that is not at the origin, set :math:`(x'_j,\quad y'_j,\quad z'_j) = (x_j-\textbf{R}_x,\quad y_j-\textbf{R}_y,\quad z_j-\textbf{R}_z)`.

.. _inertialize-example:

Example Inertial Frame Transformation
-------------------------------------

Using a calculation of the quadrupole moment of a water molecule (point group :math:`C_{2v}`) with the following coordinates:

.. code-block::

                X          Y          Z          
    
    O           5.753633   3.280382   2.728205
    H           4.931827   3.412032   2.252440
    H           5.738252   3.930645   3.432457

we see that the quadrupole tensor is (at the ωB97M-V/def2-QZVPPD level of theory)

.. code-block::

    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)
              Total:   -4.53597   -5.02123   -4.27393   -0.09199    0.86709    0.80776

which is in stark contrast to the expected tensor for a molecule with :math:`C_{2v}` symmetry (recall that molecules with :math:`C_{2v}` symmetry will have only diagonal components in the quadrupole tensor). The reason for this discrepancy is due to the alignment of the water molecule with respect to the principle unit coordinate axes. Using the function :py:meth:`Quadrupole.inertialize()`, we can rotate the quadrupole tensor into the inertial frame of the molecule, which then yields a quadrupole tensor of

.. code-block::

    Quadrupole (au):      (xx)       (yy)       (zz)       (xy)       (xz)       (yz)      
              Total:   -5.81808   -4.68670   -3.32635    0.00000   -0.00000   -0.00000

This tensor now contains only diagonal components, and aligns with expectations for the :math:`C_{2v}` point group.


.. _references:

References
==========

.. [exp] https://en.wikipedia.org/wiki/Exponential_function

.. [GeThWe95] Gelessus, A.; Thiel, W.; Weber, W. Multipoles and Symmetry. J. Chem. Educ. 1995, 72 (6), 505. https://doi.org/10.1021/ed072p505.

.. [Buck59] Buckingham, A. D. Molecular Quadrupole Moments. Q. Rev. Chem. Soc. 1959, 13 (3), 183-214. https://doi.org/10.1039/QR9591300183.

.. [Raab75] Raab, R. E. Magnetic Multipole Moments. Molecular Physics 1975, 29 (5), 1323-1331. https://doi.org/10.1080/00268977500101151.


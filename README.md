# Quadrupole Analysis
Python functions for taking molecular quadrupole tensors and converting to forms for comparison to literature.


### Inertia Tensor and Eigenvectors

The inertia tensor $\textbf{I}$ of a molecule with its center of mass at the origin is given as

$$
    \textbf{I} = \sum_j \begin{bmatrix}
        m_j \left( y^2_j+z^2_j \right) & -m_j x_j y_j                   & -m_j x_j z_j \\
        -m_j y_j x_j                   & m_j \left( x^2_j+z^2_j \right) & -m_j y_j z_j \\
        -m_j z_j x_j                   & -m_j z_j y_j                   & m_j \left( x^2_j+y^2_j \right)
    \end{bmatrix}
$$

with the index $j$ running over all atoms and $m$ being their mass. For samples with standard isotopic distributions the masses are the average atomic masses and can be accessed by the function `get_atomic_mass()`. Due to the transposition symmetry of the inertia tensor ($`\textbf{I}_{\alpha\beta} = \textbf{I}_{\beta\alpha}`$), one need only calculate the upper right (or lower left) triangular portion of the tensor, simplifying the calculations to

$$
    \begin{align*}
        \textbf{I}_{xx} &= \sum_j^N m_j \left( y'^2_j+z'^2_j \right) \\
        \textbf{I}_{yy} &= \sum_j^N m_j \left( x'^2_j+z'^2_j \right) \\
        \textbf{I}_{zz} &= \sum_j^N m_j \left( x'^2_j+y'^2_j \right) \\
        \textbf{I}_{xy} &= \textbf{I}_{yx} = -\sum_j^N m_j x'_j y'_j \\
        \textbf{I}_{xz} &= \textbf{I}_{zx} = -\sum_j^N m_j x'_j z'_j \\
        \textbf{I}_{yz} &= \textbf{I}_{zy} = -\sum_j^N m_j y'_j z'_j \\
    \end{align*}
$$

and for a system with a center of mass $\textbf{R}_\alpha = (\textbf{R}_x,\quad \textbf{R}_y,\quad \textbf{R}_z)$ given by

$$
    \textbf{R}_\alpha = \frac{1}{M}\sum_{j} m_j * \textbf{r}_j;\quad M = \sum_{j} m_j
$$

that is not at the origin, set $(x'_j,\quad y'_j,\quad z'_j) = (x_j-\textbf{R}_x,\quad y_j-\textbf{R}_y,\quad z_j-\textbf{R}_z)$.

Calculating the center of mass in Python requires a simple loop over all atoms. Here I have used the `Geometry` class which, when used as an iterator, yields individual members of the `Atom` class which have two attributes, the atomic symbol (`atom.element`) and the XYZ coordinates (`atom.xyz`), making our code into

***

```python
center_of_mass = np.zeros(3, dtype=float)
total_mass = 0.
for atom in geometry:
    mass = get_atomic_mass(atom.element)
    center_of_mass += atom.xyz * mass
    total_mass += mass
center_of_mass = center_of_mass / total_mass
```

***

Then, initialize an array of zeros for the inertia matrix and begin iterating through the atoms in the geometry once again,

***

```python
inertia_matrix = np.zeros((3, 3), dtype=float)

for atom in geometry:
    mass = get_atomic_mass(atom.element)
    x = atom.xyz[0] - center_of_mass[0]
    y = atom.xyz[1] - center_of_mass[1]
    z = atom.xyz[2] - center_of_mass[2]

    xx = mass * (y**2 + z**2)
    yy = mass * (x**2 + z**2)
    zz = mass * (x**2 + y**2)

    xy = mass * (x * y)
    xz = mass * (x * z)
    yz = mass * (y * z)

    inertia_matrix[0,0] += xx
    inertia_matrix[1,1] += yy
    inertia_matrix[2,2] += zz

    inertia_matrix[0,1] += -xy
    inertia_matrix[1,0] += -xy

    inertia_matrix[0,2] += -xz
    inertia_matrix[2,0] += -xz

    inertia_matrix[1,2] += -yz
    inertia_matrix[2,1] += -yz
```

***

Finally, to get the desired eigenvectors for our rotation matrix, use the NumPy function `numpy.linalg.eig()`, and return the output

***

```python
eigenvalues, eigenvectors = np.linalg.eig(inertia_matrix)

return eigenvalues, eigenvectors
```

***

Currently the eigenvalues are not used further in the code, but are returned nonetheless to make the code more easily extensible.


### Detracing Operation

The quadrupole tensor is a 3x3 matrix with transposition symmetry. These can be calculated as

$$
    \Theta_{\alpha\beta} = \sum_i e_i\textbf{r}_{i_\alpha}\textbf{r}_{i_\beta}
$$


There have been many papers arguing the form of this matrix, however due to the prevalence of the traceless quadrupole moment in experimental measurements of the quadrupole tensor, I have opted to include a method which can apply a detracing operation to an otherwise normal quadrupole matrix. The function `detrace_quadrupole()` performs the following operation

$$
    \mathbb{A}_{traceless} = \frac{3}{2}\left( \mathbb{A} - \mathbb{I}\frac{tr(\mathbb{A})}{3} \right)
$$

with the code

***

```python
def detrace_quadrupole(quadrupole: npt.NDArray):
    return (3 / 2) * (quadrupole - (np.eye(3,3) * (np.trace(quadrupole) / 3)))
```

***
with the standard factor of 3/2 for the detracing operation. See the associated docstring for `detrace_quadrupole()` for links to papers discussing the traceless quadrupole and for notes regarding detracing in ORCA and Quantum ESPRESSO.

### Comparing Quadrupoles

There is a significant challenge when it comes to comparing literature quadrupoles to calculated quadrupoles and that is the arbitrary alignment of the molecular coordinates. It is, in theory, possible to find the proper rotation of a molecule such that the quadrupole moment aligns with experiment, however as there is often little to no widespread agreement on how a molecule should be aligned this can be an arduous task. Similarly, multiple sources may use the same experimental data but alter the alignment of the molecule's quadrupole moment. 

For example, consider the following diagram from [a paper on the molecular Zeeman effect](https://doi.org/10.1080/00268977100100221) (Table 4, Journal page 246, PDF page 24):

![Reference Axes](/notebook/images/ref_axes.png)
![Reference Data for Water](/notebook/images/water_ref.png)

Clearly the authors have aligned the H2O molecule to be in the XY plane, with the oxygen pointing in the +X direction. If this alignment is used in a calculation (see `water_aligned.out`):

![Alignment of water molecule in XY plane](/notebook/images/water_xy.png)

the quadrupole moment (at the ωB97M-V/def2-QZVPPD level of theory) is [-0.15, 2.59, -2.44]. Clearly, when comparing this to the literature value of [-0.13, 2.63, -2.50], the product is a sensible difference of [-0.02, -0.04, 0.06]. However, if instead the calculation is run using a rotation that aligns the molecule in the XZ plane (see `water_xz_plane.out`): 

![Alignment of water molecule in XZ plane](/notebook/images/water_xz.png)

the exact same method for acquiring the quadrupole moment would yield [2.59, -2.44, -0.15], which provides a difference of [2.72, -5.07, -2.35] when compared to the literature. One may attempt to align the largest inertial axis with the Z axis, as is occasionally suggested in the literature, however there are no rules for how one may align the remaining inertial axes. Therefore without visually inspecting the paper's diagram (should it exist), there is no *a priori* way to guarantee the alignment of the quadrupole moment.

It is for this reason that I have chosen to supply a function that can attempt to align the quadrupole from a calculation with the quadrupole from an experimental source. The algorithm is not guaranteed to produce the correct alignment, and could potentially introduce error into statistical analyses, however given the absence of a conclusive method for aligning the quadrupole moment it can potentially save a significant amount of time when attempting a wide-scale analysis of quadrupole moments.

The algorithm works by purely statistical reasoning, no deeper physics at play. It starts by comparing the sign of the molecular quadrupole moments, since in theory for molecules with low symmetry a rotation of 180 degrees could result in a sign flip, and if the sign of the calculated quadrupole does not match experiment it will flip the sign of the calculated quadrupole. It then takes the 6 possible permutations of the quadrupole tensor and checks which has the lowest difference from experiment. Given that there could be ties between two permutations, it takes the list and sorts by the standard deviation of each tensor, so the quadrupole that matches most closely to the experiment in both total deviation and in lowest standard deviation will be the returned values.

To demonstrate this, I ran a calculation with a water molecule arbitrarily oriented in space. The resulting diagonal components of the quadrupole tensor, after rotation into the inertial frame, were [2.59, -0.15, -2.44]. Using the function with the literature value,
```python
>>> expt_quad = np.array([-0.13, 2.63, -2.50])
>>> calc_quad = np.array([2.59, -0.15, -2.44])
>>> best_quad, best_diff = compare_quadrupole(expt_quad, calc_quad)
>>> print(best_quad)
[-0.15, 2.59, -2.44]
>>> print(best_diff)
[-0.02, -0.04, 0.06]
```
and lo the quadrupole moment can indeed be matched to the expected value, albeit by statistical means rather than physical means.

from __future__ import annotations
from os import PathLike
from pathlib import Path
import numpy as np
import numpy.typing as npt
from numpy import sin, cos, sqrt
from .elements import Element, ElementLike


class FileFormatError(Exception):
    """Exception raised when a file is improperly formatted"""

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


class LatticeError(Exception):
    """Exception raised when trying to generate lattices with
    incompatible lattice type and cell parameters."""

    lattice_names = {
        1  : "Simple Cubic",
        2  : "Face-Centered Cubic",
        3  : "Body-Centered Cubic",
       -3  : "Body-Centered Cubic (High Symmetry)",
        4  : "Simple Hexagonal",
        5  : "Rhombohedral",
        6  : "Simple Tetragonal",
        7  : "Body-Centered Tetragonal",
        8  : "Simple Orthorhombic",
        9  : "Base-Centered Orthorhombic (c-face)",
       -9  : "Base-Centered Orthorhombic (c-face alternate)",
        91 : "Base-Centered Orthorhombic (a-face)",
        10 : "Face-Centered Orthorhombic",
        11 : "Body-Centered Orthorhombic",
        12 : "Simple Monoclinic",
        13 : "Base-Centered Monoclinic",
       -13 : "Base-Centered Monoclinic (Unique axis b)",
        14 : "Simple Triclinic",
    }

    def __init__(self, bravais_index: int, cell_params: npt.NDArray):
        self.bravais_index = bravais_index
        self.cell_params = cell_params

    def __str__(self):
        a = self.cell_params[0]
        b = self.cell_params[1]
        c = self.cell_params[2]
        alpha = self.cell_params[3]
        beta  = self.cell_params[4]
        gamma = self.cell_params[5]
        return (
            f"Can not generate {self.lattice_names[self.bravais_index]} with\n"
            f"a={a:8.5f} b={b:8.5f} c={c:8.5f}"
            f"α={alpha:8.5f} β={beta:8.5f} γ={gamma:8.5f}"
        )


class Atom:
    """Class containing the information of a single atom.

    Attributes
    ----------
    element : Element
        A member of the `Element` enum.
    xyz : NDArray of float with size 3
        The x-, y-, and z-coordinates of the atom in Ångstrom.
    """

    def __init__(
        self,
        element: ElementLike,
        xyz: npt.ArrayLike,
    ):
        """Class containing the information of a single atom.

        Parameters
        ----------
        element : ElementLike
            A member of the `Element` enum, or the atomic symbol/number.
        xyz : ArrayLike
            The x-, y-, and z-coordinates of the atom in Ångstrom.
        """
        self.element = Element(element)
        self.xyz = np.array(xyz, dtype=np.float64)

    def __repr__(self):
        return (
            f"{"Element":12}{"X":11}{"Y":11}{"Z":11}\n"
            f"{self.element:9}{self.xyz[0]:11.6f}{self.xyz[1]:11.6f}{self.xyz[2]:11.6f}\n"
        )

    def __eq__(self, other: Atom):
        return (self.element is other.element and (self.xyz == other.xyz).all())


class Geometry:
    """Class storing the geometric parameters of a molecular geometry or
    crystal structure. All quantities should be in Ångstrom.

    Attributes
    ----------
    atoms : list[Atom]
        The atoms in the geometry.

    Optional Attributes
    -------------------
    lat_vec : ndarray of float with shape (3,3)
        The primitive lattice vectors of the geometry,
    alat : float
        The lattice parameter.

    Note
    ----
    `alat` is calculated by taking the square root of the sum 
    of the first row of the lattice vector matrix.
    """

    bohr_to_angstrom = 0.529177210544

    def __init__(
        self,
        atoms: list[Atom],
        lat_vec: npt.NDArray | None = None,
        alat: float | None = None,
    ):
        """Create a new `Geometry` object.

        Parameters
        ----------
        atoms : list[Atom]
            The atoms in the geometry.
        lat_vec : ArrayLike of float with shape (3,3), optional
            The primitive lattice vectors of the geometry,
        alat : float, optional
            The lattice parameter.

        Note
        ----
        `alat` is calculated by taking the square root of the sum 
        of the first row of the lattice vector matrix.
        """
        self.atoms   = atoms
        self.lat_vec = np.array(lat_vec, dtype=float) if lat_vec is not None else None
        self.alat    = float(alat) if alat is not None else None


    @property
    def coordinates(self) -> npt.NDArray:
        return np.array([i.xyz for i in self.atoms])

    @property
    def elements(self) -> list[Element]:
        return [i.element for i in self.atoms]

    @staticmethod
    def _gen_prim_lattice(
        bravais_index: int,
        cell_params: npt.ArrayLike,
    ) -> npt.NDArray:
        """Generate a primitive unit cell. FOR INTERNAL USE ONLY, USERS
        SHOULD USE `generate_lattice()`!!
        """
        a = np.float64(cell_params[0])
        b = np.float64(cell_params[1])
        c = np.float64(cell_params[2])
        alpha = np.float64(cell_params[3])
        beta  = np.float64(cell_params[4])
        gamma = np.float64(cell_params[5])

        # region LatticeCheck
        right_angles = [
            1, 2, 3, -3,          # Cubic
            6, 7,                 # Tetragonal
            8, 9, -9, 91, 10, 11, # Orthorhombic
        ]

        # Check cell parameters to make sure they match the lattice type
        # First all cells where α = β = γ = 90°
        if bravais_index in right_angles:
            # Check all
            if not (alpha == beta == gamma == np.pi/2):
                raise LatticeError(bravais_index, cell_params)
            # Narrow down to tetragonal and cubic
            elif bravais_index in right_angles[:-6] and not (a == b):
                raise LatticeError(bravais_index, cell_params)
            # Next check cubic lattices
            elif bravais_index in right_angles[:-8] and not (a == b == c):
                raise LatticeError(bravais_index, cell_params)

        # Now check rhombohedral
        elif bravais_index in [5, -5]:
            if not (alpha == beta == gamma) or not (a == b == c):
                raise LatticeError(bravais_index, cell_params)
        # Now check hexagonal
        elif bravais_index == 4:
            if (
                not (a == b)
                or not (
                    alpha == beta == np.pi/2
                    and gamma == 2*np.pi/3
                )
                or c <= 0.0
            ):
                raise LatticeError(bravais_index, cell_params)
        elif bravais_index in [12, 13]:
            if not (alpha == beta == np.pi/2):
                raise LatticeError(bravais_index, cell_params)
        elif bravais_index in [-12, -13]:
            if not (alpha == gamma == np.pi/2):
                raise LatticeError(bravais_index, cell_params)

        # endregion LatticeCheck

        # Now that we have guaranteed the parameters will provide the correct
        # output, we can proceed with making the cell.
        match bravais_index:
            case 1: # Simple Cubic
                cell = a * np.eye(3, dtype=np.float64)
                return cell
            case 2: # Face-Centered Cubic
                cell = np.array([
                    [-1,  0,  1],
                    [ 0,  1,  1],
                    [-1,  1,  0],
                ], dtype=np.float64)
                return (a / 2) * cell
            case 3: # Body-Centered Cubic
                cell = np.array([
                    [ 1,  1,  1],
                    [-1,  1,  1],
                    [-1, -1,  1],
                ], dtype=np.float64)
                return (a / 2) * cell
            case -3: # Body-Centered Cubic, More Symmetric Axis
                cell = np.array([
                    [-1,  1,  1],
                    [ 1, -1,  1],
                    [ 1,  1, -1],
                ], dtype=np.float64)
                return (a / 2) * cell
            case 4: # Hexagonal
                cell = np.array([
                    [   a,           0,  0],
                    [-a/2, a*sqrt(3)/2,  0],
                    [   0,           0,  c],
                ], dtype=np.float64)
                return cell
            case 5 | -5: # Rhombohedral
                term1 = sqrt(1 + 2 * cos(gamma))
                term2 = sqrt(1 - cos(gamma))
                if bravais_index == 5: # Rhombohedral, Symmetry about z-axis
                    cell = np.array([
                        [ term2/sqrt(2),        -term2/sqrt(6), term1/sqrt(3)],
                        [             0, sqrt(2)*term2/sqrt(3), term1/sqrt(3)],
                        [-term2/sqrt(2),        -term2/sqrt(6), term1/sqrt(3)],
                    ], dtype=np.float64)
                    return a * cell
                else: # Rhombohedral, Symmetry about <111>
                    u = (term1 - 2*term2) / 3
                    v = (term1 + term2) / 3
                    cell = np.array([
                        [u, v, v],
                        [v, u, v],
                        [v, v, u],
                    ], dtype=np.float64)
                    return a * cell
            case 6: # Simple Tetragonal
                cell = np.array([
                    [a, 0, 0],
                    [0, a, 0],
                    [0, 0, c],
                ], dtype=np.float64)
                return cell
            case 7: # Body-Centered Tetragonal
                cell = np.array([
                    [ a, -a, c],
                    [ a,  a, c],
                    [-a, -a, c],
                ], dtype=np.float64)
                return cell / 2
            case 8: # Simple Orthorhombic
                cell = np.array([
                    [a, 0, 0],
                    [0, b, 0],
                    [0, 0, c],
                ], dtype=np.float64)
                return cell
            case 9: # Base-Centered Orthorhombic, C-type, legacy PWscf
                cell = np.array([
                    [ a/2, b/2, 0],
                    [-a/2, b/2, 0],
                    [   0,   0, c],
                ], dtype=np.float64)
                return cell
            case -9: # Base-Centered Orthorhombic, C-type
                cell = np.array([
                    [a/2, -b/2, 0],
                    [a/2,  b/2, 0],
                    [  0,    0, c],
                ], dtype=np.float64)
                return cell
            case 91: # Base-Centered Orthorhombic, A type
                cell = np.array([
                    [a,   0,    0],
                    [0, b/2, -c/2],
                    [0, b/2,  c/2],
                ], dtype=np.float64)
                return cell
            case 10: # Face-Centered Orthorhombic
                cell = np.array([
                    [a, 0, c],
                    [a, b, 0],
                    [0, b, c],
                ], dtype=np.float64)
                return cell / 2
            case 11: # Body-Centered Orthorhombic
                cell = np.array([
                    [ a,  b,  c],
                    [-a,  b,  c],
                    [-a, -b,  c],
                ], dtype=np.float64)
                return cell / 2
            case 12: # Simple Monoclinic, unique axis c (orthogonal to a)
                bcosg = b * cos(gamma)
                bsing = b * sin(gamma)
                cell = np.array([
                    [    a,     0, 0],
                    [bcosg, bsing, 0],
                    [    0,     0, c],
                ], dtype=np.float64)
                return cell
            case -12: # Simple Monoclinic, unique axis b
                ccosbe = c * cos(beta)
                csinbe = c * sin(beta)
                cell = np.array([
                    [     a,  0,      0],
                    [     0,  b,      0],
                    [ccosbe,  0, csinbe],
                ], dtype=np.float64)
                return cell
            case 13: # Base-Centered Monoclinic, unique axis c
                bcosg = b * cos(gamma)
                bsing = b * sin(gamma)
                cell = np.array([
                    [  a/2,     0, -c/2],
                    [bcosg, bsing,    0],
                    [  a/2,     0,  c/2],
                ], dtype=np.float64)
                return cell
            case -13: # Base-Centered Monoclinic, unique axis b
                ccosbe = c * cos(beta)
                csinbe = c * sin(beta)
                cell = np.array([
                    [   a/2, b/2,      0],
                    [  -a/2, b/2,      0],
                    [ccosbe,   0, csinbe],
                ], dtype=np.float64)
                return cell
            case 14: # Triclinic
                bcosg  = b * cos(gamma)
                bsing  = b * sin(gamma)
                ccosbe = c * cos(beta)
                term1 = (
                    c * (cos(alpha) - cos(beta) * cos(gamma))
                    / sin(gamma)
                )
                term2 = c * sqrt(
                    (
                        1.0 + (2.0 * cos(alpha) * cos(beta) * cos(gamma))
                        - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2
                    )
                ) / sin(gamma)

                cell = np.array([
                    [     a,     0,     0],
                    [ bcosg, bsing,     0],
                    [ccosbe, term1, term2],
                ], dtype=np.float64)
                return cell
            case _:
                raise ValueError(f"Invalid lattice type: {bravais_index}")


    @staticmethod
    def generate_lattice(
        bravais_index: int,
        cell_params: npt.ArrayLike,
        primitive: bool = False,
        espresso_like: bool = False,
    ) -> npt.NDArray:
        """Generate a 3x3 unit cell matrix from a Bravais lattice index
        and a set of cell parameters.

        Parameters
        ----------
        bravais_index : {1, 2, 3, -3, 4, 5, -5, 6, 7, 8, 9, -9, 91, 10, 11, 12, -12, 13, -13, 14}
            Integer corresponding to the type of Bravais lattice.
            Described in ``Bravais Indices``.
        cell_params : ArrayLike, dtype float, size 6
            ArrayLike of cell parameters, in order:
            (`a`, `b`, `c`, `α`, `β`, `γ`)
        primitive : bool, default=False
            Denote whether supplied cell parameters are for a primitive
            or a conventional unit cell.
        espresso_like : bool, default=False
            If `cell_params` are provided in Quantum ESPRESSO format,
            this should be set to ``True``.
            Automatically sets `primitive` to ``True`` as well.

        Notes
        -----
        Quantum ESPRESSO format is as follows:
        ```
        cell_params = (a, b/a, c/a, cos(α), cos(β), cos(γ))
        ```

        This function will cross-check all Bravais types against the
        provided cell parameters.

        Bravais Indices
        ---------------

        1
            Simple Cubic, cP
        2
            Face-Centered Cubic, cF
        3
            Body-Centered Cubic, cI
        -3
            Body-Centered Cubic, cI, Higher Symmetry
        4
            Simple Hexagonal, hP
        5
            Rhombohedral, hR, 3-fold symmetry axis c
        -5
            Rhombohedral, hR, 3-fold symmetry axis <111>
        6
            Simple Tetragonal, tP
        7
            Body-Centered Tetragonal, tI
        8
            Simple Orthorhombic, oP
        9
            Base-Centered Orthorhombic, oS, c-face
        -9
            Base-Centered Orthorhombic, oS, alternate alignment
        91
            Base-Centered Orthorhombic, oS, a-face
        10
            Face-Centered Orthorhombic, oF
        11
            Body-Centered Orthorhombic, oI
        12
            Simple Monoclinic, mP, unique axis c
        13
            Base-Centered Monoclinic, mS
        -13
            Base-Centered Monoclinic, mS, unique axis b
        14
            Simple Triclinic, aP
        """

        supported_indices = [
            1, 2, 3, -3, # Cubic
            4, # Hexagonal
            5, -5, # Rhombohedral
            6, 7, # Tetragonal
            8, 9, -9, 91, 10, 11, # Orthorhombic
            12, 13, -12, -13, # Monoclinic
            14, # Triclinic
        ]

        if bravais_index not in supported_indices:
            raise ValueError(
               f"Bravais lattice index {bravais_index} not supported!\n"
                "Please select from a supported index!"
            )

        if not espresso_like:
            lattice = Geometry._gen_prim_lattice(bravais_index, cell_params)
            if primitive or espresso_like:
                return lattice
            else:
                raise ValueError(
                    "Only primitive cells are currently supported!"
                )

        # If we are reading a QE output then we need to translate the parameters
        # to match the typical a, b, c, α, β, γ
        match bravais_index:
            case 1 | 2 | 3 | -3: # Cubic cells
                a = cell_params[0]
                b = a
                c = a
                alpha = np.pi / 2
                beta  = np.pi / 2
                gamma = np.pi / 2
            case 4: # Hexagonal
                a = cell_params[0]
                b = a
                c = cell_params[2] * a
                alpha = np.pi / 2
                beta  = np.pi / 2
                gamma = 2 * np.pi / 3
            case 5 | -5: # Rhombohedral
                a = cell_params[0]
                b = a
                c = a
                gamma = np.arccos(cell_params[3])
                alpha = gamma
                beta  = gamma
            case 6 | 7: # Tetragonal
                a = cell_params[0]
                b = a
                c = a * cell_params[2]
                alpha = np.pi / 2
                beta  = np.pi / 2
                gamma = np.pi / 2
            case 8 | 9 | -9 | 91 | 10 | 11: # Orthorhombic
                a = cell_params[0]
                b = a * cell_params[1]
                c = a * cell_params[2]
                alpha = np.pi / 2
                beta  = np.pi / 2
                gamma = np.pi / 2
            case 12 | -12: # Monoclinic
                a = cell_params[0]
                b = a * cell_params[1]
                c = a * cell_params[2]
                if bravais_index == 12: # Monoclinic, unique axis c
                    gamma = np.arccos(cell_params[3])
                    alpha = np.pi / 2
                    beta  = np.pi / 2
                else: # Monoclinic, unique axis b
                    beta = np.arccos(cell_params[4])
                    alpha = np.pi / 2
                    gamma = np.pi / 2
            case 13 | -13: # Base-Centered Monoclinic
                a = cell_params[0]
                b = a * cell_params[1]
                c = a * cell_params[2]
                if bravais_index == 13: # Base-Centered Monoclinic, unique axis c
                    gamma = np.arccos(cell_params[3])
                    alpha = np.pi / 2
                    beta  = np.pi / 2
                else: # Base-Centered Monoclinic, unique axis b
                    beta = np.arccos(cell_params[4])
                    alpha = np.pi / 2
                    gamma = np.pi / 2
            case 14:
                a = cell_params[0]
                b = a * cell_params[1]
                c = a * cell_params[2]
                alpha = np.arccos(cell_params[3])
                beta  = np.arccos(cell_params[4])
                gamma = np.arccos(cell_params[5])

        cell_params = np.array([a, b, c, alpha, beta, gamma], dtype=np.float64)
        lattice = Geometry._gen_prim_lattice(bravais_index, cell_params)

        return lattice


    @classmethod
    def from_xsf(cls, file: PathLike) -> Geometry:
        """Read in the crystallographic information from an XSF file."""
        with open(file, "r") as xsf:

            # Pulls in the lines that contain the primitive lattice vectors and the line containing the number of atoms.
            crystal_info = [next(xsf) for _ in range(7)]

            # Extract the lattice vectors
            lat_vec = np.array([line.strip().split() for line in crystal_info[2:5]], dtype=np.float64)

            # Calculate lattice parameter
            alat = sqrt(np.sum(lat_vec[0,:] ** 2))

            # Pull the number of atoms
            num_atoms = int(crystal_info[-1].split()[0])

            # Read in all of the atoms and turn it into a list of Atom objects
            atoms = [next(xsf).strip().split() for _ in range(num_atoms)]
            atoms = [Atom(element=atom[0], xyz=np.array([float(i) for i in atom[1:4]])) for atom in atoms]

        return Geometry(atoms, lat_vec, alat)


    @classmethod
    def from_xyz(cls, file: PathLike) -> Geometry:
        """Read in XYZ file and return a `Geometry` object"""

        with open(file) as xyz:
            num_atoms = xyz.readline()
            try:
                num_atoms = int(num_atoms)
            except ValueError:
                raise FileFormatError(
                    f"File {Path(file).resolve()} is improperly formatted at line 1,\n"
                    f"expected number of atoms, got '{num_atoms}' instead!"
                )

            xyz.readline() # Skip comment line

            elements = []
            xyzs = []
            for i in range(num_atoms):
                line = xyz.readline()
                if line == "":
                    raise FileFormatError(
                        f"File {Path(file).resolve()} contains less atoms than expected!"
                    )
                elif line.strip() == "":
                    raise FileFormatError(
                        f"File {Path(file).resolve()} is improperly formatted!"
                    )
                else:
                    line = line.strip().split()
                    elements.append(line[0])
                    xyzs.append(line[1:4])

        atoms = []
        for i, element in enumerate(elements):
            atoms.append(Atom(element, xyzs[i]))

        return Geometry(atoms)


    @classmethod
    def from_list(cls, elements: list[ElementLike], xyzs: npt.ArrayLike) -> Geometry:
        if len(elements) != len(xyzs):
            raise ValueError(
                "The list of elements and coordinates must be of the same size!"
            )

        atoms = []
        for i, element in enumerate(elements):
            atoms.append(Atom(element, xyzs[i]))

        return Geometry(atoms)


    @classmethod
    def from_orca(cls, file: PathLike) -> Geometry:
        xyz_data = []
        with open(file, "r") as orca_out:
            input_search = True
            while input_search:
                # Check for end of input so we can start looking for the calculation type
                line = orca_out.readline()
                if "END OF INPUT" in line:
                    input_search = False
                elif line == "":
                    raise FileFormatError(
                        f"Error reading file '{Path(file).resolve()}', did not find end of input!"
                    )

            for i in range(20):
                # The calculation type should be printed only 4 lines after the
                # end of the input, but we still go for 20 just in case
                line = orca_out.readline()
                if "Geometry Optimization Run" in line:
                    opt_run = True
                    break
                elif "Single Point Calculation" in line:
                    opt_run = False
                    break
                elif i == 19:
                    raise FileFormatError(
                        f"Error reading file '{Path(file).resolve()}' at line {orca_out.tell()}!"
                    )

            if not opt_run:
                # If this isn't an optimization, do not spin through the whole file
                # The geometry is only printed once at the beginning in an SCF calculation
                coordinate_search = True
                while coordinate_search:
                    line = orca_out.readline()
                    if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
                        orca_out.readline() # Skip forward one more line
                        coordinate_search = False

                read_data = True
                while read_data:
                    line = orca_out.readline()
                    if line == "\n":
                        read_data = False
                    else:
                        xyz_data.append(line.strip().split())
            else:
                final_geom_search = True
                while final_geom_search:
                    line = orca_out.readline()
                    if "FINAL ENERGY EVALUATION AT THE STATIONARY POINT" in line:
                        final_geom_search = False
                    elif line == "":
                        raise FileFormatError(
                            f"Error reading file '{Path(file).resolve()}', can not find final geometry"
                        )

                coordinate_search = True
                while coordinate_search:
                    line = orca_out.readline()
                    if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
                        orca_out.readline()
                        coordinate_search = False

                read_data = True
                while read_data:
                    line = orca_out.readline()
                    if line == "\n":
                        read_data = False
                    else:
                        xyz_data.append(line.strip().split())

        atoms = [Atom(i[0], np.array(i[1:4])) for i in xyz_data]
        return Geometry(atoms)


    @classmethod
    def from_cube(cls, file: PathLike) -> Geometry:
        """Read in and interpret crystallographic information from a
        CUBE file.

        Note
        ----
        This function calculates the dimensions of the unit cell by
        taking the number of grid points and muliplying it by the
        spacing of the grid points. Due to rounding errors when an
        external program prints a CUBE file, the cell dimensions can
        have inaccuracies of +/- 5e-05 angstrom.
        """

        with open(file, "r") as cube:
            # Skip over header
            for _ in range(2):
                next(cube)

            num_atoms = int(next(cube).strip().split()[0])

            grid_info = [next(cube).strip().split() for _ in range(3)]
            grid_points = [int(i[0]) for i in grid_info]

            lat_vec = []
            for i, dim in enumerate(grid_info):
                lat_vec.append(
                    [grid_points[i]*float(dim[j]) for j in range(1,4)]
                )

            lat_vec = np.array(lat_vec) * Geometry.bohr_to_angstrom

            alat = sqrt(np.sum(lat_vec[0,:] ** 2))

            atom_data = [next(cube).strip().split() for _ in range(num_atoms)]

            atoms = []
            for atom in atom_data:
                element = int(atom[0])
                coordinate = np.array(atom[2:5], dtype=float) * Geometry.bohr_to_angstrom
                atoms.append(Atom(element, coordinate))

        return Geometry(atoms, lat_vec, alat)


    @classmethod
    def from_qe_pp(cls, file: PathLike) -> Geometry:
        """Read in only the crystallographic information from a
        Quantum ESPRESSO post-processing file.
        (e.g. leaving the ``&PLOT`` blank and reading ``filplot``)
        """
        with open(file, "r") as qe_pp:
            # First line is either blank or has a title, either way, skip.
            next(qe_pp)

            sys_info = next(qe_pp).split()
            num_atoms = int(sys_info[-2])
            num_types = int(sys_info[-1])

            cell_info = next(qe_pp).split()
            ibrav = int(cell_info[0])
            alat = float(cell_info[1]) * Geometry.bohr_to_angstrom

            # Anything other than ibrav=0 is only partially supported
            match ibrav:
                case 0:
                    lat_vec = np.array(
                        [next(qe_pp).split() for _ in range(3)],
                        dtype=np.float64,
                    ) * alat
                case _:
                    lat_vec = Geometry.generate_lattice(
                        bravais_index=ibrav,
                        cell_params=np.array(cell_info[1:], dtype=np.float64),
                        primitive=True,
                        espresso_like=True,
                    ) * Geometry.bohr_to_angstrom

            # Skip over PW Parameters
            next(qe_pp)

            atom_types = [next(qe_pp).split()[:-1] for _ in range(num_types)]
            atom_types = dict(
                [[int(i[0]), Element(i[1])] for i in atom_types]
            )

            # Read in all of the atoms and turn it into a list of Atom objects
            atoms = [next(qe_pp).strip().split() for _ in range(num_atoms)]
            atoms = [
                Atom(
                    element=atom_types[int(atom[-1])],
                    xyz=np.array([float(i) for i in atom[1:4]]) * alat
                ) for atom in atoms
            ]

        return Geometry(atoms, lat_vec, alat)


    def calc_principal_moments(self):
        """Calculate the principal inertial axes for a given geometry.

        Returns
        -------
        eigenvalues : ndarray
            First output of numpy.linalg.eig(inertia_tensor)
        eigenvectors : ndarray
            Second output of numpy.linalg.eig(inertia_tensor)
        """
        center_of_mass = np.zeros(3, dtype=float)
        total_mass = 0.
        for atom in self:
            mass = atom.element.mass
            center_of_mass += atom.xyz * mass
            total_mass += mass

        center_of_mass = center_of_mass / total_mass

        inertia_matrix = np.zeros((3, 3), dtype=float)

        for atom in self:
            mass = atom.element.mass
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

        eigenvalues, eigenvectors = np.linalg.eig(inertia_matrix)

        return eigenvalues, eigenvectors


    def __repr__(self):
        self_repr = ""
        if self.lat_vec is not None:

            self_repr += f"{"Lattice":12}{"X":11}{"Y":11}{"Z":11}\n{"Vectors":11}\n"
            self_repr += f"{"":9}{self.lat_vec[0][0]:11.6f}{self.lat_vec[0][1]:11.6f}{self.lat_vec[0][2]:11.6f}\n"
            self_repr += f"{"":9}{self.lat_vec[1][0]:11.6f}{self.lat_vec[1][1]:11.6f}{self.lat_vec[1][2]:11.6f}\n"
            self_repr += f"{"":9}{self.lat_vec[2][0]:11.6f}{self.lat_vec[2][1]:11.6f}{self.lat_vec[2][2]:11.6f}\n\n"

            self_repr += f"{"Element":12}{"X":11}{"Y":11}{"Z":11}\n\n"
            for i in self.atoms:
                self_repr += f"{i.element:9}{i.xyz[0]:11.6f}{i.xyz[1]:11.6f}{i.xyz[2]:11.6f}\n"
        else:
            self_repr += f"{"Element":12}{"X":11}{"Y":11}{"Z":11}\n\n"
            for i in self.atoms:
                self_repr += f"{i.element:9}{i.xyz[0]:11.6f}{i.xyz[1]:11.6f}{i.xyz[2]:11.6f}\n"
        return self_repr


    def __iter__(self):
        yield from self.atoms


    def __len__(self):
        return len(self.atoms)


    def __getitem__(self, index):
        return self.atoms[index]
    

    def __eq__(self, other: Geometry):
        if self.alat != other.alat:
            return False
        elif not np.array_equal(self.lat_vec, other.lat_vec):
            return False
        else:
            for self_atom, other_atom in zip(self, other):
                if self_atom != other_atom:
                    return False
        return True

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from os import PathLike


# Helper functions for converting between atomic symbol and atomic number as well as getting atomic masses

def get_atomic_number(atomic_symbol: str) -> int:
    """Convert between atomic symbol and atomic number (case insensitive)."""
    elements = [
        "H" ,
        "He",
        "Li",
        "Be",
        "B" ,
        "C" ,
        "N" ,
        "O" ,
        "F" ,
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P" ,
        "S" ,
        "Cl",
        "Ar",
        "K" ,
        "Ca",
        "Sc",
        "Ti",
        "V" ,
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y" ,
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I" ,
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W" ,
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U" ,
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    ]

    return elements.index(atomic_symbol.title()) # str.title() returns string with the first letter capitalized


def get_atomic_symbol(atomic_number: int) -> str:
    """Convert between atomic number and atomic symbol (case insensitive)."""
    elements = [
        "H" ,
        "He",
        "Li",
        "Be",
        "B" ,
        "C" ,
        "N" ,
        "O" ,
        "F" ,
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P" ,
        "S" ,
        "Cl",
        "Ar",
        "K" ,
        "Ca",
        "Sc",
        "Ti",
        "V" ,
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y" ,
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I" ,
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W" ,
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U" ,
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    ]

    return elements[atomic_number - 1]


def get_atomic_mass(atomic_id: str | int):
    """Get atomic mass for an element, specified by either atomic symbol or atomic number (case insensitive)."""

    if isinstance(atomic_id, str):
        atomic_id = get_atomic_number(atomic_id)

    masses = [
        1.0080,
        4.002602,
        6.94,
        9.0121831,
        10.81,
        12.011,
        14.007,
        15.999,
        18.998403162,
        20.1797,
        22.98976928,
        24.305,
        26.9815384,
        28.085,
        30.973761998,
        32.06,
        35.45,
        39.95,
        39.0983,
        40.078,
        44.955907,
        47.867,
        50.9415,
        51.9961,
        54.938043,
        55.845,
        58.933194,
        58.6934,
        63.546,
        65.38,
        69.723,
        72.630,
        74.921595,
        78.971,
        79.904,
        83.798,
        85.4678,
        87.62,
        88.905838,
        91.222,
        92.90637,
        95.95,
        97.0,
        101.07,
        102.90549,
        106.42,
        107.8682,
        112.414,
        114.818,
        118.710,
        121.760,
        127.60,
        126.90447,
        131.293,
        132.90545196,
        137.327,
        138.90547,
        140.116,
        140.90766,
        144.242,
        145.0,
        150.36,
        151.964,
        157.249,
        158.925354,
        162.500,
        164.930329,
        167.259,
        168.934219,
        173.045,
        174.96669,
        178.486,
        180.94788,
        183.84,
        186.207,
        190.23,
        192.217,
        195.084,
        196.966570,
        200.592,
        204.38,
        207.2,
        208.98040,
        209.0,
        210.0,
        222.0,
        223.0,
        226.0,
        227.0,
        232.0377,
        231.03588,
        238.02891,
        237.0,
        244.0,
        243.0,
        247.0,
        247.0,
        251.0,
        252.0,
        257.0,
        258.0,
        259.0,
        262.0,
        267.0,
        270.0,
        269.0,
        270.0,
        270.0,
        278.0,
        281.0,
        281.0,
        285.0,
        286.0,
        289.0,
        289.0,
        293.0,
        293.0,
        294.0,
    ]

    return masses[atomic_id - 1]


class Atom:
    """Class containing the information of a single atom.

    Attributes
    ----------
    element : str
        The atomic symbol of the element.
    xyz : NDArray
        The x-, y-, and z-coordinates of the atom.
    """

    def __init__(
        self,
        element: str,
        xyz: npt.ArrayLike,
    ):
        self.element = element
        self.xyz = np.array(xyz, dtype=np.float64)

    def __repr__(self):
        return (
            f"{"Element":12}{"X":11}{"Y":11}{"Z":11}\n"
            f"{self.element:9}{self.xyz[0]:11.6f}{self.xyz[1]:11.6f}{self.xyz[2]:11.6f}\n"
        )


class Geometry:
    """Class storing the geometric parameters of a molecular or crystalline geometry.
    
    Attributes
    ----------
    atoms : list[Atom]
        The atoms in the geometry.
    lat_vec : npt.NDArray or None, default=None
        The primitive lattice vectors of the geometry in units of alat
    alat : float or None, default=None
        The lattice parameter.

    Notes
    -----
    The lattice vectors should be provided in units of alat here, which involves taking the square
    root of the sum of the first row of the lattice vector matrix.
    """

    def __init__(
        self,
        atoms: list[Atom],
        lat_vec: npt.NDArray | None = None,
        alat: float | None = None,
    ):
        self.atoms = atoms
        self.lat_vec = lat_vec
        self.alat = float(alat) if alat is not None else None


    def get_coords(self) -> npt.NDArray:
        return np.array([i.xyz for i in self.atoms])


    def get_elements(self) -> list[str]:
        return [i.element for i in self.atoms]


    @classmethod
    def from_xsf(cls, file: PathLike) -> Geometry:
        """Read in only the crystallographic information from an XSF file."""
        with open(file, "r") as xsf:

            # Pulls in the lines that contain the primitive lattice vectors and the line containing the number of atoms.
            crystal_info = [next(xsf) for _ in range(7)]

            # Extract the lattice vectors
            lat_vec = np.array([line.strip().split() for line in crystal_info[2:5]], dtype=np.float64)

            # Calculate lattice parameter
            alat = np.sqrt(np.sum(lat_vec[0,:] ** 2))

            # Pull the number of atoms
            num_atoms = int(crystal_info[-1].split()[0])

            # Read in all of the atoms and turn it into a list of Atom objects
            atoms = [next(xsf).strip().split() for _ in range(num_atoms)]
            atoms = [Atom(element=atom[0], xyz=np.array([float(i) for i in atom[1:4]])) for atom in atoms]

        return Geometry(atoms, lat_vec, alat)


    @classmethod
    def from_xyz(cls, file: PathLike) -> Geometry:
        """Read in XYZ file and return a `Geometry` object"""

        molecule_xyz = []

        with open(file) as xyz:
            for line in xyz:
                line = line.strip().split()
                molecule_xyz.append(line)

        expected_num_atoms = int(molecule_xyz[0])

        elements = []
        xyzs = []
        for index, line in enumerate(molecule_xyz[2:]):
            if len(line) == 0:
                break
            if index > expected_num_atoms:
                raise ValueError("File contains more atoms than expected!")
            elements.append(line[0])
            xyzs.append(np.array(line[1:4], dtype=float))

        atoms = []

        for i, element in enumerate(elements):
            atoms.append(Atom(element, xyzs[i]))

        return Geometry(atoms)


    @classmethod
    def from_list(cls, elements: list[str], xyzs: npt.NDArray, charge: int = 0):
        if len(elements) != len(xyzs):
            raise ValueError("The list of elements and coordinates must be of the same size!")
        else:
            atoms = []
            for i, element in enumerate(elements):
                atoms.append(Atom(element, xyzs[i]))

        return Geometry(atoms, charge)


    @classmethod
    def from_orca(cls, file: PathLike) -> Geometry:
        xyz_data = []
        with open(file, "r") as orca_out:
            found_xyz = False
            while not found_xyz:
                line = orca_out.readline().strip()
                if line == "CARTESIAN COORDINATES (ANGSTROEM)":
                    orca_out.readline()
                    found_xyz = True

            finished_xyz = False
            while not finished_xyz:
                line = orca_out.readline().strip()
                if line == "":
                    finished_xyz = True
                else:
                    xyz_data.append(line.split())

        atoms = [Atom(i[0], np.array(i[1:4])) for i in xyz_data]
        return Geometry(atoms)


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
            mass = get_atomic_mass(atom.element)
            center_of_mass += atom.xyz * mass
            total_mass += mass

        center_of_mass = center_of_mass / total_mass

        inertia_matrix = np.zeros((3, 3), dtype=float)

        for atom in self:
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


class Quadrupole:
    """Class containing data and functions required for analyzing a quadrupole moment.

    Attributes
    ----------
    quadrupole : ndarray
        Array containing the 3x3 quadrupole matrix, the diagonal components of the quadrupole (shape 3x1),
        or the 6 independent elements of the quadrupole (shape 6x1, in order, [xx, yy, zz, xy, xz, yz])
    units : {"au", "buckingham", "cm^2", "esu"}, default="buckingham"
        Units of the quadrupole matrix (case insensitive).

    Note
    ----
    The attributes specify that there are 6 independent elements of a quadrupole tensor. This is
    because a molecular quadrupole, by definition, is symmetric. It is worth noting however that a
    traceless quadrupole moment only has 5 independent elements as being traceless dictates that
    one of the diagonal components must be equal to the negative sum of the remaining two, i.e. it
    is required that :math:`Q_{aa} + Q_{bb} = -2Q_{cc}`, therefore :math:`Q_{cc}` depends on 
    :math:`Q_{aa}` and :math:`Q_{bb}`
    """

    au_cm2_conversion   = 4.4865515185e-40
    esu_cm2_conversion  = 2.99792458e13
    esu_buck_conversion = 1e-26

    def __init__(self, quadrupole: npt.ArrayLike, units: str = "buckingham"):
        quadrupole = np.array(quadrupole, dtype=float)
        if quadrupole.shape == (3, 3):
            self.quadrupole = quadrupole
        elif quadrupole.shape == (3,):
            self.quadrupole = np.diag(quadrupole)
        elif quadrupole.shape == (6,):
            self.quadrupole = np.array(
                [
                    [quadrupole[0], quadrupole[3], quadrupole[4]],
                    [quadrupole[3], quadrupole[1], quadrupole[5]],
                    [quadrupole[4], quadrupole[5], quadrupole[2]],
                ]
            )
        else:
            raise ValueError(f"Cannot cast array of shape {quadrupole.shape} to a quadrupole, supply either shape (3, 3) or (3,) or (6,)!")

        units = units.lower()
        if units not in ["au", "buckingham", "cm^2", "esu"]:
            raise ValueError("Invalid units, please select from ( 'au', 'buckingham', 'cm^2', 'esu' )")
        else:
            self.units = units


    #-----------------------------------------------------------#
    def au_to_cm2(self) -> Quadrupole:                          #
        """Convert from Hartree atomic units to Coulomb•m²"""   #
        q = self.quadrupole * Quadrupole.au_cm2_conversion      #
        return Quadrupole(q, units="cm^2")                      #
                                                                # https://physics.nist.gov/cgi-bin/cuu/Value?aueqm
    def cm2_to_au(self) -> Quadrupole:                          #
        """Convert from Coulomb•m² to Hartree atomic units"""   #
        q = self.quadrupole / Quadrupole.au_cm2_conversion      #
        return Quadrupole(q, units="au")                        #
    #-----------------------------------------------------------#

    #-----------------------------------------------------------#
    def cm2_to_esu(self) -> Quadrupole:                         #
        """Convert from Coulomb•m² to e.s.u•cm²"""              #
        q = self.quadrupole * Quadrupole.esu_cm2_conversion     #
        return Quadrupole(q, units="esu")                       # CGS statCoulomb/cm^2 to Coulomb/m^2
                                                                # Factor of c * (100cm)^2/m^2
    def esu_to_cm2(self) -> Quadrupole:                         # c taken from https://physics.nist.gov/cgi-bin/cuu/Value?c
        """Convert from e.s.u•cm² to Coulomb•m²"""              #
        q = self.quadrupole / Quadrupole.esu_cm2_conversion     #
        return Quadrupole(q, units="cm^2")                      #
    #-----------------------------------------------------------#

    #-----------------------------------------------------------#
    def buck_to_esu(self) -> Quadrupole:                        #
        """Convert from Buckingham to e.s.u•cm²"""              #
        q = self.quadrupole * Quadrupole.esu_buck_conversion    #
        return Quadrupole(q, units="esu")                       # Suggested by Peter J. W. Debye in 1963
                                                                # https://doi.org/10.1021/cen-v041n016.p040
    def esu_to_buck(self) -> Quadrupole:                        #
        """Convert from e.s.u•cm² to Buckingham"""              #
        q = self.quadrupole / Quadrupole.esu_buck_conversion    #
        return Quadrupole(q, units="Buckingham")                #
    #-----------------------------------------------------------#

    #-----------------------------------------------------------#
    def cm2_to_buck(self) -> Quadrupole:                        #
        """Convert from Buckingham to Coulomb•m²"""             #
        q = self.cm2_to_esu()                                   #
        return q.esu_to_buck()                                  #
                                                                #
    def buck_to_cm2(self) -> Quadrupole:                        #
        """Convert from Coulomb•m² to Buckingham"""             #
        q = self.buck_to_esu()                                  #
        return q.esu_to_cm2()                                   #
    #-----------------------------------------------------------#

    #-----------------------------------------------------------#
    def au_to_esu(self) -> Quadrupole:                          #
        """Convert from Hartree atomic units to e.s.u•cm²"""    #
        q = self.au_to_cm2()                                    #
        return q.cm2_to_esu()                                   #
                                                                #
    def esu_to_au(self) -> Quadrupole:                          #
        """Convert from Hartree atomic units to e.s.u•cm²"""    #
        q = self.esu_to_cm2()                                   #
        return q.cm2_to_au()                                    #
    #-----------------------------------------------------------#

    #-----------------------------------------------------------#
    def au_to_buck(self) -> Quadrupole:                         #
        """Convert from Hartree atomic units to Buckingham"""   #
        q = self.au_to_cm2()                                    #
        q = q.cm2_to_esu()                                      #
        return q.esu_to_buck()                                  #
                                                                #
    def buck_to_au(self) -> Quadrupole:                         #
        """Convert from Buckingham to Hartree atomic units"""   #
        q = self.buck_to_esu()                                  #
        q = q.esu_to_cm2()                                      #
        return q.cm2_to_au()                                    #
    #-----------------------------------------------------------#

    def as_unit(self, units: str) -> Quadrupole:
        """Return quadrupole as a specified unit"""
        self_units = self.units
        new_units = units.lower()
        if new_units not in ["au", "buckingham", "cm^2", "esu"]:
            raise ValueError(f"Unit {units} not recognized, please select from ( 'au', 'buckingham', 'cm^2', 'esu' )")
        
        if self_units == new_units:
            return self

        match (self_units, new_units):
            case ("buckingham", "au"):
                return self.buck_to_au()
            case ("buckingham", "cm^2"):
                return self.buck_to_cm2()
            case ("buckingham", "esu"):
                return self.buck_to_esu()
            case ("au", "buckingham"):
                return self.au_to_buck()
            case ("au", "cm^2"):
                return self.au_to_cm2()
            case ("au", "esu"):
                return self.au_to_esu()
            case ("esu", "buckingham"):
                return self.esu_to_buck()
            case ("esu", "cm^2"):
                return self.esu_to_cm2()
            case ("esu", "au"):
                return self.esu_to_au()
            case ("cm^2", "buck"):
                return self.cm2_to_buck()
            case ("cm^2", "au"):
                return self.cm2_to_au()
            case ("cm^2", "esu"):
                return self.cm2_to_esu()


    @classmethod
    def from_orca(cls, file: PathLike) -> tuple[Quadrupole]:
        """Read an ORCA output and pull the quadrupole moment(s) from it.

        Returns
        -------
        quad_matrices : tuple[Quadrupole]
            Tuple containing quadrupoles. See Notes for explanation of why
            this can return multiple matrices instead of just one.

        Note
        ----
        For ORCA outputs with methods that produce multiple densities (post-HF methods usually),
        there can be multiple quadrupoles listed. In this case it is up to the user to pull the correct
        quadrupole from the output. Typically, whichever is listed last is the one that is the most accurate
        to the given level of theory of the calculation, but this should be double checked.
        """

        with open(file, "r") as output:
            quadrupoles = []
            for line in output:
                if line.strip().endswith("(Buckingham)"):
                    quadrupoles.append(line.strip().split()[:-1])

        quad_matrices = []
        for quad in quadrupoles[::2]:
            quad_matrix = np.array(
                [
                    [quad[0], quad[3], quad[4]],
                    [quad[3], quad[1], quad[5]],
                    [quad[4], quad[5], quad[2]],
                ], dtype=np.float64
            )
            quad_matrices.append(quad_matrix)

        quads = tuple(Quadrupole(quad, units="Buckingham") for quad in quad_matrices)

        return quads


    def inertialize(self, geometry: Geometry) -> Quadrupole:
        """Rotate the quadrupole into the inertial frame of the given molecular geometry."""
        eigenvalues, eigenvectors = geometry.calc_principal_moments()
        q = np.real_if_close(np.linalg.inv(eigenvectors) @ self.quadrupole @ eigenvectors, tol=1e-8)
        return Quadrupole(q, units=self.units)


    def detrace(self) -> Quadrupole:
        """Apply detracing operation to a quadrupole

        Notes
        -----
        This detracing operator subtracts out the average of the trace of the quadrupole matrix, then multiplies by 3/2.
        The factor of 3/2 comes from the definition of the traceless quadrupole moment from
        Buckingham (1959) (https://doi.org/10.1039/QR9591300183). This is also the form that the NIST CCCBDB
        reports quadrupole moments in.

        It is also important to note that while this is a common definition, there are arguments both for and against the use of the
        traceless quadrupole moment. See https://doi.org/10.1080/00268977500101151 for further discussion.

        ORCA uses a similar definition but instead uses a factor of 3 instead of 3/2.
        Quantum ESPRESSO does not detrace the quadrupole moment.
        """
        q = (3 / 2) * (self.quadrupole - (np.eye(3,3) * (np.trace(self.quadrupole) / 3)))
        return Quadrupole(q, units=self.units)


    def compare(self, expt: Quadrupole):
        """Attempt to align a diagonal calculated quadrupole moment with an experimental quadrupole moment.
        
        Note
        ----
        This code does not guarantee a correct comparison, it simply uses statistical analysis to attempt to
        rotate a calculated quadrupole into the correct frame to be compared to an experimental quadrupole.
        """

        calc_quad = np.diag(self.quadrupole)
        expt_quad = np.diag(expt.quadrupole)

        expt_signs = np.sign(expt_quad)
        calc_signs = np.sign(calc_quad)

        if expt_signs.sum() != calc_signs.sum():
            calc_quad = calc_quad * np.array([-1., -1., -1.])

        permutations = [
            np.array([calc_quad[0], calc_quad[1], calc_quad[2]]), # abc
            np.array([calc_quad[0], calc_quad[2], calc_quad[1]]), # acb
            np.array([calc_quad[2], calc_quad[1], calc_quad[0]]), # cba
            np.array([calc_quad[2], calc_quad[0], calc_quad[1]]), # cab
            np.array([calc_quad[1], calc_quad[0], calc_quad[2]]), # bac
            np.array([calc_quad[1], calc_quad[2], calc_quad[0]]), # bca
        ]

        diffs = []
        for perm in permutations:
            diffs.append([perm, perm - np.array(expt_quad)])

        diffs.sort(key=lambda x: np.std(x[1]))

        best_match = min(diffs, key=lambda x: np.sum(np.abs(x[1])))

        best_quad = best_match[0]

        return Quadrupole(best_quad)


    def __repr__(self):
        quad = self.quadrupole
        self_str  = ""
        if self.units in ["buckingham", "au"]:
            self_str += f"{"Quadrupole Moment":18}({self.units}):      {"(xx)":10} {"(yy)":10} {"(zz)":10} {"(xy)":10} {"(xz)":10} {"(yz)":10}\n"
            self_str += f"{"":15}{" "*len(self.units)}Total: {quad[0,0]:10.5f} {quad[1,1]:10.5f} {quad[2,2]:10.5f} {quad[0,1]:10.5f} {quad[0,2]:10.5f} {quad[1,2]:10.5f}\n"
        else:
            self_str += f"{"Quadrupole Moment":18}({self.units}):      {"(xx)":13} {"(yy)":13} {"(zz)":13} {"(xy)":13} {"(xz)":13} {"(yz)":13}\n"
            self_str += f"{"":15}{" "*len(self.units)}Total: {quad[0,0]:13.5e} {quad[1,1]:13.5e} {quad[2,2]:13.5e} {quad[0,1]:13.5e} {quad[0,2]:13.5e} {quad[1,2]:13.5e}\n"
        return self_str
    

    def __add__(self, quad: Quadrupole) -> Quadrupole:
        q1 = self.quadrupole
        q2 = quad.as_unit(self.units)
        q2 = q2.quadrupole
        return Quadrupole(quadrupole=q1+q2, units=self.units)
    

    def __sub__(self, quad: Quadrupole) -> Quadrupole:
        q1 = self.quadrupole
        q2 = quad.as_unit(self.units)
        q2 = q2.quadrupole
        return Quadrupole(quadrupole=q1-q2, units=self.units)


    def __getitem__(self, index):
        return self.quadrupole[index]

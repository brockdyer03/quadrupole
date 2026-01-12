from __future__ import annotations
import numpy as np
import numpy.typing as npt
from os import PathLike
from pathlib import Path
from enum import Enum
from dataclasses import dataclass


class FileFormatError(Exception):
    """Exception raised when a file is improperly formatted"""

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


@dataclass
class ElementData:
    symbol: str
    number: int
    mass: float


class Element(ElementData, Enum):
    def __new__(cls, symbol: str, number: int, mass: float):
        element = ElementData.__new__(cls)
        element._value_ = ElementData(symbol, number, mass)
        element._add_alias_(symbol)
        element._add_value_alias_(symbol)
        element._add_value_alias_(number)
        return element

    def __str__(self):
        return self.symbol

    Unknown       = "Xx", 0,   0.0
    Hydrogen      = "H",  1,   1.0080
    Helium        = "He", 2,   4.002602
    Lithium       = "Li", 3,   6.94
    Beryllium     = "Be", 4,   9.0121831
    Boron         = "B",  5,   10.81
    Carbon        = "C",  6,   12.011
    Nitrogen      = "N",  7,   14.007
    Oxygen        = "O",  8,   15.999
    Fluorine      = "F",  9,   18.998403162
    Neon          = "Ne", 10,  20.1797
    Sodium        = "Na", 11,  22.98976928
    Magnesium     = "Mg", 12,  24.305
    Aluminum      = "Al", 13,  26.9815384
    Silicon       = "Si", 14,  28.085
    Phosphorus    = "P",  15,  30.973761998
    Sulfur        = "S",  16,  32.06
    Chlorine      = "Cl", 17,  35.45
    Argon         = "Ar", 18,  39.95
    Potassium     = "K",  19,  39.0983
    Calcium       = "Ca", 20,  40.078
    Scandium      = "Sc", 21,  44.955907
    Titanium      = "Ti", 22,  47.867
    Vanadium      = "V",  23,  50.9415
    Chromium      = "Cr", 24,  51.9961
    Manganese     = "Mn", 25,  54.938043
    Iron          = "Fe", 26,  55.845
    Cobalt        = "Co", 27,  58.933194
    Nickel        = "Ni", 28,  58.6934
    Copper        = "Cu", 29,  63.546
    Zinc          = "Zn", 30,  65.38
    Gallium       = "Ga", 31,  69.723
    Germanium     = "Ge", 32,  72.630
    Arsenic       = "As", 33,  74.921595
    Selenium      = "Se", 34,  78.971
    Bromine       = "Br", 35,  79.904
    Krypton       = "Kr", 36,  83.798
    Rubidium      = "Rb", 37,  85.4678
    Strontium     = "Sr", 38,  87.62
    Yttrium       = "Y",  39,  88.905838
    Zirconium     = "Zr", 40,  91.222
    Niobium       = "Nb", 41,  92.90637
    Molybdenum    = "Mo", 42,  95.95
    Technetium    = "Tc", 43,  97.0
    Ruthenium     = "Ru", 44,  101.07
    Rhodium       = "Rh", 45,  102.90549
    Palladium     = "Pd", 46,  106.42
    Silver        = "Ag", 47,  107.8682
    Cadmium       = "Cd", 48,  112.414
    Indium        = "In", 49,  114.818
    Tin           = "Sn", 50,  118.710
    Antimony      = "Sb", 51,  121.760
    Tellurium     = "Te", 52,  127.60
    Iodine        = "I",  53,  126.90447
    Xenon         = "Xe", 54,  131.293
    Cesium        = "Cs", 55,  132.90545196
    Barium        = "Ba", 56,  137.327
    Lanthanum     = "La", 57,  138.90547
    Cerium        = "Ce", 58,  140.116
    Praseodymium  = "Pr", 59,  140.90766
    Neodymium     = "Nd", 60,  144.242
    Promethium    = "Pm", 61,  145.0
    Samarium      = "Sm", 62,  150.36
    Europium      = "Eu", 63,  151.964
    Gadolinium    = "Gd", 64,  157.249
    Terbium       = "Tb", 65,  158.925354
    Dysprosium    = "Dy", 66,  162.500
    Holmium       = "Ho", 67,  164.930329
    Erbium        = "Er", 68,  167.259
    Thulium       = "Tm", 69,  168.934219
    Ytterbium     = "Yb", 70,  173.045
    Lutetium      = "Lu", 71,  174.96669
    Hafnium       = "Hf", 72,  178.486
    Tantalum      = "Ta", 73,  180.94788
    Tungsten      = "W",  74,  183.84
    Rhenium       = "Re", 75,  186.207
    Osmium        = "Os", 76,  190.23
    Iridium       = "Ir", 77,  192.217
    Platinum      = "Pt", 78,  195.084
    Gold          = "Au", 79,  196.966570
    Mercury       = "Hg", 80,  200.592
    Thallium      = "Tl", 81,  204.38
    Lead          = "Pb", 82,  207.2
    Bismuth       = "Bi", 83,  208.98040
    Polonium      = "Po", 84,  209.0
    Astatine      = "At", 85,  210.0
    Radon         = "Rn", 86,  222.0
    Francium      = "Fr", 87,  223.0
    Radium        = "Ra", 88,  226.0
    Actinium      = "Ac", 89,  227.0
    Thorium       = "Th", 90,  232.0377
    Protactinium  = "Pa", 91,  231.03588
    Uranium       = "U",  92,  238.02891
    Neptunium     = "Np", 93,  237.0
    Plutonium     = "Pu", 94,  244.0
    Americium     = "Am", 95,  243.0
    Curium        = "Cm", 96,  247.0
    Berkelium     = "Bk", 97,  247.0
    Californium   = "Cf", 98,  251.0
    Einsteinium   = "Es", 99,  252.0
    Fermium       = "Fm", 100, 257.0
    Mendelevium   = "Md", 101, 258.0
    Nobelium      = "No", 102, 259.0
    Lawrencium    = "Lr", 103, 262.0
    Rutherfordium = "Rf", 104, 267.0
    Dubnium       = "Db", 105, 270.0
    Seaborgium    = "Sg", 106, 269.0
    Bohrium       = "Bh", 107, 270.0
    Hassium       = "Hs", 108, 270.0
    Meitnerium    = "Mt", 109, 278.0
    Darmstadtium  = "Ds", 110, 281.0
    Roentgenium   = "Rg", 111, 281.0
    Copernicium   = "Cn", 112, 285.0
    Nihonium      = "Nh", 113, 286.0
    Flerovium     = "Fl", 114, 289.0
    Moscovium     = "Mc", 115, 289.0
    Livermorium   = "Lv", 116, 293.0
    Tennessine    = "Ts", 117, 293.0
    Oganesson     = "Og", 118, 294.0


class Atom:
    """Class containing the information of a single atom.

    Attributes
    ----------
    element : Element | str | int
        A member of the `Element` enum, or the atomic symbol/number.
    xyz : ArrayLike
        The x-, y-, and z-coordinates of the atom in Ångstrom.
    """

    def __init__(
        self,
        element: Element | str | int,
        xyz: npt.ArrayLike,
    ):
        self.element = Element(element)
        self.xyz = np.array(xyz, dtype=np.float64)

    def __repr__(self):
        return (
            f"{"Element":12}{"X":11}{"Y":11}{"Z":11}\n"
            f"{self.element:9}{self.xyz[0]:11.6f}{self.xyz[1]:11.6f}{self.xyz[2]:11.6f}\n"
        )

    def __eq__(self, other: Atom):
        return (self.element == other.element and (self.xyz == other.xyz).all())


class Geometry:
    """Class storing the geometric parameters of a molecular geometry or crystal structure.
    All quantities should be in Ångstrom
    
    Attributes
    ----------
    atoms : list[Atom]
        The atoms in the geometry.
    lat_vec : npt.NDArray or None, default=None
        The primitive lattice vectors of the geometry,
    alat : float or None, default=None
        The lattice parameter.

    Notes
    -----
    Alat is calculated by taking the square root of the sum 
    of the first row of the lattice vector matrix.
    """

    bohr_to_angstrom = 0.529177210544

    def __init__(
        self,
        atoms: list[Atom],
        lat_vec: npt.NDArray | None = None,
        alat: float | None = None,
    ):
        self.atoms   = atoms
        self.lat_vec = np.array(lat_vec, dtype=float) if lat_vec is not None else None
        self.alat    = float(alat) if alat is not None else None


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

        expected_num_atoms = int(molecule_xyz[0][0])

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
    def from_list(cls, elements: list[str], xyzs: npt.NDArray) -> Geometry:
        if len(elements) != len(xyzs):
            raise ValueError("The list of elements and coordinates must be of the same size!")

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
                            f"Error reading file '{Path(file).resolve()}', "
                            "can not find final geometry"
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
    def from_cube(cls, cube_file: PathLike) -> Geometry:
        """Read in and interpret crystallographic information from a CUBE file.

        Note
        ----
        This function calculates the dimensions of the unit cell by taking the number
        of grid points and muliplying it by the spacing of the grid points. Due to rounding
        errors when an external program prints a CUBE file, the cell dimensions can have 
        inaccuracies of +/- 5e-05 angstrom.
        """

        with open(cube_file, "r") as cube:
            # Skip over header
            for _ in range(2):
                next(cube)

            num_atoms = int(next(cube).strip().split()[0])

            grid_info = [next(cube).strip().split() for _ in range(3)]
            grid_points = [int(i[0]) for i in grid_info]

            lat_vec = []
            for i, dim in enumerate(grid_info):
                lat_vec.append([grid_points[i]*float(dim[j]) for j in range(1,4)])

            lat_vec = np.array(lat_vec) * Geometry.bohr_to_angstrom

            alat = np.sqrt(np.sum(lat_vec[0,:] ** 2))

            atom_data = [next(cube).strip().split() for _ in range(num_atoms)]

            atoms = []
            for atom in atom_data:
                element = int(atom[0])
                coordinate = np.array(atom[2:5], dtype=float) * Geometry.bohr_to_angstrom
                atoms.append(Atom(element, coordinate))

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


class Quadrupole:
    """Class containing data and functions required for analyzing a quadrupole moment.

    Attributes
    ----------
    quadrupole : ndarray
        Array containing the 3x3 quadrupole matrix, the diagonal components of the quadrupole (shape 3x1),
        or the 6 independent elements of the quadrupole (shape 6x1, in order, [xx, yy, zz, xy, xz, yz])
    units : {"au", "buckingham", "cm2", "esu"}, default="buckingham"
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

    # https://physics.nist.gov/cgi-bin/cuu/Value?aueqm
    au_to_cm2_conversion   = 4.4865515185e-40

    # Coulomb*m^2 to CGS statCoulomb*cm^2
    # Factor of c * (100cm)^2/m^2
    # c taken from https://physics.nist.gov/cgi-bin/cuu/Value?c
    cm2_to_esu_conversion  = 2.99792458e13

    # Suggested by Peter J. W. Debye in 1963
    # https://doi.org/10.1021/cen-v041n016.p040
    esu_to_buck_conversion = 1e-26

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
        if units not in ["au", "buckingham", "cm2", "esu"]:
            raise ValueError("Invalid units, please select from ( 'au', 'buckingham', 'cm2', 'esu' )")
        else:
            self.units = units


    #-----------------------------------------------------------#
    def au_to_cm2(self) -> Quadrupole:                          #
        """Convert from Hartree atomic units to Coulomb•m²"""   #
        q = self.quadrupole * Quadrupole.au_to_cm2_conversion   #
        return Quadrupole(q, units="cm2")                      #
                                                                #
    def cm2_to_au(self) -> Quadrupole:                          #
        """Convert from Coulomb•m² to Hartree atomic units"""   #
        q = self.quadrupole / Quadrupole.au_to_cm2_conversion   #
        return Quadrupole(q, units="au")                        #
    #-----------------------------------------------------------#

    #-----------------------------------------------------------#
    def cm2_to_esu(self) -> Quadrupole:                         #
        """Convert from Coulomb•m² to e.s.u•cm²"""              #
        q = self.quadrupole * Quadrupole.cm2_to_esu_conversion  #
        return Quadrupole(q, units="esu")                       #
                                                                #
    def esu_to_cm2(self) -> Quadrupole:                         #
        """Convert from e.s.u•cm² to Coulomb•m²"""              #
        q = self.quadrupole / Quadrupole.cm2_to_esu_conversion  #
        return Quadrupole(q, units="cm2")                      #
    #-----------------------------------------------------------#

    #-----------------------------------------------------------#
    def buck_to_esu(self) -> Quadrupole:                        #
        """Convert from Buckingham to e.s.u•cm²"""              #
        q = self.quadrupole * Quadrupole.esu_to_buck_conversion #
        return Quadrupole(q, units="esu")                       #
                                                                #
    def esu_to_buck(self) -> Quadrupole:                        #
        """Convert from e.s.u•cm² to Buckingham"""              #
        q = self.quadrupole / Quadrupole.esu_to_buck_conversion #
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
        if new_units not in ["au", "buckingham", "cm2", "esu"]:
            raise ValueError(f"Unit {units} not recognized, please select from ( 'au', 'buckingham', 'cm2', 'esu' )")

        if self_units == new_units:
            return self

        match (self_units, new_units):
            case ("buckingham", "au"):
                return self.buck_to_au()
            case ("buckingham", "cm2"):
                return self.buck_to_cm2()
            case ("buckingham", "esu"):
                return self.buck_to_esu()
            case ("au", "buckingham"):
                return self.au_to_buck()
            case ("au", "cm2"):
                return self.au_to_cm2()
            case ("au", "esu"):
                return self.au_to_esu()
            case ("esu", "buckingham"):
                return self.esu_to_buck()
            case ("esu", "cm2"):
                return self.esu_to_cm2()
            case ("esu", "au"):
                return self.esu_to_au()
            case ("cm2", "buckingham"):
                return self.cm2_to_buck()
            case ("cm2", "au"):
                return self.cm2_to_au()
            case ("cm2", "esu"):
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
        if not isinstance(expt, Quadrupole):
            expt = Quadrupole(expt)

        calc_quad = np.diag(self.quadrupole)
        expt_quad = np.diag(expt.quadrupole)

        expt_signs = np.sign(expt_quad)
        calc_signs = np.sign(calc_quad)

        invert_sign = False
        if expt_signs.sum() != calc_signs.sum():
            invert_sign = True
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

        if invert_sign:
            best_quad = -best_match[0]
        else:
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

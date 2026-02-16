from __future__ import annotations
from os import PathLike
from pathlib import Path
import numpy as np
import numpy.typing as npt
from .geometry import Geometry, FileFormatError


class Quadrupole:
    """Class containing data and functions required for analyzing a
    quadrupole moment.

    Parameters
    ----------
    quadrupole : ArrayLike
        Sequence containing the 3x3 quadrupole matrix, the diagonal
        components of the quadrupole (shape 3x1, ``[aa, bb, cc]``),
        or the 6 independent elements of the quadrupole
        (shape 6x1, in order, ``[aa, bb, cc, ab, ac, bc]``).
    units : {"au", "buckingham", "cm2", "esu"}, default="buckingham"
        Units of the quadrupole matrix (case insensitive).

    Attributes
    ----------
    quadrupole : NDArray
        3x3 array of floats.
    units : {"au", "buckingham", "cm2", "esu"}
        Units of the quadrupole matrix (case insensitive).

    Methods
    -------
    as_unit(units)
        Represent the quadrupole in a given unit.
    from_orca(output_path)
        Read all quadrupoles from an ORCA output.
    inertialize(geometry)
        Rotate a quadrupole into a molecule's inertial frame.
    detrace()
        Apply a detracing operator to a quadrupole
    compare(expt)
        Statistically compare two quadrupoles and return the best match.

    Notes
    -----
    The attributes specify that there are 6 independent elements of a
    quadrupole tensor. This is because a molecular quadrupole, by
    definition, is symmetric. It is worth noting however that a
    traceless quadrupole moment only has 5 independent elements as being
    traceless dictates that one of the diagonal components must be equal
    to the negative sum of the remaining two, i.e. it is required that
    :math:`Q_{aa} + Q_{bb} = -2Q_{cc}`, therefore 
    :math:`Q_{cc}` depends on :math:`Q_{aa}`
    and :math:`Q_{bb}`.
    """

    au_to_cm2_conversion   = 4.4865515185e-40
    cm2_to_esu_conversion  = 2.99792458e13
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
            raise ValueError(
                f"Cannot cast array of shape {quadrupole.shape} to a quadrupole, supply either shape (3, 3) or (3,) or (6,)!"
            )

        units = units.lower()
        if units not in ["au", "buckingham", "cm2", "esu"]:
            raise ValueError(
                "Invalid units, please select from ( 'au', 'buckingham', 'cm2', 'esu' )"
            )
        else:
            self.units = units


    #-----------------------------------------------------------#
    def au_to_cm2(self) -> Quadrupole:                          #
        """Convert from Hartree atomic units to Coulomb•m²"""   #
        q = self.quadrupole * Quadrupole.au_to_cm2_conversion   #
        return Quadrupole(q, units="cm2")                       #
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
        return Quadrupole(q, units="cm2")                       #
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
        """Return quadrupole as a specified unit.
        
        Parameters
        ----------
        units : {"au", "buckingham", "cm2", "esu"}
            Name of the desired units.

        Notes
        -----
        au_to_cm2_conversion : 4.4865515185e-40
            NIST CODATA [1]_.
        cm2_to_esu_conversion : 2.99792458e13
            Conversion from Coulomb•m² to statCoulomb•cm². Equal to a
            factor of `c`•(100 cm)² / m². Value of `c` from the NIST
            CODATA [2]_.
        esu_to_buck_conversion : 1e-26
            Suggested by Peter J. W. Debye in 1963 [3]_.

        References
        ----------
        .. [1] CODATA Value: atomic unit of electric quadrupole moment.
           https://physics.nist.gov/cgi-bin/cuu/Value?aueqm (accessed 2026-02-13).
        .. [2] CODATA Value: speed of light in vacuum.
           https://physics.nist.gov/cgi-bin/cuu/Value?c (accessed 2026-02-13).
        .. [3] Birefringence Gives C02's Molecular Quadrupole Moment.
           Chem. Eng. News Archive 1963, 41 (16), 40-43.
           https://doi.org/10.1021/cen-v041n016.p040.
        """
        self_units = self.units
        new_units = units.lower()
        if new_units not in ["au", "buckingham", "cm2", "esu"]:
            raise ValueError(
                f"Unit {units} not recognized, please pick from ('au', 'buckingham', 'cm2', 'esu')"
            )

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
    def from_orca(cls, file: PathLike):
        """Read an ORCA output and pull out the quadrupole moment(s).

        Returns
        -------
        quad_matrices : tuple[Quadrupole]
            Tuple containing quadrupoles. See Notes for explanation of 
            why this can return multiple matrices instead of just one.

        Notes
        -----
        For ORCA outputs with methods that produce multiple densities
        (post-HF methods usually), there can be multiple quadrupoles
        listed. In this case it is up to the user to pull the correct
        quadrupole from the output. Typically, whichever is listed last
        is the one that is the most accurate to the given level of
        theory of the calculation, but this should be double checked.
        """

        with open(file, "r") as output:
            quadrupoles = []
            for line in output:
                if line.strip().endswith("(Buckingham)"):
                    quadrupoles.append(line.strip().split()[:-1])

        if len(quadrupoles) == 0:
            raise FileFormatError(
                "Could not locate a quadrupole moment in output "
               f"{Path(file).resolve()}"
            )

        quad_matrices = []
        # This list slice is a neat trick to only grab every other element.
        for quad in quadrupoles[::2]:
            quad_matrix = np.array(
                [
                    [quad[0], quad[3], quad[4]],
                    [quad[3], quad[1], quad[5]],
                    [quad[4], quad[5], quad[2]],
                ], dtype=np.float64
            )
            quad_matrices.append(quad_matrix)

        quads = tuple(
            cls(quad, units="Buckingham") for quad in quad_matrices
        )

        return quads


    def inertialize(self, geometry: Geometry) -> Quadrupole:
        """Rotate the quadrupole into the inertial frame of the given
        molecular geometry.
        """
        eigenvalues, eigenvectors = geometry.calc_principal_moments()
        q = np.real_if_close(
            np.linalg.inv(eigenvectors) @ self.quadrupole @ eigenvectors, tol=1e-8
        )
        return Quadrupole(q, units=self.units)


    def detrace(self) -> Quadrupole:
        """Apply detracing operation to a quadrupole.

        Notes
        -----
        This detracing operator subtracts out the average of the trace
        of the quadrupole matrix, then multiplies by 3/2. The factor of
        3/2 comes from the definition of the traceless quadrupole moment
        from Buckingham (1959) [1]_.
        This is also the form that the NIST CCCBDB reports quadrupole
        moments in.

        It is also important to note that while this is a common
        definition, there are arguments both for and against the use of
        the traceless quadrupole moment. See Raab (1974) for further
        discussion [2]_.

        ORCA uses a similar definition but uses a factor of 3 instead
        of 3/2.
        Quantum ESPRESSO does not detrace the quadrupole moment.

        References
        ----------
        .. [1] Buckingham, A. D. Molecular Quadrupole Moments. 
           Q. Rev. Chem. Soc. 1959, 13 (3), 183-214. 
           https://doi.org/10.1039/QR9591300183.

        .. [2] Raab, R. E. Magnetic Multipole Moments.
           Molecular Physics 1975, 29 (5), 1323-1331.
           https://doi.org/10.1080/00268977500101151.

        """
        q = (
            (3 / 2)
            * (
                self.quadrupole
                - (np.eye(3,3) * (np.trace(self.quadrupole) / 3))
            )
        )
        return Quadrupole(q, units=self.units)


    def compare(self, expt: Quadrupole):
        """Attempt to align a diagonal calculated quadrupole moment with
        an experimental quadrupole moment.

        Notes
        -----
        This function defaults to ensuring all units are those of
        ``expt`` and will return a ``Quadrupole`` with those units.

        This code does not guarantee a correct comparison, it simply
        uses statistical analysis to attempt to permute a calculated
        quadrupole into the correct frame to be compared to an
        experimental quadrupole.
        """
        if not isinstance(expt, Quadrupole):
            expt = Quadrupole(expt, units=self.units)
        
        calc_quad = np.diag(self.as_unit(expt.units).quadrupole)
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

        return Quadrupole(best_quad, expt.units)


    def __repr__(self):
        quad = self.quadrupole
        self_str  = ""
        if self.units in ["buckingham", "au"]:
            self_str += f"{"Quadrupole":11}({self.units}):      {"(xx)":10} {"(yy)":10} {"(zz)":10} {"(xy)":10} {"(xz)":10} {"(yz)":10}\n"
            self_str += f"{"":8}{" "*len(self.units)}Total: {quad[0,0]:10.5f} {quad[1,1]:10.5f} {quad[2,2]:10.5f} {quad[0,1]:10.5f} {quad[0,2]:10.5f} {quad[1,2]:10.5f}\n"
        else:
            self_str += f"{"Quadrupole":11}({self.units}):      {"(xx)":13} {"(yy)":13} {"(zz)":13} {"(xy)":13} {"(xz)":13} {"(yz)":13}\n"
            self_str += f"{"":8}{" "*len(self.units)}Total: {quad[0,0]:13.5e} {quad[1,1]:13.5e} {quad[2,2]:13.5e} {quad[0,1]:13.5e} {quad[0,2]:13.5e} {quad[1,2]:13.5e}\n"
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

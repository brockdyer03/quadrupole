from enum import Enum
from dataclasses import dataclass


@dataclass(frozen=True)
class ElementData:
    """Dataclass for storing element data."""
    symbol: str
    number: int
    mass: float


class Element(ElementData, Enum):
    """Enumeration of all elements on the periodic table.
    All element data here was taken from the 
    International Union of Pure and Applied Chemistry (IUPAC) [1]_.

    Notes
    -----
    If you are going to use the ``Element`` enumeration in your own
    project be aware that, as it inherits from ``Enum``, its members
    are singletons [2]_. This means that performing comparisons
    between elements should be done using ``is`` and not ``==``. The
    comparison with ``==`` invokes the ``__eq__`` method of the object
    which can take nearly twice as long as the identity comparison
    with ``is``.

    References
    ----------
    .. [1] https://iupac.qmul.ac.uk/AtWt/
    .. [2] https://docs.python.org/3/howto/enum.html#enum-members-aka-instances

    Examples
    --------
    Directly accessing an element can be done by name or symbol.
    Programmatic access can be done by symbol (as ``str``) or
    number (as ``int``).

    >>> Element.Hydrogen is Element.H is Element("H") is Element(1)
    True 

    Element access by string is case-insensitive:

    >>> Element.Ruthenium is Element("Ru") is Element("ru") is Element("rU") is Element("RU")
    True 

    You can use dot access to get an element's data:

    >>> Element.Hydrogen
    <Element.Hydrogen: symbol='H', number=1, mass=1.008>
    >>> Element.Hydrogen.name
    'Hydrogen'
    >>> Element.Hydrogen.symbol
    'H'
    >>> Element.Hydrogen.number
    1
    >>> Element.Hydrogen.mass
    1.008

    A "zero" element is available as a placeholder:

    >>> Element.Unknown is Element.Xx is Element("Xx") is Element(0)
    True 
    >>> Element.Xx.name
    'Unknown'
    >>> Element.Xx.symbol
    'Xx'
    >>> Element.Xx.number
    0
    >>> Element.Xx.mass
    0.0

    Printing an element directly calls its `__str__()` method, which
    returns the atomic symbol.

    >>> str(hydrogen)
    'H'
    >>> print(hydrogen)
    H 

    Since the ``ElementData`` dataclass is marked as frozen, you can
    create unordered sets of elements:

    >>> elem = [Element.H, Element.H, Element.C, Element.C, Element.N]
    >>> elem_set = set(elem)
    >>> len(elem_set)
    3
    >>> for e in elem_set:
    ...     print(e)
    C 
    H 
    N 

    You can also iterate through all of the elements in order of their
    atomic number:

    >>> for elem in Element:
    ...     print(elem)
    Xx 
    H 
    He 
    Li 
    Be 
    B 
    C 
    ...
    """
    def __new__(cls, symbol: str, number: int, mass: float):
        element = ElementData.__new__(cls)
        element._value_ = ElementData(symbol, number, mass)
        element._add_alias_(symbol)
        element._add_value_alias_(symbol)
        element._add_value_alias_(number)
        return element

    def __str__(self):
        return self.symbol
    
    @classmethod
    def _missing_(cls, value: str):
        return cls(value.title())

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


#: Union of all types that could be coerced into an Element.
#: These afford programmatic access by calling ``Element(ElementLike)``.
type ElementLike = Element | str | int

#[allow(unused_imports)]
use std::str::FromStr;

use std::io::{self, Read};
use std::fs::File;
use std::path::Path;

use strum_macros;



#[derive(Debug, PartialEq, Eq, strum_macros::EnumString, strum_macros::Display, strum_macros::AsRefStr)]
pub enum Element {
    H,
    He,
    Li,
    Be,
    B,
    C,
    N,
    O,
    F,
    Ne,
    Na,
    Mg,
    Al,
    Si,
    P,
    S,
    Cl,
    Ar,
    K ,
    Ca,
    Sc,
    Ti,
    V,
    Cr,
    Mn,
    Fe,
    Co,
    Ni,
    Cu,
    Zn,
    Ga,
    Ge,
    As,
    Se,
    Br,
    Kr,
    Rb,
    Sr,
    Y,
    Zr,
    Nb,
    Mo,
    Tc,
    Ru,
    Rh,
    Pd,
    Ag,
    Cd,
    In,
    Sn,
    Sb,
    Te,
    I,
    Xe,
    Cs,
    Ba,
    La,
    Ce,
    Pr,
    Nd,
    Pm,
    Sm,
    Eu,
    Gd,
    Tb,
    Dy,
    Ho,
    Er,
    Tm,
    Yb,
    Lu,
    Hf,
    Ta,
    W,
    Re,
    Os,
    Ir,
    Pt,
    Au,
    Hg,
    Tl,
    Pb,
    Bi,
    Po,
    At,
    Rn,
    Fr,
    Ra,
    Ac,
    Th,
    Pa,
    U,
    Np,
    Pu,
    Am,
    Cm,
    Bk,
    Cf,
    Es,
    Fm,
    Md,
    No,
    Lr,
    Rf,
    Db,
    Sg,
    Bh,
    Hs,
    Mt,
    Ds,
    Rg,
    Cn,
    Nh,
    Fl,
    Mc,
    Lv,
    Ts,
    Og,
}

impl Element {

    pub fn symbol(&self) -> &str {
        match *self {
            Element::H  => "H",
            Element::He => "He",
            Element::Li => "Li",
            Element::Be => "Be",
            Element::B  => "B",
            Element::C  => "C",
            Element::N  => "N",
            Element::O  => "O",
            Element::F  => "F",
            Element::Ne => "Ne",
            Element::Na => "Na",
            Element::Mg => "Mg",
            Element::Al => "Al",
            Element::Si => "Si",
            Element::P  => "P",
            Element::S  => "S",
            Element::Cl => "Cl",
            Element::Ar => "Ar",
            Element::K  => "K" ,
            Element::Ca => "Ca",
            Element::Sc => "Sc",
            Element::Ti => "Ti",
            Element::V  => "V",
            Element::Cr => "Cr",
            Element::Mn => "Mn",
            Element::Fe => "Fe",
            Element::Co => "Co",
            Element::Ni => "Ni",
            Element::Cu => "Cu",
            Element::Zn => "Zn",
            Element::Ga => "Ga",
            Element::Ge => "Ge",
            Element::As => "As",
            Element::Se => "Se",
            Element::Br => "Br",
            Element::Kr => "Kr",
            Element::Rb => "Rb",
            Element::Sr => "Sr",
            Element::Y  => "Y",
            Element::Zr => "Zr",
            Element::Nb => "Nb",
            Element::Mo => "Mo",
            Element::Tc => "Tc",
            Element::Ru => "Ru",
            Element::Rh => "Rh",
            Element::Pd => "Pd",
            Element::Ag => "Ag",
            Element::Cd => "Cd",
            Element::In => "In",
            Element::Sn => "Sn",
            Element::Sb => "Sb",
            Element::Te => "Te",
            Element::I  => "I",
            Element::Xe => "Xe",
            Element::Cs => "Cs",
            Element::Ba => "Ba",
            Element::La => "La",
            Element::Ce => "Ce",
            Element::Pr => "Pr",
            Element::Nd => "Nd",
            Element::Pm => "Pm",
            Element::Sm => "Sm",
            Element::Eu => "Eu",
            Element::Gd => "Gd",
            Element::Tb => "Tb",
            Element::Dy => "Dy",
            Element::Ho => "Ho",
            Element::Er => "Er",
            Element::Tm => "Tm",
            Element::Yb => "Yb",
            Element::Lu => "Lu",
            Element::Hf => "Hf",
            Element::Ta => "Ta",
            Element::W  => "W",
            Element::Re => "Re",
            Element::Os => "Os",
            Element::Ir => "Ir",
            Element::Pt => "Pt",
            Element::Au => "Au",
            Element::Hg => "Hg",
            Element::Tl => "Tl",
            Element::Pb => "Pb",
            Element::Bi => "Bi",
            Element::Po => "Po",
            Element::At => "At",
            Element::Rn => "Rn",
            Element::Fr => "Fr",
            Element::Ra => "Ra",
            Element::Ac => "Ac",
            Element::Th => "Th",
            Element::Pa => "Pa",
            Element::U  => "U",
            Element::Np => "Np",
            Element::Pu => "Pu",
            Element::Am => "Am",
            Element::Cm => "Cm",
            Element::Bk => "Bk",
            Element::Cf => "Cf",
            Element::Es => "Es",
            Element::Fm => "Fm",
            Element::Md => "Md",
            Element::No => "No",
            Element::Lr => "Lr",
            Element::Rf => "Rf",
            Element::Db => "Db",
            Element::Sg => "Sg",
            Element::Bh => "Bh",
            Element::Hs => "Hs",
            Element::Mt => "Mt",
            Element::Ds => "Ds",
            Element::Rg => "Rg",
            Element::Cn => "Cn",
            Element::Nh => "Nh",
            Element::Fl => "Fl",
            Element::Mc => "Mc",
            Element::Lv => "Lv",
            Element::Ts => "Ts",
            Element::Og => "Og",
        }
    }

    pub fn name(&self) -> &str {
        match *self {
            Element::H  => "Hydrogen",
            Element::He => "Helium",
            Element::Li => "Lithium",
            Element::Be => "Beryllium",
            Element::B  => "Boron",
            Element::C  => "Carbon",
            Element::N  => "Nitrogen",
            Element::O  => "Oxygen",
            Element::F  => "Fluorine",
            Element::Ne => "Neon",
            Element::Na => "Sodium",
            Element::Mg => "Magnesium",
            Element::Al => "Aluminum",
            Element::Si => "Silicon",
            Element::P  => "Phosphorus",
            Element::S  => "Sulfur",
            Element::Cl => "Chlorine",
            Element::Ar => "Argon",
            Element::K  => "Potassium",
            Element::Ca => "Calcium",
            Element::Sc => "Scandium",
            Element::Ti => "Titanium",
            Element::V  => "Vanadium",
            Element::Cr => "Chromium",
            Element::Mn => "Manganese",
            Element::Fe => "Iron",
            Element::Co => "Cobalt",
            Element::Ni => "Nickel",
            Element::Cu => "Copper",
            Element::Zn => "Zinc",
            Element::Ga => "Gallium",
            Element::Ge => "Germanium",
            Element::As => "Arsenic",
            Element::Se => "Selenium",
            Element::Br => "Bromine",
            Element::Kr => "Krypton",
            Element::Rb => "Rubidium",
            Element::Sr => "Strontium",
            Element::Y  => "Yttrium",
            Element::Zr => "Zirconium",
            Element::Nb => "Niobium",
            Element::Mo => "Molybdenum",
            Element::Tc => "Technetium",
            Element::Ru => "Ruthenium",
            Element::Rh => "Rhodium",
            Element::Pd => "Palladium",
            Element::Ag => "Silver",
            Element::Cd => "Cadmium",
            Element::In => "Indium",
            Element::Sn => "Tin",
            Element::Sb => "Antimony",
            Element::Te => "Tellurium",
            Element::I  => "Iodine",
            Element::Xe => "Xenon",
            Element::Cs => "Cesium",
            Element::Ba => "Barium",
            Element::La => "Lanthanum",
            Element::Ce => "Cerium",
            Element::Pr => "Praseodymium",
            Element::Nd => "Neodymium",
            Element::Pm => "Promethium",
            Element::Sm => "Samarium",
            Element::Eu => "Europium",
            Element::Gd => "Gadolinium",
            Element::Tb => "Terbium",
            Element::Dy => "Dysprosium",
            Element::Ho => "Holmium",
            Element::Er => "Erbium",
            Element::Tm => "Thulium",
            Element::Yb => "Ytterbium",
            Element::Lu => "Lutetium",
            Element::Hf => "Hafnium",
            Element::Ta => "Tantalum",
            Element::W  => "Tungsten",
            Element::Re => "Rhenium",
            Element::Os => "Osmium",
            Element::Ir => "Iridium",
            Element::Pt => "Platinum",
            Element::Au => "Gold",
            Element::Hg => "Mercury",
            Element::Tl => "Thallium",
            Element::Pb => "Lead",
            Element::Bi => "Bismuth",
            Element::Po => "Polonium",
            Element::At => "Astatine",
            Element::Rn => "Radon",
            Element::Fr => "Francium",
            Element::Ra => "Radium",
            Element::Ac => "Actinium",
            Element::Th => "Thorium",
            Element::Pa => "Protactinium",
            Element::U  => "Uranium",
            Element::Np => "Neptunium",
            Element::Pu => "Plutonium",
            Element::Am => "Americium",
            Element::Cm => "Curium",
            Element::Bk => "Berkelium",
            Element::Cf => "Californium",
            Element::Es => "Einsteinium",
            Element::Fm => "Fermium",
            Element::Md => "Mendelevium",
            Element::No => "Nobelium",
            Element::Lr => "Lawerencium",
            Element::Rf => "Rutherforium",
            Element::Db => "Dubnium",
            Element::Sg => "Seaborgium",
            Element::Bh => "Bohrium",
            Element::Hs => "Hassium",
            Element::Mt => "Meitnerium",
            Element::Ds => "Darmstadtium",
            Element::Rg => "Roentgenium",
            Element::Cn => "Copernicium",
            Element::Nh => "Nihonium",
            Element::Fl => "Flerovium",
            Element::Mc => "Moscovium",
            Element::Lv => "Livermorium",
            Element::Ts => "Tennessine",
            Element::Og => "Oganesson",
        }
    }

    pub fn number(&self) -> u8 {
        match *self {
            Element::H  => 1,
            Element::He => 2,
            Element::Li => 3,
            Element::Be => 4,
            Element::B  => 5,
            Element::C  => 6,
            Element::N  => 7,
            Element::O  => 8,
            Element::F  => 9,
            Element::Ne => 10,
            Element::Na => 11,
            Element::Mg => 12,
            Element::Al => 13,
            Element::Si => 14,
            Element::P  => 15,
            Element::S  => 16,
            Element::Cl => 17,
            Element::Ar => 18,
            Element::K  => 19,
            Element::Ca => 20,
            Element::Sc => 21,
            Element::Ti => 22,
            Element::V  => 23,
            Element::Cr => 24,
            Element::Mn => 25,
            Element::Fe => 26,
            Element::Co => 27,
            Element::Ni => 28,
            Element::Cu => 29,
            Element::Zn => 30,
            Element::Ga => 31,
            Element::Ge => 32,
            Element::As => 33,
            Element::Se => 34,
            Element::Br => 35,
            Element::Kr => 36,
            Element::Rb => 37,
            Element::Sr => 38,
            Element::Y  => 39,
            Element::Zr => 40,
            Element::Nb => 41,
            Element::Mo => 42,
            Element::Tc => 43,
            Element::Ru => 44,
            Element::Rh => 45,
            Element::Pd => 46,
            Element::Ag => 47,
            Element::Cd => 48,
            Element::In => 49,
            Element::Sn => 50,
            Element::Sb => 51,
            Element::Te => 52,
            Element::I  => 53,
            Element::Xe => 54,
            Element::Cs => 55,
            Element::Ba => 56,
            Element::La => 57,
            Element::Ce => 58,
            Element::Pr => 59,
            Element::Nd => 60,
            Element::Pm => 61,
            Element::Sm => 62,
            Element::Eu => 63,
            Element::Gd => 64,
            Element::Tb => 65,
            Element::Dy => 66,
            Element::Ho => 67,
            Element::Er => 68,
            Element::Tm => 69,
            Element::Yb => 70,
            Element::Lu => 71,
            Element::Hf => 72,
            Element::Ta => 73,
            Element::W  => 74,
            Element::Re => 75,
            Element::Os => 76,
            Element::Ir => 77,
            Element::Pt => 78,
            Element::Au => 79,
            Element::Hg => 80,
            Element::Tl => 81,
            Element::Pb => 82,
            Element::Bi => 83,
            Element::Po => 84,
            Element::At => 85,
            Element::Rn => 86,
            Element::Fr => 87,
            Element::Ra => 88,
            Element::Ac => 89,
            Element::Th => 90,
            Element::Pa => 91,
            Element::U  => 92,
            Element::Np => 93,
            Element::Pu => 94,
            Element::Am => 95,
            Element::Cm => 96,
            Element::Bk => 97,
            Element::Cf => 98,
            Element::Es => 99,
            Element::Fm => 100,
            Element::Md => 101,
            Element::No => 102,
            Element::Lr => 103,
            Element::Rf => 104,
            Element::Db => 105,
            Element::Sg => 106,
            Element::Bh => 107,
            Element::Hs => 108,
            Element::Mt => 109,
            Element::Ds => 110,
            Element::Rg => 111,
            Element::Cn => 112,
            Element::Nh => 113,
            Element::Fl => 114,
            Element::Mc => 115,
            Element::Lv => 116,
            Element::Ts => 117,
            Element::Og => 118,
        }
    }

    pub fn mass(&self) -> f32 {
        match *self {
            Element::H  => 1.0080,
            Element::He => 4.002602,
            Element::Li => 6.94,
            Element::Be => 9.0121831,
            Element::B  => 10.81,
            Element::C  => 12.011,
            Element::N  => 14.007,
            Element::O  => 15.999,
            Element::F  => 18.998403162,
            Element::Ne => 20.1797,
            Element::Na => 22.98976928,
            Element::Mg => 24.305,
            Element::Al => 26.9815384,
            Element::Si => 28.085,
            Element::P  => 30.973761998,
            Element::S  => 32.06,
            Element::Cl => 35.45,
            Element::Ar => 39.95,
            Element::K  => 39.0983,
            Element::Ca => 40.078,
            Element::Sc => 44.955907,
            Element::Ti => 47.867,
            Element::V  => 50.9415,
            Element::Cr => 51.9961,
            Element::Mn => 54.938043,
            Element::Fe => 55.845,
            Element::Co => 58.933194,
            Element::Ni => 58.6934,
            Element::Cu => 63.546,
            Element::Zn => 65.38,
            Element::Ga => 69.723,
            Element::Ge => 72.630,
            Element::As => 74.921595,
            Element::Se => 78.971,
            Element::Br => 79.904,
            Element::Kr => 83.798,
            Element::Rb => 85.4678,
            Element::Sr => 87.62,
            Element::Y  => 88.905838,
            Element::Zr => 91.222,
            Element::Nb => 92.90637,
            Element::Mo => 95.95,
            Element::Tc => 97.0,
            Element::Ru => 101.07,
            Element::Rh => 102.90549,
            Element::Pd => 106.42,
            Element::Ag => 107.8682,
            Element::Cd => 112.414,
            Element::In => 114.818,
            Element::Sn => 118.710,
            Element::Sb => 121.760,
            Element::Te => 127.60,
            Element::I  => 126.90447,
            Element::Xe => 131.293,
            Element::Cs => 132.90545196,
            Element::Ba => 137.327,
            Element::La => 138.90547,
            Element::Ce => 140.116,
            Element::Pr => 140.90766,
            Element::Nd => 144.242,
            Element::Pm => 145.0,
            Element::Sm => 150.36,
            Element::Eu => 151.964,
            Element::Gd => 157.249,
            Element::Tb => 158.925354,
            Element::Dy => 162.500,
            Element::Ho => 164.930329,
            Element::Er => 167.259,
            Element::Tm => 168.934219,
            Element::Yb => 173.045,
            Element::Lu => 174.96669,
            Element::Hf => 178.486,
            Element::Ta => 180.94788,
            Element::W  => 183.84,
            Element::Re => 186.207,
            Element::Os => 190.23,
            Element::Ir => 192.217,
            Element::Pt => 195.084,
            Element::Au => 196.966570,
            Element::Hg => 200.592,
            Element::Tl => 204.38,
            Element::Pb => 207.2,
            Element::Bi => 208.98040,
            Element::Po => 209.0,
            Element::At => 210.0,
            Element::Rn => 222.0,
            Element::Fr => 223.0,
            Element::Ra => 226.0,
            Element::Ac => 227.0,
            Element::Th => 232.0377,
            Element::Pa => 231.03588,
            Element::U  => 238.02891,
            Element::Np => 237.0,
            Element::Pu => 244.0,
            Element::Am => 243.0,
            Element::Cm => 247.0,
            Element::Bk => 247.0,
            Element::Cf => 251.0,
            Element::Es => 252.0,
            Element::Fm => 257.0,
            Element::Md => 258.0,
            Element::No => 259.0,
            Element::Lr => 262.0,
            Element::Rf => 267.0,
            Element::Db => 270.0,
            Element::Sg => 269.0,
            Element::Bh => 270.0,
            Element::Hs => 270.0,
            Element::Mt => 278.0,
            Element::Ds => 281.0,
            Element::Rg => 281.0,
            Element::Cn => 285.0,
            Element::Nh => 286.0,
            Element::Fl => 289.0,
            Element::Mc => 289.0,
            Element::Lv => 293.0,
            Element::Ts => 293.0,
            Element::Og => 294.0,
        }
    }
}


#[derive(Debug)]
pub struct Atom {
    pub element: Element,
    pub xyz: [f64; 3],
}


impl Atom {
    pub fn new(element: Element, xyz: [f64; 3]) -> Atom {
        Atom {
            element,
            xyz,
        }
    }

    pub fn element(&self) -> &Element {
        &self.element
    }

    pub fn xyz(&self) -> &[f64; 3] {
        &self.xyz
    }

    pub fn element_str(&self) -> &str  {
        self.element.as_ref()
    }
}

#[derive(Debug)]
pub struct Geometry {
    pub atoms: Vec<Atom>,
}


impl Geometry {
    pub fn new(atoms: Vec<Atom>) -> Geometry {
        Geometry {
            atoms
        }
    }


    pub fn from_vecs(elements: Vec<Element>, xyzs: Vec<[f64; 3]>) -> Geometry {
        if elements.len() != xyzs.len() {
            panic!("There must be as many elements as coordinates, but you supplied {0} element(s) and {1} coordinate(s)", elements.len(), xyzs.len());
        }

        let mut atoms: Vec<Atom> = vec![];

        let atom_iter = elements.into_iter().zip(xyzs);

        for (element, xyz) in atom_iter {
            atoms.push(
                Atom {
                    element,
                    xyz
                }
            )
        }

        Geometry::new(atoms)
    }


    pub fn from_xyz(xyz_path: &Path) -> Result<Geometry, io::Error> {
        let mut xyz_string = String::new();
        
        File::open(&xyz_path)?.read_to_string(&mut xyz_string)?;

        let mut xyz_string = xyz_string.lines();

        let num_atoms= match xyz_string.next() {
            Some(num) => num,
            None => panic!("XYZ File {} is improperly formatted!", xyz_path.display()),
        };

        let num_atoms: u32 = match num_atoms.trim().parse() {
            Ok(num) => num,
            Err(_) => panic!("The first line of the file should contain the number of atoms as an integer!")
        };

        let mut atoms: Vec<Atom> = vec![];

        xyz_string.next().expect("XYZ file should have a comment line after the number of atoms!");

        for row in xyz_string {
            if row.trim().len() == 0 {
                continue
            }
            let row: Vec<&str> = row.trim().split_whitespace().collect();
            let element = Element::from_str(row[0]).expect("Invalid Element in File!");
            let x: f64 = row[1].parse().expect("Improperly formatted float for coordinate!");
            let y: f64 = row[2].parse().expect("Improperly formatted float for coordinate!");
            let z: f64 = row[3].parse().expect("Improperly formatted float for coordinate!");
            let xyz: [f64; 3] = [x, y, z];

            atoms.push(Atom::new(element, xyz));
        }

        if num_atoms != atoms.len().try_into().unwrap() {
            panic!("Number of atoms does not equal number specified in file header!");
        }

        Ok(Geometry::new(atoms))

    }


    pub fn from_xsf(xsf_path: &Path) -> Result<Geometry, io::Error> {
        let mut xsf_string = String::new();
        
        File::open(&xsf_path)?.read_to_string(&mut xsf_string)?;

        let xsf_string = xsf_string.lines();

        let xsf_iter = xsf_string.into_iter().skip_while(|x| x.trim() != "PRIMCOORD");
        
        let mut xsf_iter = xsf_iter.skip(1);

        let num_atoms= match xsf_iter.next() {
            Some(num_atoms) => num_atoms,
            None => panic!("XSF File {} is improperly formatted!", xsf_path.canonicalize().unwrap().display()),
        };

        let prim_coord_header: Vec<&str> = num_atoms.trim().split_whitespace().collect();
        let prim_coord_header_check: u8 = prim_coord_header[1]
            .parse()
            .expect("XSF file is improperly formatted at PRIMCOORD header, the number of atoms must be followed by 1!");

        if prim_coord_header_check != 1 {
            panic!("XSF file is improperly formatted at PRIMCOORD header, the number of atoms must be followed by 1!")
        }

        let num_atoms: u32 = match prim_coord_header[0].parse() {
            Ok(num) => num,
            Err(_) => panic!("The first number in the PRIMCOORD header should be the number of atoms as an integer!")
        };

        let mut atoms: Vec<Atom> = vec![];

        for _ in 0..num_atoms {
            let row = xsf_iter.next().expect("Atoms should begin directly after PRIMCOORD header but they do not!");

            let row: Vec<&str> = row.trim().split_whitespace().collect();
            
            let element = Element::from_str(row[0]).expect("Invalid Element in File!");
            let x: f64 = row[1].parse().expect("Improperly formatted float for coordinate!");
            let y: f64 = row[2].parse().expect("Improperly formatted float for coordinate!");
            let z: f64 = row[3].parse().expect("Improperly formatted float for coordinate!");
            let xyz: [f64; 3] = [x, y, z];

            atoms.push(Atom::new(element, xyz));
        }

        if num_atoms != atoms.len().try_into().unwrap() {
            panic!("Number of atoms does not equal number specified in file header!");
        }

        Ok(Geometry::new(atoms))

    }

}


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn get_element_symbol() {
        let symbol = Element::H.symbol();
        assert_eq!(symbol, "H");
        let symbol = Element::Ru.symbol();
        assert_eq!(symbol, "Ru");
    }


    #[test]
    fn get_element_name() {
        let name = Element::H.name();
        assert_eq!(name, "Hydrogen");
        let name = Element::Ru.name();
        assert_eq!(name, "Ruthenium");
    }


    #[test]
    fn get_element_number() {
        let number = Element::H.number();
        assert_eq!(number, 1);
        let number = Element::Ru.number();
        assert_eq!(number, 44)
    }


    #[test]
    fn get_element_mass() {
        let mass = Element::H.mass();
        assert_eq!(mass, 1.0080);
        let mass = Element::Ru.mass();
        assert_eq!(mass, 101.07);
    }


    #[test]
    fn element_display() {
        let display_name = format!("{}", Element::H);
        assert_eq!(display_name, "H");
        let display_name = format!("{}", Element::Ru);
        assert_eq!(display_name, "Ru");
    }


    #[test]
    fn element_as_ref_str() {
        let ref_str = Element::H.as_ref();
        assert_eq!(ref_str, "H");
        let ref_str = Element::Ru.as_ref();
        assert_eq!(ref_str, "Ru");
    }


    #[test]
    fn element_from_string() {
        let element = Element::from_str("H").unwrap();
        assert_eq!(element, Element::H);
        let element = Element::from_str("Ru").unwrap();
        assert_eq!(element, Element::Ru);
    }


    #[test]
    fn atom_init() {
        let atom = Atom::new(
            Element::Ru,
            [0.0, 42.0, 137.0]
        );
        assert_eq!(atom.element, Element::Ru);
        assert_eq!(atom.xyz, [0.0, 42.0, 137.0]);
        assert_eq!(atom.xyz[0], 0.0);
        assert_eq!(atom.xyz[1], 42.0);
        assert_eq!(atom.xyz[2], 137.0);
    }


    #[test]
    fn atom_get_coords() {
        let atom = Atom::new(
            Element::Ru,
            [0.0, 42.0, 137.0],
        );
        let xyz = atom.xyz();
        assert_eq!(*xyz, [0.0, 42.0, 137.0]);
    }


    #[test]
    fn atom_replace_coords() {
        let mut atom = Atom::new(
            Element::Ru,
            [0.0, 42.0, 137.0]
        );

        atom.xyz = [42.0, 137.0, 0.0];

        assert_eq!(atom.xyz[0], 42.0);
        assert_eq!(atom.xyz[1], 137.0);
        assert_eq!(atom.xyz[2], 0.0);
    }


    #[test]
    fn atom_element_str() {
        let atom = Atom::new(
            Element::Ru,
            [0.0, 42.0, 137.0]
        );
        let name = atom.element_str();
        assert_eq!(name, "Ru");
    }


    #[test]
    fn geometry_init() {
        let atoms = vec![
            Atom::new(Element::H , [0.0, 1.0, 2.0]),
            Atom::new(Element::Ru, [3.0, 4.0, 5.0]),
        ];
        let geometry = Geometry {atoms};

        assert_eq!(geometry.atoms[0].element, Element::H);
        assert_eq!(geometry.atoms[1].element, Element::Ru);
        
        assert_eq!(geometry.atoms[0].xyz, [0.0, 1.0, 2.0]);
        assert_eq!(geometry.atoms[1].xyz, [3.0, 4.0, 5.0]);
    }


    #[test]
    fn geometry_from_vec() {
        let elements = vec![Element::H, Element::Ru];
        let xyzs = vec![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];

        let geometry = Geometry::from_vecs(elements, xyzs);

        assert_eq!(geometry.atoms[0].element, Element::H);
        assert_eq!(geometry.atoms[1].element, Element::Ru);
        
        assert_eq!(geometry.atoms[0].xyz, [0.0, 1.0, 2.0]);
        assert_eq!(geometry.atoms[1].xyz, [3.0, 4.0, 5.0]);
    }


    #[test]
    fn geometry_from_xyz_file() {
        let xyz_path = Path::new("./test_files/water.xyz");
        let geometry = Geometry::from_xyz(xyz_path).unwrap();

        assert_eq!(geometry.atoms[0].element, Element::O);
        assert_eq!(geometry.atoms[1].element, Element::H);
        assert_eq!(geometry.atoms[2].element, Element::H);

        assert_eq!(geometry.atoms[0].xyz, [-0.000000, -0.000000, -0.000000]);
        assert_eq!(geometry.atoms[1].xyz, [-0.584028, -0.760231,  0.000000]);
        assert_eq!(geometry.atoms[2].xyz, [-0.584028,  0.760231, -0.000000]);
        
        let num_atoms: u32 = geometry.atoms.len().try_into().unwrap();
        assert_eq!(num_atoms, 3);
    }


    #[test]
    fn geometry_from_xsf_file() {
        let xsf_path = Path::new("./test_files/water.xsf");
        let geometry = Geometry::from_xsf(xsf_path).unwrap();

        assert_eq!(geometry.atoms[0].element, Element::O);
        assert_eq!(geometry.atoms[1].element, Element::H);
        assert_eq!(geometry.atoms[2].element, Element::H);

        assert_eq!(geometry.atoms[0].xyz, [8.770451552, 9.174907782, 8.771938274]);
        assert_eq!(geometry.atoms[1].xyz, [9.544851428, 8.573414302, 8.771938274]);
        assert_eq!(geometry.atoms[2].xyz, [8.000511848, 8.567492748, 8.771938274]);
        
        let num_atoms: u32 = geometry.atoms.len().try_into().unwrap();
        assert_eq!(num_atoms, 3);
    }

}

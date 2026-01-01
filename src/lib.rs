mod elements;

#[allow(unused_imports)]
use std::str::FromStr;

use std::{
    fmt::{Debug},
    ops::{Add, AddAssign, Neg, Sub, SubAssign, Div, Mul, Rem},
    cmp::PartialEq,
    path::Path,
    fs::File,
    io::{self, Read},
};

use crate::elements::Element;


#[derive(Debug, Copy, Clone)]
pub struct Coordinate {
    x: f64,
    y: f64,
    z: f64,
}

impl Coordinate {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Coordinate{ x, y, z }
    }
}

impl From<[f64; 3]> for Coordinate {
    fn from(value: [f64; 3]) -> Self {
        Coordinate::new(value[0], value[1], value[2])
    }
}

impl TryFrom<Vec<f64>> for Coordinate {
    type Error = ();
    fn try_from(value: Vec<f64>) -> Result<Self, Self::Error> {
        match value.len() {
            3 => Ok(Coordinate::new(value[0], value[1], value[2])),
            _ => Err(()),
        }
    }
}

impl Neg for Coordinate {
    type Output = Self;

    fn neg(mut self) -> Self {
        self.x = -self.x;
        self.y = -self.y;
        self.z = -self.z;
        self
    }
}

impl Add for Coordinate {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Add for &Coordinate {
    type Output = Coordinate;

    fn add(self, rhs: Self) -> Self::Output {
        Coordinate::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign<&Coordinate> for Coordinate {
    fn add_assign(&mut self, rhs: &Coordinate) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl AddAssign for Coordinate {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Coordinate {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Coordinate::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Sub for &Coordinate {
    type Output = Coordinate;

    fn sub(self, rhs: Self) -> Self::Output {
        Coordinate::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for Coordinate {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl SubAssign<&Coordinate> for Coordinate {
    fn sub_assign(&mut self, rhs: &Coordinate) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl PartialEq for Coordinate {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }

    fn ne(&self, other: &Self) -> bool {
        self.x != other.x && self.y != other.y && self.z != other.z
    }
}

#[derive(Debug)]
pub struct Atom {
    pub element: Element,
    pub coord: Coordinate,
}


impl Atom {
    pub fn new(element: Element, coord: Coordinate) -> Atom {
        Atom {
            element,
            coord,
        }
    }

    pub fn element(&self) -> &Element {
        &self.element
    }

    pub fn coord(&self) -> &Coordinate {
        &self.coord
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


    pub fn from_vecs(elements: Vec<Element>, xyzs: Vec<Coordinate>) -> Geometry {
        if elements.len() != xyzs.len() {
            panic!("There must be as many elements as coordinates, but you supplied {0} element(s) and {1} coordinate(s)", elements.len(), xyzs.len());
        }

        let mut atoms: Vec<Atom> = vec![];

        let atom_iter = elements.into_iter().zip(xyzs);

        for (element, coord) in atom_iter {
            atoms.push(Atom { element, coord })
        }

        Geometry::new(atoms)
    }


    pub fn from_xyz(xyz_path: &Path) -> Result<Geometry, io::Error> {
        let mut xyz_string = String::new();
        
        File::open(&xyz_path)?.read_to_string(&mut xyz_string)?;

        let mut xyz_string = xyz_string.lines();

        let num_atoms= match xyz_string.next() {
            Some(num) => num,
            None => panic!("XYZ File {} is improperly formatted!", xyz_path.canonicalize()?.display()),
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
            let coord = Coordinate::new(x, y, z);

            atoms.push(Atom::new(element, coord));
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
            let coord = Coordinate::new(x, y, z);

            atoms.push(Atom::new(element, coord));
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
    fn element_from_string() {
        let element = Element::from_str("H").unwrap();
        assert_eq!(element, Element::H);
        let element = Element::from_str("Ru").unwrap();
        assert_eq!(element, Element::Ru);
    }


    #[test]
    fn element_from_usize() {
        let element = Element::try_from(1).unwrap();
        assert_eq!(element, Element::H);
        let element = Element::try_from(44).unwrap();
        assert_eq!(element, Element::Ru);
    }


    #[test]
    #[should_panic(expected = "Out of bounds error")]
    fn element_out_of_bounds() {
        Element::try_from(137).expect("Out of bounds error");
    }


    #[test]
    fn atom_init() {
        let atom = Atom::new(
            Element::Ru,
            Coordinate::new(0.0, 42.0, 137.0)
        );
        assert_eq!(atom.element, Element::Ru);
        assert_eq!(atom.coord, Coordinate::new(0.0, 42.0, 137.0));
        assert_eq!(atom.coord.x, 0.0);
        assert_eq!(atom.coord.y, 42.0);
        assert_eq!(atom.coord.z, 137.0);
    }


    #[test]
    fn atom_get_coords() {
        let atom = Atom::new(
            Element::Ru,
            Coordinate::new(0.0, 42.0, 137.0),
        );
        let xyz = atom.coord();
        assert_eq!(*xyz, Coordinate::new(0.0, 42.0, 137.0));
    }


    #[test]
    fn atom_replace_coords() {
        let mut atom = Atom::new(
            Element::Ru,
            Coordinate::new(0.0, 42.0, 137.0)
        );

        atom.coord = Coordinate::new(42.0, 137.0, 0.0);

        assert_eq!(atom.coord.x, 42.0);
        assert_eq!(atom.coord.y, 137.0);
        assert_eq!(atom.coord.z, 0.0);
    }


    #[test]
    fn atom_element_str() {
        let atom = Atom::new(
            Element::Ru,
            Coordinate::new(0.0, 42.0, 137.0)
        );
        let name = atom.element_str();
        assert_eq!(name, "Ru");
    }


    #[test]
    fn geometry_init() {
        let atoms = vec![
            Atom::new(Element::H , Coordinate::new(0.0, 1.0, 2.0)),
            Atom::new(Element::Ru, Coordinate::new(3.0, 4.0, 5.0)),
        ];
        let geometry = Geometry {atoms};

        assert_eq!(geometry.atoms[0].element, Element::H);
        assert_eq!(geometry.atoms[1].element, Element::Ru);
        
        assert_eq!(geometry.atoms[0].coord, Coordinate::new(0.0, 1.0, 2.0));
        assert_eq!(geometry.atoms[1].coord, Coordinate::new(3.0, 4.0, 5.0));
    }


    #[test]
    fn geometry_from_vec() {
        let elements = vec![Element::H, Element::Ru];
        let xyzs = vec![Coordinate::new(0.0, 1.0, 2.0), Coordinate::new(3.0, 4.0, 5.0)];

        let geometry = Geometry::from_vecs(elements, xyzs);

        assert_eq!(geometry.atoms[0].element, Element::H);
        assert_eq!(geometry.atoms[1].element, Element::Ru);
        
        assert_eq!(geometry.atoms[0].coord, Coordinate::new(0.0, 1.0, 2.0));
        assert_eq!(geometry.atoms[1].coord, Coordinate::new(3.0, 4.0, 5.0));
    }


    #[test]
    fn geometry_from_xyz_file() {
        let xyz_path = Path::new("./test_files/water.xyz");
        let geometry = Geometry::from_xyz(xyz_path).unwrap();

        assert_eq!(geometry.atoms[0].element, Element::O);
        assert_eq!(geometry.atoms[1].element, Element::H);
        assert_eq!(geometry.atoms[2].element, Element::H);

        assert_eq!(geometry.atoms[0].coord, Coordinate::new(-0.000000, -0.000000, -0.000000));
        assert_eq!(geometry.atoms[1].coord, Coordinate::new(-0.584028, -0.760231,  0.000000));
        assert_eq!(geometry.atoms[2].coord, Coordinate::new(-0.584028,  0.760231, -0.000000));
        
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

        assert_eq!(geometry.atoms[0].coord, Coordinate::new(8.770451552, 9.174907782, 8.771938274));
        assert_eq!(geometry.atoms[1].coord, Coordinate::new(9.544851428, 8.573414302, 8.771938274));
        assert_eq!(geometry.atoms[2].coord, Coordinate::new(8.000511848, 8.567492748, 8.771938274));
        
        let num_atoms: u32 = geometry.atoms.len().try_into().unwrap();
        assert_eq!(num_atoms, 3);
    }

}

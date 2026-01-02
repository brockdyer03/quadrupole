

use std::str::FromStr;

use std::{
    fmt::{self, Debug, Display},
    ops::{
        Neg,
        Add, AddAssign,
        Sub, SubAssign, 
        Mul, MulAssign,
        Div, DivAssign,
        Rem,
        Index,
    },
    cmp::PartialEq,
    path::Path,
    fs::File,
    io::{self, Read},
};

use crate::elements::Element;

use ndarray::{self, Array2, arr2};


#[derive(Debug, PartialEq)]
pub struct Tensor3x3(Array2<f64>);

impl Tensor3x3 {
    pub fn new(array: [[f64; 3]; 3]) -> Tensor3x3 {
        Tensor3x3(arr2(&array))
    }
}


#[derive(Debug, Copy, Clone)]
pub struct Coordinate {
    pub x: f64,
    pub y: f64,
    pub z: f64,
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

impl Index<usize> for Coordinate {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index {index} out of bounds for Coordinate (max is 2)!")
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

impl AddAssign for Coordinate {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl AddAssign<&Coordinate> for Coordinate {
    fn add_assign(&mut self, rhs: &Coordinate) {
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

impl Mul for Coordinate {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Coordinate::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Mul for &Coordinate {
    type Output = Coordinate;

    fn mul(self, rhs: Self) -> Self::Output {
        Coordinate::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl MulAssign for Coordinate {
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
    }
}

impl MulAssign<&Coordinate> for Coordinate {
    fn mul_assign(&mut self, rhs: &Coordinate) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
    }
}

impl Div for Coordinate {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Coordinate::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl Div for &Coordinate {
    type Output = Coordinate;

    fn div(self, rhs: Self) -> Self::Output {
        Coordinate::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl DivAssign for Coordinate {
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}

impl DivAssign<&Coordinate> for Coordinate {
    fn div_assign(&mut self, rhs: &Coordinate) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}

impl Rem for Coordinate {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Coordinate::new(self.x % rhs.x, self.y % rhs.y, self.z % rhs.z)
    }
}

impl Rem for &Coordinate {
    type Output = Coordinate;

    fn rem(self, rhs: Self) -> Self::Output {
        Coordinate::new(self.x % rhs.x, self.y % rhs.y, self.z % rhs.z)
    }
}

impl Coordinate {
    pub fn outer_product(&self, rhs: &Coordinate) -> Tensor3x3 {
        let mut outer: [[f64; 3]; 3] = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                outer[i][j] = self[i] * rhs[j]
            }
        }
        Tensor3x3::new(outer)
    }

    pub fn nearest_int(&self) -> Coordinate {
        Coordinate::new(self.x.round(), self.y.round(), self.z.round())
    }
}


#[derive(Debug, Clone, Copy)]
pub struct Atom {
    element: Element,
    coord: Coordinate,
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

    pub fn new_coord(mut self, new_coord: Coordinate) -> Self {
        self.coord = new_coord;
        self
    }

    pub fn element_str(&self) -> &str  {
        self.element.as_ref()
    }
}


#[derive(Debug)]
pub struct Geometry {
    atoms: Vec<Atom>,
}

#[derive(Debug, Clone)]
pub struct LenError;

impl Display for LenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Length of atoms must match length of coordinates!")
    }
}

impl Geometry {

    pub fn new(atoms: Vec<Atom>) -> Self {
        Geometry {
            atoms
        }
    }

    pub fn atoms(&self) -> &Vec<Atom> {
        &self.atoms
    }

    pub fn push(mut self, atom: Atom) -> Self {
        self.atoms.push(atom);
        self
    }

    pub fn from_vecs(elements: Vec<Element>, xyzs: Vec<Coordinate>) -> Result<Self, LenError> {
        if elements.len() != xyzs.len() {
            return Err(LenError);
        }

        let mut atoms: Vec<Atom> = vec![];

        let atom_iter = elements.into_iter().zip(xyzs);

        for (element, coord) in atom_iter {
            atoms.push(Atom { element, coord })
        }

        Ok(Geometry::new(atoms))
    }

    pub fn from_xyz(xyz_path: &Path) -> Result<Self, io::Error> {
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

    pub fn from_xsf(xsf_path: &Path) -> Result<Self, io::Error> {
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

impl Index<usize> for Geometry {
    type Output = Atom;

    fn index(&self, index: usize) -> &Self::Output {
        &self.atoms[index]
    }
}


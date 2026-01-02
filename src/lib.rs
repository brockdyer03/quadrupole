pub mod geometry;
pub mod elements;

pub use crate::elements::Element;
pub use crate::geometry::{
    Tensor3x3,
    Coordinate,
    Atom,
    Geometry,
};

#[allow(unused_imports)]
use std::{
    path::Path,
    str::FromStr,
};


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
    fn get_element_weight() {
        let weight = Element::H.weight();
        assert_eq!(weight, 1.0080);
        let weight = Element::Ru.weight();
        assert_eq!(weight, 101.07);
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
    fn coordinate_outer_product() {
        let coord = Coordinate::new(1.0, 2.0, 3.0);
        let outer = coord.outer_product(&coord);
        assert_eq!(outer, Tensor3x3::new([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0]
        ]));
    }

    #[test]
    fn atom_init() {
        let atom = Atom::new(
            Element::Ru,
            Coordinate::new(0.0, 42.0, 137.0)
        );
        assert_eq!(*atom.element(), Element::Ru);
        assert_eq!(*atom.coord(), Coordinate::new(0.0, 42.0, 137.0));
        assert_eq!(atom.coord().x, 0.0);
        assert_eq!(atom.coord().y, 42.0);
        assert_eq!(atom.coord().z, 137.0);
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

        atom = atom.new_coord(Coordinate::new(42.0, 137.0, 0.0));

        assert_eq!(atom.coord().x, 42.0);
        assert_eq!(atom.coord().y, 137.0);
        assert_eq!(atom.coord().z, 0.0);
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
        let geometry = Geometry::new(atoms);

        assert_eq!(*geometry.atoms()[0].element(), Element::H);
        assert_eq!(*geometry.atoms()[1].element(), Element::Ru);

        assert_eq!(*geometry.atoms()[0].coord(), Coordinate::new(0.0, 1.0, 2.0));
        assert_eq!(*geometry.atoms()[1].coord(), Coordinate::new(3.0, 4.0, 5.0));
    }

    #[test]
    fn geometry_from_vec() {
        let elements = vec![Element::H, Element::Ru];
        let xyzs = vec![Coordinate::new(0.0, 1.0, 2.0), Coordinate::new(3.0, 4.0, 5.0)];

        let geometry = Geometry::from_vecs(elements, xyzs).expect("Length of atoms does not match length of coordinates");

        assert_eq!(*geometry.atoms()[0].element(), Element::H);
        assert_eq!(*geometry.atoms()[1].element(), Element::Ru);
        
        assert_eq!(*geometry.atoms()[0].coord(), Coordinate::new(0.0, 1.0, 2.0));
        assert_eq!(*geometry.atoms()[1].coord(), Coordinate::new(3.0, 4.0, 5.0));
    }

    #[test]
    fn geometry_from_xyz_file() {
        let xyz_path = Path::new("./test_files/water.xyz");
        let geometry = Geometry::from_xyz(xyz_path).unwrap();

        assert_eq!(*geometry.atoms()[0].element(), Element::O);
        assert_eq!(*geometry.atoms()[1].element(), Element::H);
        assert_eq!(*geometry.atoms()[2].element(), Element::H);

        assert_eq!(*geometry.atoms()[0].coord(), Coordinate::new(-0.000000, -0.000000, -0.000000));
        assert_eq!(*geometry.atoms()[1].coord(), Coordinate::new(-0.584028, -0.760231,  0.000000));
        assert_eq!(*geometry.atoms()[2].coord(), Coordinate::new(-0.584028,  0.760231, -0.000000));
        
        let num_atoms: u32 = geometry.atoms().len().try_into().unwrap();
        assert_eq!(num_atoms, 3);
    }

    #[test]
    fn geometry_from_xsf_file() {
        let xsf_path = Path::new("./test_files/water.xsf");
        let geometry = Geometry::from_xsf(xsf_path).unwrap();

        assert_eq!(*geometry.atoms()[0].element(), Element::O);
        assert_eq!(*geometry.atoms()[1].element(), Element::H);
        assert_eq!(*geometry.atoms()[2].element(), Element::H);

        assert_eq!(*geometry.atoms()[0].coord(), Coordinate::new(8.770451552, 9.174907782, 8.771938274));
        assert_eq!(*geometry.atoms()[1].coord(), Coordinate::new(9.544851428, 8.573414302, 8.771938274));
        assert_eq!(*geometry.atoms()[2].coord(), Coordinate::new(8.000511848, 8.567492748, 8.771938274));
        
        let num_atoms: u32 = geometry.atoms().len().try_into().unwrap();
        assert_eq!(num_atoms, 3);
    }

}

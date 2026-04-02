use nalgebra::{Matrix3, Vector3, Vector6};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum QuadrupoleUnits {
    AtomicUnits,
    Buckingham,
    Cm2,
    ESU,
}

impl TryFrom<&str> for QuadrupoleUnits {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "atomicunits" => Ok(QuadrupoleUnits::AtomicUnits),
            "au" => Ok(QuadrupoleUnits::AtomicUnits),
            "buckingham" => Ok(QuadrupoleUnits::Buckingham),
            "buck" => Ok(QuadrupoleUnits::Buckingham),
            "cm2" => Ok(QuadrupoleUnits::Cm2),
            "cm^2" => Ok(QuadrupoleUnits::Cm2),
            "esu" => Ok(QuadrupoleUnits::ESU),
            "e.s.u." => Ok(QuadrupoleUnits::ESU),
            _ => Err(()),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Quadrupole {
    pub quadrupole: Matrix3<f64>,
    pub units: QuadrupoleUnits,
}

impl Quadrupole {

    pub fn new(quadrupole: Matrix3<f64>, units: QuadrupoleUnits) -> Self {
        Quadrupole{ quadrupole, units }
    }

    pub fn from_diagonal(diagonal: Vector3<f64>, units: QuadrupoleUnits) -> Self {
        Quadrupole::new(Matrix3::from_diagonal(&diagonal), units)
    }

    pub fn from_triangular(triangle: Vector6<f64>, units: QuadrupoleUnits) -> Self {
        let matrix: Matrix3<f64> = Matrix3::new(
            triangle.x, triangle.w, triangle.a,
            triangle.w, triangle.y, triangle.b,
            triangle.a, triangle.b, triangle.z,
        );
        Quadrupole::new(matrix, units)
    }

}


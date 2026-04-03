use std::fmt;

use nalgebra::{Matrix3, Vector3, Vector6};

#[derive(Debug, Clone)]
pub struct UnitError;

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid unit for quadrupole moments, please select from AtomicUnits, Buckingham, Cm2, or ESU!")
    }
}

const AU_TO_CM2_CONVERSION:   f64 = 4.4865515185e-40;
const CM2_TO_ESU_CONVERSION:  f64 = 2.99792458e13;
const ESU_TO_BUCK_CONVERSION: f64 = 1e-26;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum QuadUnits {
    AtomicUnits,
    Buckingham,
    Cm2,
    ESU,
}

impl TryFrom<&str> for QuadUnits {
    type Error = UnitError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "atomicunits" => Ok(QuadUnits::AtomicUnits),
            "au" => Ok(QuadUnits::AtomicUnits),
            "buckingham" => Ok(QuadUnits::Buckingham),
            "buck" => Ok(QuadUnits::Buckingham),
            "cm2" => Ok(QuadUnits::Cm2),
            "cm^2" => Ok(QuadUnits::Cm2),
            "esu" => Ok(QuadUnits::ESU),
            "e.s.u." => Ok(QuadUnits::ESU),
            _ => Err(UnitError),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Quadrupole {
    pub quadrupole: Matrix3<f64>,
    pub units: QuadUnits,
}

impl Quadrupole {

    pub fn new(quadrupole: Matrix3<f64>, units: QuadUnits) -> Self {
        Quadrupole{ quadrupole, units }
    }

    pub fn from_diagonal(diagonal: Vector3<f64>, units: QuadUnits) -> Self {
        Quadrupole::new(Matrix3::from_diagonal(&diagonal), units)
    }

    pub fn from_triangular(triangle: Vector6<f64>, units: QuadUnits) -> Self {
        let matrix: Matrix3<f64> = Matrix3::new(
            triangle.x, triangle.w, triangle.a,
            triangle.w, triangle.y, triangle.b,
            triangle.a, triangle.b, triangle.z,
        );
        Quadrupole::new(matrix, units)
    }

}

impl Quadrupole {

    fn au_to_cm2(quad: &Self) -> Self {
        let q = quad.quadrupole * AU_TO_CM2_CONVERSION;
        Quadrupole::new(q, QuadUnits::Cm2)
    }

    fn cm2_to_au(quad: &Self) -> Self {
        let q = quad.quadrupole / AU_TO_CM2_CONVERSION;
        Quadrupole::new(q, QuadUnits::AtomicUnits)
    }

    fn cm2_to_esu(quad: &Self) -> Self {
        let q = quad.quadrupole * CM2_TO_ESU_CONVERSION;
        Quadrupole::new(q, QuadUnits::ESU)
    }

    fn esu_to_cm2(quad: &Self) -> Self {
        let q = quad.quadrupole / CM2_TO_ESU_CONVERSION;
        Quadrupole::new(q, QuadUnits::Cm2)
    }

    fn buck_to_esu(quad: &Self) -> Self {
        let q = quad.quadrupole * ESU_TO_BUCK_CONVERSION;
        Quadrupole::new(q, QuadUnits::ESU)
    }

    fn esu_to_buck(quad: &Self) -> Self {
        let q = quad.quadrupole / ESU_TO_BUCK_CONVERSION;
        Quadrupole::new(q, QuadUnits::Buckingham)
    }

    fn cm2_to_buck(quad: &Self) -> Self {
        let q = Self::cm2_to_esu(quad);
        Self::esu_to_buck(&q)
    }

    fn buck_to_cm2(quad: &Self) -> Self {
        let q = Self::buck_to_esu(quad);
        Self::esu_to_cm2(&q)
    }

    fn au_to_esu(quad: &Self) -> Self {
        let q = Self::au_to_cm2(quad);
        Self::cm2_to_esu(&q)
    }

    fn esu_to_au(quad: &Self) -> Self {
        let q = Self::esu_to_cm2(quad);
        Self::cm2_to_au(&q)
    }

    fn au_to_buck(quad: &Self) -> Self {
        let q = Self::au_to_cm2(quad);
        let q = Self::cm2_to_esu(&q);
        Self::esu_to_buck(&q)
    }

    fn buck_to_au(quad: &Self) -> Self {
        let q = Self::buck_to_esu(quad);
        let q = Self::esu_to_cm2(&q);
        Self::cm2_to_au(&q)
    }

    pub fn as_unit(quad: &Self, new_units: &str) -> Result<Self, UnitError>  {
        let current_units = quad.units;
        let new_units = QuadUnits::try_from(new_units)?;

        match (current_units, new_units) {
            (QuadUnits::Buckingham, QuadUnits::Buckingham)  => Ok(*quad),
            (QuadUnits::AtomicUnits, QuadUnits::AtomicUnits)=> Ok(*quad),
            (QuadUnits::ESU, QuadUnits::ESU)                => Ok(*quad),
            (QuadUnits::Cm2, QuadUnits::Cm2)                => Ok(*quad),
            (QuadUnits::Buckingham, QuadUnits::AtomicUnits) => Ok(Self::buck_to_au(quad)),
            (QuadUnits::Buckingham, QuadUnits::Cm2)         => Ok(Self::buck_to_cm2(quad)),
            (QuadUnits::Buckingham, QuadUnits::ESU)         => Ok(Self::buck_to_esu(quad)),
            (QuadUnits::AtomicUnits, QuadUnits::Buckingham) => Ok(Self::au_to_buck(quad)),
            (QuadUnits::AtomicUnits, QuadUnits::Cm2)        => Ok(Self::au_to_cm2(quad)),
            (QuadUnits::AtomicUnits, QuadUnits::ESU)        => Ok(Self::au_to_esu(quad)),
            (QuadUnits::ESU, QuadUnits::Buckingham)         => Ok(Self::esu_to_buck(quad)),
            (QuadUnits::ESU, QuadUnits::Cm2)                => Ok(Self::esu_to_cm2(quad)),
            (QuadUnits::ESU, QuadUnits::AtomicUnits)        => Ok(Self::esu_to_au(quad)),
            (QuadUnits::Cm2, QuadUnits::Buckingham)         => Ok(Self::cm2_to_buck(quad)),
            (QuadUnits::Cm2, QuadUnits::AtomicUnits)        => Ok(Self::cm2_to_au(quad)),
            (QuadUnits::Cm2, QuadUnits::ESU)                => Ok(Self::cm2_to_au(quad)),
        }
    }

}

impl Quadrupole {

    pub fn compare(calc: &Self, expt: &Self) -> Self {
        todo!()
    }

}


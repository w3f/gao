use ark_ff::{FftField, Field, Zero};
use ark_poly::{DenseUVPolynomial, Polynomial};
use crate::{M, P};

/// Represents a Bezout matrix `B_i` or a composition of Bezout matrices `B_ij = B_{j-1}*...*B_i`.
#[derive(Debug, PartialEq)] // todo
pub struct BezoutMatrix<F: Field>(pub M<F>);

impl<F: FftField> BezoutMatrix<F> {
    pub fn new(r0: &P<F>, r1: &P<F>) -> Result<Self, String> {
        let minus_q = -(r0 / r1);
        Self::from_minus_quotient(minus_q)
    }

    pub fn from_minus_quotient(minus_q: P<F>) -> Result<Self, String> {
        if minus_q.is_zero() {
            return Err("zero quotient".to_string());
        }
        Ok(Self([
            P::zero(),
            P::from_coefficients_vec(vec![F::one()]),
            P::from_coefficients_vec(vec![F::one()]),
            minus_q
        ]))
    }

    pub fn id() -> Self {
        Self([
            P::from_coefficients_vec(vec![F::one()]),
            P::zero(),
            P::zero(),
            P::from_coefficients_vec(vec![F::one()])
        ])
    }

    pub fn find_r1(&self, r0: &P<F>, r1: &P<F>) -> P<F> {
        &self.0[0] * r0 + &self.0[1] * r1 // `= 0.r0 + 1.r1 = r1`
    }

    pub fn next_remainder(&self, r0: &P<F>, r1: &P<F>) -> P<F> {
        &self.0[2] * r0 + &self.0[3] * r1 // `= 1.r0 + (-r0/r1).r1 = r0 - q.r1 = r2`
    }

    // let `B := B(r0, r1)`, and `A1 := (r0, r1)`. Then `B * A1 =: A2 = (r1, r2)`, where `r2 = r0 mod r1`
    pub fn apply(&self, r0: &P<F>, r1: &P<F>) -> (P<F>, P<F>) {
        let r2 = self.next_remainder(r0, r1);
        let r1 = self.find_r1(r0, r1);
        (r1, r2)
    }

    pub fn compose(&self, other: &Self) -> Self {
        Self([
            &self.0[0] * &other.0[0] + &self.0[1] * &other.0[2],
            &self.0[0] * &other.0[1] + &self.0[1] * &other.0[3],
            &self.0[2] * &other.0[0] + &self.0[3] * &other.0[2],
            &self.0[2] * &other.0[1] + &self.0[3] * &other.0[3],
        ])
    }

    pub fn degree(&self) -> usize {
        // `B_22` should have the highest degree
        self.0[3].degree()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ark_bls12_381::Fr;
    use ark_std::test_rng;
    use crate::div;

    #[test]
    fn test_bezout_matrices() {
        let rng = &mut test_rng();

        let r0 = P::<Fr>::rand(2, rng);
        let r1 = P::<Fr>::rand(1, rng);

        let B1 = BezoutMatrix::new(&r0, &r1).unwrap();
        assert_eq!(B1.find_r1(&r0, &r1), r1);
        let r2 = B1.next_remainder(&r0, &r1);
        assert_eq!(r2, div(&r0, &r1).1);

        let B2 = BezoutMatrix::new(&r1, &r2).unwrap();
        let (r2_, r3) = B2.apply(&r1, &r2);
        assert_eq!(r2_, r2);
        assert!(r3.is_zero());

        let B13 = B2.compose(&B1);
        let (r2_, r3_) = B13.apply(&r0, &r1);
        assert_eq!(r2_, r2);
        assert_eq!(r3_, r3);
        assert_eq!(B13.degree(), 2);
        assert_eq!(B13.degree(), B13.0.iter().map(|p| p.degree()).max().unwrap());
    }
}
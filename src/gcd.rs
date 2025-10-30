use ark_ff::{FftField, Zero};
use ark_poly::{DenseUVPolynomial, Polynomial};
use crate::P;

/// For a pair of polynomials `(f, g)`, computes their greatest common divisor `gcd(f, g)`
/// and their Bézout coefficients, i.e. a pair of polynomials `(s, t)` such that `sf + tg = gcd(f,g)`,
/// in `O(d^2)` in the degree of polynomials using the Extended Euclidean Algorithm.

// ALGORITHM 3.6 from the book.
pub fn euclid<F: FftField>(f: &P<F>, g: &P<F>) -> (P<F>, P<F>, P<F>) {
    let mut r0 = f.clone();
    let mut r1 = g.clone();
    let (mut s0, mut s1) = (P::from_coefficients_slice(&[F::one()]),  P::zero());
    let (mut t0, mut t1) = (P::zero(), P::from_coefficients_slice(&[F::one()]));

    // while (k.is_some() && r0.degree() >= k.unwrap()) || (k.is_none() && !r1.is_zero()) {
    while !r1.is_zero() {
        // todo: compute the quotient and the remainder separately
        let q = &r0 / &r1;
        // let (q1, r2) = DenseOrSparsePolynomial::divide_with_q_and_r(&r0.into(), &r1.clone().into()).unwrap();
        assert_eq!(q.degree(), 1); // guarantees a "normal degree sequence"
        let r2 = r0 - &q * &r1;
        let s2 = s0 - &q * &s1;
        let t2 = t0 - &q * &t1;
        (r0, r1) = (r1, r2);
        (s0, s1) = (s1, s2);
        (t0, t1) = (t1, t2);
    }
    (r0, s0, t0)
}

#[cfg(test)]
mod tests {
    use ark_bls12_381::Fr;
    use ark_poly::Polynomial;
    use ark_std::{end_timer, start_timer, test_rng};

    use super::*;

    #[test]
    fn extended_gcd() {
        let rng = &mut test_rng();
        let d = 1023;
        let f = P::<Fr>::rand(d, rng);
        let g = P::<Fr>::rand(d - 1, rng);
        let _t_gcd = start_timer!(|| format!("EEA, deg = {d}"));
        let (r, s, t) = euclid(&f, &g);
        end_timer!(_t_gcd); // 686.490ms
        assert_eq!(r, &s * &f + &t * &g);
        assert_eq!(r.degree(), 0); // chances to get smth else with random coefficients are small
    }
}
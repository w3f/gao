use crate::{M, ME, P};
use ark_ff::{FftField, Field, Zero};
use ark_poly::{DenseUVPolynomial, EvaluationDomain, Polynomial, Radix2EvaluationDomain};
use ark_poly::univariate::DenseOrSparsePolynomial;

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
            minus_q,
        ]))
    }

    pub fn id() -> Self {
        Self([
            P::from_coefficients_vec(vec![F::one()]),
            P::zero(),
            P::zero(),
            P::from_coefficients_vec(vec![F::one()]),
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

fn quotient_sequence<F: FftField>(r0: &P<F>, r1: &P<F>) -> Vec<P<F>> {
    let d = r0.degree();
    let mut qs = Vec::with_capacity(d);
    let mut r0 = DenseOrSparsePolynomial::from(r0);
    let mut r1 = DenseOrSparsePolynomial::from(r1);
    while !r1.is_zero() {
        let (q1, r2) = r0.divide_with_q_and_r(&r1).unwrap();
        qs.push(q1);
        r0 = r1.into();
        r1 = r2.into();
    }
    qs
}

pub fn eval<F: FftField>(a: &M<F>) -> ME<F> {
    let deg2 = 2 * a[3].degree() + 1;
    let d2 = Radix2EvaluationDomain::new(deg2).unwrap();
    a.iter().map(|pi: &P<F>| pi.evaluate_over_domain_by_ref(d2))
        .collect::<Vec<_>>().try_into().unwrap()
}

pub fn interpolate<F: FftField>(a: &ME<F>) -> M<F> {
    a.iter().map(|ei| ei.interpolate_by_ref())
        .collect::<Vec<_>>().try_into().unwrap()
}

pub fn mul<F: FftField>(a: &M<F>, b: &M<F>) -> M<F> {
    let a = eval(a);
    let b = eval(b);
    let c: ME<F> = [
        &(&a[0] * &b[0]) + &(&a[1] * &b[2]),
        &(&a[0] * &b[1]) + &(&a[1] * &b[3]),
        &(&a[2] * &b[0]) + &(&a[3] * &b[2]),
        &(&a[2] * &b[1]) + &(&a[3] * &b[3]),
    ];
    interpolate(&c)
}


pub fn matrix_product_tree<F: FftField>(bs: &[M<F>]) -> M<F> {
    let n = bs.len();
    match n {
        1 => bs[0].clone(),
        _ => {
            let h = n / 2;
            let b1 = matrix_product_tree(&bs[0..h]);
            let b2 = matrix_product_tree(&bs[h..n]);
            mul(&b2, &b1)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    use ark_bls12_381::Fr;
    use ark_std::{end_timer, start_timer, test_rng};
    use crate::gcd::euclid;

    #[test]
    fn test_bezout_matrices() {
        let rng = &mut test_rng();

        let r0 = P::<Fr>::rand(2, rng);
        let r1 = P::<Fr>::rand(1, rng);

        let B1 = BezoutMatrix::new(&r0, &r1).unwrap();
        assert_eq!(B1.find_r1(&r0, &r1), r1);
        let r2 = B1.next_remainder(&r0, &r1);
        assert_eq!(r2, crate::poly_div::div(&r0, &r1).1);

        let B2 = BezoutMatrix::new(&r1, &r2).unwrap();
        let (r2_, r3) = B2.apply(&r1, &r2);
        assert_eq!(r2_, r2);
        assert!(r3.is_zero());

        let B13 = B2.compose(&B1);
        let (r2_, r3_) = B13.apply(&r0, &r1);
        assert_eq!(r2_, r2);
        assert_eq!(r3_, r3);
        assert_eq!(B13.degree(), 2);
        assert_eq!(
            B13.degree(),
            B13.0.iter().map(|p| p.degree()).max().unwrap()
        );
    }

    #[test]
    fn test_prod() {
        let rng = &mut test_rng();

        let r0 = P::<Fr>::rand(2, rng);
        let r1 = P::<Fr>::rand(1, rng);

        let bis = quotient_sequence(&r0, &r1).iter()
            .map(|qi| BezoutMatrix::from_minus_quotient(-qi.clone()).unwrap())
            .collect::<Vec<_>>();

        let a = &bis[0];
        let b = &bis[1];
        let c = a.compose(&b);
        let c_ = mul(&a.0, &b.0);
        assert_eq!(c_, c.0);
    }

    #[test]
    fn extended_gcd_2() {
        let rng = &mut test_rng();
        let d = 1024;
        let f = P::<Fr>::rand(d, rng);
        let g = P::<Fr>::rand(d - 1, rng);
        let _t_gcd = start_timer!(|| format!("naive EEA, deg = {d}"));
        let (r, s, t) = euclid(&f, &g);
        end_timer!(_t_gcd); // 686.490ms
        assert_eq!(r, &s * &f + &t * &g);
        assert_eq!(r.degree(), 0); // chances to get smth else with random coefficients are small

        let _t_fft_gcd = start_timer!(|| format!("fft EEA, deg = {d}"));
        let qs = quotient_sequence(&f, &g);
        let bs = qs.iter()
            .map(|qi| BezoutMatrix::from_minus_quotient(-qi.clone()).unwrap().0)
            .collect::<Vec<_>>();

        // let B1 = BezoutMatrix::new(&f, &g).unwrap();
        // assert_eq!(B1.0, bs[0]);
        // let r2 = B1.next_remainder(&f, &g);
        // let B2 = BezoutMatrix::new(&g, &r2).unwrap();
        // assert_eq!(B2.0, bs[1]);
        // let B13 = B2.compose(&B1);
        // assert_eq!(B13.0, mul(&bs[1], &bs[0]));
        //
        let prod = matrix_product_tree(&bs);
        // assert_eq!(B13.0, prod);
        // assert_eq!(&prod[2], &s);
        // assert_eq!(&prod[3], &t);
        let (gcd, z) = BezoutMatrix(prod).apply(&f, &g);
        end_timer!(_t_fft_gcd); // 686.490ms
        assert!(z.is_zero());
        assert_eq!(gcd, r);
    }
}

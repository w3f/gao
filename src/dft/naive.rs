use crate::dft::DftDomain;
use crate::{Poly, P};
use ark_ff::{FftField, Field};
use ark_poly::Polynomial;

pub struct NaiveDomain<F: Field> {
    n: usize,
    w: F,
}

impl<F: Field> NaiveDomain<F> {
    pub fn new(w: F, n: usize) -> Self {
        debug_assert!(w.pow([n as u64]).is_one());
        Self { n, w }
    }
}

impl<F: FftField> DftDomain<F> for NaiveDomain<F> {
    fn dft(&self, coeffs: &[F]) -> Vec<F> {
        assert_eq!(coeffs.len(), self.n);
        let poly = P::with_coeffs(coeffs.to_vec());
        let mut res = vec![F::zero(); self.n];
        res[0] = poly.evaluate(&F::one());
        res[1] = poly.evaluate(&self.w);
        let mut wi = self.w;
        for i in 2..self.n {
            wi *= self.w;
            res[i] = poly.evaluate(&wi);
        }
        res
    }

    fn idft(&self, _evals: &[F]) -> Vec<F> {
        todo!()
    }

    fn n(&self) -> usize {
        self.n
    }

    fn w(&self) -> F {
        self.w
    }
}

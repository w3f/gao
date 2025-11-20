#![allow(non_snake_case)]

mod bezout;
pub mod gao;
mod gcd;
pub mod half_gcd;
pub mod interpolation;
pub mod lagrange;
pub mod poly_mul;
pub mod product_tree;
mod poly_div;

use ark_ff::{FftField, Field, One};
use ark_poly::{DenseUVPolynomial, Evaluations, Radix2EvaluationDomain};
use ark_poly::univariate::{DenseOrSparsePolynomial, DensePolynomial};
pub type P<F> = DensePolynomial<F>;
pub type PE<F> = Evaluations<F, Radix2EvaluationDomain<F>>;
pub type M<F> = [P<F>; 4];
pub type ME<F> = [PE<F>; 4];

trait Poly<F: Field> {
    fn c(c: F) -> Self;
    fn xk(k: usize) -> Self;
    fn one() -> Self;
    fn x() -> Self;
    fn slice(&self, from: usize, to: usize) -> Self;
    fn lc(&self) -> F;
    fn ct(&self) -> F;
}

impl<F: FftField> Poly<F> for P<F> {
    fn c(c: F) -> Self {
        Self::from_coefficients_vec(vec![c])
    }

    fn xk(k: usize) -> Self {
        let mut coeffs = vec![F::zero(); k + 1];
        coeffs[k] = F::one();
        Self::from_coefficients_vec(coeffs)
    }

    fn one() -> Self {
        Self::c(F::one())
    }

    fn x() -> Self {
        Self::xk(1)
    }

    /// Returns a subarray of coefficients `f_k,...,f_l` as a degree `l - k` polynomial.
    fn slice(&self, from: usize, to: usize) -> Self {
        Self::from_coefficients_slice(&self.coeffs[from..to + 1])
    }

    /// The leading coefficient `f_n`.
    fn lc(&self) -> F {
        self.coeffs[self.coeffs.len() - 1]
    }

    /// The constant term `f_0 = f(0)`.
    fn ct(&self) -> F {
        self.coeffs[0]
    }
}


fn div<F: FftField>(p: &P<F>, q: &P<F>) -> (P<F>, P<F>) {
    let p = DenseOrSparsePolynomial::from(p);
    let q = DenseOrSparsePolynomial::from(q);
    p.divide_with_q_and_r(&q).unwrap()
}

#[cfg(test)]
mod tests {}

#![allow(non_snake_case)]

mod bezout;
pub mod gao;
mod gcd;
pub mod half_gcd;
pub mod interpolation;
pub mod lagrange;
pub mod poly_mul;
pub mod product_tree;
pub mod poly_div;

use ark_ff::{FftField, Field};
use ark_poly::{DenseUVPolynomial, Evaluations, Polynomial, Radix2EvaluationDomain};
use ark_poly::univariate::{DenseOrSparsePolynomial, DensePolynomial};
pub type P<F> = DensePolynomial<F>;
pub type PE<F> = Evaluations<F, Radix2EvaluationDomain<F>>;
pub type M<F> = [P<F>; 4];
pub type ME<F> = [PE<F>; 4];

pub trait Poly<F: Field> {
    fn with_coeffs(coeffs: Vec<F>) -> Self;
    fn constant(c: F) -> Self;
    fn xk(k: usize) -> Self;
    fn one() -> Self;
    fn x() -> Self;
    fn slice(&self, from: usize, to: usize) -> Self;
    fn div_xk(&self, n: usize) -> Self;
    fn mul_xk(&self, n: usize) -> Self;
    fn mod_xk(&self, n: usize) -> Self;
    fn lc(&self) -> F;
    fn ct(&self) -> F;
}

impl<F: FftField> Poly<F> for P<F> {
    fn with_coeffs(coeffs: Vec<F>) -> Self {
        Self::from_coefficients_vec(coeffs)
    }

    fn xk(k: usize) -> Self {
        let mut coeffs = vec![F::zero(); k + 1];
        coeffs[k] = F::one();
        Self::with_coeffs(coeffs)
    }

    fn x() -> Self {
        Self::xk(1)
    }

    fn constant(c: F) -> Self {
        Self::with_coeffs(vec![c])
    }

    fn one() -> Self {
        Self::constant(F::one())
    }

    /// Returns a subarray of coefficients `f_k,...,f_{l-1}` as a degree `l - 1 - k` polynomial.
    fn slice(&self, from: usize, to: usize) -> Self {
        Self::with_coeffs(self.coeffs[from..to].to_vec())
    }

    fn div_xk(&self, k: usize) -> Self {
        self.slice(k, self.coeffs.len())
    }

    fn mod_xk(&self, k: usize) -> Self {
        self.slice(0, k)
    }

    fn mul_xk(&self, k: usize) -> Self {
        let d = self.degree();
        let mut coeffs = Vec::with_capacity(d + k + 1);
        coeffs.extend(vec![F::zero(); k]);
        coeffs.extend_from_slice(&self.coeffs);
        Self::with_coeffs(coeffs)
    }

    /// The leading coefficient `f_d`.
    fn lc(&self) -> F {
        self.coeffs[self.degree()]
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

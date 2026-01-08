#![allow(non_snake_case)]

pub mod bezout;
pub mod dft;
pub mod gao;
pub mod gcd;
pub mod half_gcd;
pub mod interpolation;
pub mod lagrange;
pub mod poly_div;
mod poly_gcd;
pub mod poly_mul;
pub mod product_tree;

use ark_ff::{FftField, Field, Zero};
use ark_poly::univariate::DensePolynomial;
use ark_poly::{DenseUVPolynomial, Evaluations, Polynomial, Radix2EvaluationDomain};
use std::ops::RangeBounds;
use std::slice::SliceIndex;

pub type P<F> = DensePolynomial<F>;
pub type PE<F> = Evaluations<F, Radix2EvaluationDomain<F>>;
pub type M<F> = [P<F>; 4];
pub type ME<F> = [PE<F>; 4];

pub trait Poly<F: Field> {
    fn with_coeffs(coeffs: Vec<F>) -> Self;
    fn c_xk(c: F, k: usize) -> Self;
    fn xk(k: usize) -> Self;
    fn constant(c: F) -> Self;
    fn one() -> Self;
    fn x() -> Self;
    // + SliceIndex<[usize]>
    fn slice<R: RangeBounds<usize> + SliceIndex<[F], Output = [F]>>(&self, range: R) -> Self;
    fn div_xk(&self, n: usize) -> Self;
    fn mul_xk(&self, n: usize) -> Self;
    fn mod_xk(&self, n: usize) -> Self;
    fn lc(&self) -> F;
    fn ct(&self) -> F;
    /// `c_k + c_{k+1}.X + ... + c_d.X{d-k}`
    /// Panics if `k > d`. Otherwise returns `f(X)/X^k`
    fn l_shift(&self, k: usize) -> Self;
    /// `c_0.X^k + c_1.X^{k+1} + ... + c_d.X^{d+k} = f(X).X^k`
    fn r_shift(&self, k: usize) -> Self;
}

impl<F: FftField> Poly<F> for P<F> {
    /// `c[0] + c[1]X + ... + c[n-1]X^{n-1}`
    fn with_coeffs(c: Vec<F>) -> Self {
        Self::from_coefficients_vec(c)
    }

    /// `cX^k`
    fn c_xk(c: F, k: usize) -> Self {
        let mut coeffs = vec![F::zero(); k + 1];
        coeffs[k] = c;
        Self::with_coeffs(coeffs)
    }

    /// `X^k`
    fn xk(k: usize) -> Self {
        Self::c_xk(F::one(), k)
    }

    /// `X`
    fn x() -> Self {
        Self::xk(1)
    }

    /// `c`
    fn constant(c: F) -> Self {
        Self::c_xk(c, 0)
    }

    /// `1`
    fn one() -> Self {
        Self::constant(F::one())
    }

    /// Returns a subarray of coefficients as polynomial.
    fn slice<R: RangeBounds<usize> + SliceIndex<[F], Output = [F]>>(&self, range: R) -> Self {
        Self::with_coeffs(self.coeffs[range].to_vec())
    }

    /// `c_k + c_{k+1}.X + ... + c_d.X{d-k}`
    /// Panics if `k > d`. Otherwise returns `f(X)/X^k`
    fn l_shift(&self, k: usize) -> Self {
        self.slice(k..)
    }

    /// `c_0.X^k + c_1.X^{k+1} + ... + c_d.X^{d+k} = f(X).X^k`
    fn r_shift(&self, k: usize) -> Self {
        let d = self.degree();
        let mut coeffs = Vec::with_capacity(d + k + 1);
        coeffs.extend(vec![F::zero(); k]);
        coeffs.extend_from_slice(&self.coeffs);
        Self::with_coeffs(coeffs)
    }

    /// `f(X) / X^k`
    fn div_xk(&self, k: usize) -> Self {
        if k > self.degree() {
            Self::zero()
        } else {
            self.l_shift(k)
        }
    }

    /// `f(X) * X^k`
    fn mul_xk(&self, k: usize) -> Self {
        self.r_shift(k)
    }

    /// `f(X) mod X^k`
    fn mod_xk(&self, k: usize) -> Self {
        if k > self.degree() {
            self.clone()
        } else {
            self.slice(..k)
        }
    }

    /// The leading coefficient `c_d`.
    fn lc(&self) -> F {
        self.coeffs[self.degree()]
    }

    /// The constant term `c_0 = f(0)`.
    fn ct(&self) -> F {
        self.coeffs[0]
    }
}

#[cfg(test)]
mod tests {
    use ark_bls12_381::Fr;
    use ark_ff::FftField;
    use ark_poly::univariate::DenseOrSparsePolynomial;

    pub type P = crate::P<Fr>;
    // pub type PM = crate::poly_gcd::PM<Fr>;

    pub fn ark_div<F: FftField>(p: &crate::P<F>, q: &crate::P<F>) -> (crate::P<F>, crate::P<F>) {
        let p = DenseOrSparsePolynomial::from(p);
        let q = DenseOrSparsePolynomial::from(q);
        p.divide_with_q_and_r(&q).unwrap()
    }
}

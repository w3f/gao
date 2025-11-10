#![allow(non_snake_case)]

mod bezout;
pub mod gao;
mod gcd;
pub mod half_gcd;
pub mod interpolation;
pub mod lagrange;
pub mod poly_mul;
pub mod product_tree;

use ark_ff::FftField;
use ark_poly::univariate::{DenseOrSparsePolynomial, DensePolynomial};
pub type P<F> = DensePolynomial<F>;
pub type M<F> = [P<F>; 4];

fn div<F: FftField>(p: &P<F>, q: &P<F>) -> (P<F>, P<F>) {
    let p = DenseOrSparsePolynomial::from(p);
    let q = DenseOrSparsePolynomial::from(q);
    p.divide_with_q_and_r(&q).unwrap()
}

#[cfg(test)]
mod tests {}

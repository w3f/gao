#![allow(non_snake_case)]

mod lagrange;
mod gcd;
mod bezout;
mod half_gcd;
mod gao;

use ark_ff::FftField;
use ark_poly::univariate::{DenseOrSparsePolynomial, DensePolynomial};


type P<F> = DensePolynomial<F>;
type M<F> = [P<F>; 4];

fn div<F: FftField>(p: &P<F>, q: &P<F>) -> (P<F>, P<F>) {
    let p = DenseOrSparsePolynomial::from(p);
    let q = DenseOrSparsePolynomial::from(q);
    p.divide_with_q_and_r(&q).unwrap()
}


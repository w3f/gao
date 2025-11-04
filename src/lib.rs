#![allow(non_snake_case)]

pub mod lagrange;
mod gcd;
mod bezout;
pub mod half_gcd;
pub mod gao;
pub mod product_tree;
pub mod poly_mul;

use ark_ff::FftField;
use ark_poly::{EvaluationDomain, Evaluations, Polynomial, Radix2EvaluationDomain};
use ark_poly::univariate::{DenseOrSparsePolynomial, DensePolynomial};
use crate::product_tree::ProductTree;

pub type P<F> = DensePolynomial<F>;
pub type M<F> = [P<F>; 4];

fn div<F: FftField>(p: &P<F>, q: &P<F>) -> (P<F>, P<F>) {
    let p = DenseOrSparsePolynomial::from(p);
    let q = DenseOrSparsePolynomial::from(q);
    p.divide_with_q_and_r(&q).unwrap()
}

// We want to interpolate a degree `t-1` polynomial `f`
// using evaluations of `f` at not more than 'n >= t' predefined points.

// Consider an FFT domain `[w_0, ..., w_{d-1}], where d >= n`.
// The evaluations have form `(i_k, v_k)` where `0 <= i_k < n`, and
// mean allegedly that `f(w_{i_k}) = v_k` for k = 1,...,s`.
// 0 < t <= s <= n <= d.

// Let `S = {i_k|k = 1,...,s}` be the set of indices of the domain elements for which the evaluations are given.
// Let `C = [0,...,d-1]\S` be the complement to `S`. Then to compute `f` we
// 1. Compute `z_c` - the vanishing polynomial of `{w_i|i in C}`.
//    - a product tree of degree `d - s`, or
//    - a product tree of degree `n - s`, given the product tree `d - n` is precomputed.
// 2. Evaluate `z_c` over the domain using an FFT. // `d-FFT`
// 3. Compute `f * z_c` in evaluation form. // `s` field multiplications
// 4. Interpolate `f * z_c`. // `d-FFT`
// 5. Evaluate `f * z_c` and `z_c` over a coset of the domain. // `2 x d-FFT`
// 6. Compute `f = (f * z_c) / z_c` over the coset in the evaluation form and interpolate it. // `d-FFT`
pub fn interpolation_a_la_al<F: FftField>(n: usize, t: usize, f_on_s: &[(usize, F)]) -> P<F>
{
    debug_assert!(0 < t);
    let s = f_on_s.len();
    debug_assert!(t <= s);
    debug_assert!(s <= n);
    let domain = Radix2EvaluationDomain::<F>::new(n).unwrap();
    let d = domain.size();
    debug_assert!(n <= d);

    let mut f_on_d = vec![None; d];
    for (i, vi) in f_on_s {
        f_on_d[*i] = Some(vi);
    }

    let mut ws = domain.elements();
    let mut complement: Vec<F> = ws.by_ref()
        .take(n)
        .enumerate()
        .filter_map(|(i, wi)| f_on_d[i].is_none().then_some(wi))
        .collect();
    debug_assert_eq!(complement.len(), n - s);
    complement.extend(ws); // TODO: precompute the tree
    debug_assert_eq!(complement.len(), d - s);

    // 1. Compute `z_c` - the vanishing polynomial of `{w_i|i in C}`.
    let tree_on_c = ProductTree::new(&complement).unwrap();
    let zc = tree_on_c.root_poly();
    debug_assert_eq!(zc.degree(), d - s);

    // 2. Evaluate `z_c` over the domain using an FFT.
    let zc_on_d = zc.evaluate_over_domain_by_ref(domain).evals;
    // 3. Compute `f * z_c` in evaluation form.
    let f_zc_on_d: Vec<F> = f_on_d.into_iter()
        .zip(zc_on_d)
        .map(|(vi, zi)| {
            match vi {
                Some(vi) => zi * vi,
                None => F::zero()
            }
        }).collect();
    let f_zc_on_d = Evaluations::from_vec_and_domain(f_zc_on_d, domain);
    // 4. Interpolate `f * z_c`.
    let f_zc = f_zc_on_d.interpolate();
    // 5. Evaluate `f * z_c` and `z_c` over a coset of the domain.
    let coset = domain.get_coset(F::GENERATOR).unwrap();
    let f_zc_on_coset = f_zc.evaluate_over_domain_by_ref(coset);
    let zc_on_coset = zc.evaluate_over_domain_by_ref(coset);
    // 6. Compute `f = (f * z_c) / z_c` over the coset in the evaluation form and interpolate it.
    let f_on_coset = &f_zc_on_coset / &zc_on_coset;
    f_on_coset.interpolate()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use ark_bls12_381::Fr;
    use ark_ff::{FftField, Zero};
    use ark_poly::{EvaluationDomain, Evaluations, GeneralEvaluationDomain, Radix2EvaluationDomain};
    use ark_std::{end_timer, rand, start_timer, test_rng};
    use crate::{interpolation_a_la_al, P};
    use ark_poly::Polynomial;
    use crate::product_tree::ProductTree;
    use ark_poly::DenseUVPolynomial;
    use ark_std::iterable::Iterable;
    use rand::Rng;
    use ark_std::rand::prelude::SliceRandom;


    // d - domain.size()
    // max_us - maximal number of evaluation points, max_us <= d
    // n_vs - number of alleged evaluations, n_vs <= max_us
    fn split_domain<R: Rng>(d: usize, max_us: usize, n_vs: usize, rng: &mut R) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        assert!(max_us <= d);
        let is: Vec<usize> = (0..d).collect();
        let (us, not_us) = is.split_at(max_us);
        let mut us = us.to_vec();
        us.shuffle(rng);
        assert!(n_vs <= max_us);
        let (us_with_v, us_no_v) = us.split_at(n_vs);
        let mut us_with_v = us_with_v.to_vec();
        us_with_v.sort();
        let mut us_no_v = us_no_v.to_vec();
        us_no_v.sort();
        assert_eq!(us_with_v.len() + us_no_v.len(), d - not_us.len());
        (us_with_v, us_no_v, not_us.to_vec())
    }


    fn _test_interpolation_a_la_al<F: FftField>(n: usize, t: usize) {
        let rng = &mut test_rng();
        debug_assert!(t <= n);
        let f = P::<F>::rand(t - 1, rng);
        let s = rng.gen_range(t..=n); // number of alleged evaluations
        let S = {
            let mut legit_is: Vec<usize> = (0..n).collect();
            legit_is.shuffle(rng);
            legit_is.truncate(s);
            legit_is
        };
        debug_assert_eq!(S.len(), s);

        let domain = Radix2EvaluationDomain::<F>::new(n).unwrap();
        let ws: Vec<F> = domain.elements().collect();

        let f_on_S: Vec<(usize, F)> = S.iter()
            .map(|&i| (i, f.evaluate(&ws[i])))
            .collect();

        let f_ = interpolation_a_la_al(n, t, &f_on_S);
        assert_eq!(f_, f);
    }

    #[test]
    fn test_interpolation_a_la_al() {
        _test_interpolation_a_la_al::<Fr>(1000, 667);
    }
}
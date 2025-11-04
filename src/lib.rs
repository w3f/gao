#![allow(non_snake_case)]

pub mod lagrange;
mod gcd;
mod bezout;
pub mod half_gcd;
pub mod gao;
pub mod product_tree;
pub mod poly_mul;

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
mod tests {
    use ark_bls12_381::Fr;
    use ark_ff::{FftField, Zero};
    use ark_poly::{EvaluationDomain, Evaluations, GeneralEvaluationDomain, Radix2EvaluationDomain};
    use ark_std::{end_timer, rand, start_timer, test_rng};
    use crate::P;
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

    fn _it_works<F: FftField>(n: usize, t: usize) {
        let rng = &mut test_rng();
        assert!(t <= n); // `t` is the threshold
        let D = Radix2EvaluationDomain::<F>::new(n).unwrap();
        let mut ws = D.elements();
        let us: Vec<F> = ws.by_ref().take(n).collect(); // legit evaluation points
        let not_us: Vec<F> = ws.collect(); // excess domain elements
        let v = P::<F>::rand(t - 1, rng);
        let n_vs = rng.gen_range(t..=n); // number of alleged evaluations

        // S is the set of indices for which evaluations are provided
        // C is the set of indices for which evaluations are missing
        // T is the indices that are not evaluation points
        // S + C  = D \ T
        let (S, C, T) = split_domain(D.size(), n, n_vs, rng);

        let mut v_on_D: Vec<Option<F>> = vec![None; D.size()];
        for i in S.clone() {
            v_on_D[i] = Some(v.evaluate(&us[i]));
        }
        let v_on_S = S.iter().map(|i| v.evaluate(&us[*i])).collect::<Vec<_>>();
        let C_points: Vec<F> = C.iter().map(|i| us[*i]).collect();
        let T_points = not_us;


        let _t_v = start_timer!(|| format!("Interpolation: deg(v) = {t}, |d| = {n}"));
        let tree_on_C = ProductTree::new(&C_points).unwrap();
        let zC = tree_on_C.root_poly();
        let zC_on_D = zC.evaluate_over_domain_by_ref(D).evals;
        // let zC_on_S: Vec<F> = S.iter().map(|i| zC_on_D[i]).collect();
        // assert_eq!(zC_on_S.len(), v_on_S.len());

        assert_eq!(zC_on_D.len(), v_on_D.len());

        let vzC_on_D: Vec<F> = zC_on_D.iter()
            .zip(v_on_D)
            .map(|(zi, vi)| {
                match vi {
                    Some(vi) => vi * zi,
                    None => {
                        assert!(zi.is_zero());
                        F::zero()
                    }
                }
            }).collect();
        assert_eq!(vzC_on_D.len(), D.size());



        let mut vzC_on_D2 = vec![F::zero(); D.size()];
        let mut v_on_S_iter = v_on_S.iter();
        for i in S {
            vzC_on_D2[i] = zC_on_D[i] * v_on_S_iter.next().unwrap();
        }

        assert_eq!(vzC_on_D2, vzC_on_D);

        let vzC_on_D = Evaluations::from_vec_and_domain(vzC_on_D, D);



        let vzC = vzC_on_D.interpolate();
        let coset = D.get_coset(F::GENERATOR).unwrap();
        let vzC_on_coset = vzC.evaluate_over_domain_by_ref(coset);
        let zC_on_coset = zC.evaluate_over_domain_by_ref(coset);
        let v_on_coset = &vzC_on_coset / &zC_on_coset;
        let v_ = v_on_coset.interpolate();
        end_timer!(_t_v);
        assert_eq!(v_, v);
    }

    #[test]
    fn it_works() {
        // _it_works();
    }
}
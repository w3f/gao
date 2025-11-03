#![allow(non_snake_case)]

mod lagrange;
mod gcd;
mod bezout;
mod half_gcd;
mod gao;
mod product_tree;
mod poly_mul;

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
    use ark_poly::{EvaluationDomain, Evaluations, GeneralEvaluationDomain};
    use ark_std::{end_timer, start_timer, test_rng, UniformRand};
    use crate::P;
    use ark_poly::Polynomial;
    use crate::product_tree::ProductTree;
    use ark_poly::DenseUVPolynomial;

    #[test]
    fn it_works() {
        let rng = &mut test_rng();

        let log_n = 10;
        let n = 2usize.pow(log_n);
        let p = P::<Fr>::rand(n / 2 - 1, rng);
        assert_eq!(p.degree(), n / 2 - 1);
        let domain = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
        let us: Vec<Fr> = domain.elements().collect();
        let (t, c) = us.split_at(n / 2); // todo: shuffle
        let v_on_t = t.iter().map(|x| p.evaluate(x)).collect::<Vec<_>>();

        let _t_v = start_timer!(|| format!("Interpolation: deg(p) = {}, |d| = {n}", n / 2 - 1));
        let tree = ProductTree::new(c).unwrap();
        let z = tree.root_poly();
        let z_on_d = z.evaluate_over_domain_by_ref(domain).evals;
        let vz_on_t = v_on_t.iter()
            .zip(z_on_d)
            .map(|(v, z)| z * v)
            .collect::<Vec<_>>();
        let vz_on_d = [vz_on_t, vec![Fr::zero(); n / 2]].concat();
        let vz_on_d = Evaluations::from_vec_and_domain(vz_on_d, domain);
        let vz = vz_on_d.interpolate();
        let coset = domain.get_coset(Fr::GENERATOR).unwrap();
        let vz_on_c = vz.evaluate_over_domain_by_ref(coset);
        let z_on_c = z.evaluate_over_domain_by_ref(coset);
        let v_on_c = &vz_on_c / &z_on_c;
        let v = v_on_c.interpolate();
        end_timer!(_t_v);
        assert_eq!(p, v);
    }
}
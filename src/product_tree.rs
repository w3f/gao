use ark_ff::{FftField, Field};
use ark_poly::univariate::{DenseOrSparsePolynomial, DensePolynomial};
use ark_poly::{DenseUVPolynomial, Polynomial};
use crate::P;

/// The vanishing polynomial of `u`.
/// `z(X) = X - u`
fn z<F: Field>(u: F) -> DensePolynomial<F> {
    DensePolynomial::from_coefficients_slice(&[-u, F::one()]) // `= -u.0 + 1.X = X - u`
}


pub struct ProductTree<F: Field>(pub Vec<DensePolynomial<F>>);

impl<F: FftField> ProductTree<F> {
    // TODO: us should be distinct
    pub fn new(us: &[F]) -> Result<Self, ()> {
        let n = us.len();
        match n {
            0 => Err(()),
            1 => Ok(Self(vec![z(us[0])])),
            _ => {
                let h = n / 2;
                let subtree_0 = Self::new(&us[0..h])?;
                let subtree_1 = Self::new(&us[h..n])?;
                let root = subtree_0.root() * subtree_1.root();
                Ok(Self([
                    subtree_0.0,
                    subtree_1.0,
                    vec![root]
                ].concat()))
            }
        }
    }

    pub fn root(&self) -> &DensePolynomial<F> {
        &self.0[self.0.len() - 1]
    }

    pub fn evaluate(&self, f: &DensePolynomial<F>) -> Result<Vec<F>, ()> {
        Self::_evaluate(&self.0[0..self.0.len() - 1], f)
    }

    fn _evaluate(products: &[DensePolynomial<F>], f: &DensePolynomial<F>) -> Result<Vec<F>, ()> {
        let d = f.degree();
        if d == 0 {
            return Ok(f.coeffs.clone());
        }

        let m = products.len();
        // if m != 2 * d {
        //     return Err(());
        // }
        let p: DenseOrSparsePolynomial<F> = f.into();
        let (subtree_0, subtree_1) = products.split_at(m / 2);
        let (root_0, products_0) = subtree_0.split_last().unwrap();
        let (root_1, products_1) = subtree_1.split_last().unwrap();
        let (_q0, r0) = p.divide_with_q_and_r(&root_0.into()).unwrap();
        let (_q1, r1) = p.divide_with_q_and_r(&root_1.into()).unwrap();
        let v0 = Self::_evaluate(products_0, &r0)?;
        let v1 = Self::_evaluate(products_1, &r1)?;
        Ok([v0, v1].concat())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use ark_std::{end_timer, start_timer, test_rng, UniformRand};

    #[test]
    // n = 2^k
    fn test_subproduct_tree() {
        let log_n = 4;
        let n = 2usize.pow(log_n);
        let fft_domain = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
        let xs = fft_domain.elements().collect::<Vec<_>>();
        let tree = ProductTree::new(&xs).unwrap();

        assert_eq!(*tree.root(), fft_domain.vanishing_polynomial().into());
        let products = tree.0;
        let m = products.len();
        assert_eq!(m, 2 * n - 1);
        let xs_l = &xs[0..n / 2];
        let xs_r = &xs[n / 2..n];
        let h = (m - 1) / 2;
        let products_l = &products[0..h];
        let products_r = &products[h..m - 1]; // skip the root
        assert_eq!(products_l, ProductTree::new(&xs_l).unwrap().0);
        assert_eq!(products_r, ProductTree::new(&xs_r).unwrap().0);
    }

    fn _bench_subproduct_tree<F: FftField>(log_n: u32) {
        let n = 2usize.pow(log_n);
        let fft_domain = GeneralEvaluationDomain::<F>::new(n).unwrap();
        let xs = fft_domain.elements().collect::<Vec<_>>();

        let _t_product_tree = start_timer!(|| format!("Product tree, n = {n}"));
        let tree = ProductTree::new(&xs).unwrap();
        end_timer!(_t_product_tree);

        let u = Fr::rand(&mut test_rng());
        let mut m = z(u);
        for d in 1..log_n + 1 {
            let _t_mul = start_timer!(|| format!("Polynomial multiplication, d = {d}"));
            m = &m * &m;
            end_timer!(_t_mul);
        }
        assert_eq!(m.degree(), tree.root().degree());
    }

    #[test]
    #[ignore]
    fn bench_subproduct_tree() {
        let log_n = 10;
        // cargo test bench_subproduct_tree --release --features="print-trace" -- --ignored --show-output
        _bench_subproduct_tree::<Fr>(log_n); // 8.546ms
    }
}
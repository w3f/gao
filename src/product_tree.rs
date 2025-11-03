use crate::poly_mul::Monic;
use crate::P;
use ark_ff::{FftField, Field};
use ark_poly::univariate::{DenseOrSparsePolynomial, DensePolynomial};
use ark_poly::{DenseUVPolynomial, Polynomial};
use ark_std::{end_timer, start_timer};

/// The vanishing polynomial of `u`.
/// `z(X) = X - x`
fn z<F: Field>(x: F) -> DensePolynomial<F> {
    DensePolynomial::from_coefficients_slice(&[-x, F::one()]) // `= -x.0 + 1.X = X - u`
}

pub fn products<F: FftField>(xs: &[F]) -> Vec<DensePolynomial<F>> {
    let n = xs.len();
    let mut products = Vec::with_capacity(2 * n - 1);
    products.extend(xs.iter().map(|&xi| z(xi)));
    for i in 0..n - 1 {
        products.push(&products[2 * i] * &products[2 * i + 1])
    }
    products
}

pub fn z_of<F: FftField>(xs: &[F]) -> DensePolynomial<F> {
    products(xs).last().unwrap().clone()
}

pub struct ProductTree<F: FftField>(pub Vec<Monic<F>>);

impl<F: FftField> ProductTree<F> {
    // TODO: xs should be distinct
    pub fn new(xs: &[F]) -> Result<Self, ()> {
        let n = xs.len();
        match n {
            0 => Err(()),
            1 => Ok(Self(vec![Monic::new(z(xs[0]))])),
            _ => {
                let h = n / 2;
                let subtree_0 = Self::new(&xs[0..h])?;
                let subtree_1 = Self::new(&xs[h..n])?;
                let root = Monic::mul(subtree_0.root(), subtree_1.root());
                Ok(Self([
                    subtree_0.0,
                    subtree_1.0,
                    vec![root]
                ].concat()))
            }
        }
    }

    fn split_root(&self) -> (&Monic<F>, &[Monic<F>]) {
        self.0.split_last().unwrap()
    }

    pub fn root(&self) -> &Monic<F> {
        self.split_root().0
    }

    pub fn root_poly(&self) -> &P<F> {
        &self.root().poly
    }

    pub fn child_polys(&self) -> Vec<P<F>> {
        self.split_root().1.iter()
            .map(|n| n.poly.clone())
            .collect::<Vec<_>>()
    }

    pub fn evaluate(&self, f: &DensePolynomial<F>) -> Result<Vec<F>, ()> {
        let child_nodes = self.child_polys();
        Self::_evaluate(&child_nodes, f)
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
    use ark_std::{end_timer, start_timer};

    #[test]
    // n = 2^k
    fn test_subproduct_tree() {
        let log_n = 4;
        let n = 2usize.pow(log_n);
        let fft_domain = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
        let xs = fft_domain.elements().collect::<Vec<_>>();
        let tree = ProductTree::new(&xs).unwrap();

        assert_eq!(*tree.root_poly(), fft_domain.vanishing_polynomial().into());
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

    fn _bench_product_tree<F: FftField>(log_n: u32) {
        let n = 2usize.pow(log_n);
        let fft_domain = GeneralEvaluationDomain::<F>::new(n).unwrap();
        let xs = fft_domain.elements().collect::<Vec<_>>();

        let _t_product_tree = start_timer!(|| format!("Product tree, n = {n}"));
        let tree = ProductTree::new(&xs).unwrap();
        end_timer!(_t_product_tree);
    }

    // cargo test bench_product_tree --release --features="print-trace" -- --ignored --show-output
    #[test]
    #[ignore]
    fn bench_product_tree() {
        let log_n = 10;
        _bench_product_tree::<Fr>(log_n); // 8.546ms // 3.360ms
    }
}
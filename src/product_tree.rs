use crate::poly_mul::Monic;
use crate::P;
use ark_ff::{FftField, Field};
use ark_poly::univariate::{DenseOrSparsePolynomial};
use ark_poly::{DenseUVPolynomial, Polynomial};

/// The vanishing polynomial of a point `x`.
/// `z(X) = X - x`
fn z<F: Field>(x: F) -> P<F> {
    P::from_coefficients_slice(&[-x, F::one()]) // `= -x.0 + 1.X = X - x`
}

/// Let `Z = [z_0, ..., z_{n-1}], z_i != z_j` for `i != j`.
/// The tree is represented as an array, constructed recursively.
///  If `n = 1`, `T([z_0]) = [z_0]`.
///  Otherwise, let `h = n // 2`, `Z_L = [z_0, ..., z_{h-1}]`, and `Z_R = [z_h, ..., z_{n-1}]`.
///  Then `T(Z) = T(Z_L) || T(Z_R) || [z_{h-1} * z_{n-1}]`.
/// `len(T(Z)) = 2.len(Z) - 1`.
///
/// Consider a product tree `T = [t_0, ..., t_{m-1}]`.
/// `m = len(T) = 2n - 1`, `n = (m + 1) / 2`, `h = n // 2`, `n_l = h`, `n_r = n - h`
/// Therefore:
/// * `root(T) := t_{m-1}`,
/// * `len(T_L) = 2.n_l - 1`, so `T_L = [t_0,...,t_{2h-2}]`, and
/// * `len(T_R) = 2.n_r - 1`, so `T_R = [t_{2h-1},...,t_{m-2}]`.

/// By the construction the tree is:
/// 1. *full*, i.e. every inner node (not a leaf) has exactly `2` children,
/// 2. somewhere between *balanced* and *complete* -- it may have gaps anywhere at the last row.

/// The code below assumes that `z_i = z(x_i) = X - x_i` is a monic linear polynomial for any `i`.

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

    pub(crate) fn unwrap_polys(&self) -> Vec<P<F>> {
        self.0.iter()
            .map(|m| m.poly.clone())
            .collect()
    }

    // TODO: Should be smth like:
    // fn split(&self) -> (&ProductTree<F>, &ProductTree<F>, &Monic<F>) {
    pub(crate) fn split(tree: &[P<F>]) -> (&[P<F>], &[P<F>], &P<F>) {
        let m = tree.len();
        // println!("{m}");
        if m == 1 {
            return (&[], &[], &tree[0]);
        }
        let n = (m + 1) / 2;
        let h = n / 2;
        let m_l = 2 * h - 1; // nodes in the left subtree
        let (root, children) = tree.split_last().unwrap();
        let (subtree_l, subtree_r) = children.split_at(m_l);
        (subtree_l, subtree_r, root)
    }

    pub fn root(&self) -> &Monic<F> {
        self.0.split_last().unwrap().0
    }

    pub fn root_poly(&self) -> &P<F> {
        &self.root().poly
    }

    pub fn evaluate(&self, f: &P<F>) -> Result<Vec<F>, ()> {
        let tree = self.unwrap_polys();
        let (subtree_0, subtree_1, _) = Self::split(&tree);
        Self::_evaluate(&subtree_0, &subtree_1, f)
    }

    fn _evaluate(subtree_0: &[P<F>], subtree_1: &[P<F>], f: &P<F>) -> Result<Vec<F>, ()> {
        let d = f.degree();
        if d == 0 {
            // the degree necessarily drops to `0`, as long as the corresponding leaf has degree `1`
            return Ok(f.coeffs.clone());
        }
        let p: DenseOrSparsePolynomial<F> = f.into();
        let (subtree_00, subtree_01, root_0) = Self::split(subtree_0);
        let (subtree_10, subtree_11, root_1) = Self::split(subtree_1);
        let (_q0, r0) = p.divide_with_q_and_r(&root_0.into()).unwrap();
        let (_q1, r1) = p.divide_with_q_and_r(&root_1.into()).unwrap();
        let v0 = Self::_evaluate(subtree_00, subtree_01, &r0)?;
        let v1 = Self::_evaluate(subtree_10, subtree_11, &r1)?;
        Ok([v0, v1].concat())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use ark_std::{end_timer, start_timer, test_rng};

    fn check_tree<F: FftField>(tree: &[P<F>], n: usize) {
        assert_eq!(tree.len(), 2 * n - 1);
        let (subtree_0, subtree_1, root) = ProductTree::split(&tree);
        assert_eq!(root.degree(), n);
        if n == 1 {
            return;
        }
        let h = n / 2;
        check_tree(subtree_0, h);
        check_tree(subtree_1, n - h);
    }

    fn _product_tree<F: FftField>(n: usize) {
        let rng = &mut test_rng();

        let xs = (0..n).map(|_| F::rand(rng)).collect::<Vec<_>>();
        let tree = ProductTree::new(&xs).unwrap();
        check_tree(&tree.unwrap_polys(), n);
    }

    #[test]
    fn product_tree() {
        for n in [1, 2, 3, 4, 15, 16, 17] {
            _product_tree::<Fr>(n);
        }
    }

    fn _multipoint_evaluation<F: FftField>(n: usize) {
        let rng = &mut test_rng();

        let fft_domain = GeneralEvaluationDomain::<F>::new(n).unwrap();
        let xs = fft_domain.elements().take(n).collect::<Vec<_>>();
        let tree = ProductTree::new(&xs).unwrap();

        let p = P::<F>::rand(n - 1, rng);

        let vs = tree.evaluate(&p).unwrap();
        let vs_ = p.evaluate_over_domain_by_ref(fft_domain).evals;
        assert_eq!(vs, vs_[0..n].to_vec());
    }

    #[test]
    fn multipoint_evaluation() {
        for n in [1, 2, 3, 4, 15, 16, 17] {
            _multipoint_evaluation::<Fr>(n);
        }
    }

    fn _bench_product_tree<F: FftField>(log_n: u32) {
        let n = 2usize.pow(log_n);
        let fft_domain = GeneralEvaluationDomain::<F>::new(n).unwrap();
        let xs = fft_domain.elements().collect::<Vec<_>>();

        let _t_product_tree = start_timer!(|| format!("Product tree, n = {n}"));
        let _tree = ProductTree::new(&xs).unwrap();
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
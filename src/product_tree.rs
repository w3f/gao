use ark_ff::{FftField, Field};
use ark_poly::univariate::{DenseOrSparsePolynomial, DensePolynomial};
use ark_poly::{DenseUVPolynomial, EvaluationDomain, Evaluations, GeneralEvaluationDomain, Polynomial};
use ark_std::{end_timer, start_timer};
use crate::P;

/// The vanishing polynomial of `u`.
/// `z(X) = X - u`
fn z<F: Field>(u: F) -> DensePolynomial<F> {
    DensePolynomial::from_coefficients_slice(&[-u, F::one()]) // `= -u.0 + 1.X = X - u`
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node<F: FftField>(pub DensePolynomial<F>, Option<Evaluations<F>>);

pub struct ProductTree<F: FftField>(pub Vec<Node<F>>);

impl<F: FftField> ProductTree<F> {
    // TODO: us should be distinct
    pub fn new(us: &[F]) -> Result<Self, ()> {
        let n = us.len();
        match n {
            0 => Err(()),
            1 => Ok(Self(vec![Node(z(us[0]), None)])),
            _ => {
                let h = n / 2;
                let subtree_0 = Self::new(&us[0..h])?;
                let subtree_1 = Self::new(&us[h..n])?;
                let root = mul_nodes(subtree_0.root_node(), subtree_1.root_node());
                Ok(Self([
                    subtree_0.0,
                    subtree_1.0,
                    vec![root]
                ].concat()))
            }
        }
    }

    fn split_root(&self) -> (&Node<F>, &[Node<F>]) {
        self.0.split_last().unwrap()
    }

    pub fn root_node(&self) -> &Node<F> {
        self.split_root().0
    }

    pub fn root(&self) -> &P<F> {
        &self.root_node().0
    }

    pub fn child_polys(&self) -> Vec<P<F>> {
        self.split_root().1.iter().map(|n| n.0.clone()).collect()
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

fn double_evals<F: FftField>(p: &P<F>, evals: &Evaluations<F>) -> Evaluations<F> {
    let n = evals.domain().size();

    // `(1, w, w^2, ..., w^{2n-1})`
    let domain_2x = GeneralEvaluationDomain::<F>::new(2 * n).unwrap();
    // Let `W := w^2`
    // `(1, W, W^2, ..., W^{n-1}) = (1, w^2, ..., w^{2n-2})`
    let domain = evals.domain();

    // The evaluations `(p(w), p(w^3), ..., p(w^{2n-1}))` are missing.
    // Consider a polynomial `pw(X) := p(w.X)`.
    // Then `(p(w), p(w^3), ..., p(w^{2n-1})) = (pw(1), pw(W), ..., pw(W^{n-1}))`.
    // `pw(X) = p(w.X) = p_0 + p_1(w.X) + ... + p_{n-1}(w.X)^{n-1} = p_0 + (p_1.w)X + ... + (p^{n-1}.w^{n-1})X^{n-1}`
    let pw_coeffs = p.coeffs.iter() // `= (p_0, p_1, ..., p_{n-1})`
        .zip(domain_2x.elements()) // ` = (1, w, ..., w^{n-1}, ..., w^{2n-1})`
        .map(|(ci, wi)| wi * ci)
        .collect::<Vec<_>>();
    let pw = P::<F>::from_coefficients_vec(pw_coeffs);

    let _t_fft = start_timer!(|| format!("FFT, n = {n}"));
    let evals_w = pw.evaluate_over_domain_by_ref(domain);
    end_timer!(_t_fft);

    // `evals2x` = `evals` interleaved with `evals_w`
    let evals_2x: Vec<F> = evals.evals.iter().cloned()
        .zip(evals_w.evals)
        .flat_map(|(e1, e2)| vec![e1, e2])
        .collect();
    let evals_2x = Evaluations::from_vec_and_domain(evals_2x, domain_2x);
    evals_2x
}

fn monic_interpolation<F: FftField>(p_evals: &Evaluations<F>) -> P<F> {
    // Let `p` be a degree `d` monic polynomial.
    // Then `p(X) = X^d + f(X)`, where `deg(f) <= d-1`.
    // Then `f` can be interpolated using `d` evaluations.
    // `f(X) = p(X) - X^d`
    // `f(xi) = vi - xi^d, i = 1,...,d`
    let d = p_evals.evals.len();
    let domain = p_evals.domain();
    let f_evals = p_evals.evals.iter()
        .zip(domain.elements())
        .map(|(vi, xi)| *vi - xi.pow([d as u64]))
        .collect::<Vec<_>>();
    let f_evals = Evaluations::from_vec_and_domain(f_evals, domain);
    let mut f = f_evals.interpolate();

    // p(X) = f(X) + X^d
    f.coeffs.resize(d + 1, F::zero());
    f.coeffs[d] = F::one();
    f
}

fn mul_nodes<F: FftField>(a: &Node<F>, b: &Node<F>) -> Node<F> {
    let d = a.0.degree();
    assert_eq!(b.0.degree(), d);

    if d < 128 {
        let c = a.0.naive_mul(&b.0);
        return Node(c, None);
    }

    let n = d; // a monic degree `d` polynomials is uniquely defined by `d` evaluations
    let domain_2x = GeneralEvaluationDomain::<F>::new(2 * n).unwrap();

    let a_evals_2x = match &a.1 {
        Some(a_evals) => double_evals(&a.0, &a_evals),
        None => a.0.evaluate_over_domain_by_ref(domain_2x),
    };

    let b_evals_2x = match &b.1 {
        Some(b_evals) => double_evals(&b.0, &b_evals),
        None => b.0.evaluate_over_domain_by_ref(domain_2x),
    };
    let c_evals = &a_evals_2x * &b_evals_2x;
    let c = monic_interpolation(&c_evals);
    Node(c, Some(c_evals))
}

fn products<F: FftField>(xs: &[F]) -> Vec<DensePolynomial<F>> {
    let n = xs.len();
    let mut products = Vec::with_capacity(2 * n - 1);
    products.extend(xs.iter().map(|&xi| z(xi)));
    for i in 0..n - 1 {
        products.push(&products[2 * i] * &products[2 * i + 1])
    }
    products
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::{One, Zero};
    use ark_poly::{EvaluationDomain, Evaluations, GeneralEvaluationDomain};
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
    }

    // cargo test bench_subproduct_tree --release --features="print-trace" -- --ignored --show-output
    #[test]
    #[ignore]
    fn bench_subproduct_tree() {
        let log_n = 10;
        _bench_subproduct_tree::<Fr>(log_n); // 8.546ms // 3.503ms
    }

    #[test]
    fn test_monic_interpolation() {
        let rng = &mut test_rng();

        let n = 4;
        let domain = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
        let xs = domain.elements().take(n).collect::<Vec<_>>();
        let products = products(&xs);
        let p = products[products.len() - 1].clone();
        assert_eq!(p.degree(), n);
        assert!(p.coeffs[n].is_one());
        let p_evals = p.evaluate_over_domain_by_ref(domain);

        let p_ = monic_interpolation(&p_evals);
        assert_eq!(p_, p);
    }

    #[test]
    fn test_fft_doubling() {
        let rng = &mut test_rng();

        let n = 10;
        let p = P::<Fr>::rand(n - 1, rng);
        let domain = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
        let domain_2x = GeneralEvaluationDomain::<Fr>::new(2 * n).unwrap();
        let p_evals = p.evaluate_over_domain_by_ref(domain);
        let p_evals_2x = p.evaluate_over_domain_by_ref(domain_2x);

        let p_evals_2x_ = double_evals(&p, &p_evals);
        assert_eq!(p_evals_2x_, p_evals_2x);
    }

    // cargo test bench_fft_doubling --release --features="print-trace" -- --ignored --show-output
    #[test]
    #[ignore]
    fn bench_fft_doubling() {
        let rng = &mut test_rng();

        let log_n = 9;
        let n = 2usize.pow(log_n);
        let p = P::<Fr>::rand(n - 1, rng);
        let domain = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
        let domain_2x = GeneralEvaluationDomain::<Fr>::new(2 * n).unwrap();
        let p_evals = p.evaluate_over_domain_by_ref(domain);

        let _t_fft = start_timer!(|| format!("FFT, 2n = {}", 2 * n));
        let p_evals_2x = p.evaluate_over_domain_by_ref(domain_2x);
        end_timer!(_t_fft);

        let _t_fft_doubling = start_timer!(|| format!("Doubling evals, n = {n}"));
        let p_evals_2x_ = double_evals(&p, &p_evals);
        end_timer!(_t_fft_doubling);

        assert_eq!(p_evals_2x_, p_evals_2x);
    }

    // cargo test bench_mul --release --features="print-trace" -- --ignored --show-output
    #[test]
    #[ignore]
    fn bench_mul() {
        let rng = &mut test_rng();

        let log_n = 9;
        let n = 2usize.pow(log_n);
        let domain = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
        let domain_2x = GeneralEvaluationDomain::<Fr>::new(2 * n).unwrap();

        let mut a = P::<Fr>::rand(n, rng);
        let mut b = P::<Fr>::rand(n, rng);
        a.coeffs[n] = Fr::one();
        b.coeffs[n] = Fr::one();

        let a_evals_2x = a.evaluate_over_domain_by_ref(domain_2x);
        let b_evals_2x = b.evaluate_over_domain_by_ref(domain_2x);

        let _t_smart_fft = start_timer!(|| format!("Smart mul, n = {n}"));
        let c_evals = &a_evals_2x * &b_evals_2x;
        let c = monic_interpolation(&c_evals);
        let c_node = Node(c, Some(c_evals));
        end_timer!(_t_smart_fft);

        let _t_naive_fft = start_timer!(|| format!("Naive mul, n = {n}"));
        let c_ = &a * &b;
        end_timer!(_t_naive_fft);

        assert_eq!(c_, c_node.0);
    }
}
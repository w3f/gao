use crate::Poly;
use crate::P;
use ark_ff::FftField;
use ark_poly::{DenseUVPolynomial, EvaluationDomain, Polynomial};
use ark_std::iterable::Iterable;
use std::iter;

/// `X^k.a(1/X)`
fn rev<F: FftField>(k: usize, a: &P<F>) -> P<F> {
    let d = a.degree();
    debug_assert!(k >= d);
    let zeros = iter::repeat(F::zero())
        .take(k - d);
    let a_coeffs_rev = a.coeffs().iter()
        .copied()
        .rev();
    let a_rev_coeffs: Vec<F> = zeros.chain(a_coeffs_rev).collect();
    let a_rev = P::from_coefficients_vec(a_rev_coeffs);
    a_rev
}

/// `a(X) rem X^k`
fn rem<F: FftField>(a: &P<F>, k: usize) -> P<F> {
    a.slice(0, k - 1)
}

pub fn div<F: FftField>(a: &P<F>, b: &P<F>, log_l: usize) -> (P<F>, P<F>) {
    let m = a.degree();
    let n = b.degree();
    debug_assert!(m > n);
    let l = m - n + 1;
    assert_eq!(2usize.pow(log_l as u32), l);
    let rev_a = rev(m, &a);
    let rev_b = rev(n, &b);
    // let rev_b_inv = inv_mod_fft(&rev_b, log_l);
    let rev_b_inv = inv_mod(&rev_b, log_l);
    let rev_q = rem(&(rev_a * rev_b_inv), l);
    let q = rev(l - 1, &rev_q);
    let r = a - b * &q;
    (q, r)
}

// uses whatever arkworks does for multiplying polynomials
fn inv_mod_fft<F: FftField>(f: &P<F>, log_l: usize) -> P<F> {
    assert_eq!(f.coeffs[0], F::one());
    let mut g = P::one();
    let mut d = 1;
    for _ in 0..log_l {
        d = d << 1;
        g = &(&g * F::from(2)) - &(f * &(&g * &g));
        g = rem(&g, d);
    }
    g
}

/// `g` such that `fg = 1 mod X^l`
// uses quadratic multiplication
fn inv_mod<F: FftField>(f: &P<F>, log_l: usize) -> P<F> {
    assert_eq!(f.coeffs[0], F::one());
    let l = 2usize.pow(log_l as u32);
    let mut gi = P::one();
    let mut li = 1;
    let mut g_coeffs = Vec::with_capacity(l); // `g_coeffs = gi.coeffs`
    g_coeffs.push(gi.coeffs[0]); // `= P::one()`
    for _ in 0..log_l {
        li = li << 1;
        let g2_high = hensel_lift(&rem(&f, li), &gi);
        g_coeffs.extend(g2_high.coeffs); // `g2 = (g1.coeffs ||  g2_high.coeffs) = g1 + g2_high.X^i`
        gi = P::from_coefficients_slice(&g_coeffs);
    }
    let g = P::from_coefficients_vec(g_coeffs);
    g
}

// See https://people.csail.mit.edu/madhu/ST12/scribe/lect06.pdf, section 2.1
/// Given
/// `g1` such that `f.g1 = 1 mod X^l`,
/// computes
/// `g2` such that `f.g2 = 1 mod X^2l`.
/// Returns the upper half of `g2`.
fn hensel_lift<F: FftField>(f: &P<F>, g1: &P<F>) -> P<F> {
    let i = g1.degree() + 1; // mod X^{i-1}
    let i2 = 2 * i;
    assert_eq!(f.degree(), i2 - 1);

    let g1sq = g1 * g1;
    assert_eq!(g1sq.degree(), i2 - 2);
    let f_g1sq_high = half_mul_mod(f, &g1sq, i2);
    assert_eq!(f_g1sq_high.degree(), i - 1);
    let g2_high: Vec<F> = f_g1sq_high.coeffs.iter().map(|&c| -c).collect();
    let g2_high = P::from_coefficients_vec(g2_high);
    g2_high
}

/// `a.b mod X^l`
pub fn mul_mod<F: FftField>(a: &P<F>, b: &P<F>, l: usize) -> P<F> {
    let mut res = vec![F::zero(); l];
    for (i, &ai) in a.coeffs.iter().take(l).enumerate() {
        for (j, bj) in b.coeffs.iter().take(l - i).enumerate() {
            assert!(i + j < l);
            res[i + j] += ai * bj;
        }
    }
    P::from_coefficients_vec(res)
}

// High half of `a.b mod X^2l`
pub fn half_mul_mod<F: FftField>(a: &P<F>, b: &P<F>, l2: usize) -> P<F> {
    let l = l2 / 2;
    let mut res = vec![F::zero(); l];
    for (i, &ai) in a.coeffs.iter().enumerate().take(l2) {
        let skip = l.saturating_sub(i); // j >= l - i
        for (j, bj) in b.coeffs.iter().enumerate().skip(skip).take((l2 - skip) - i) {
            assert!(i + j >= l);
            assert!(i + j < l2);
            res[i + j - l] += ai * bj;
        }
    }
    P::from_coefficients_vec(res)
}

// pub fn div_fft<F: FftField>(a: &P<F>, b: &P<F>) -> (P<F>, P<F>) {
//     debug_assert!(a.degree() > b.degree());
//     let m = a.degree() - b.degree();
//     let rev_a = rev(a.degree(), &a);
//     let rev_b = rev(b.degree(), &b);
//     let rev_b_inv = inv_mod_fft(&rev_b, 3);
//     let rev_q = rem(&(rev_a * rev_b_inv), m + 1);
//     let q = rev(m, &rev_q);
//     let r = a - b * &q;
//     (q, r)
// }

// fn inv_mod_fft<F: FftField>(f: &P<F>, log_l: usize) -> P<F> {
//     assert_eq!(f.coeffs[0], F::one());
//     let mut g = Monic::with_evals(P::one(), Evaluations::from_vec_and_domain(vec![F::one(), F::one()], GeneralEvaluationDomain::<F>::new(2).unwrap()));
//     let mut n = 1;
//     for _ in 0..log_l {
//         n = n << 1;
//         g = fft_step(&rem(&f, n), g);
//         assert_eq!(g.poly.degree(), n - 1);
//     }
//     g.poly
// }

// fn fft_step<F: FftField>(f: &P<F>, g0: Monic<F>) -> Monic<F> {
//     let n = g0.poly.degree() + 1;
//     assert_eq!(f.degree(), 2 * n - 1);
//
//     let g_evals = double_evals(&g0.poly, g0.evals.as_ref().unwrap());
//     assert_eq!(g_evals.domain().size(), 4 * n);
//     let g2_evals = &g_evals * &g_evals; // 2n
//     // let fg2 = mul_mod(f, &g2, 2 * n);
//     // let fg2 = mul2(f, &g2, n, 2 * n);
//     let f_evals = f.evaluate_over_domain_by_ref(g2_evals.domain());
//     // let s = f_evals.domain().size() / g2_evals.domain().size();
//     // let f_evals: Vec<F> = f_evals.evals.iter().step_by(s).collect();
//     // assert_eq!(f_evals.len(), g_evals.domain().size());
//     // let f_evals = Evaluations::from_vec_and_domain(f_evals, g2_evals.domain());
//     let fg2_evals = &f_evals * &g2_evals;
//     let g_evals = &(&g_evals * F::from(2)) - &fg2_evals;
//     let g = g_evals.interpolate_by_ref();
//     assert_eq!(g.degree(), 4 * n - 3);
//     assert_eq!(g.slice(0, n - 1), g0.poly);
//
//     Monic::with_evals(g, g_evals)
// }




#[cfg(test)]
mod tests {
    use super::*;
    use crate::Poly;
    use ark_bls12_381::Fr;
    use ark_ff::One;
    use ark_std::{end_timer, start_timer, test_rng};


    #[test]
    fn test_rem() {
        let rng = &mut test_rng();
        let n = 10;
        let a = P::<Fr>::rand(n, rng);
        // assert_eq!(rem(&a, 0), P::zero());
        assert_eq!(rem(&a, 1), P::c(a.ct()));
        assert_eq!(rem(&a, n), a.slice(0, n - 1));
        assert_eq!(rem(&a, n + 1), a);
    }

    #[test]
    fn test_rev() {
        let rng = &mut test_rng();
        let (n, m) = (10, 5);
        let a = P::<Fr>::rand(n, rng);
        let rev_a = rev(n, &a);
        assert_eq!(rev(n, &rev_a), a);
        assert_eq!(rev(n + 1, &a), &rev_a * &P::x());


        let b = P::<Fr>::rand(m, rng);
        let q = &a / &b;
        let a_rev = rev_a;
        let qb_rev = &rev(n - m, &q) * &rev(m, &b);
        assert_eq!(rem(&a_rev, n - m + 1), rem(&qb_rev, n - m + 1)); // `rev(a) = rev(qb) mod X^{n-m+1}`
    }

    #[test]
    fn test_inv_mod() {
        let rng = &mut test_rng();
        let log_l = 10;
        let l = 2usize.pow(log_l as u32);
        let mut f = P::<Fr>::rand(l - 1, rng);
        f.coeffs[0] = Fr::one();
        let g = inv_mod(&f, log_l);
        let fg = &f * &g;
        assert_eq!(rem(&fg, l), P::one())
    }

    #[test]
    fn test_mul_mod() {
        let rng = &mut test_rng();

        let log_l = 1;
        let l = 2usize.pow(log_l as u32);
        let l2 = 2 * l;
        let a = P::<Fr>::rand(l2 - 1, rng);
        let b = P::<Fr>::rand(l2 - 1, rng);

        let ab = mul_mod(&a, &b, l2);
        let ab_high = half_mul_mod(&a, &b, l2);

        assert_eq!(ab_high.coeffs, ab.coeffs[l..]);
    }

    // cargo test test_div --release --features="print-trace" -- --show-output
    #[test]
    fn test_div() {
        let rng = &mut test_rng();

        let log_l = 10; // l = deg(a) - deg(b) + 1
        let l = 2usize.pow(log_l as u32);
        let (m, n) = (2 * l - 1, l);
        assert_eq!(l, m - n + 1);
        let a = P::<Fr>::rand(m, rng);
        let mut b = P::<Fr>::rand(n, rng);
        b.coeffs[n] = Fr::one();

        let _t_ark_div = start_timer!(|| format!("Arkworks division, deg = {m} / {n}"));
        let res_ = crate::div(&a, &b);
        end_timer!(_t_ark_div);

        let _t_div = start_timer!(|| format!("Custom division, deg = {m} / {n}"));
        let (q, r) = div(&a, &b, log_l);
        end_timer!(_t_div);

        assert_eq!((q, r), res_);
    }

    // #[test]
    // fn test_fft() {
    //     let rng = &mut test_rng();
    //
    //     let log_n = 2;
    //     let n = 2usize.pow(log_n);
    //     let d1 = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
    //     let d2 = Radix2EvaluationDomain::<Fr>::new(2 * n).unwrap();
    //     let p = P::<Fr>::rand(n - 1, rng);
    //     println!("p = {p:?}");
    //     let p_over_d2 = p.evaluate_over_domain_by_ref(d2);
    //     let p2_over_d2 = &p_over_d2 * &p_over_d2;
    //     let p2 = p2_over_d2.interpolate_by_ref();
    //     println!("p^2 = {p2:?}\n");
    //
    //     let p1 = p.slice(2, 3);
    //     println!("p1 = {p1:?}");
    //     let p_over_d1 = p1.evaluate_over_domain_by_ref(d1);
    //     let p_over_d1 = p1.evaluate_over_domain_by_ref(d1);
    //     let p2_over_d1 = &p_over_d1 * &p_over_d1;
    //     let p2 = p2_over_d1.interpolate_by_ref();
    //     println!("p^2 = {p2:?}\n");
    // }
}
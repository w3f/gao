use crate::Poly;
use crate::P;
use ark_ff::FftField;
use ark_poly::{DenseUVPolynomial, EvaluationDomain, Evaluations, GeneralEvaluationDomain, Polynomial, Radix2EvaluationDomain};
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

pub fn quotient<F: FftField>(a: &P<F>, b: &P<F>) -> P<F> {
    let m = a.degree();
    let n = b.degree();
    debug_assert!(m > n);
    let l = m - n + 1;
    let rev_a = rev(m, &a);
    let rev_b = rev(n, &b);
    let rev_b_inv = inv_mod(&rev_b, l);
    let rev_q = (rev_a * rev_b_inv).mod_xk(l);
    let q = rev(l - 1, &rev_q);
    q
}

pub fn remainder<F: FftField>(a: &P<F>, b: &P<F>, q: &P<F>) -> P<F> {
    let r = a - b * q;
    r
}

pub fn div<F: FftField>(a: &P<F>, b: &P<F>) -> (P<F>, P<F>) {
    let q = quotient(a, b);
    let r = remainder(a, b, &q);
    (q, r)
}

/// `g` such that `fg = 1 mod X^l`
// uses quadratic multiplication
pub fn inv_mod<F: FftField>(f: &P<F>, l: usize) -> P<F> {
    let log_l = ark_std::log2(l);
    let mut gi = P::constant(f.ct().inverse().unwrap());
    let mut li = 1;
    let mut g_coeffs = Vec::with_capacity(l); // `g_coeffs = gi.coeffs`
    g_coeffs.push(gi.coeffs[0]); // `= P::one()`
    for k in 0..log_l {
        li = li << 1;
        let fi = f.mod_xk(li);
        let g2_high = if k < 7 {
            hensel_lift(&fi, &gi, li >> 1)
        } else {
            hensel_lift_fft(&fi, &gi, li >> 1)
        };
        g_coeffs.extend(g2_high.coeffs); // `g2 = (g1.coeffs ||  g2_high.coeffs) = g1 + g2_high.X^i`
        gi = P::from_coefficients_slice(&g_coeffs);
    }
    let g = P::from_coefficients_vec(g_coeffs);
    g
}

// `g2 = 2.g1 - f.g1^2 mod X^2n = g1(2 - f.g1) mod X^2n`
// But `f.g1 = 1 mod X^n`, so `g2 = g1 mod X^n`
// Thus `g2 = g2_low + g2_high.X^n`, where `g2_low = g1` and `g2_high = -(f.g1^2 / X^n)` `
pub fn hensel_lift_fft<F: FftField>(f: &P<F>, g1: &P<F>, n: usize) -> P<F> {
    debug_assert!(g1.degree() < n);
    debug_assert!(f.degree() < 2 * n);
    let d = GeneralEvaluationDomain::new(2 * n).unwrap();
    let g1 = g1.evaluate_over_domain_by_ref(d);
    let fg_high = middle_prod(&g1, &f);
    let fg = P::one() + fg_high.mul_xk(n); // f.g1 = 1 mod X^n
    let fg2_high = middle_prod(&g1, &fg);
    -fg2_high
}

fn middle_prod<F: FftField>(p1_over_d: &Evaluations<F>, p2: &P<F>) -> P<F> {
    let d = p1_over_d.domain();
    let n2 = d.size();
    debug_assert!(p2.degree() < n2);
    let p2_over_d = p2.evaluate_over_domain_by_ref(d);
    let p_over_d = p1_over_d * &p2_over_d;
    let p = p_over_d.interpolate();
    p.div_xk(n2 / 2)
}

fn middle_prod2<F: FftField>(n: usize, p1: &P<F>, p2: &P<F>) -> P<F> {
    debug_assert!(p1.degree() < n);
    debug_assert!(p2.degree() < 2 * n);
    let d = Radix2EvaluationDomain::new(2 * n).unwrap();
    let p1_over_d = p1.evaluate_over_domain_by_ref(d);
    let p2_over_d = p2.evaluate_over_domain_by_ref(d);
    let p_over_d = &p1_over_d * &p2_over_d;
    let p = p_over_d.interpolate();
    p.div_xk(n)
}

// See https://people.csail.mit.edu/madhu/ST12/scribe/lect06.pdf, section 2.1
/// Given
/// `g1` such that `f.g1 = 1 mod X^l`,
/// computes
/// `g2` such that `f.g2 = 1 mod X^2l`.
/// Returns the upper half of `g2`.
pub fn hensel_lift<F: FftField>(f: &P<F>, g1: &P<F>, n: usize) -> P<F> {
    debug_assert!(g1.degree() < n);
    debug_assert!(f.degree() < 2 * n);
    let g1sq = g1.naive_mul(g1);
    let f_g1sq_high = half_mul_mod(f, &g1sq, 2 * n);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Poly;
    use ark_bls12_381::Fr;
    use ark_std::{end_timer, start_timer, test_rng};

    // #[test]
    // fn test_rem() {
    //     let rng = &mut test_rng();
    //     let n = 10;
    //     let a = P::<Fr>::rand(n, rng);
    //     // assert_eq!(rem(&a, 0), P::zero());
    //     assert_eq!(rem(&a, 1), P::constant(a.ct()));
    //     assert_eq!(rem(&a, n), a.slice(0, n - 1));
    //     assert_eq!(rem(&a, n + 1), a);
    // }
    //
    // #[test]
    // fn test_rev() {
    //     let rng = &mut test_rng();
    //     let (n, m) = (10, 5);
    //     let a = P::<Fr>::rand(n, rng);
    //     let rev_a = rev(n, &a);
    //     assert_eq!(rev(n, &rev_a), a);
    //     assert_eq!(rev(n + 1, &a), &rev_a * &P::x());
    //
    //
    //     let b = P::<Fr>::rand(m, rng);
    //     let q = &a / &b;
    //     let a_rev = rev_a;
    //     let qb_rev = &rev(n - m, &q) * &rev(m, &b);
    //     assert_eq!(rem(&a_rev, n - m + 1), rem(&qb_rev, n - m + 1)); // `rev(a) = rev(qb) mod X^{n-m+1}`
    // }

    #[test]
    fn test_inv_mod() {
        let rng = &mut test_rng();
        let log_l = 10;
        let l = 2usize.pow(log_l as u32);
        let f = P::<Fr>::rand(l - 1, rng);
        let g = inv_mod(&f, l);
        let fg = &f * &g;
        assert_eq!(fg.mod_xk(l), P::one())
    }

    #[test]
    fn test_mul_mod() {
        let rng = &mut test_rng();

        let log_l = 9;
        let l = 2usize.pow(log_l as u32);
        let l2 = 2 * l;
        let a = P::<Fr>::rand(l2 - 1, rng);
        let b = P::<Fr>::rand(l2 - 1, rng);

        let ab = mul_mod(&a, &b, l2);
        let ab_high = half_mul_mod(&a, &b, l2);

        assert_eq!(ab_high.coeffs, ab.coeffs[l..]);
    }

    // RUST_BACKTRACE=1 cargo test test_div --release --features="print-trace" -- --show-output
    #[test]
    fn test_div() {
        let rng = &mut test_rng();

        let log_l = 16; // l = deg(a) - deg(b) + 1
        let l = 2usize.pow(log_l as u32);
        let (m, n) = (2 * l - 1, l);
        assert_eq!(l, m - n + 1);
        let a = P::<Fr>::rand(m, rng);
        let b = P::<Fr>::rand(n, rng);

        let _t_ark_div = start_timer!(|| format!("Arkworks division, deg = {m} / {n}"));
        let res_ = crate::tests::ark_div(&a, &b);
        end_timer!(_t_ark_div);

        let _t_div = start_timer!(|| format!("Custom division, deg = {m} / {n}"));
        let (q, r) = div(&a, &b);
        end_timer!(_t_div);

        assert_eq!((q, r), res_);
    }

    // #[test]
    // fn test_middle_product() {
    //     let rng = &mut test_rng();
    //
    //     let i = 12;
    //     let n = 2usize.pow(i as u32);
    //     let domain = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    //     let domain_2n = GeneralEvaluationDomain::<Fr>::new(2 * n).unwrap();
    //
    //     let f = P::<Fr>::rand(n - 1, rng);
    //     let g = inv_mod(&rem(&f, n / 2), i - 1);
    //
    //     let _t_lp = start_timer!(|| format!("Long product"));
    //
    //     let _t_g2_1 = start_timer!(|| format!("1"));
    //     let g_over_2n = g.evaluate_over_domain_by_ref(domain_2n); // 2n-FFT
    //     let g2_over_2n = &g_over_2n * &g_over_2n; // 2n x M
    //     end_timer!(_t_g2_1);
    //
    //     let _t_g2_2 = start_timer!(|| format!("2"));
    //     let g_over_n = g.evaluate_over_domain_by_ref(domain);
    //     let g2_over_n = &g_over_n * &g_over_n;
    //     let g2 = g2_over_n.interpolate_by_ref();
    //     let g2_over_2n_ = double_evals(&g2, &g2_over_n);
    //     end_timer!(_t_g2_2);
    //     assert_eq!(g2_over_2n, g2_over_2n_);
    //
    //     let f_over_2n = f.evaluate_over_domain_by_ref(domain_2n); // 2n-FFT
    //     let fg2_over_2n = &f_over_2n * &g2_over_2n; // 2n x M
    //     let fg2 = fg2_over_2n.interpolate_by_ref(); // 2n-iFFT
    //     let fg2_ = fg2.mod_xn(n);
    //     end_timer!(_t_lp);
    //
    //     let _t_mp = start_timer!(|| format!("Middle product"));
    //     let f_over_n = f.evaluate_over_domain_by_ref(domain); // n-FFT
    //     let g_over_n = g.evaluate_over_domain_by_ref(domain); // n-FFT
    //     let fg_over_n = &f_over_n * &g_over_n; // n x M
    //     let fg = fg_over_n.interpolate_by_ref(); // n-iFFT
    //     let fg_high = fg.div_xn(n / 2);
    //     let fg = P::one() + fg_high.mul_xn(n / 2);
    //     // assert_eq!(fg, (&f * &g).mod_xn(n));
    //     let fg_over_n = fg.evaluate_over_domain_by_ref(domain); // n-FFT
    //     let fg2_over_n = &fg_over_n * &g_over_n; // n x M
    //     let fg2 = fg2_over_n.interpolate_by_ref();  // n-iFFT
    //     let fg2_high = fg2.div_xn(n / 2);
    //     let fg2 = g + fg2_high.mul_xn(n / 2);
    //     end_timer!(_t_mp);
    //
    //     assert_eq!(fg2, fg2_);
    // }


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
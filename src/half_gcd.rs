use crate::bezout::{coeff_at, double, eval_over_k, interpolate, middle_prod, mul, mul_evals, BezoutMatrix};
use crate::Poly;
use crate::{M, ME, P};
use ark_ff::{FftField, Field, Zero};
use ark_poly::EvaluationDomain;
use ark_poly::{DenseUVPolynomial, Polynomial};

/// For a degree `d` polynomial `p` removes its `i < d + 1` lower coefficients
/// and returns a polynomial of degree `d - i` retaining `d - i + 1` higher coefficients.
/// Equivalently, returns `p(X) / X^{i}`.
/// This notation from the paper, page 9 ("U_i = ..."), is different from one in the book.
fn truncate_i<F: Field>(p: &P<F>, i: usize) -> P<F> {
    truncate_ij(p, i, p.degree() + 1)
}

fn truncate_ij<F: Field>(p: &P<F>, i: usize, j: usize) -> P<F> {
    assert!(j <= p.coeffs.len());
    if i < j {
        P::from_coefficients_slice(&p.coeffs[i..j])
    } else {
        P::zero()
    }
}

/// Consistently truncates the coefficients of 2 polynomials `(r0, r1)`.
/// `(r0_k, r1_k)` contain enough information to compute `r_{k+1}`, see Lemma 4 in the paper.
fn truncate_pair<F: Field>(p: &P<F>, q: &P<F>, k: usize) -> (P<F>, P<F>) {
    let dp = p.degree();
    let dq = q.degree();
    assert!(dq < dp); // TODO: why?
    if dp > 2 * k {
        let i = dp - 2 * k;
        let p = truncate_i(&p, i);
        let q = truncate_i(&q, i);
        (p, q)
    } else {
        (p.clone(), q.clone())
    }
}

/// Algorithm 1 from the article.
/// Works for "normal degree sequences" only.
pub fn simple_half_gcd<F: FftField>(
    p: &P<F>,
    q: &P<F>,
    k: usize,
) -> BezoutMatrix<F> {
    let d = p.degree();
    // println!("k = {k}, deg(P) = {d}, deg(Q) = {}", q.degree());
    assert_eq!(d, q.degree() + 1);
    assert!(k > 0);
    assert!(d >= k);

    if k == 1 {
        let (p1, q1) = truncate_pair(p, q, 1);
        debug_assert!(p1.degree() <= 2);
        debug_assert_eq!(q1.degree() + 1, p1.degree());
        return BezoutMatrix::new(&p1, &q1).unwrap(); // B_{1; 2} = B_1
    }

    let h1 = k.div_ceil(2);
    let h2 = k - h1;
    // println!("h1 = {h1}, h2 = {h2}");

    let (p1, q1) = truncate_pair(&p, &q, h1);
    debug_assert_eq!(p1.degree(), q1.degree() + 1);
    // println!("deg(P1) = {}, deg(Q1) = {}", p1.degree(), q1.degree());
    let m1 = simple_half_gcd(&p1, &q1, h1); // `= B_{1; h1+1}(p, q)`
    let (p2, q2) = m1.apply(&p, &q); // `= B_{1; h1+1} * A_1 = A_{h1+1} = (r_h1, r_{h1+1})`
    // println!("deg(P2) = {}, deg(Q2) = {}", p2.degree(), q2.degree());
    debug_assert_eq!(p2.degree(), d - h1); // `deg(r_i) = d - i`
    debug_assert_eq!(p2.degree(), q2.degree() + 1);
    let (p2_tr, q2_tr) = truncate_pair(&p2, &q2, h2);
    let m2 = simple_half_gcd(&p2_tr, &q2_tr, h2); // `= B_{1; h2+1}(p2', q2') = B_{1; h2+1}(p2, q2) by the lemma
    // But `deg(p2) = d - h1`, so B_{1; h2+1}(p2, q2) = B_{1+h1; h2+1+h1}(p, q)
    m2.compose(&m1)
    //`B_{h1+1; k+1}(p, q) * B_{1; h1+1}(p, q) = B_{1; k+1}(p, q)`
}

fn split_one<F: FftField>(f: &P<F>, d: usize, h: usize) -> [P<F>; 3] {
    let res = f.coeffs[d - 4 * h..].chunks(h)
        .map(|coeffs| P::with_coeffs(coeffs.to_vec()))
        .take(3)
        .collect::<Vec<_>>();

    assert_eq!(res[0], truncate_ij(f, d - 4 * h, d - 3 * h));
    res.try_into().unwrap()
}

fn split_pair<F: FftField>(p: &P<F>, q: &P<F>, h: usize) -> M<F> {
    let d = p.degree();
    debug_assert!(h > 0);
    debug_assert_eq!(d, q.degree() + 1);
    debug_assert_eq!(d, 4 * h); // d >= 2k = 4h
    let p_chunks = split_one(p, d, h);
    let q_chunks = split_one(q, d, h);
    let res = [
        &p_chunks[0] + &p_chunks[1].mul_xk(h),
        &p_chunks[1] + &p_chunks[2].mul_xk(h),
        &q_chunks[0] + &q_chunks[1].mul_xk(h),
        &q_chunks[1] + &q_chunks[2].mul_xk(h),
    ];
    assert_eq!(res[3].degree(), 2 * h - 1);
    res
}

fn conv<F: FftField>(a: &P<F>, b: &P<F>, d: usize, h: usize) -> F {
    a.coeffs.iter()
        .zip(b.coeffs[0..d - h + 1].iter().rev())
        .map(|(&mi, pj)| mi * pj)
        .take(h + 1)
        .sum()
}

fn truncated_pq<F: FftField>(M: &M<F>, M_evals: &ME<F>, p: &P<F>, q: &P<F>, h: usize) -> (P<F>, P<F>) {
    let pq = split_pair(p, q, h);
    let pq2_tr = middle_prod(h, &pq, M_evals);
    let d = p.degree();
    let extra_p = conv(&M[0], p, d, h) + conv(&M[1], q, d, h);
    let p2_tr = &pq2_tr[0] + &pq2_tr[1].mul_xk(h) + &P::constant(extra_p).mul_xk(2 * h);
    let q2_tr = &pq2_tr[2] + &pq2_tr[3].mul_xk(h);
    (p2_tr, q2_tr)
}

/// Algorithm 2 from the article.
/// Works for "normal degree sequences" and `k=2^l`.
pub fn simple_half_gcd2<F: FftField>(
    p: &P<F>,
    q: &P<F>,
    k: usize,
) -> (BezoutMatrix<F>, ME<F>) {
    let d = p.degree();
    // println!("k = {k}, deg(P) = {d}, deg(Q) = {}", q.degree());
    assert_eq!(d, q.degree() + 1);
    assert!(k > 0);
    assert!(d >= 2 * k);

    if k == 1 {
        let (p1, q1) = truncate_pair(p, q, 1);
        debug_assert!(p1.degree() <= 2);
        debug_assert_eq!(q1.degree() + 1, p1.degree());
        let B1 = BezoutMatrix::new(&p1, &q1).unwrap(); // B_{1; 2} = B_1
        let B1_evals = eval_over_k(1, &B1.0);
        return (B1, B1_evals);
    }

    let h = k / 2;
    // println!("h = {h}");

    let (p1, q1) = truncate_pair(&p, &q, h);
    debug_assert_eq!(p1.degree(), q1.degree() + 1);
    // println!("deg(P1) = {}, deg(Q1) = {}", p1.degree(), q1.degree());
    let (M1, M1_evals) = simple_half_gcd2(&p1, &q1, h); // `= B_{1; h+1}(p, q)`
    assert_eq!(M1.degree(), h);
    let M1_cap = double(&M1.0, &M1_evals);
    // let (p2, q2) = M1.apply(&p, &q); // `= B_{1; h+1} * A_1 = A_{h+1} = (r_h, r_{h+1})`
    // println!("deg(P2) = {}, deg(Q2) = {}", p2.degree(), q2.degree());
    // debug_assert_eq!(p2.degree(), d - h);
    // debug_assert_eq!(p2.degree(), q2.degree() + 1);

    let (p2_tr, q2_tr) = truncated_pq(&M1.0, &M1_cap, &p, &q, h);
    // let (p2_tr_, q2_tr_) = truncate_pair(&p2, &q2, h);
    // assert_eq!(p2_tr_, p2_tr);
    // assert_eq!(q2_tr_, q2_tr);

    // debug_assert_eq!(p2_tr.degree(), 2 * h);
    debug_assert_eq!(p2_tr.degree(), q2_tr.degree() + 1);
    let (M2, M2_evals) = simple_half_gcd2(&p2_tr, &q2_tr, h); // `= B_{1; h+1}(p2, q2)
    assert_eq!(M2.degree(), h);
    let M2_cap = double(&M2.0, &M2_evals);
    // But `deg(p2) = d - h1`, so B_{1; h2+1}(p2, q2) = B_{1+h1; h2+1+h1}(p, q)
    let M2M1_cap = mul_evals(&M2_cap, &M1_cap);
    let M1_h = coeff_at(&M1.0, h);
    let M2_h = coeff_at(&M2.0, h);
    let M2M1_h = mul(&M2_h, &M1_h);
    let mut M2M1 = interpolate(&M2M1_cap);
    assert_eq!(M2M1[3].degree(), k - 1);
    let modulus = P::xk(k) - P::one();
    for (i, lc_i) in M2M1_h.iter().enumerate() {
        M2M1[i] += &(lc_i * &modulus);
    }
    //`B_{h1+1; k+1}(p, q) * B_{1; h1+1}(p, q) = B_{1; k+1}(p, q)`
    assert_eq!(M2M1_cap[3].domain().size(), k);
    (BezoutMatrix(M2M1), M2M1_cap)
}

// pub fn half_euclid2<F: FftField>(p: DensePolynomial<F>, q: DensePolynomial<F>, k: usize, max_deg: Option<usize>) -> BezoutMatrix<F> {
//     let d = p.degree();
//     println!("k = {k}, deg(P) = {d}, deg(Q) = {}", q.degree());
//     let qq = truncate(&q, d - k);
//     if qq.is_zero() {
//         return BezoutMatrix::id();
//     }
//
//     if max_deg.is_some() && q.degree() < max_deg.unwrap() {
//         return BezoutMatrix::quotient_matrix(&p, &q);
//     }
//
//     assert!(k <= d);
//     assert!(q.degree() < d);
//     assert!(k != 0);
//
//     if k == 1 {
//         let (p1, q1) = truncate_pair(&p, &q, 1);
//         return BezoutMatrix::quotient_matrix(&p1, &q1);
//     }
//
//     let h1 = k.div_ceil(2);
//     let mut h2 = k - h1;
//     // println!("h1 = {h1}, h2 = {h2}");
//
//     let (p1, q1) = truncate_pair(&p, &q, h1);
//     // println!("deg(P1) = {}, deg(Q1) = {}", p1.degree(), q1.degree());
//     let mut m1 = half_euclid2(p1, q1, h1, max_deg);
//     let delta = h1 - m1.degree();
//     let (mut p2, mut q2) = m1.apply(&p, &q);
//     // println!("deg(P2) = {}, deg(Q2) = {}", p2.degree(), q2.degree());
//     assert_eq!(p2.degree(), d - h1 + delta);
//     assert!(q2.degree() < d - h1);
//     if truncate(&q2, d - k).is_zero() {
//         return m1;
//     }
//
//     let i = if d > h1 + 2 * h2 + delta { d - h1 - 2 * h2 - delta } else { 0 };
//     let p21 = truncate(&p2, i);
//     let q21 = truncate(&q2, i);
//     let mut h3 = h2;
//
//     if delta > 0 {
//         let j = BezoutMatrix::quotient_matrix(&p21, &q21);
//         assert_eq!(j.0[3], -(&p2 / &q2));
//         m1 = j.multiply(&m1);
//         h3 = k - m1.degree();
//         assert!(h3 <= h2);
//         let p3 = q2.clone();
//         let q3 = p2 + &j.0[3] * &q2;
//         assert_eq!(m1.apply(&p, &q), (p3.clone(), q3.clone()));
//         p2 = p3;
//         q2 = q3;
//
//         let i = if d > k + h3 { d - k - h3 } else { 0 };
//         let p2 = truncate(&p2, i);
//         let q2 = truncate(&q2, i);
//         let m2 = half_euclid2(p2, q2, h3, max_deg);
//         m2.multiply(&m1)
//     } else {
//         let m2 = half_euclid2(p21, q21, h2, max_deg);
//         m2.multiply(&m1)
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bezout::BezoutMatrix;
    use crate::gcd;
    use ark_bls12_381::Fr;
    use ark_ff::{One, Zero};
    use ark_poly::univariate::DensePolynomial;
    use ark_poly::DenseUVPolynomial;
    use ark_std::{end_timer, start_timer, test_rng};

    #[test]
    fn truncation() {
        let rng = &mut test_rng();
        let d = 10;
        let p = DensePolynomial::<Fr>::rand(d, rng);
        let (i, j) = (5, 8);

        let p_i = truncate_i(&p, i);
        let mut Xi = vec![Fr::zero(); i + 1];
        Xi[i] = Fr::one();
        let Xi = DensePolynomial::from_coefficients_vec(Xi);
        assert_eq!(p_i, &p / &Xi);

        let p_ij = truncate_ij(&p, i, j);
        assert_eq!(p_ij.coeffs[0], p.coeffs[i]);
        assert_eq!(p_ij.coeffs.len(), j - i);
        assert_eq!(p_ij.degree(), j - i - 1);
        assert_eq!(p_ij.coeffs[j - i - 1], p.coeffs[j - 1]);
    }

    #[test]
    fn test_lemma() {
        let rng = &mut test_rng();
        let r0 = DensePolynomial::<Fr>::rand(100, rng);
        let r1 = DensePolynomial::<Fr>::rand(99, rng);
        let B1 = BezoutMatrix::new(&r0, &r1).unwrap();
        let (r0_88, r1_88) = truncate_pair(&r0, &r1, 1);
        let B1_ = BezoutMatrix::new(&r0_88, &r1_88).unwrap();
        assert_eq!(B1_, B1);

        // let r2 = B1.next_remainder(&r0, &r1);
        // let B2 = BezoutMatrix::new(&r1, &r2).unwrap();
        // let r3 = B2.next_remainder(&r1, &r2);
        // let B3 = BezoutMatrix::new(&r2, &r3).unwrap();
        //
        // let B14 = B3.compose(&B2.compose(&B1));
    }


    #[test]
    fn test_simple_half_gcd1() {
        let rng = &mut test_rng();
        let d = 1024;

        let f = P::<Fr>::rand(d, rng);
        let g = P::<Fr>::rand(d - 1, rng);
        let _t_gcd = start_timer!(|| format!("Half-GCD for normal degree sequences, deg = {d}"));
        let B = simple_half_gcd(&f, &g, d);
        let (gcd, zero) = B.apply(&f, &g);
        end_timer!(_t_gcd);
        assert_eq!(zero.degree(), 0);

        let _t_gcd = start_timer!(|| format!("EEA, deg = {d}"));
        let (r, s, t) = gcd::euclid(&f, &g);
        end_timer!(_t_gcd);
        assert_eq!(gcd, r);
        assert_eq!(B.0[0], s);
        assert_eq!(B.0[1], t);
    }

    // cargo test test_simple_half_gcd --release --features="print-trace" -- --show-output
    #[test]
    fn test_simple_half_gcd2() {
        let rng = &mut test_rng();
        let d = 1024;
        let k = d;
        let f = P::<Fr>::rand(d, rng);
        let g = P::<Fr>::rand(d - 1, rng);
        let _t_gcd = start_timer!(|| format!("Half-GCD with FFTs, deg = {d}"));
        let (B, _) = simple_half_gcd2(&f.mul_xk(k), &g.mul_xk(k), k);
        let (gcd, zero) = B.apply(&f, &g);
        end_timer!(_t_gcd);
        assert_eq!(zero.degree(), 0);

        let _t_gcd = start_timer!(|| format!("EEA, deg = {d}"));
        let (r, s, t) = gcd::euclid(&f, &g);
        end_timer!(_t_gcd);
        assert_eq!(gcd, r);
        assert_eq!(B.0[0], s);
        assert_eq!(B.0[1], t);
    }
}

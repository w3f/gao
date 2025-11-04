use crate::P;
use ark_ff::{FftField, Field, Zero};
use ark_poly::{DenseUVPolynomial, Polynomial};
use ark_poly::univariate::DensePolynomial;
use crate::bezout::BezoutMatrix;

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

/// Works for "normal degree sequences" only.
pub fn simple_half_gcd<F: FftField>(p: &DensePolynomial<F>, q: &DensePolynomial<F>, k: usize) -> BezoutMatrix<F> {
    let d = p.degree();
    // println!("k = {k}, deg(P) = {d}, deg(Q) = {}", q.degree());
    assert!(k <= d && k > 0);
    assert!(q.degree() < d);

    if k == 1 {
        let (p1, q1) = truncate_pair(p, q, k);
        return BezoutMatrix::new(&p1, &q1).unwrap();
    }

    let h1 = k.div_ceil(2);
    let h2 = k - h1;
    // println!("h1 = {h1}, h2 = {h2}");

    let (p1, q1) = truncate_pair(&p, &q, h1);
    // println!("deg(P1) = {}, deg(Q1) = {}", p1.degree(), q1.degree());
    let m1 = simple_half_gcd(&p1, &q1, h1); // `= B_{1; h1+1}(p, q)`
    let (p2, q2) = m1.apply(&p, &q); // `= B_{1; h1+1} * A_1 = A_{h1+1} = (r_h1, r_{h1+1})`
    // println!("deg(P2) = {}, deg(Q2) = {}", p2.degree(), q2.degree());
    // assert_eq!(p2.degree(), d - h1); // `deg(r_i) = d - i`
    let (p2, q2) = truncate_pair(&p2, &q2, h2);
    let m2 = simple_half_gcd(&p2, &q2, h2); // `= B_{1; h2+1}(p2, q2)
    // But `deg(p2) = d - h1`, so B_{1; h2+1}(p2, q2) = B_{1+h1; h2+1+h1}(p, q)
    m2.compose(&m1)
    //`B_{h1+1; k+1}(p, q) * B_{1; h1+1}(p, q) = B_{1; k+1}(p, q)`
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
    use ark_bls12_381::Fr;
    use ark_ff::{One, Zero};
    use ark_poly::univariate::DensePolynomial;
    use ark_poly::DenseUVPolynomial;
    use ark_std::{end_timer, start_timer, test_rng};
    use crate::gcd;

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
    fn simple_half_gcd() {
        let rng = &mut test_rng();
        let d = 1023;
        let f = P::<Fr>::rand(d, rng);
        let g = P::<Fr>::rand(d - 1, rng);
        let _t_gcd = start_timer!(|| format!("Half-GCD for normal degree sequences, deg = {d}"));
        let B = super::simple_half_gcd(&f, &g, d);
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
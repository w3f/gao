use crate::{Poly, M, P};
use ark_ff::{FftField, Zero};
use ark_poly::Polynomial;

/// 2x2 matrix with polynomial entries, sometimes viewed as a polynomial with matrix coefficients.
#[derive(Debug, PartialEq)]
pub struct PM<F: FftField>(pub M<F>);

impl<F: FftField> PM<F> {
    pub fn id() -> Self {
        Self([P::one(), P::zero(), P::zero(), P::one()])
    }

    /// Let `-q = -(r0 / r1)`.
    /// Then `Q(-q) * (r0, r1)T = (r1, r2)T`,
    /// where `r2 = r0 - (r0/r1).r1 = r0 mod r1`.
    pub fn quotient(minus_q: P<F>) -> Self {
        Self([P::zero(), P::one(), P::one(), minus_q])
    }

    pub fn deg(&self) -> usize {
        self.0.iter().map(|p| p.degree()).max().unwrap()
    }

    pub fn apply(&self, r0: &P<F>, r1: &P<F>) -> (P<F>, P<F>) {
        (
            &self.0[0] * r0 + &self.0[1] * r1,
            &self.0[2] * r0 + &self.0[3] * r1,
        )
    }

    pub fn compose(&self, other: &Self) -> Self {
        Self([
            &self.0[0] * &other.0[0] + &self.0[1] * &other.0[2],
            &self.0[0] * &other.0[1] + &self.0[1] * &other.0[3],
            &self.0[2] * &other.0[0] + &self.0[3] * &other.0[2],
            &self.0[2] * &other.0[1] + &self.0[3] * &other.0[3],
        ])
    }
}

// TODO: for d = 64 textbook EEA is faster

/// *Algorithm 4* from the paper.
///
/// The input is `(P, Q), deg(Q) < deg(P) = d` and `k <= d`.
///
/// Let `R[0] := P, R[1] := Q`;
/// `R[i] := R[i-2] % R[i-1], i = 2,...,l` where `R[l-1] != 0` and `R[l] = 0`.
/// `R` is the remainder sequence of `(P,Q)`.
///
/// Let `Q[i] := R[i-1] / R[i], i = 1,...,l-1`,
/// then `R[i+1] = R[i-1] % R[i] = R[i-1] - Q[i]R[i]`.
///
/// Define `A[i] := (R[i-1], R[i])T` the pair of remainders at the step `i = 1...,l`
/// (e.g., `A[1] = (P, Q)T` and `A[l] = (gcd(P,Q), 0)T`).
///
/// Then `B[i] := ((0, 1), (1, -Q[i])T` is such that `A[i+1] = B[i]A[i]` for `i = 1,...,l-1`,   (**)
/// `deg(B[i]) = deg(Q[i]) = deg(R[i-1]) - deg(R[i])`.
///
/// Set `B[i,j) := B[j-1]...B[i], i < j`.
/// Then `A[j] = B[i,j)A[i]`,                                                                    (*)
/// `B[i,k) = B[j,k)B[i,j), i < j < k`,                                                          (2)
/// and `B[1,k-i+1)(R[i], R[i+1]) = B[i+1,k+1)(R[0], R[1])`.                                    (3)
///
/// `deg(B[i,j)) = deg(B[i]) + ... + deg(B[j-1]) = deg(R[i-1]) - deg(R[j-1])`
/// setting `i := 1, j := i+1` gives `deg(R[i]) = d - deg(B[1,i+1))`
///
/// Define `k(i) := d - deg(R[i]) = deg(B[1,i+1)), i = 0,...,l-1`,
/// and `i(k) := max(i | k(i) <= k)` for `k >= 0`.
/// `i(k) = i <=> k(i) <= k < k(i+1)`.
///
/// Finally, the output is `B[1,i(k)+1)(P,Q) = B[1,j+1)` for `j := i(k)`.
/// In particular `B[1,j+1) * (P,Q)T = B[1,j+1) * A[1] = A[j+1] = (R[j], R[j+1])T`.
/// Then `deg(B[1,j+1)) <= k`, `deg(R[j]) = d - deg(B[1,j+1)) >= d - k`, and `deg(R[j+1]) < d - k`
///
///
/// The basic idea of the algorithm is:
/// 1. Compute `M1 := eea(P, Q, k/2) = B[1, i(k/2)+1)(P,Q)` recursively.
/// 2. Compute `A = M1 * (P,Q)T`.
///    Let `j = i(k/2)+1`, then `A = B[1,j) * A[1] = A[j] = (R[j-1], R[j])T`.
/// 3. If `k/2 > deg(M)` compute `B[j]`.
/// 4. Compute `A[j+1] = B[j] * A[j]` by (**) and `B[1,j+1) = B[j] * B[1, j)`
/// 5. Set `k' := k - deg(B[1,j+1))` and
///    compute `M2 := eea(R[j], R[j+1], k') = B[1, i'(k')+1)(R[j], R[j+1])` recursively.
///    Notice that `i'(k') = i'(k - deg(B[1,j+1)) = i(k) - j`,
///    therefore `M2 = B[1, i(k)-j+1)(R[j], R[j+1]) = B[j+1, i(k)+1)(R[0], R[1])` by (3).
/// 6. Return `M2 * M1 = B[j+1, i(k)+1) * B[1,j+1) = B[1, i(k)+1)` by (2)
pub fn eea<F: FftField>(p: &P<F>, q: &P<F>, k: usize) -> (PM<F>, Vec<P<F>>) {
    let d = p.degree();
    // println!("k = {k}, deg(P) = {d}, deg(Q) = {}", q.degree());
    // assert!(d > q.degree(), "deg(P) = {d} <= {} = deg(Q)", q.degree());
    // assert!(k > 0, "k = 0");
    assert!(d >= k, "k > {d} = deg(P)");

    if q.div_xk(d - k).is_zero() {
        debug_assert!(q.is_zero() || q.degree() < d - k);
        return (PM::id(), vec![]);
    }
    debug_assert!(p.degree() - q.degree() <= k);

    if k == 1 {
        debug_assert_eq!(p.degree() - q.degree(), 1); //TODO: remove
        let (p_t, q_t) = truncate(p, q, 1);
        debug_assert!(p_t.degree() <= 2);
        debug_assert_eq!(p_t.degree(), q_t.degree() + 1); //TODO: remove
                                                          // let (p_t, q_t) = (p , q);
        let q = p_t / q_t; //TODO: specify div
        let minus_q = -q.clone();
        let Q = PM::quotient(minus_q); // `B[1] = B[1,2)(P, Q)`
        return (Q, vec![q]);
    }

    let h1 = k.div_ceil(2);
    let h2 = k - h1;

    let (p_t, q_t) = truncate(&p, &q, h1);
    // let (p_t, q_t) = (p , q);
    let (mut M1, mut qs) = eea(&p_t, &q_t, h1); // `= B[1,j+1)(P,Q), j = i(h1)`
    debug_assert!(M1.deg() <= h1);
    // let (p_t, q_t) = truncate(&p, &q, d - M1.deg());
    let (mut p1, mut q1) = M1.apply(&p, &q); // ` = (R[j], R[j+1])T`
    debug_assert_eq!(p1.degree(), d - M1.deg());
    debug_assert!(q1.degree() < d - h1);

    if q1.div_xk(d - k).is_zero() {
        debug_assert!(q1.is_zero() || d - q1.degree() > k);
        // `d - deg(R[j+1]) = k(j+1) > k`, therefore
        // `i(k) = max(i | k(i) <= k) = j = i(h1)`.
        // Thus `B[1,i(k)+1)(P,Q) = B[1,i(h1)+1)(P,Q)`.
        return (M1, qs);
    }
    debug_assert!(q1.degree() >= d - k);

    let delta = h1 - M1.deg();

    // deg(p1) = d - deg(M1) = d - h1 + delta
    // deg(q1) < d - h1
    // deg(q1) >= d - k
    // deg(p1) - deg(q1) > (d - h1 + delta) - (d - h1) = delta
    // deg(p1) - deg(q1) <= (d - h1 + delta) - (d - k) = (k - h1) + delta = h2 + delta
    debug_assert!(p1.degree() - q1.degree() <= h2 + delta);
    debug_assert!(p1.degree() - q1.degree() > delta);

    if delta > 0 {
        let (p1_t, q1_t) = truncate(&p1, &q1, h2 + delta);
        let d = &p1_t / &q1_t;
        debug_assert_eq!(d, &p1 / &q1);
        let D = PM::quotient(-d.clone());
        (p1, q1) = D.apply(&p1_t, &q1_t);
        M1 = D.compose(&M1);
        qs.push(d);
    }

    let h2 = k - M1.deg();
    let (p1_t, q1_t) = truncate(&p1, &q1, h2);
    let (M2, qs2) = eea(&p1_t, &q1_t, h2);
    let M = M2.compose(&M1);
    qs.extend(qs2);
    (M, qs)
}

pub fn truncate<F: FftField>(p: &P<F>, q: &P<F>, i: usize) -> (P<F>, P<F>) {
    let d = p.degree();
    debug_assert!(q.degree() < d); // TODO: make not strict
    let (p_t, q_t) = if d > 2 * i {
        let k = d - 2 * i;
        (p.div_xk(k), q.div_xk(k))
    } else {
        (p.clone(), q.clone())
    };
    // println!("dep(p) - dep(p_t) = {}", p.degree() - p_t.degree());
    (p_t, q_t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ark_div, P};
    use ark_bls12_381::Fr;
    use ark_poly::DenseUVPolynomial;

    use crate::gcd;
    use ark_std::rand::Rng;
    use ark_std::{end_timer, start_timer, test_rng};

    // fn check_res(B: &PM, p: &P, q: &P, k: usize) {
    //     assert!(B.deg() <= k);
    //     let (r1, r2) = B.apply(&p, &q);
    //     assert_eq!(r1.degree(), p.degree() - B.deg());
    //     let r3 = ark_div(&r1, &r2).1;
    //     assert!(r3.degree() < p.degree() - k);
    // }

    // #[test]
    // fn test_k_one() {
    //     let rng = &mut test_rng();
    //
    //     let k = 1;
    //
    //     // if `deg(p) - deg(q) > k`, then `Q = Id` is returned.
    //     let r0 = P::rand(2, rng);
    //     let r1 = P::rand(0, rng);
    //     let (r1_, r2) = eea(&r0, &r1, k).apply(&r0, &r1);
    //     assert!(r2.degree() < r0.degree() - k);
    //
    //     let r0 = P::rand(1, rng);
    //     let r1 = P::rand(0, rng);
    //     let (r1_, r2) = eea(&r0, &r1, k).apply(&r0, &r1);
    //     assert_eq!(r1_, r1);
    //     assert!(r2.is_zero());
    //
    //     let r0 = P::rand(2, rng);
    //     let r1 = P::rand(1, rng);
    //     let (r1_, r2) = eea(&r0, &r1, k).apply(&r0, &r1);
    //     assert_eq!(r1_, r1);
    //     assert!(r2.degree() < r0.degree() - k);
    //     assert!(r2.degree() < r1.degree());
    //     assert_eq!(r2, &r0 - (&r0 / &r1) * r1);
    //
    //     let r0 = P::rand(3, rng);
    //     let r1 = P::rand(2, rng);
    //     let (r1_, r2) = eea(&r0, &r1, k).apply(&r0, &r1);
    //     assert_eq!(r1_, r1);
    //     assert!(r2.degree() < r0.degree() - k);
    //     assert!(r2.degree() < r1.degree());
    // }

    fn get_polys<R: Rng>(gcd: &P, deg_qs: &[usize], rng: &mut R) -> (Vec<P>, Vec<P>) {
        let mut rems = Vec::with_capacity(deg_qs.len() + 2);
        rems.push(P::zero());
        rems.push(gcd.clone());
        let qs: Vec<P> = deg_qs.iter().map(|deg_qi| P::rand(*deg_qi, rng)).collect();
        for (i, qi) in qs.iter().rev().enumerate() {
            // `R[i-2] = R[i] + Q[i-1]R[i-1]`
            let r2 = &rems[i] + &(qi * &rems[i + 1]);
            rems.push(r2);
        }
        rems.reverse();
        (qs, rems)
    }

    #[test]
    fn test_eea() {
        let rng = &mut test_rng();
        let gcd = P::rand(1, rng);
        _test_eea(&gcd, &[1, 2, 3], rng);
        _test_eea(&gcd, &[1, 1, 1], rng);
        _test_eea(&P::rand(0, rng), &[3, 2, 1], rng);
        _test_eea(&P::rand(1, rng), &[2, 2, 2], rng);
        _test_eea(&P::rand(1, rng), &[10, 1, 1], rng);
    }

    fn _test_eea<R: Rng>(gcd: &P, deg_qs: &[usize], rng: &mut R) {
        let (qs, rems) = get_polys(&gcd, &deg_qs, rng);
        let l = deg_qs.len() + 2;
        assert_eq!(rems.len(), l);
        let r0 = &rems[0];
        let r1 = &rems[1];

        assert_eq!(r0.degree(), deg_qs.iter().sum::<usize>() + gcd.degree());
        let (q1, r2) = ark_div(&r0, &r1);
        assert_eq!(q1.degree(), deg_qs[0]);
        assert_eq!(rems[2], r2);

        assert_eq!(rems[l - 2], *gcd);
        assert_eq!(rems[l - 1], P::zero());

        let mut sum_qi = deg_qs.iter().scan(0, |s, qi| {
            *s += qi;
            Some(*s)
        });
        let mut i = 0;
        let mut q_sum = sum_qi.next();
        for k in 1..r0.degree() {
            if k >= q_sum.unwrap() {
                i = i + 1;
                q_sum = sum_qi.next();
            }

            let (B_k, qs_) = eea(r0, r1, k);
            assert!(B_k.deg() <= k);
            assert_eq!(qs_.len(), i);
            if i > 1 {
                assert_eq!(qs_[i - 1], qs[i - 1]);
            }

            let (r, r_next) = B_k.apply(r0, r1);

            assert_eq!(r, rems[i]);
            assert_eq!(r_next, rems[i + 1]);
        }
    }

    // cargo test bench_generic_eea --release --features="print-trace" -- --show-output
    #[test]
    fn bench_generic_eea() {
        let rng = &mut test_rng();
        let d = 1024;
        let k = d;
        let gcd = P::rand(1, rng);
        let deg_qs = vec![1; d - gcd.degree()];
        let (_, rems) = get_polys(&gcd, deg_qs.as_slice(), rng);
        let r0 = &rems[0];
        let r1 = &rems[1];
        assert_eq!(r0.degree(), d);

        let _t1 = start_timer!(|| format!("Generic Half-GCD, deg(r0) = {d}, deg(qi) = 1, k = {k}"));
        let (B, _) = eea(r0, r1, k);
        let (gcd_, zero) = B.apply(r0, r1);
        assert_eq!(zero, P::zero());
        assert_eq!(gcd_, gcd);
        end_timer!(_t1);

        let gcd = P::rand(2, rng);
        let deg_qs = vec![2; (d - gcd.degree()) / 2];
        let (_, rems) = get_polys(&gcd, deg_qs.as_slice(), rng);
        let r0 = &rems[0];
        let r1 = &rems[1];
        assert_eq!(r0.degree(), d);

        let _t2 = start_timer!(|| format!("Generic Half-GCD, deg(r0) = {d}, deg(qi) = 2, k = {k}"));
        let (B, _) = eea(r0, r1, k);
        let (gcd_, zero) = B.apply(r0, r1);
        assert_eq!(zero, P::zero());
        assert_eq!(gcd_, gcd);
        end_timer!(_t2);

        let gcd = P::rand(1, rng);
        let deg_qs = vec![3; (d - gcd.degree()) / 3];
        let (_, rems) = get_polys(&gcd, deg_qs.as_slice(), rng);
        let r0 = &rems[0];
        let r1 = &rems[1];
        assert_eq!(r0.degree(), d);

        let _t3 = start_timer!(|| format!("Generic Half-GCD, deg(r0) = {d}, deg(qi) = 3, k = {k}"));
        let (B, _) = eea(r0, r1, k);
        let (gcd_, zero) = B.apply(r0, r1);
        assert_eq!(zero, P::zero());
        assert_eq!(gcd_, gcd);
        end_timer!(_t3);

        let mut deg_qs = Vec::with_capacity(d / 4);
        let mut sum = gcd.degree();
        while sum < d {
            let deg_qi = rng.gen_range(1..10);
            sum += deg_qi;
            deg_qs.push(deg_qi)
        }
        let (_, rems) = get_polys(&gcd, deg_qs.as_slice(), rng);
        let r0 = &rems[0];
        let r1 = &rems[1];

        let d = r0.degree();
        let k = d;
        let _t4 =
            start_timer!(|| format!("Generic Half-GCD, deg(r0) = {d}, 1 <= deg(qi) < 10, k = {k}"));
        let (B, _) = eea(r0, r1, k);
        let (gcd_, zero) = B.apply(r0, r1);
        assert_eq!(zero, P::zero());
        assert_eq!(gcd_, gcd);
        end_timer!(_t4);
    }

    // cargo test bench_eea --release --features="print-trace" -- --show-output
    #[test]
    fn bench_eea() {
        for log_d in 4..=10 {
            let d = 2usize.pow(log_d);
            println!("d = {d}");
            _bench_eea(d);
            println!("\n")
        }
    }
    fn _bench_eea(d: usize) {
        let rng = &mut test_rng();

        let f = crate::P::<Fr>::rand(d, rng);
        let g = crate::P::<Fr>::rand(d - 1, rng);
        let k = d;

        let _t_gcd = start_timer!(|| format!("EEA, deg = {d}"));
        let (gcd, _s, _t) = gcd::euclid(&f, &g);
        end_timer!(_t_gcd);

        let _t_alg_1 = start_timer!(|| format!("Half-GCD, deg = {d}, k = {k}"));
        let B = crate::half_gcd::simple_half_gcd(&f, &g, k);
        let (gcd_, zero) = B.apply(&f, &g);
        end_timer!(_t_alg_1);
        assert_eq!(zero, P::zero());
        assert_eq!(gcd_, gcd);

        let _t_alg_2 = start_timer!(|| format!("FFT-Half-GCD, deg = {d}, k = {k}"));
        let (B, _) = crate::half_gcd::simple_half_gcd2(&f.mul_xk(k), &g.mul_xk(k), k);
        let (gcd_, zero) = B.apply(&f, &g);
        assert_eq!(zero, P::zero());
        assert_eq!(gcd_, gcd);
        end_timer!(_t_alg_2);

        let _t_alg_4 = start_timer!(|| format!("Generic Half-GCD, deg = {d}, k = {k}"));
        let (B, _) = eea(&f, &g, k);
        let (gcd_, zero) = B.apply(&f, &g);
        assert_eq!(zero, P::zero());
        assert_eq!(gcd_, gcd);
        end_timer!(_t_alg_4);
    }
}

// d = 16
// Start:   EEA, deg = 16
// ··End:     EEA, deg = 16 ...........................................................310.833µs
// Start:   Half-GCD, deg = 16, k = 16
// ··End:     Half-GCD, deg = 16, k = 16 ..............................................693.459µs
// Start:   FFT-Half-GCD, deg = 16, k = 16
// ··End:     FFT-Half-GCD, deg = 16, k = 16 ..........................................910.792µs
// Start:   Generic Half-GCD, deg = 16, k = 16
// ··End:     Generic Half-GCD, deg = 16, k = 16 ......................................650.750µs
//
//
// d = 32
// Start:   EEA, deg = 32
// ··End:     EEA, deg = 32 ...........................................................731.416µs
// Start:   Half-GCD, deg = 32, k = 32
// ··End:     Half-GCD, deg = 32, k = 32 ..............................................1.489ms
// Start:   FFT-Half-GCD, deg = 32, k = 32
// ··End:     FFT-Half-GCD, deg = 32, k = 32 ..........................................1.836ms
// Start:   Generic Half-GCD, deg = 32, k = 32
// ··End:     Generic Half-GCD, deg = 32, k = 32 ......................................1.471ms
//
//
// d = 64
// Start:   EEA, deg = 64
// ··End:     EEA, deg = 64 ...........................................................2.239ms
// Start:   Half-GCD, deg = 64, k = 64
// ··End:     Half-GCD, deg = 64, k = 64 ..............................................3.362ms
// Start:   FFT-Half-GCD, deg = 64, k = 64
// ··End:     FFT-Half-GCD, deg = 64, k = 64 ..........................................4.019ms
// Start:   Generic Half-GCD, deg = 64, k = 64
// ··End:     Generic Half-GCD, deg = 64, k = 64 ......................................3.319ms
//
//
// d = 128
// Start:   EEA, deg = 128
// ··End:     EEA, deg = 128 ..........................................................7.909ms
// Start:   Half-GCD, deg = 128, k = 128
// ··End:     Half-GCD, deg = 128, k = 128 ............................................7.658ms
// Start:   FFT-Half-GCD, deg = 128, k = 128
// ··End:     FFT-Half-GCD, deg = 128, k = 128 ........................................8.242ms
// Start:   Generic Half-GCD, deg = 128, k = 128
// ··End:     Generic Half-GCD, deg = 128, k = 128 ....................................7.625ms
//
//
// d = 256
// Start:   EEA, deg = 256
// ··End:     EEA, deg = 256 ..........................................................30.984ms
// Start:   Half-GCD, deg = 256, k = 256
// ··End:     Half-GCD, deg = 256, k = 256 ............................................17.253ms
// Start:   FFT-Half-GCD, deg = 256, k = 256
// ··End:     FFT-Half-GCD, deg = 256, k = 256 ........................................17.066ms
// Start:   Generic Half-GCD, deg = 256, k = 256
// ··End:     Generic Half-GCD, deg = 256, k = 256 ....................................17.356ms
//
//
// d = 512
// Start:   EEA, deg = 512
// ··End:     EEA, deg = 512 ..........................................................157.896ms
// Start:   Half-GCD, deg = 512, k = 512
// ··End:     Half-GCD, deg = 512, k = 512 ............................................38.843ms
// Start:   FFT-Half-GCD, deg = 512, k = 512
// ··End:     FFT-Half-GCD, deg = 512, k = 512 ........................................35.496ms
// Start:   Generic Half-GCD, deg = 512, k = 512
// ··End:     Generic Half-GCD, deg = 512, k = 512 ....................................39.103ms
//
//
// d = 1024
// Start:   EEA, deg = 1024
// ··End:     EEA, deg = 1024 .........................................................698.373ms
// Start:   Half-GCD, deg = 1024, k = 1024
// ··End:     Half-GCD, deg = 1024, k = 1024 ..........................................86.754ms
// Start:   FFT-Half-GCD, deg = 1024, k = 1024
// ··End:     FFT-Half-GCD, deg = 1024, k = 1024 ......................................73.972ms
// Start:   Generic Half-GCD, deg = 1024, k = 1024
// ··End:     Generic Half-GCD, deg = 1024, k = 1024 ..................................87.266ms

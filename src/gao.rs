use crate::half_gcd::simple_half_gcd;
use crate::interpolation::{z_xs, Domain};
use crate::P;
use ark_ff::{FftField, Zero};
use ark_poly::Polynomial;
use ark_poly::{DenseUVPolynomial, EvaluationDomain};
use ark_std::{end_timer, start_timer};

// pub fn decode1<F: FftField>(b: &[F], k: usize) -> Result<P<F>, String> {
//     let n = b.len();
//
//     let _t_interpolate = start_timer!(|| format!("Interpolation, n = {n}"));
//     let fft_domain = GeneralEvaluationDomain::<F>::new(n).unwrap();
//     let bitmask = &vec![true; n];
//     let domain = InterpolationDomain::from_subset(fft_domain, bitmask).unwrap();
//     let g1 = domain.interpolate(&b).unwrap();
//     end_timer!(_t_interpolate);
//     assert_eq!(g1.degree(), n - 1);
//
//     let g0: P<F> = fft_domain.vanishing_polynomial().into();
//     assert_eq!(g0.degree(), n);
//
//     let d = (n + k) / 2;
//     let _t_gcd = start_timer!(|| format!("Half-GCD for normal degree sequences, deg = {n}"));
//     let B = simple_half_gcd(&g0, &g1, n - d);
//     let (_, g) = B.apply(&g0, &g1);
//     end_timer!(_t_gcd);
//     assert!(g.degree() < d);
//
//     let v = &B.0[3];
//     println!("deg(g) = {}, deg(v) = {}", g.degree(), v.degree());
//
//     let _t_div = start_timer!(|| "Division");
//     let (f1, r) = div(&g, v);
//     end_timer!(_t_div);
//     assert!(f1.degree() < k);
//     assert!(r.is_zero());
//     Ok(f1)
// }

// pub fn decode2<F: FftField>(b: &[(usize, F)], k: usize, n: usize) -> Result<P<F>, String> {
//     let domain = Domain::new(k, n);
//     let n = b.len();
//
//     let _t_g1 = start_timer!(|| format!("g1, supp(g1) = {}", b.len()));
//     let (g1, _) = domain.interpolate(b);
//     assert!(g1.degree() < n);
//     end_timer!(_t_g1);
//
//     let _t_g0 = start_timer!(|| format!("g0, deg(g0) = {n}"));
//     let ws: Vec<F> = domain.fft_domain.elements().collect();
//     let xs: Vec<F> = b.iter().map(|(i, _)| ws[*i]).collect();
//     let g0 = z_xs(&xs).poly;
//     assert_eq!(g0.degree(), n);
//     end_timer!(_t_g0);
//
//     let d = (n + k) / 2;
//     let _t_gcd = start_timer!(|| format!("Half-GCD for normal degree sequences, deg = {n}"));
//     let B = simple_half_gcd(&g0, &g1, n - d);
//     let (_, g) = B.apply(&g0, &g1);
//     end_timer!(_t_gcd);
//     assert_eq!(g.degree(), d - 1);
//
//     let v = &B.0[3];
//     println!("deg(g) = {}, deg(v) = {}", g.degree(), v.degree());
//
//     let _t_div = start_timer!(|| "Division");
//     let (f1, r) = div(&g, v);
//     end_timer!(_t_div);
//     assert!(f1.degree() < k);
//     assert!(r.is_zero());
//     Ok(f1)
// }

pub fn decode<F: FftField>(b: &[(usize, F)], k: usize, n: usize) -> Result<P<F>, String> {
    let domain = Domain::new(k, n);
    let n = b.len();
    let d = n - k + 1;

    let _t_g1 = start_timer!(|| format!("g1, supp(g1) = {}", b.len()));
    let (g1, _) = domain.interpolate(b);
    assert_eq!(g1.degree(), n - 1);
    end_timer!(_t_g1);

    let _t_g0 = start_timer!(|| format!("g0, deg(g0) = {n}"));
    let ws: Vec<F> = domain.fft_domain.elements().collect();
    let xs: Vec<F> = b.iter().map(|(i, _)| ws[*i]).collect();
    let g0 = z_xs(&xs).poly;
    assert_eq!(g0.degree(), n);
    end_timer!(_t_g0);

    let s0 = P::from_coefficients_slice(&g0.coeffs[k..]);
    assert_eq!(s0.degree(), d - 1);
    let s1 = P::from_coefficients_slice(&g1.coeffs[k..]);
    assert_eq!(s1.degree(), d - 2);

    let h = (d + 1) / 2;
    let _t_gcd = start_timer!(|| format!("Half-GCD for normal degree sequences, deg = {}", d - 1));
    let B = simple_half_gcd(&s0, &s1, d - h);
    let (_, g) = B.apply(&s0, &s1);
    end_timer!(_t_gcd);
    assert!(g.degree() < h - 1);
    let (u, v) = (&B.0[2], &B.0[3]);

    let _t_div =
        start_timer!(|| format!("g0 / v, deg(g0) = {}, deg(v) = {}", g0.degree(), v.degree()));
    let (h1, r) = crate::poly_div::div(&g0, v);
    // println!("{r:?}");
    assert!(r.is_zero());
    end_timer!(_t_div);
    let f1 = &g1 + &(&h1 * u);
    assert_eq!(f1.degree(), k - 1);
    Ok(f1)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ark_bls12_381::Fr;
    use ark_ff::Zero;
    use ark_poly::univariate::{DenseOrSparsePolynomial, DensePolynomial};
    use ark_poly::DenseUVPolynomial;
    use ark_std::rand::prelude::SliceRandom;
    use ark_std::{test_rng, UniformRand};

    // RUST_BACKTRACE=1 cargo test gaos_decoder --release --features="print-trace" -- --show-output
    #[test]
    fn gaos_decoder() {
        let rng = &mut test_rng();
        let n = 1000;
        let k = 668;
        let s = 100; // erasures
        let t = (n - s - k) / 2; // error threshold

        let domain = Domain::new(k, n);
        let f = DensePolynomial::<Fr>::rand(k - 1, rng); // message
        let c = f.evaluate_over_domain_by_ref(domain.fft_domain).evals[0..n].to_vec(); // codeword
        let mut b: Vec<(usize, Fr)> = c.into_iter().enumerate().collect(); // codeword with errors
        b.shuffle(rng);
        b.truncate(n - s);
        for i in 0..t {
            b[i].1 = Fr::rand(rng);
        }
        assert_eq!(2 * t + s, n - k);
        let _t_decode = start_timer!(|| format!("RS decoding, (k,n,s,t) = ({k},{n},{s},{t})"));
        let f1 = decode(&b, k, n).unwrap();
        end_timer!(_t_decode);
        assert_eq!(f1, f);
    }

    #[test]
    fn bench_z_s() {
        let rng = &mut test_rng();
        let n = 1000;
        let k = 668;
        let s = 100; // erasures
        let domain = Domain::new(k, n);

        let zd: DenseOrSparsePolynomial<Fr> = domain.fft_domain.vanishing_polynomial().into();
        let mut ws = domain.fft_domain.elements();
        let mut xs: Vec<_> = ws.by_ref().take(n).collect();
        let c2s: Vec<_> = ws.collect();
        let zc2 = z_xs(&c2s).poly;
        let (zn, _r) = zd.divide_with_q_and_r(&zc2.into()).unwrap();
        assert!(_r.is_zero());
        assert_eq!(zn.degree(), n);
        // erasures
        xs.shuffle(rng);
        let (ss, c1s) = xs.split_at(n - s);
        let zc1 = z_xs(c1s).poly;

        let _t_zs_mul = start_timer!(|| format!("z_S by mul, deg(z_S) = {}", n - s));
        let zs = z_xs(ss).poly;
        end_timer!(_t_zs_mul);
        println!();
        assert_eq!(zs.degree(), n - s);

        let _t_zs_div = start_timer!(|| format!("z_S by div, deg(z_S) = {n} - {s}"));
        let zs_ = &zn / &zc1;
        end_timer!(_t_zs_div);
        assert_eq!(zs_, zs);
    }
}

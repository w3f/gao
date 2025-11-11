use crate::half_gcd::simple_half_gcd;
use crate::lagrange::InterpolationDomain;
use crate::{div, gcd, P};
use ark_ff::{FftField, Zero};
use ark_poly::EvaluationDomain;
use ark_poly::{GeneralEvaluationDomain, Polynomial};
use ark_std::{end_timer, start_timer};
use crate::interpolation::{z_xs, Domain};

pub fn decode<F: FftField>(b: &[(usize, F)], k: usize, n: usize) -> Result<P<F>, String> {
    let domain = Domain::new(k, n);
    let n = b.len();

    let _t_g1 = start_timer!(|| format!("g1, supp(g1) = {}", b.len()));
    let (g1, _) = domain.interpolate(b);
    assert!(g1.degree() < n);
    end_timer!(_t_g1);

    let _t_g0 = start_timer!(|| format!("g0, deg(g0) = {n}"));
    let ws: Vec<F> = domain.fft_domain.elements().collect();
    let xs: Vec<F> = b.iter().map(|(i, _)| ws[*i]).collect();
    let g0 = z_xs(&xs).poly;
    assert_eq!(g0.degree(), n);
    end_timer!(_t_g0);

    let d = (n + k) / 2;
    let _t_gcd = start_timer!(|| format!("Half-GCD for normal degree sequences, deg = {n}"));
    let B = simple_half_gcd(&g0, &g1, n - d);
    let (_, g) = B.apply(&g0, &g1);
    end_timer!(_t_gcd);
    assert_eq!(g.degree(), d - 1);

    let v = &B.0[3];
    println!("deg(g) = {}, deg(v) = {}", g.degree(), v.degree());

    let _t_div = start_timer!(|| "Division");
    let (f1, r) = div(&g, v);
    end_timer!(_t_div);
    assert!(f1.degree() < k);
    assert!(r.is_zero());
    Ok(f1)
}

pub fn decode2<F: FftField>(b: &[F], k: usize) -> Result<P<F>, String> {
    let n = b.len();

    let _t_interpolate = start_timer!(|| format!("Interpolation, n = {n}"));
    let fft_domain = GeneralEvaluationDomain::<F>::new(n).unwrap();
    let bitmask = &vec![true; n];
    let domain = InterpolationDomain::from_subset(fft_domain, bitmask).unwrap();
    let g1 = domain.interpolate(&b).unwrap();
    end_timer!(_t_interpolate);
    assert_eq!(g1.degree(), n - 1);

    let g0: P<F> = fft_domain.vanishing_polynomial().into();
    assert_eq!(g0.degree(), n);

    let d = (n + k) / 2;
    let _t_gcd = start_timer!(|| format!("Half-GCD for normal degree sequences, deg = {n}"));
    let B = simple_half_gcd(&g0, &g1, n - d);
    let (_, g) = B.apply(&g0, &g1);
    end_timer!(_t_gcd);
    assert!(g.degree() < d);

    let v = &B.0[3];
    println!("deg(g) = {}, deg(v) = {}", g.degree(), v.degree());

    let _t_div = start_timer!(|| "Division");
    let (f1, r) = div(&g, v);
    end_timer!(_t_div);
    assert!(f1.degree() < k);
    assert!(r.is_zero());
    Ok(f1)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ark_bls12_381::Fr;
    use ark_poly::univariate::DensePolynomial;
    use ark_poly::DenseUVPolynomial;
    use ark_std::{test_rng, UniformRand}; use ark_std::rand::prelude::SliceRandom;

    // RUST_BACKTRACE=1 cargo test gaos_decoder --release --features="print-trace" -- --show-output
    #[test]
    fn gaos_decoder() {
        let rng = &mut test_rng();
        let n = 1000;
        let k = 668;

        let domain = Domain::new(k, n);
        let f = DensePolynomial::<Fr>::rand(k - 1, rng); // message
        let c = f.evaluate_over_domain_by_ref(domain.fft_domain).evals; // codeword

        let s = 100; // erasures
        let t = (n - s - k) / 2; // error threshold
        let b: Vec<Fr> = c.into_iter().take(n).collect(); // codeword with errors
        let mut b: Vec<(usize, Fr)> = b.into_iter().enumerate().collect();
        b.shuffle(rng);
        b.truncate(n - s);
        for i in 0..t {
            b[i] = (b[i].0, Fr::rand(rng));
        }
        let _t_decode = start_timer!(|| format!("RS decoding, (k,n,s,t) = ({k},{n},{s},{t})"));
        let f1 = decode(&b, k, n).unwrap();
        end_timer!(_t_decode);
        assert_eq!(f1, f);
    }
}

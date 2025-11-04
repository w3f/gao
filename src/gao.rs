use crate::half_gcd::simple_half_gcd;
use crate::lagrange::InterpolationDomain;
use crate::{div, P};
use ark_ff::{FftField, Zero};
use ark_poly::EvaluationDomain;
use ark_poly::{GeneralEvaluationDomain, Polynomial};
use ark_std::{end_timer, start_timer};

pub fn decode<F: FftField>(b: &[F], k: usize) -> Result<P<F>, String> {
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
    use ark_poly::GeneralEvaluationDomain;
    use ark_std::{test_rng, UniformRand};

    #[test]
    fn gaos_decoder() {
        let rng = &mut test_rng();
        let n = 1024;
        let k = 1000;

        let f = DensePolynomial::<Fr>::rand(k - 1, rng); // message
        let fft_domain = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
        let c = f.evaluate_over_domain_by_ref(fft_domain).evals; // codeword

        let t = (n - k) / 2; // error threshold
        let mut b = c.clone(); // codeword with errors
        for i in 0..t {
            b[i] = Fr::rand(rng);
        }

        let f1 = decode(&b, k).unwrap();
        assert_eq!(f1, f);
    }
}
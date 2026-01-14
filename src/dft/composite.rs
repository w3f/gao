use crate::dft::{roots, DftDomain};
use ark_ff::Field;
use ark_std::{end_timer, start_timer};

#[derive(Debug, PartialEq)]
pub struct CooleyTukeyDomain<F: Field, D1: DftDomain<F>, D2: DftDomain<F>> {
    n: usize,
    w: F,
    d1: D1,
    d2: D2,
    twiddles: Vec<Vec<F>>,
}

impl<F: Field, D1: DftDomain<F>, D2: DftDomain<F>> CooleyTukeyDomain<F, D1, D2> {
    pub fn new(w: F, d1: D1, d2: D2) -> Self {
        let (n1, n2) = (d1.n(), d2.n());
        let n = n1 * n2;
        debug_assert!(w.pow([n as u64]).is_one());
        let twiddles = Self::twiddles(w, n1, n2);

        Self {
            n,
            w,
            d1,
            d2,
            twiddles,
        }
    }

    fn twiddles(w: F, n1: usize, n2: usize) -> Vec<Vec<F>> {
        let roots = roots((n1 - 1) * (n2 - 1) + 1, w);
        let mut twiddles = vec![vec![F::zero(); n1]; n2];
        for i2 in 0..n2 {
            let mut inner = vec![F::zero(); n1];
            for k1 in 0..n1 {
                inner[k1] = roots[k1 * i2];
            }
            twiddles[i2] = inner;
        }
        twiddles
    }
}

impl<F: Field, D1: DftDomain<F>, D2: DftDomain<F>> DftDomain<F> for CooleyTukeyDomain<F, D1, D2> {
    fn dft(&self, coeffs: &[F]) -> Vec<F> {
        let n = self.n;
        let n1 = self.d1.n();
        let n2 = self.d2.n();
        debug_assert_eq!(coeffs.len(), n);

        let _t_dft = start_timer!(|| format!("{n1}n-DFT, n = {n2}"));
        let _t_inner_dfts = start_timer!(|| format!("{n1} x n-DFT, n = {n2}"));
        let mut inner_dfts = vec![vec![F::zero(); n2]; n1];
        for k1 in 0..n1 {
            let inner_coeffs: Vec<_> = coeffs.iter().cloned().skip(k1).step_by(n1).collect();
            debug_assert_eq!(inner_coeffs.len(), n2);
            inner_dfts[k1] = self.d2.dft(&inner_coeffs);
        }
        end_timer!(_t_inner_dfts);

        let mut res = vec![F::zero(); n];
        for i2 in 0..n2 {
            let mut outer_coeffs = vec![F::zero(); n1];
            for k1 in 0..n1 {
                outer_coeffs[k1] = self.twiddles[i2][k1] * inner_dfts[k1][i2];
            }
            let outer_dft = self.d1.dft(&outer_coeffs);
            for i1 in 0..n1 {
                res[i1 * n2 + i2] = outer_dft[i1];
            }
        }
        end_timer!(_t_dft);
        res
    }

    fn n(&self) -> usize {
        self.n
    }

    fn w(&self) -> F {
        self.w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::dft::radix_3_2k::Radix3_2k;
    use ark_bls12_381::Fr;
    use ark_ff::UniformRand;
    use ark_poly::{EvaluationDomain, MixedRadixEvaluationDomain};
    use ark_std::{end_timer, start_timer, test_rng};

    #[test]
    fn test_3n_fft() {
        let rng = &mut test_rng();

        let log_n = 10;
        let n = 1 << log_n;
        let m = 3 * n;
        let coeffs: Vec<_> = (0..m).map(|_| Fr::rand(rng)).collect();

        let _t_arkworks = start_timer!(|| format!("Arkworks 3n-FFT, n = {n}"));
        let _t_precomp = start_timer!(|| format!("pre-computation"));
        let fft_domain = MixedRadixEvaluationDomain::<Fr>::new(m).unwrap();
        end_timer!(_t_precomp);
        let _t_fft = start_timer!(|| format!("forward FFT"));
        let evals = fft_domain.fft(&coeffs);
        end_timer!(_t_fft);
        let _t_ifft = start_timer!(|| format!("inverse FFT"));
        let coeffs_ = fft_domain.ifft(&evals);
        end_timer!(_t_ifft);
        end_timer!(_t_arkworks);
        assert_eq!(coeffs_, coeffs);

        let _t_custom = start_timer!(|| format!("Custom 3n-FFT, n = {n}"));
        let _t_precomp = start_timer!(|| format!("pre-computation"));
        let fft_domain = Radix3_2k::new(log_n).unwrap();
        end_timer!(_t_precomp);
        let _t_fft = start_timer!(|| format!("forward FFT"));
        let evals = fft_domain.dft(&coeffs);
        end_timer!(_t_fft);
        let _t_ifft = start_timer!(|| format!("inverse FFT"));
        let coeffs_ = fft_domain.idft(&evals);
        end_timer!(_t_ifft);
        end_timer!(_t_custom);
        println!("\n");
        assert_eq!(coeffs_, coeffs);
    }
}

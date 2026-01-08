use crate::dft::DftDomain;
use ark_ff::Field;

pub struct CooleyTukeyDomain<F: Field, D1: DftDomain<F>, D2: DftDomain<F>> {
    n: usize,
    w: F,
    d1: D1,
    d2: D2,
    twiddles: Vec<Vec<F>>,
}

impl<F: Field, D1: DftDomain<F>, D2: DftDomain<F>> CooleyTukeyDomain<F, D1, D2> {
    pub fn new(w: F, d1: D1, d2: D2) -> Self {
        let n = d1.n() * d2.n();
        debug_assert!(w.pow([n as u64]).is_one());
        let mut twiddles = vec![vec![F::zero(); d1.n()]; d2.n()];
        for i2 in 0..d2.n() {
            let mut inner = vec![F::zero(); d1.n()];
            for k1 in 0..d1.n() {
                inner[k1] = w.pow([(k1 * i2) as u64]);
            }
            twiddles[i2] = inner;
        }

        Self {
            n,
            w,
            d1,
            d2,
            twiddles,
        }
    }
}

impl<F: Field, D1: DftDomain<F>, D2: DftDomain<F>> DftDomain<F> for CooleyTukeyDomain<F, D1, D2> {
    fn dft(&self, coeffs: &[F]) -> Vec<F> {
        let n = self.n;
        let n1 = self.d1.n();
        let n2 = self.d2.n();
        debug_assert_eq!(coeffs.len(), n);

        let mut inner_dfts = vec![vec![F::zero(); n2]; n1];
        for k1 in 0..n1 {
            let inner_coeffs: Vec<_> = coeffs.iter().cloned().skip(k1).step_by(n1).collect();
            debug_assert_eq!(inner_coeffs.len(), n2);
            inner_dfts[k1] = self.d2.dft(&inner_coeffs);
        }

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
        res
    }

    fn idft(&self, _evals: &[F]) -> Vec<F> {
        todo!()
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

    use crate::dft::radix3::Radix3;
    use ark_bls12_381::Fr;
    use ark_ff::{FftField, UniformRand};
    use ark_poly::{EvaluationDomain, MixedRadixEvaluationDomain, Radix2EvaluationDomain};
    use ark_std::{end_timer, start_timer, test_rng};

    #[test]
    fn test_3n_fft() {
        let rng = &mut test_rng();

        let log_n = 10;
        let n = 1 << log_n;
        let m = 3 * n;
        let coeffs: Vec<_> = (0..m).map(|_| Fr::rand(rng)).collect();

        let _t_fft = start_timer!(|| format!("3n-FFT, n = {n}"));
        let _t_precomp = start_timer!(|| format!("pre-computation"));
        let fft_domain = {
            let _t_precomp = start_timer!(|| format!("pre-computation"));
            let w = Fr::get_root_of_unity(m as u64).unwrap();
            let d_3 = Radix3::new().unwrap();
            debug_assert_eq!(d_3.w(), w.pow([n as u64]));
            let d_n = Radix2EvaluationDomain::new(n).unwrap();
            debug_assert_eq!(d_n.w(), w.pow([3]));
            let fft_domain = CooleyTukeyDomain::new(w, d_3, d_n);
            end_timer!(_t_precomp);
            fft_domain
        };
        let _t_comp = start_timer!(|| format!("computation"));
        let fft = fft_domain.dft(&coeffs);
        end_timer!(_t_comp);
        end_timer!(_t_fft);

        let _t_fft = start_timer!(|| format!("Arkworks 3n-FFT, n = {n}"));
        let _t_precomp = start_timer!(|| format!("pre-computation"));
        let fft_domain = MixedRadixEvaluationDomain::<Fr>::new(m).unwrap();
        end_timer!(_t_precomp);
        let _t_comp = start_timer!(|| format!("computation"));
        let fft_ = fft_domain.fft(&coeffs);
        end_timer!(_t_comp);
        end_timer!(_t_fft);

        assert_eq!(fft, fft_);
    }
}

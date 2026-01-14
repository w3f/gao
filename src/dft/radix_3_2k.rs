use crate::dft::radix_2k::{bitreverse, Radix2k};
use crate::dft::radix_3::Radix3;
use crate::dft::{roots, transpose, DftDomain};
use ark_ff::{FftField, Field};
use ark_std::{end_timer, start_timer};

pub struct Radix3_2k<F: Field> {
    n: usize,
    w: F,
    radix3: Radix3<F>,
    radix2k: Radix2k<F>,
    twiddles: Vec<Vec<F>>,
}

impl<F: FftField> Radix3_2k<F> {
    pub fn new(k: usize) -> Option<Self> {
        let n = 1 << k;
        let w = F::get_root_of_unity(3 * n as u64)?;
        let roots = roots(3 * n, w);
        let radix3 = Radix3::new()?;
        let radix2k = Radix2k {
            ws: roots.iter().copied().step_by(3).collect(),
        };
        let twiddles = Self::twiddles(n, roots);
        Some(Self {
            n,
            w,
            radix3,
            radix2k,
            twiddles,
        })
    }

    fn twiddles(n: usize, roots: Vec<F>) -> Vec<Vec<F>> {
        let roots_2 = roots
            .iter()
            .copied()
            .step_by(2)
            .chain(roots.iter().skip(n / 2).map(|&f| f * f))
            .collect();
        let twiddles = transpose(&[vec![F::one(); n], roots, roots_2]);
        twiddles
    }
}

impl<F: FftField> DftDomain<F> for Radix3_2k<F> {
    fn dft(&self, coeffs: &[F]) -> Vec<F> {
        let n = self.n;
        debug_assert_eq!(coeffs.len(), n * 3);

        let _t_3n_dft = start_timer!(|| format!("3n-DFT, n = {n}"));
        let _t_n_dfts = start_timer!(|| format!("3 x n-DFT"));
        let mut inner_dfts = vec![vec![F::zero(); n]; 3];
        for i in 0..3 {
            let inner_coeffs: Vec<_> = coeffs.iter().copied().skip(i).step_by(3).collect();
            debug_assert_eq!(inner_coeffs.len(), n);
            let mut dft_i = self.radix2k.dft(&inner_coeffs);
            bitreverse(&mut dft_i);
            inner_dfts[i] = dft_i;
        }
        end_timer!(_t_n_dfts);

        let mut res = vec![F::zero(); 3 * n];

        let _t_3_dfts = start_timer!(|| format!("n x 3-DFT"));
        for j in 0..n {
            let mut outer_coeffs = vec![F::zero(); 3];
            for i in 0..3 {
                outer_coeffs[i] = self.twiddles[j][i] * inner_dfts[i][j];
            }
            let outer_dft = self.radix3.dft(&outer_coeffs);
            for i in 0..3 {
                res[i * n + j] = outer_dft[i];
            }
        }
        end_timer!(_t_3_dfts);
        end_timer!(_t_3n_dft);
        res
    }

    fn n(&self) -> usize {
        self.n * 3
    }

    fn w(&self) -> F {
        self.w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::tests::dft_idft_roundtrip;
    use ark_bls12_381::Fr;
    use ark_poly::{EvaluationDomain, MixedRadixEvaluationDomain};
    use ark_std::{end_timer, start_timer, test_rng, UniformRand};

    #[test]
    fn test_3_2k_fft() {
        let log_n = 8;
        let d = Radix3_2k::<Fr>::new(log_n).unwrap();
        dft_idft_roundtrip(&d);
    }

    #[test]
    fn test_bench_3_2k_fft() {
        let rng = &mut test_rng();

        let log_n = 12;
        let n = 1 << log_n;
        let m = 3 * n;
        let coeffs: Vec<_> = (0..m).map(|_| Fr::rand(rng)).collect();

        let _t_arkworks = start_timer!(|| format!("Arkworks 3*2^k-FFT, k = {log_n}"));
        let _t_precomp = start_timer!(|| format!("pre-computation"));
        let d = MixedRadixEvaluationDomain::<Fr>::new(m).unwrap();
        end_timer!(_t_precomp);
        let _t_fft = start_timer!(|| format!("forward FFT"));
        let evals = d.fft(&coeffs);
        end_timer!(_t_fft);
        let _t_ifft = start_timer!(|| format!("inverse FFT"));
        let coeffs_ = d.ifft(&evals);
        end_timer!(_t_ifft);
        end_timer!(_t_arkworks);
        println!("\n");
        assert_eq!(coeffs_, coeffs);

        let _t_custom = start_timer!(|| format!("Custom 3*2^k-FFT, k = {log_n}"));
        let _t_precomp = start_timer!(|| format!("pre-computation"));
        let d = Radix3_2k::<Fr>::new(log_n).unwrap();
        end_timer!(_t_precomp);
        let _t_fft = start_timer!(|| format!("forward FFT"));
        let evals_ = d.dft(&coeffs);
        end_timer!(_t_fft);
        assert_eq!(evals_, evals);
        let _t_ifft = start_timer!(|| format!("inverse FFT"));
        let coeffs_ = d.idft(&evals_);
        end_timer!(_t_ifft);
        end_timer!(_t_custom);
        println!("\n");
        assert_eq!(coeffs_, coeffs);
    }
}

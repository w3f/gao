pub mod composite;
pub mod naive;
pub mod radix3;

use ark_ff::{FftField, Field};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};

pub trait DftDomain<F: Field> {
    fn dft(&self, coeffs: &[F]) -> Vec<F>;
    fn n(&self) -> usize;
    fn w(&self) -> F;

    fn n_inv(&self) -> F {
        F::from(self.n() as u64).inverse().unwrap()
    }

    // FFT_ω(p)[k] = p(ω^k), k = 0,...,n−1
    // FFT_{ω^{-1}}(p)[k] = p(ω^{-k}) = p(ω^{n−k mod n}), k = 0,...,n−1
    //   = FFT_ω(p)[0],       k = 0
    //   = FFT_ω(p)[n − k],   k = 0,...,n−1
    //
    // IFFT(x) = (1/n)·FFT_{ω^{-1}}(x)
    fn idft(&self, evals: &[F]) -> Vec<F> {
        let n_inv = self.n_inv();
        let mut dft = self.dft(evals);
        dft[1..].reverse();
        dft.into_iter().map(|c| c * n_inv).collect()
    }
}

pub fn roots<F: Field>(n: usize, w: F) -> Vec<F> {
    let mut roots = Vec::with_capacity(n);
    roots.push(F::one());
    roots.push(w);
    let mut wi = w;
    for _ in 2..n {
        wi *= w;
        roots.push(wi);
    }
    roots
}

pub fn transpose<F: Copy>(rows: &[Vec<F>]) -> Vec<Vec<F>> {
    let n_cols = rows[0].len();
    (0..n_cols)
        .map(|j| rows.iter().map(|m_i| m_i[j]).collect())
        .collect()
}

impl<F: FftField> DftDomain<F> for Radix2EvaluationDomain<F> {
    fn dft(&self, coeffs: &[F]) -> Vec<F> {
        self.fft(coeffs)
    }

    fn idft(&self, evals: &[F]) -> Vec<F> {
        self.ifft(evals)
    }

    fn n(&self) -> usize {
        self.size()
    }

    fn w(&self) -> F {
        self.group_gen
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use ark_std::test_rng;

    pub fn dft_idft_roundtrip<F: Field, D: DftDomain<F>>(d: D) {
        let rng = &mut test_rng();

        let coeffs: Vec<_> = (0..d.n()).map(|_| F::rand(rng)).collect();

        let evals = d.dft(&coeffs);
        let coeffs_ = d.idft(&evals);

        assert_eq!(coeffs, coeffs_);
    }
}

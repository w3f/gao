pub mod composite;
pub mod naive;
pub mod radix3;

use ark_ff::{FftField, Field};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};

pub trait DftDomain<F: Field> {
    fn dft(&self, coeffs: &[F]) -> Vec<F>;
    fn idft(&self, evals: &[F]) -> Vec<F>;
    fn n(&self) -> usize;
    fn w(&self) -> F;
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
mod tests {}

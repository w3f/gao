use crate::dft::DftDomain;
use ark_ff::{FftField, Field};

/// Radix-3 FFT domain.
///
/// Assumes:
/// - char(F) ≠ 2,3
/// - F contains a primitive 3rd root of unity
pub struct Radix3<F: Field> {
    omega: F,
    inv_2: F,  // 1/2
    inv_3: F,  // 1/3
    half_c: F, // (ω − ω²)/2
}

impl<F: FftField> Radix3<F> {
    pub fn new() -> Option<Self> {
        let omega = F::get_root_of_unity(3)?;
        let inv_2 = F::from(2u64).inverse()?;
        let inv_3 = F::from(3u64).inverse()?;
        let half_c = (omega - omega.square()) * inv_2;
        Some(Self {
            omega,
            inv_2,
            inv_3,
            half_c,
        })
    }

    /// 3-point DFT.
    ///
    /// Let ω be a primitive 3rd root of unity (ω³ = 1, ω ≠ 1).
    /// For input (x0, x1, x2),
    /// let p(Z) = x0 + x1.Z + x2.Z².
    /// Then DFT(x0, x1, x2) = (X0, X1, X2), where
    ///   X0 = p(1) = x0 + x1 + x2
    ///   X1 = p(ω) = x0 + ω·x1 + ω²·x2
    ///   X2 = p(ω²) = x0 + ω²·x1 + ω·x2
    ///
    /// Letting:
    ///   a = x1 + x2
    ///   b = x1 − x2
    ///   c = ω − ω²
    ///
    /// one may compute:
    ///   X0 = x0 + a
    ///   X1 = (x0 − 1/2.a) + (c/2)·b = s + t
    ///   X2 = (x0 − 1/2.a) − (c/2)·b = s - t
    ///
    /// Cost: exactly 2 field multiplications.
    fn fft3(&self, x: [F; 3]) -> [F; 3] {
        let a = x[1] + x[2];
        let b = x[1] - x[2];
        let s = x[0] - a * self.inv_2;
        let t = self.half_c * b;
        [x[0] + a, s + t, s - t]
    }
}

impl<F: FftField> DftDomain<F> for Radix3<F> {
    fn dft(&self, coeffs: &[F]) -> Vec<F> {
        assert_eq!(coeffs.len(), 3);
        self.fft3([coeffs[0], coeffs[1], coeffs[2]]).to_vec()
    }

    // IFFT(x) = (1/n) · FFT_{ω⁻¹}(x)
    // For n = 3, FFT_{ω⁻¹}(x) = [y0, y2, y1], where y = FFT_ω(x).
    fn idft(&self, evals: &[F]) -> Vec<F> {
        let y = self.dft(evals);
        vec![y[0] * self.inv_3, y[2] * self.inv_3, y[1] * self.inv_3]
    }

    fn n(&self) -> usize {
        3
    }

    fn w(&self) -> F {
        self.omega
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::UniformRand;
    use ark_poly::EvaluationDomain;
    use ark_poly::MixedRadixEvaluationDomain;
    use ark_std::test_rng;
    #[test]
    fn radix3_dft_idft_roundtrip() {
        let rng = &mut test_rng();

        let n = 3;
        let d3 = Radix3::<Fr>::new().unwrap();
        let x: Vec<_> = (0..n).map(|_| Fr::rand(rng)).collect();

        let x_dft = d3.dft(&x);
        let x_ = d3.idft(&x_dft);

        assert_eq!(x, x_);
    }

    #[test]
    fn matches_arkworks() {
        let rng = &mut test_rng();

        let n = 3;
        let d3 = Radix3::<Fr>::new().unwrap();
        let d = MixedRadixEvaluationDomain::<Fr>::new(n).unwrap();

        let x: Vec<_> = (0..n).map(|_| Fr::rand(rng)).collect();
        let dft = d3.dft(&x);
        let dft_ = d.fft(&x);

        assert_eq!(dft, dft_);
    }
}

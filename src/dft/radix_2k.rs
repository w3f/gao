use crate::dft::{roots, DftDomain};
use ark_ff::{FftField, Field};

pub struct Radix2k<F: Field> {
    pub ws: Vec<F>,
}

impl<F: FftField> Radix2k<F> {
    pub fn new(n: usize) -> Option<Self> {
        let w = F::get_root_of_unity(n as u64)?;
        Some(Self::from_root(w, n))
    }
}

impl<F: Field> Radix2k<F> {
    pub fn from_root(w: F, n: usize) -> Self {
        debug_assert!(w.pow([n as u64]).is_one());
        let ws = roots(n, w);
        Self { ws }
    }

    pub fn dft_in_place_dif(&self, f: &mut [F]) {
        let n = f.len();
        debug_assert_eq!(self.ws.len(), n);
        let mut m = n;
        let mut r = 1;
        while m > 1 {
            let half_m = m >> 1;
            for k in (0..n).step_by(m) {
                for j in 0..half_m {
                    let w = self.ws[j * r];
                    let i1 = k + j;
                    let i2 = i1 + half_m;
                    let u = f[i1];
                    let v = f[i2];
                    f[i1] = u + v;
                    f[i2] = (u - v) * w;
                }
            }
            m = half_m;
            r <<= 1;
        }
    }

    pub fn dft_in_place_dit(&self, f: &mut [F], ws: &[F]) {
        let n = f.len();
        debug_assert_eq!(ws.len(), n);
        let mut m = 2;
        let mut r = n >> 1;
        while m <= n {
            let half_m = m >> 1;
            for k in (0..n).step_by(m) {
                for j in 0..half_m {
                    let w = ws[j * r];
                    let i1 = k + j;
                    let i2 = i1 + half_m;
                    let u = f[i1];
                    let v = f[i2] * w;
                    f[i1] = u + v;
                    f[i2] = u - v;
                }
            }
            m <<= 1;
            r >>= 1;
        }
    }

    pub fn dft_rec(&self, f: &[F]) -> Vec<F> {
        self._dft_rec(f, 1)
    }

    fn _dft_rec(&self, f: &[F], r: usize) -> Vec<F> {
        let n = f.len();
        debug_assert_eq!(self.ws.len(), n);
        if n == 1 {
            return f.to_vec();
        }
        let half_n = n >> 1;
        let (f0, f1): (Vec<_>, Vec<_>) = f[..half_n]
            .iter()
            .zip(f[half_n..].iter())
            .zip(self.ws.iter().step_by(r))
            .map(|((&l, h), w)| (l + h, (l - h) * w))
            .unzip();
        let r2 = r << 1;
        let g0 = self._dft_rec(&f0, r2);
        let g1 = self._dft_rec(&f1, r2);
        let mut res = Vec::with_capacity(n);
        for (a, b) in g0.into_iter().zip(g1) {
            res.push(a);
            res.push(b);
        }
        res
    }
}

impl<F: FftField> DftDomain<F> for Radix2k<F> {
    fn dft(&self, coeffs: &[F]) -> Vec<F> {
        let mut coeffs = coeffs.to_vec();
        self.dft_in_place_dif(&mut coeffs);
        coeffs
    }

    fn n(&self) -> usize {
        self.ws.len()
    }

    fn w(&self) -> F {
        self.ws[1]
    }

    fn idft(&self, evals: &[F]) -> Vec<F> {
        let mut ws = self.ws.clone();
        ws[1..].reverse();
        let mut coeffs = evals.to_vec();
        self.dft_in_place_dit(&mut coeffs, &ws);
        self.normalize(&mut coeffs);
        coeffs
    }
}

pub fn bitreverse<F: Field>(f: &mut [F]) {
    let n = f.len();
    let log_n = n.trailing_zeros();
    let s = usize::BITS - log_n;
    for i in 0..n {
        let j = i.reverse_bits() >> s;
        if j > i {
            f.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::tests::dft_idft_roundtrip;
    use ark_bls12_381::Fr;
    use ark_poly::EvaluationDomain;
    use ark_poly::Radix2EvaluationDomain;
    use ark_std::{end_timer, start_timer, test_rng, UniformRand};

    #[test]
    fn test_2k_fft() {
        let rng = &mut test_rng();

        let log_n = 8;
        let n = 1 << log_n;

        let d = Radix2k::<Fr>::new(n).unwrap();
        dft_idft_roundtrip(&d);

        let coeffs: Vec<_> = (0..n).map(|_| Fr::rand(rng)).collect();
        let evals = Radix2EvaluationDomain::<Fr>::new(n).unwrap().fft(&coeffs);

        let mut evals_ = coeffs.clone();
        d.dft_in_place_dif(&mut evals_);
        bitreverse(&mut evals_);
        assert_eq!(evals_, evals);

        let mut evals_ = coeffs.clone();
        bitreverse(&mut evals_);
        d.dft_in_place_dit(&mut evals_, &d.ws);
        assert_eq!(evals_, evals);
    }

    #[test]
    fn test_bench_2k_fft() {
        let rng = &mut test_rng();

        let log_n = 10;
        let n = 1 << log_n;
        let coeffs: Vec<_> = (0..n).map(|_| Fr::rand(rng)).collect();

        let _t_arkworks = start_timer!(|| format!("Arkworks 2^k-FFT, k = {log_n}"));
        let _t_precomp = start_timer!(|| format!("pre-computation"));
        let d = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        end_timer!(_t_precomp);
        let _t_fft = start_timer!(|| format!("forward FFT"));
        let evals_ = d.dft(&coeffs);
        end_timer!(_t_fft);
        let _t_ifft = start_timer!(|| format!("inverse FFT"));
        let coeffs_ = d.ifft(&evals_);
        end_timer!(_t_ifft);
        end_timer!(_t_arkworks);
        println!("\n");
        assert_eq!(coeffs_, coeffs);

        let _t_custom = start_timer!(|| format!("Custom 2^k-FFT, k = {log_n}"));
        let _t_precomp = start_timer!(|| format!("pre-computation"));
        let d = Radix2k::<Fr>::new(n).unwrap();
        end_timer!(_t_precomp);
        let _t_fft = start_timer!(|| format!("forward FFT"));
        let evals_ = d.dft(&coeffs);
        end_timer!(_t_fft);
        let _t_ifft = start_timer!(|| format!("inverse FFT"));
        let coeffs_ = d.idft(&evals_);
        end_timer!(_t_ifft);
        end_timer!(_t_custom);
        println!("\n");
        assert_eq!(coeffs_, coeffs);
    }
}

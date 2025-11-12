use crate::poly_mul::Monic;
use crate::P;
use ark_ff::FftField;
use ark_poly::{
    DenseUVPolynomial, EvaluationDomain, Evaluations, Polynomial, Radix2EvaluationDomain,
};
use ark_std::{end_timer, start_timer};

// TODO: multipoint_evaluation for Al's optimization

/// Constant polynomial
/// `one(X) = 1`.
fn one<F: FftField>() -> Monic<F> {
    let z_0 = P::from_coefficients_vec(vec![F::one()]);
    Monic::new(z_0)
}

/// Vanishing polynomial of a point `x`.
/// `z(X) = X - x`
pub fn z_x<F: FftField>(x: F) -> Monic<F> {
    let z_1 = P::from_coefficients_slice(&[-x, F::one()]); // `= -x.0 + 1.X = X - x`
    Monic::new(z_1)
}

/// Vanishing polynomial of a set `xs`.
/// `z(X) = (X - x1)...(X - xn)`
#[cfg(feature = "parallel")]
pub fn z_xs<F: FftField>(xs: &[F]) -> Monic<F> {
    let n = xs.len();
    match n {
        0 => one(),
        1 => z_x(xs[0]),
        _ => {
            let h = n / 2;
            let (mut l, mut r) = if n < 256 {
                (z_xs(&xs[0..h]), z_xs(&xs[h..n]))
            } else {
                rayon::join(
                    || z_xs(&xs[0..h]),
                    || z_xs(&xs[h..n]))
            };
            Monic::mul(&mut l, &mut r)
        }
    }
}

/// Vanishing polynomial of a set `xs`.
/// `z(X) = (X - x1)...(X - xn)`
#[cfg(not(feature = "parallel"))]
pub fn z_xs<F: FftField>(xs: &[F]) -> Monic<F> {
    let n = xs.len();
    match n {
        0 => one(),
        1 => z_x(xs[0]),
        _ => {
            let h = n / 2;
            let left = z_xs(&xs[0..h]);
            let right = z_xs(&xs[h..n]);
            Monic::mul(&left, &right)
        }
    }
}

pub fn _mul_by_z_xs<F: FftField>(xs: &[F], p: &Monic<F>) -> Monic<F> {
    let m = xs.len();
    let k = p.poly.degree();
    let n = m + k;
    let h = n / 2;
    let (left, right) = if h >= m {
        (z_xs(xs), p)
    } else {
        (z_xs(&xs[0..h]), &_mul_by_z_xs(&xs[h..m], p))
    };
    Monic::mul(&left, right)
}

/// Computes the vanishing polynomial of a set `xs`,
/// and multiplies it by a monic polynomial `p`.
/// TODO: now that doesn't make any sense
pub fn mul_by_z_xs<F: FftField>(xs: &[F], p: &Monic<F>) -> (Monic<F>, Monic<F>) {
    let m = xs.len();
    let k = p.poly.degree();
    let n = m + k;
    let h = n / 2;
    if h >= m {
        let z = z_xs(xs);
        let zp = Monic::mul(&z, p);
        (zp, z)
    } else {
        let l = z_xs(&xs[0..h]);
        let (zp, z) = mul_by_z_xs(&xs[h..m], p);
        let zp = Monic::mul(&l, &zp);
        let z = Monic::mul(&l, &z);
        (zp, z)
    }
}

pub struct Domain<F: FftField> {
    t: usize,
    n: usize,
    pub fft_domain: Radix2EvaluationDomain<F>,
    z_c2: Monic<F>,
}

impl<F: FftField> Domain<F> {
    pub fn new(t: usize, n: usize) -> Self {
        let domain = Radix2EvaluationDomain::<F>::new(n).unwrap();
        let c2: Vec<F> = domain.elements().skip(n).collect();
        let z_c2 = z_xs(&c2);
        Self {
            t,
            n,
            fft_domain: domain,
            z_c2,
        }
    }

    // We want to interpolate a degree `t-1` polynomial `f`
    // using evaluations of `f` at not more than 'n >= t' predefined points.

    // Consider an FFT domain `[w_0, ..., w_{d-1}], where d >= n`.
    // The evaluations have form `(i_k, v_k)` where `0 <= i_k < n`, and
    // mean allegedly that `f(w_{i_k}) = v_k` for k = 1,...,s`.
    // 0 < t <= s <= n <= d.

    // Let `S = {i_k|k = 1,...,s}` be the set of indices of the domain elements for which the evaluations are given.
    // Let `C = [0,...,d-1]\S` be the complement to `S`. Then to compute `f` we
    // 1. Compute `z_c` - the vanishing polynomial of `{w_i|i in C}`.
    //    - a product tree of degree `d - s`, or
    //    - a product tree of degree `n - s`, given the product tree `d - n` is precomputed.
    // 2. Evaluate `z_c` over the domain using an FFT. // `d-FFT`
    // 3. Compute `f * z_c` in evaluation form. // `s` field multiplications
    // 4. Interpolate `f * z_c`. // `d-FFT`
    // 5. Evaluate `f * z_c` and `z_c` over a coset of the domain. // `2 x d-FFT`
    // 6. Compute `f = (f * z_c) / z_c` over the coset in the evaluation form and interpolate it. // `d-FFT`

    /// Returns the interpolating polynomial, and the erasure polynomial `z_c1(X) = (X - w_j_1)...(X - w_j_{n-s})`
    pub fn interpolate(&self, f_on_s: &[(usize, F)]) -> (P<F>, P<F>) {
        let s = f_on_s.len();
        debug_assert!(self.t <= s);
        debug_assert!(s <= self.n);
        let d = self.fft_domain.size();
        // TODO: check `f_on_s` is valid
        let mut f_on_d = vec![None; d];
        for (i, vi) in f_on_s {
            f_on_d[*i] = Some(*vi);
        }
        // `C1 = {w_i | i in [0,...,n-1]\S}`
        let c1: Vec<F> = self
            .fft_domain
            .elements()
            .take(self.n)
            .enumerate()
            .filter_map(|(i, wi)| f_on_d[i].is_none().then_some(wi))
            .collect();
        debug_assert_eq!(c1.len(), self.n - s);

        // 1. Compute `z_C` - the vanishing polynomial of `{w_i | i in C}`.
        let _t_zc = start_timer!(|| format!("Compute z_C, deg(z_C) = {}", d - s));
        // `C = {w_i | i in [0,...,d-1]\S} = C1 + C2`, where C2 = {w_i | n <= i < d}`
        let (zc, z_c1) = mul_by_z_xs(&c1, &self.z_c2);
        let zc = zc.poly;
        let z_c1 = z_c1.poly;
        debug_assert_eq!(zc.degree(), d - s);
        end_timer!(_t_zc);

        // 2. Evaluate `z_c` over the domain using an FFT.
        let _t_zc_on_d = start_timer!(|| format!("Evaluate z_C, deg(z_C) = {}", zc.degree()));
        let zc_on_d = zc.evaluate_over_domain_by_ref(self.fft_domain).evals;
        end_timer!(_t_zc_on_d);

        // 3. Compute `f * z_c` in evaluation form.
        let f_zc_on_d: Vec<F> = f_on_d
            .iter()
            .zip(zc_on_d)
            .map(|(vi, zi)| match vi {
                Some(vi) => zi * vi,
                None => F::zero(),
            })
            .collect();
        let f_zc_on_d = Evaluations::from_vec_and_domain(f_zc_on_d, self.fft_domain);

        // 4. Interpolate `f * z_c`.
        let _t_f_zc = start_timer!(|| format!("Interpolate f.z_C, supp(f.z_C) = {s}"));
        let f_zc = f_zc_on_d.interpolate();
        end_timer!(_t_f_zc);

        let _t_option_1 = start_timer!(|| "Option №1");
        // 5. Evaluate `f * z_c` and `z_c` over a coset of the domain.
        let coset = self.fft_domain.get_coset(F::GENERATOR).unwrap();
        let _t_f_zc_on_coset =
            start_timer!(|| format!("Evaluate f.z_C over gD, deg(f.z_C) = {}", f_zc.degree()));
        let f_zc_on_coset = f_zc.evaluate_over_domain_by_ref(coset);
        end_timer!(_t_f_zc_on_coset);
        let _t_zc_on_coset =
            start_timer!(|| format!("Evaluate z_C over gD, deg(z_C) = {}", zc.degree()));
        let zc_on_coset = zc.evaluate_over_domain_by_ref(coset);
        end_timer!(_t_zc_on_coset);

        // 6. Compute `f = (f * z_c) / z_c` over the coset in the evaluation form and interpolate it.
        let f_on_coset = &f_zc_on_coset / &zc_on_coset;
        let f = f_on_coset.interpolate();
        end_timer!(_t_option_1);
        (f, z_c1)
        // let _t_option_2 = start_timer!(|| "Option №2");
        // let d_zc = lagrange::d(&zc);
        // let _t_tree = start_timer!(|| format!("Tree of size = {d} - {s}, deg(z') = {}", d_zc.degree()));
        // let d_zc_on_d_ = tree_on_c.evaluate(&d_zc);
        // end_timer!(_t_tree);
        // let _t_fft = start_timer!(|| format!("iFFT, size = {d}"));
        // let d_zc_on_d = d_zc.evaluate_over_domain_by_ref(domain);
        // // assert_eq!(d_zc_on_d_.unwrap().len(), d_zc_on_d.evals.len()); // TODO
        // end_timer!(_t_fft);
        // let d_f_zc = lagrange::d(&f_zc);
        // let d_f_zc_on_d = d_f_zc.evaluate_over_domain_by_ref(domain);
        // let f_on_c = &d_f_zc_on_d / &d_zc_on_d;
        // let mut f_on_d = f_on_d;
        // for i in 0..d {
        //     match f_on_d[i] {
        //         None => f_on_d[i] = Some(f_on_c.evals[i]),
        //         Some(_) => {}
        //     }
        // }
        // let f_on_d: Vec<F> = f_on_d.into_iter().flatten().collect();
        // debug_assert_eq!(f_on_d.len(), d);
        // let f_on_d = Evaluations::from_vec_and_domain(f_on_d, domain);
        // let f2 = f_on_d.interpolate();
        // debug_assert_eq!(f1, f2);
        // end_timer!(_t_option_2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_poly::DenseUVPolynomial;
    use ark_std::rand::prelude::SliceRandom;
    use ark_std::rand::Rng;
    use ark_std::{test_rng, UniformRand};

    // `t - 1` -- degree of the polynomial `f`
    // `s` -- number of alleged evaluations `(i, v_i)`
    // `n` -- number of evaluation points `w_i`
    fn _test_interpolation<F: FftField>(t: usize, n: usize, s: usize) {
        let rng = &mut test_rng();
        debug_assert!(t <= s);
        debug_assert!(s <= n);
        let f = P::<F>::rand(t - 1, rng);
        let mut is: Vec<usize> = (0..n).collect();
        is.shuffle(rng);
        let (is_in_s, is_in_c) = is.split_at(s);
        debug_assert_eq!(is_in_s.len(), s);
        debug_assert_eq!(is_in_c.len(), n - s);
        let domain = Domain::new(t, n);
        let ws: Vec<F> = domain.fft_domain.elements().collect();
        let f_on_s: Vec<(usize, F)> = is_in_s.iter().map(|&i| (i, f.evaluate(&ws[i]))).collect();
        let c: Vec<F> = is_in_c.iter().map(|&i| ws[i]).collect();
        let zc = z_xs(&c).poly;

        let _t = start_timer!(|| format!("Interpolation, (t, s, n) = ({t},{s},{n})"));
        let (f_, zc_) = domain.interpolate(&f_on_s);
        assert_eq!(f_, f);
        assert_eq!(zc_, zc);
        end_timer!(_t);
        println!("\n");
    }

    // RUST_BACKTRACE=1 cargo test test_interpolation --release --features="print-trace" -- --show-output
    #[test]
    fn test_interpolation() {
        let (t, n) = (667, 1000);
        _test_interpolation::<Fr>(t, n, t);
        _test_interpolation::<Fr>(t, n, n);
        let s = test_rng().gen_range(t..=n);
        _test_interpolation::<Fr>(t, n, s);
    }

    #[test]
    fn test_z_poly() {
        let rng = &mut test_rng();

        let n = 123;
        let k = 23;
        let m = n - k;

        let xs: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();

        let z = z_xs(&xs);
        let z_l = z_xs(&xs[0..m]);
        let z_r = z_xs(&xs[m..n]);
        let (z_, z_l_) = mul_by_z_xs(&xs[0..m], &z_r);
        assert_eq!(z_, z);
        assert_eq!(z_l_, z_l);

        assert_eq!(mul_by_z_xs(&[], &one::<Fr>()).0, one());
    }
}

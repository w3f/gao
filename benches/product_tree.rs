use ark_bls12_381::Fr;
use ark_ff::FftField;
use ark_poly::polynomial::DenseUVPolynomial;
use ark_poly::Polynomial;
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gao::poly_div::{hensel_lift, hensel_lift_fft, inv_mod};
use gao::{Poly, P};

fn polynomial_mul<F: FftField>(c: &mut Criterion) {
    let rng = &mut test_rng();
    let mut group = c.benchmark_group("polynomial-mul");

    let ds = [1, 2, 4, 8, 16, 32, 64, 128];

    let random_polys = ds
        .into_iter()
        .map(|d| {
            let mut p = P::<F>::rand(d, rng);
            p.coeffs[d] = F::one();
            p
        })
        .collect::<Vec<_>>();

    for p in random_polys.iter() {
        group.bench_with_input(BenchmarkId::new("fft-mul", p.degree()), p, |b, p| {
            b.iter(|| p * p)
        });

        group.bench_with_input(BenchmarkId::new("naive-mul", p.degree()), p, |b, p| {
            b.iter(|| p.naive_mul(p))
        });
    }
}

fn polynomial_div<F: FftField>(c: &mut Criterion) {
    let rng = &mut test_rng();
    let mut group = c.benchmark_group("polynomial-div");
    let inputs = (5..9)
        .map(|log_l| {
            let l = 2usize.pow(log_l as u32);
            let f = P::<F>::rand(l - 1, rng);
            let g = inv_mod(&f.mod_xk(l / 2), l / 2);
            (f, g, log_l)
        })
        .collect::<Vec<_>>();

    for (f, g, log_l) in inputs.iter() {
        group.bench_with_input(
            BenchmarkId::new("quadratic-lift", log_l),
            &(f, g),
            |b, (f, g)| b.iter(|| hensel_lift(&f, &g, 1 << log_l)),
        );

        group.bench_with_input(BenchmarkId::new("fft-lift", log_l), &(f, g), |b, (f, g)| {
            b.iter(|| hensel_lift_fft(&f, &g, 1 << log_l))
        });
    }
}

criterion_group!(
    benches,
    // polynomial_mul::<Fr>,
    polynomial_div::<Fr>,
);
criterion_main!(benches);

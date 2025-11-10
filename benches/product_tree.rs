use ark_bls12_381::Fr;
use ark_ff::FftField;
use ark_poly::polynomial::DenseUVPolynomial;
use ark_poly::Polynomial;

use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gao::P;

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

criterion_group!(benches, polynomial_mul::<Fr>);
criterion_main!(benches);

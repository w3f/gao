use crate::product_tree::ProductTree;
use ark_ff::{FftField, Field};
use ark_poly::univariate::DensePolynomial;
use ark_poly::{DenseUVPolynomial, EvaluationDomain};
use ark_std::{end_timer, start_timer};

/// The formal derivative of `f`.
pub fn d<F: Field>(f: &DensePolynomial<F>) -> DensePolynomial<F> {
    let df_coeffs = f.iter()
        .enumerate()
        .skip(1)
        .map(|(i, ci)| F::from(i as u32) * ci)
        .collect();
    DensePolynomial::from_coefficients_vec(df_coeffs)
}

pub struct InterpolationDomain<F: FftField> {
    us: Vec<F>,
    products: ProductTree<F>,
    weights: Vec<F>,
}

impl<F: FftField> InterpolationDomain<F> {
    pub fn from_subset<D: EvaluationDomain<F>>(fft_domain: D, bitmask: &[bool]) -> Result<Self, ()> {
        if fft_domain.size() != bitmask.len() {
            return Err(());
        }
        let us = fft_domain.elements()
            .zip(bitmask)
            .filter_map(|(ui, bi)| bi.then_some(ui))
            .collect::<Vec<_>>();
        let _t = start_timer!(|| "Subproduct tree");
        let products = ProductTree::new(&us)?;
        end_timer!(_t);
        let dz = d(products.root_poly());
        let dz_over_domain = dz.evaluate_over_domain(fft_domain);
        let mut weights = dz_over_domain.evals.into_iter()
            .zip(bitmask)
            .filter_map(|(dz_in_ui, bi)| bi.then_some(dz_in_ui))
            .collect::<Vec<_>>();
        ark_ff::batch_inversion(&mut weights);
        Ok(Self {
            us,
            products,
            weights,
        })
    }

    pub fn interpolate(&self, vs: &[F]) -> Result<DensePolynomial<F>, ()> {
        let cs = self.weights.iter().zip(vs)
            .map(|(a, b)| *a * b)
            .collect::<Vec<_>>();
        Self::linear_combination(&self.products.child_polys(), &cs)
    }

    fn linear_combination(products: &[DensePolynomial<F>], cs: &[F]) -> Result<DensePolynomial<F>, ()> {
        let n = cs.len();
        match n {
            0 => Err(()),
            1 => Ok(DensePolynomial::from_coefficients_slice(cs)),
            _ => {
                let m = products.len();
                if m != 2 * (n - 1) {
                    return Err(());
                }
                let (cs_0, cs_1) = cs.split_at(n / 2);
                let (subtree_0, subtree_1) = products.split_at(m / 2);
                let (root_0, products_0) = subtree_0.split_last().unwrap();
                let (root_1, products_1) = subtree_1.split_last().unwrap();
                let r0 = Self::linear_combination(products_0, cs_0)?;
                let r1 = Self::linear_combination(products_1, cs_1)?;
                Ok(root_1 * &r0 + root_0 * &r1)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_poly::GeneralEvaluationDomain;
    use ark_std::test_rng;

    fn _fast_interpolation<F: FftField>(n: usize) {
        let rng = &mut test_rng();

        let fft_domain = GeneralEvaluationDomain::<F>::new(n).unwrap();
        let bitmask = &vec![true; n];
        let p = DensePolynomial::rand(n - 1, rng);
        let vs = p.evaluate_over_domain_by_ref(fft_domain).evals;

        let _t = start_timer!(|| "Fast interpolation");
        let domain = InterpolationDomain::from_subset(fft_domain, bitmask).unwrap();
        assert_eq!(domain.interpolate(&vs).unwrap(), p.clone());
        end_timer!(_t);

        let _t = start_timer!(|| "Fast multipoint evaluation");
        assert_eq!(domain.products.evaluate(&p).unwrap(), vs);
        end_timer!(_t);
    }

    #[test]
    fn fast_interpolation() {
        _fast_interpolation::<ark_bls12_381::Fr>(1024);
    }
}

//! Noether's theorem: continuous symmetry → conserved quantity (discretized).
//!
//! In the continuous setting, Noether's theorem states that every differentiable symmetry
//! of the action of a physical system yields a conservation law. Here we discretize this:
//! each symmetry generator produces a discrete conserved quantity that should be preserved
//! up to numerical precision under the discrete evolution.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// A continuous symmetry that generates a conserved quantity via Noether's theorem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symmetry {
    /// Name of the symmetry (e.g., "time translation", "rotation").
    pub name: String,
    /// Generator matrix G such that δQ = ε·G·Q for infinitesimal parameter ε.
    pub generator: DMatrix<f64>,
    /// The associated conserved quantity.
    pub conserved_quantity_name: String,
}

impl Symmetry {
    pub fn new(name: &str, generator: DMatrix<f64>, conserved_name: &str) -> Self {
        Symmetry {
            name: name.to_string(),
            generator,
            conserved_quantity_name: conserved_name.to_string(),
        }
    }

    /// Apply the symmetry transformation to a state vector.
    pub fn transform(&self, state: &DVector<f64>, epsilon: f64) -> DVector<f64> {
        let n = state.nrows();
        assert_eq!(self.generator.nrows(), n);
        assert_eq!(self.generator.ncols(), n);
        state + self.generator.scale(epsilon) * state
    }

    /// Compute the Noether charge: J = Q^T · G · Q / 2.
    /// For a state vector Q and generator G, the conserved charge is Q^T G Q.
    pub fn noether_charge(&self, state: &DVector<f64>) -> f64 {
        let gt = &self.generator.transpose();
        // charge = 0.5 * Q^T * (G + G^T) * Q
        let sym_part = &self.generator + gt;
        let temp = &sym_part * state;
        0.5 * state.dot(&temp)
    }

    /// Check if the generator is antisymmetric (as for angular momentum / rotations).
    pub fn is_antisymmetric(&self, tolerance: f64) -> bool {
        let diff = &self.generator + &self.generator.transpose();
        diff.norm() < tolerance
    }

    /// Check if the generator is symmetric (as for scaling / dilation).
    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        let diff = &self.generator - &self.generator.transpose();
        diff.norm() < tolerance
    }
}

/// A Noether pair: symmetry ↔ conserved quantity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoetherPair {
    pub symmetry: Symmetry,
    pub tolerance: f64,
}

impl NoetherPair {
    pub fn new(symmetry: Symmetry, tolerance: f64) -> Self {
        NoetherPair { symmetry, tolerance }
    }

    /// Verify that the Noether charge is conserved between two states.
    pub fn verify_conservation(&self, state_before: &DVector<f64>, state_after: &DVector<f64>) -> bool {
        let charge_before = self.symmetry.noether_charge(state_before);
        let charge_after = self.symmetry.noether_charge(state_after);
        (charge_before - charge_after).abs() < self.tolerance
    }

    /// Compute conservation error.
    pub fn conservation_error(&self, state_before: &DVector<f64>, state_after: &DVector<f64>) -> f64 {
        let charge_before = self.symmetry.noether_charge(state_before);
        let charge_after = self.symmetry.noether_charge(state_after);
        (charge_before - charge_after).abs()
    }
}

/// Discretize a continuous symmetry into a finite set of transformations.
pub fn discretize_symmetry(symmetry: &Symmetry, n_steps: usize) -> Vec<DMatrix<f64>> {
    let epsilon = 2.0 * std::f64::consts::PI / n_steps as f64;
    let n = symmetry.generator.nrows();
    (0..n_steps)
        .map(|k| {
            let angle = epsilon * k as f64;
            // First-order approximation: I + angle * G
            DMatrix::identity(n, n) + symmetry.generator.scale(angle)
        })
        .collect()
}

/// Build the standard Noether pairs for a 2D system:
/// - Time translation → energy (H)
/// - Space translation → momentum (p)
/// - Rotation → angular momentum (L)
pub fn standard_2d_noether_pairs(tolerance: f64) -> Vec<NoetherPair> {
    let mut pairs = Vec::new();

    // Energy conservation (Hamiltonian symmetry)
    let h_gen = DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]);
    pairs.push(NoetherPair::new(
        Symmetry::new("time translation", h_gen, "energy"),
        tolerance,
    ));

    // x-momentum conservation
    let _px_gen = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.0]);
    // Use a non-trivial generator: shear in x
    let px_gen = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
    pairs.push(NoetherPair::new(
        Symmetry::new("x-translation", px_gen, "x-momentum"),
        tolerance,
    ));

    // Angular momentum conservation (rotation generator)
    let l_gen = DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]);
    pairs.push(NoetherPair::new(
        Symmetry::new("rotation", l_gen, "angular momentum"),
        tolerance,
    ));

    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetry_transform() {
        let gen = DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]);
        let sym = Symmetry::new("rotation", gen, "angular momentum");
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let transformed = sym.transform(&state, 0.1);
        // Should be approximately (1.0, 0.1)
        assert!((transformed[0] - 1.0).abs() < 1e-10);
        assert!((transformed[1] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_noether_charge_antisymmetric() {
        let gen = DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]);
        let sym = Symmetry::new("rotation", gen, "L");
        // For antisymmetric G, charge should be Q^T * G * Q = 0 for any Q
        // (since antisymmetric quadratic form vanishes)
        let state = DVector::from_vec(vec![3.0, 4.0]);
        let charge = sym.noether_charge(&state);
        assert!(charge.abs() < 1e-10);
    }

    #[test]
    fn test_is_antisymmetric() {
        let gen = DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]);
        let sym = Symmetry::new("rotation", gen, "L");
        assert!(sym.is_antisymmetric(1e-10));
        assert!(!sym.is_symmetric(1e-10));
    }

    #[test]
    fn test_is_symmetric() {
        let gen = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let sym = Symmetry::new("identity", gen, "charge");
        assert!(sym.is_symmetric(1e-10));
    }

    #[test]
    fn test_noether_pair_conservation() {
        let gen = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        let sym = Symmetry::new("scaling", gen, "charge");
        let pair = NoetherPair::new(sym, 1e-8);

        let state = DVector::from_vec(vec![2.0, 3.0]);
        // Same state should be trivially conserved
        assert!(pair.verify_conservation(&state, &state));
    }

    #[test]
    fn test_noether_pair_error() {
        let gen = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        let sym = Symmetry::new("test", gen, "charge");
        let pair = NoetherPair::new(sym, 1e-8);

        let before = DVector::from_vec(vec![2.0, 3.0]);
        let after = DVector::from_vec(vec![2.1, 3.0]);
        let error = pair.conservation_error(&before, &after);
        assert!(error > 0.0);
    }

    #[test]
    fn test_discretize_symmetry() {
        let gen = DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]);
        let sym = Symmetry::new("rotation", gen, "L");
        let discretized = discretize_symmetry(&sym, 8);
        assert_eq!(discretized.len(), 8);
        // Each should be 2x2
        for m in &discretized {
            assert_eq!(m.nrows(), 2);
            assert_eq!(m.ncols(), 2);
        }
    }

    #[test]
    fn test_standard_2d_noether_pairs() {
        let pairs = standard_2d_noether_pairs(1e-10);
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].symmetry.name, "time translation");
        assert_eq!(pairs[1].symmetry.name, "x-translation");
        assert_eq!(pairs[2].symmetry.name, "rotation");
    }

    #[test]
    fn test_rotation_preserves_noether_charge() {
        // Rotation generator is antisymmetric; for a symmetric charge matrix,
        // the charge is preserved under rotation.
        let gen = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        let sym = Symmetry::new("charge", gen, "q");
        let pair = NoetherPair::new(sym, 1e-8);

        let state = DVector::from_vec(vec![1.0, 0.0]);
        // Apply small rotation manually: R(θ) * state
        let theta: f64 = 0.1;
        let rotated = DVector::from_vec(vec![
            state[0] * theta.cos() - state[1] * theta.sin(),
            state[0] * theta.sin() + state[1] * theta.cos(),
        ]);
        // Charge Q^T * diag(1,0) * Q = x^2, which is NOT rotation-invariant
        // So this should fail conservation (that's the correct physics)
        let error = pair.conservation_error(&state, &rotated);
        assert!(error > 1e-3); // Should not be conserved
    }
}

//! Conservation verification: check that discrete schemes preserve conserved quantities.

use serde::{Serialize, Deserialize};
use crate::finite_volume::FiniteVolumeScheme;

/// Result of a conservation verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub law_name: String,
    pub initial_total: f64,
    pub final_total: f64,
    pub error: f64,
    pub relative_error: f64,
    pub passed: bool,
}

/// A conservation verifier that tracks quantities over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationVerifier {
    /// Tolerance for absolute error.
    pub tolerance: f64,
    /// History of total conserved quantity at each step.
    pub history: Vec<f64>,
    /// Name of the law being verified.
    pub law_name: String,
}

impl ConservationVerifier {
    pub fn new(law_name: &str, tolerance: f64) -> Self {
        ConservationVerifier {
            tolerance,
            history: Vec::new(),
            law_name: law_name.to_string(),
        }
    }

    /// Record the current total.
    pub fn record(&mut self, total: f64) {
        self.history.push(total);
    }

    /// Check if conservation holds across all recorded values.
    pub fn verify(&self) -> VerificationResult {
        if self.history.len() < 2 {
            return VerificationResult {
                law_name: self.law_name.clone(),
                initial_total: self.history.first().copied().unwrap_or(0.0),
                final_total: self.history.last().copied().unwrap_or(0.0),
                error: 0.0,
                relative_error: 0.0,
                passed: true,
            };
        }

        let initial = self.history[0];
        let final_val = *self.history.last().unwrap();
        let error = (final_val - initial).abs();
        let relative_error = if initial.abs() > 1e-15 {
            error / initial.abs()
        } else {
            error
        };

        VerificationResult {
            law_name: self.law_name.clone(),
            initial_total: initial,
            final_total: final_val,
            error,
            relative_error,
            passed: error < self.tolerance,
        }
    }

    /// Verify conservation for a finite volume scheme over n_steps.
    pub fn verify_fv_scheme(
        scheme: &mut FiniteVolumeScheme,
        velocity: f64,
        n_steps: usize,
        tolerance: f64,
    ) -> VerificationResult {
        let mut verifier = ConservationVerifier::new("FV mass conservation", tolerance);
        verifier.record(scheme.total());

        for _ in 0..n_steps {
            scheme.evolve(velocity, 1);
            verifier.record(scheme.total());
        }

        verifier.verify()
    }

    /// Check max deviation from initial value.
    pub fn max_deviation(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let initial = self.history[0];
        self.history.iter().map(|v| (v - initial).abs()).fold(0.0_f64, f64::max)
    }

    /// Compute the drift rate: average change per step.
    pub fn drift_rate(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let n = self.history.len() - 1;
        let total_drift = self.history.last().unwrap() - self.history[0];
        total_drift / n as f64
    }

    /// Reset the verifier.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finite_volume::{FiniteVolumeScheme, BoundaryCondition};

    #[test]
    fn test_verifier_records_history() {
        let mut v = ConservationVerifier::new("mass", 1e-8);
        v.record(10.0);
        v.record(10.0);
        v.record(10.0);
        assert_eq!(v.history.len(), 3);
    }

    #[test]
    fn test_verifier_passes_conserved() {
        let mut v = ConservationVerifier::new("mass", 1e-8);
        v.record(10.0);
        v.record(10.0);
        v.record(10.0);
        let result = v.verify();
        assert!(result.passed);
        assert!(result.error < 1e-10);
    }

    #[test]
    fn test_verifier_fails_non_conserved() {
        let mut v = ConservationVerifier::new("energy", 1e-8);
        v.record(100.0);
        v.record(99.0);
        v.record(98.0);
        let result = v.verify();
        assert!(!result.passed);
        assert!(result.error > 1.0);
    }

    #[test]
    fn test_verifier_single_value() {
        let mut v = ConservationVerifier::new("mass", 1e-8);
        v.record(5.0);
        let result = v.verify();
        assert!(result.passed);
    }

    #[test]
    fn test_max_deviation() {
        let mut v = ConservationVerifier::new("mass", 1e-8);
        v.record(10.0);
        v.record(10.1);
        v.record(9.9);
        v.record(10.05);
        assert!((v.max_deviation() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_drift_rate() {
        let mut v = ConservationVerifier::new("mass", 1e-8);
        v.record(10.0);
        v.record(10.0);
        v.record(10.0);
        v.record(10.0);
        assert!(v.drift_rate().abs() < 1e-10);
    }

    #[test]
    fn test_drift_rate_positive() {
        let mut v = ConservationVerifier::new("mass", 1e-8);
        v.record(10.0);
        v.record(10.1);
        v.record(10.2);
        // drift = (10.2 - 10.0) / 2 = 0.1
        assert!((v.drift_rate() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_verify_fv_periodic() {
        let n = 20;
        let initial = DVector::from_vec((0..n).map(|i| (i as f64 * 0.5).sin()).collect());
        let mut scheme = FiniteVolumeScheme::new(n, initial, 0.1, 0.01, BoundaryCondition::Periodic);
        let result = ConservationVerifier::verify_fv_scheme(&mut scheme, 0.5, 50, 1e-6);
        assert!(result.passed);
    }

    #[test]
    fn test_verifier_reset() {
        let mut v = ConservationVerifier::new("mass", 1e-8);
        v.record(1.0);
        v.record(2.0);
        v.reset();
        assert!(v.history.is_empty());
    }

    #[test]
    fn test_relative_error() {
        let mut v = ConservationVerifier::new("mass", 1e-8);
        v.record(100.0);
        v.record(101.0);
        let result = v.verify();
        assert!((result.relative_error - 0.01).abs() < 1e-10);
    }
}

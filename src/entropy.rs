//! Entropy production tracking: second law of thermodynamics in discrete systems.

use serde::{Serialize, Deserialize};

/// Entropy tracker for a discrete system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyTracker {
    /// Entropy values at each time step.
    pub history: Vec<f64>,
    /// Tolerance for non-decrease check.
    pub tolerance: f64,
}

impl EntropyTracker {
    pub fn new(tolerance: f64) -> Self {
        EntropyTracker {
            history: Vec::new(),
            tolerance,
        }
    }

    /// Compute Shannon entropy of a probability distribution.
    pub fn shannon_entropy(probs: &[f64]) -> f64 {
        probs.iter()
            .filter(|&&p| p > 1e-15)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Compute Boltzmann entropy: S = ln(W) where W is the number of microstates.
    pub fn boltzmann_entropy(n_microstates: u64) -> f64 {
        if n_microstates <= 1 { 0.0 } else { (n_microstates as f64).ln() }
    }

    /// Record entropy at current step.
    pub fn record(&mut self, entropy: f64) {
        self.history.push(entropy);
    }

    /// Check that entropy is non-decreasing (second law).
    pub fn is_non_decreasing(&self) -> bool {
        for w in self.history.windows(2) {
            if w[1] < w[0] - self.tolerance {
                return false;
            }
        }
        true
    }

    /// Total entropy produced from first to last recorded value.
    pub fn total_production(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        self.history.last().unwrap() - self.history.first().unwrap()
    }

    /// Average entropy production rate.
    pub fn production_rate(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        self.total_production() / (self.history.len() - 1) as f64
    }

    /// Compute entropy from a discrete state (normalize to probabilities first).
    pub fn entropy_from_state(state: &[f64]) -> f64 {
        let total: f64 = state.iter().sum();
        if total.abs() < 1e-15 {
            return 0.0;
        }
        let probs: Vec<f64> = state.iter().map(|x| x / total).collect();
        Self::shannon_entropy(&probs)
    }

    /// Check if total entropy production is positive (as required by 2nd law).
    pub fn is_production_positive(&self) -> bool {
        self.total_production() >= -self.tolerance
    }

    /// Reset tracker.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy_uniform() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let h = EntropyTracker::shannon_entropy(&probs);
        // H = -4 * 0.25 * ln(0.25) = ln(4)
        assert!((h - 4.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_shannon_entropy_deterministic() {
        let probs = vec![1.0, 0.0, 0.0];
        let h = EntropyTracker::shannon_entropy(&probs);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_shannon_entropy_binary() {
        let probs = vec![0.5, 0.5];
        let h = EntropyTracker::shannon_entropy(&probs);
        assert!((h - std::f64::consts::LN_2).abs() < 1e-10);
    }

    #[test]
    fn test_boltzmann_entropy() {
        let s = EntropyTracker::boltzmann_entropy(100);
        assert!((s - 100.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_boltzmann_entropy_one() {
        let s = EntropyTracker::boltzmann_entropy(1);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_non_decreasing_true() {
        let mut tracker = EntropyTracker::new(1e-10);
        tracker.record(1.0);
        tracker.record(1.5);
        tracker.record(2.0);
        assert!(tracker.is_non_decreasing());
    }

    #[test]
    fn test_non_decreasing_false() {
        let mut tracker = EntropyTracker::new(1e-10);
        tracker.record(2.0);
        tracker.record(1.0);
        assert!(!tracker.is_non_decreasing());
    }

    #[test]
    fn test_total_production() {
        let mut tracker = EntropyTracker::new(1e-10);
        tracker.record(1.0);
        tracker.record(2.0);
        tracker.record(3.5);
        assert!((tracker.total_production() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_production_rate() {
        let mut tracker = EntropyTracker::new(1e-10);
        tracker.record(0.0);
        tracker.record(1.0);
        tracker.record(2.0);
        // rate = 2.0 / 2 = 1.0
        assert!((tracker.production_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_from_state() {
        let state = vec![1.0, 1.0, 1.0, 1.0];
        let h = EntropyTracker::entropy_from_state(&state);
        assert!((h - 4.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_is_production_positive() {
        let mut tracker = EntropyTracker::new(1e-10);
        tracker.record(1.0);
        tracker.record(2.0);
        assert!(tracker.is_production_positive());
    }

    #[test]
    fn test_reset() {
        let mut tracker = EntropyTracker::new(1e-10);
        tracker.record(1.0);
        tracker.reset();
        assert!(tracker.history.is_empty());
    }
}

//! Spectral conservation: energy conservation in spectral (Fourier) methods.
//!
//! For linear PDEs with constant coefficients, spectral methods exactly conserve
//! the L2 norm (Parseval's theorem). This module verifies spectral conservation.

use serde::{Serialize, Deserialize};

/// Spectral conservation tracker: monitors L2 norm in physical and spectral space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralConservation {
    /// L2 norm history in physical space.
    pub physical_norms: Vec<f64>,
    /// L2 norm history in spectral space.
    pub spectral_norms: Vec<f64>,
    /// Tolerance for conservation check.
    pub tolerance: f64,
}

impl SpectralConservation {
    pub fn new(tolerance: f64) -> Self {
        SpectralConservation {
            physical_norms: Vec::new(),
            spectral_norms: Vec::new(),
            tolerance,
        }
    }

    /// Record physical-space L2 norm.
    pub fn record_physical(&mut self, state: &[f64]) {
        let l2: f64 = state.iter().map(|x| x * x).sum::<f64>().sqrt();
        self.physical_norms.push(l2);
    }

    /// Record spectral-space L2 norm (Parseval: should equal physical).
    pub fn record_spectral(&mut self, coefficients: &[f64]) {
        let l2: f64 = coefficients.iter().map(|x| x * x).sum::<f64>().sqrt();
        self.spectral_norms.push(l2);
    }

    /// Check Parseval's theorem: physical L2 ≈ spectral L2.
    pub fn check_parseval(&self) -> bool {
        if self.physical_norms.len() != self.spectral_norms.len() {
            return false;
        }
        self.physical_norms.iter().zip(self.spectral_norms.iter())
            .all(|(p, s)| (p - s).abs() < self.tolerance)
    }

    /// Check that physical L2 norm is conserved over time.
    pub fn check_conservation(&self) -> bool {
        if self.physical_norms.len() < 2 {
            return true;
        }
        let first = self.physical_norms[0];
        self.physical_norms.iter().all(|n| (n - first).abs() < self.tolerance)
    }

    /// Compute the relative L2 error between two states.
    pub fn relative_l2_error(a: &[f64], b: &[f64]) -> f64 {
        let diff: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a < 1e-15 { diff } else { diff / norm_a }
    }

    /// Spectral derivative via DFT (naive, for verification).
    /// For mode k: d/dx → ik
    pub fn spectral_derivative(coeffs: &[Complex64], dx: f64) -> Vec<Complex64> {
        let n = coeffs.len();
        let k_factor = 2.0 * std::f64::consts::PI / (n as f64 * dx);
        coeffs.iter().enumerate().map(|(k, c)| {
            let freq = if k <= n / 2 { k as f64 } else { k as f64 - n as f64 };
            c.scale(Complex64::new(0.0, k_factor * freq))
        }).collect()
    }
}

/// Minimal complex number (avoids extra dependency).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub fn new(re: f64, im: f64) -> Self { Complex64 { re, im } }
    pub fn scale(&self, factor: Complex64) -> Complex64 {
        Complex64 {
            re: self.re * factor.re - self.im * factor.im,
            im: self.re * factor.im + self.im * factor.re,
        }
    }
    pub fn norm_sq(&self) -> f64 { self.re * self.re + self.im * self.im }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_physical() {
        let mut sc = SpectralConservation::new(1e-10);
        sc.record_physical(&[1.0, 2.0, 3.0]);
        assert_eq!(sc.physical_norms.len(), 1);
        let expected = (1.0_f64 + 4.0 + 9.0).sqrt();
        assert!((sc.physical_norms[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_record_spectral() {
        let mut sc = SpectralConservation::new(1e-10);
        sc.record_spectral(&[3.0, 4.0]);
        let expected = (9.0_f64 + 16.0).sqrt();
        assert!((sc.spectral_norms[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_parseval_holds() {
        let mut sc = SpectralConservation::new(0.1);
        sc.record_physical(&[1.0, 2.0, 3.0]);
        sc.record_spectral(&[1.0, 2.0, 3.0]); // same values → Parseval holds
        assert!(sc.check_parseval());
    }

    #[test]
    fn test_parseval_fails() {
        let mut sc = SpectralConservation::new(1e-10);
        sc.record_physical(&[1.0, 2.0, 3.0]);
        sc.record_spectral(&[10.0, 20.0, 30.0]);
        assert!(!sc.check_parseval());
    }

    #[test]
    fn test_conservation_holds() {
        let mut sc = SpectralConservation::new(0.01);
        sc.record_physical(&[1.0, 2.0, 3.0]);
        sc.record_physical(&[1.0, 2.0, 3.0]);
        sc.record_physical(&[1.0, 2.0, 3.0]);
        assert!(sc.check_conservation());
    }

    #[test]
    fn test_conservation_fails() {
        let mut sc = SpectralConservation::new(1e-10);
        sc.record_physical(&[1.0, 2.0, 3.0]);
        sc.record_physical(&[2.0, 3.0, 4.0]);
        assert!(!sc.check_conservation());
    }

    #[test]
    fn test_relative_l2_error_zero() {
        let err = SpectralConservation::relative_l2_error(&[1.0, 2.0], &[1.0, 2.0]);
        assert!(err.abs() < 1e-10);
    }

    #[test]
    fn test_relative_l2_error_nonzero() {
        let err = SpectralConservation::relative_l2_error(&[1.0, 0.0], &[0.0, 1.0]);
        // diff = sqrt(2), norm_a = 1 → error = sqrt(2)
        assert!((err - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_complex64_multiply() {
        let a = Complex64::new(1.0, 1.0);
        let b = Complex64::new(0.0, 1.0); // i
        let c = a.scale(b); // (1+i)*i = -1+i
        assert!((c.re - (-1.0)).abs() < 1e-10);
        assert!((c.im - 1.0).abs() < 1e-10);
    }
}

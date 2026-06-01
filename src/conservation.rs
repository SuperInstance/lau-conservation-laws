//! Conservation law types: scalar, vector, and tensor conserved quantities.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use std::fmt;

/// A conserved quantity tracked through discrete time steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConservedQuantity {
    Scalar(f64),
    Vector(DVector<f64>),
    Tensor(DMatrix<f64>),
}

impl ConservedQuantity {
    /// Compute the total (sum of all components).
    pub fn total(&self) -> f64 {
        match self {
            ConservedQuantity::Scalar(s) => *s,
            ConservedQuantity::Vector(v) => v.iter().sum(),
            ConservedQuantity::Tensor(m) => m.iter().sum(),
        }
    }

    /// Number of degrees of freedom.
    pub fn dof(&self) -> usize {
        match self {
            ConservedQuantity::Scalar(_) => 1,
            ConservedQuantity::Vector(v) => v.nrows(),
            ConservedQuantity::Tensor(m) => m.nrows() * m.ncols(),
        }
    }

    /// Add another conserved quantity (must be same variant).
    pub fn add(&self, other: &ConservedQuantity) -> Option<ConservedQuantity> {
        match (self, other) {
            (ConservedQuantity::Scalar(a), ConservedQuantity::Scalar(b)) => {
                Some(ConservedQuantity::Scalar(a + b))
            }
            (ConservedQuantity::Vector(a), ConservedQuantity::Vector(b)) if a.nrows() == b.nrows() => {
                Some(ConservedQuantity::Vector(a + b))
            }
            (ConservedQuantity::Tensor(a), ConservedQuantity::Tensor(b))
                if a.nrows() == b.nrows() && a.ncols() == b.ncols() =>
            {
                Some(ConservedQuantity::Tensor(a + b))
            }
            _ => None,
        }
    }

    /// Subtract another conserved quantity.
    pub fn sub(&self, other: &ConservedQuantity) -> Option<ConservedQuantity> {
        match (self, other) {
            (ConservedQuantity::Scalar(a), ConservedQuantity::Scalar(b)) => {
                Some(ConservedQuantity::Scalar(a - b))
            }
            (ConservedQuantity::Vector(a), ConservedQuantity::Vector(b)) if a.nrows() == b.nrows() => {
                Some(ConservedQuantity::Vector(a - b))
            }
            (ConservedQuantity::Tensor(a), ConservedQuantity::Tensor(b))
                if a.nrows() == b.nrows() && a.ncols() == b.ncols() =>
            {
                Some(ConservedQuantity::Tensor(a - b))
            }
            _ => None,
        }
    }

    /// Scale by a factor.
    pub fn scale(&self, factor: f64) -> ConservedQuantity {
        match self {
            ConservedQuantity::Scalar(s) => ConservedQuantity::Scalar(s * factor),
            ConservedQuantity::Vector(v) => ConservedQuantity::Vector(v.scale(factor)),
            ConservedQuantity::Tensor(m) => ConservedQuantity::Tensor(m.scale(factor)),
        }
    }

    /// L2 norm of the quantity.
    pub fn norm(&self) -> f64 {
        match self {
            ConservedQuantity::Scalar(s) => s.abs(),
            ConservedQuantity::Vector(v) => v.norm(),
            ConservedQuantity::Tensor(m) => {
                let mut sum = 0.0;
                for val in m.iter() {
                    sum += val * val;
                }
                sum.sqrt()
            }
        }
    }

    /// Check if this is a scalar variant.
    pub fn is_scalar(&self) -> bool {
        matches!(self, ConservedQuantity::Scalar(_))
    }

    /// Check if this is a vector variant.
    pub fn is_vector(&self) -> bool {
        matches!(self, ConservedQuantity::Vector(_))
    }

    /// Check if this is a tensor variant.
    pub fn is_tensor(&self) -> bool {
        matches!(self, ConservedQuantity::Tensor(_))
    }

    /// Zero quantity of the same shape.
    pub fn zero_like(&self) -> ConservedQuantity {
        match self {
            ConservedQuantity::Scalar(_) => ConservedQuantity::Scalar(0.0),
            ConservedQuantity::Vector(v) => {
                ConservedQuantity::Vector(DVector::zeros(v.nrows()))
            }
            ConservedQuantity::Tensor(m) => {
                ConservedQuantity::Tensor(DMatrix::zeros(m.nrows(), m.ncols()))
            }
        }
    }
}

impl fmt::Display for ConservedQuantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConservedQuantity::Scalar(s) => write!(f, "Scalar({:.6})", s),
            ConservedQuantity::Vector(v) => write!(f, "Vector(d={}, total={:.6})", v.nrows(), v.iter().sum::<f64>()),
            ConservedQuantity::Tensor(m) => write!(f, "Tensor({}x{}, total={:.6})", m.nrows(), m.ncols(), m.iter().sum::<f64>()),
        }
    }
}

impl PartialEq for ConservedQuantity {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ConservedQuantity::Scalar(a), ConservedQuantity::Scalar(b)) => (a - b).abs() < 1e-10,
            (ConservedQuantity::Vector(a), ConservedQuantity::Vector(b)) => {
                a.nrows() == b.nrows() && (a - b).norm() < 1e-10
            }
            (ConservedQuantity::Tensor(a), ConservedQuantity::Tensor(b)) => {
                a.shape() == b.shape() && (a - b).norm() < 1e-10
            }
            _ => false,
        }
    }
}

/// A conservation law: dQ/dt = -∇·F(Q) where Q is conserved and F is the flux.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationLaw {
    /// Name of the conserved quantity.
    pub name: String,
    /// The quantity being conserved.
    pub quantity: ConservedQuantity,
    /// Description of the conservation law.
    pub description: String,
}

impl ConservationLaw {
    pub fn new(name: &str, quantity: ConservedQuantity) -> Self {
        ConservationLaw {
            name: name.to_string(),
            quantity,
            description: format!("Conservation of {}", name),
        }
    }

    /// Verify that the law holds between two states.
    pub fn verify(&self, before: &ConservedQuantity, after: &ConservedQuantity, tolerance: f64) -> bool {
        let diff = before.sub(after);
        match diff {
            Some(d) => d.norm() < tolerance,
            None => false,
        }
    }

    /// Compute conservation error: ||Q_after - Q_before||.
    pub fn error(&self, before: &ConservedQuantity, after: &ConservedQuantity) -> Option<f64> {
        before.sub(after).map(|d| d.norm())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_total() {
        let q = ConservedQuantity::Scalar(42.0);
        assert!((q.total() - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_total() {
        let q = ConservedQuantity::Vector(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        assert!((q.total() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_total() {
        let q = ConservedQuantity::Tensor(DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]));
        assert!((q.total() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_scalars() {
        let a = ConservedQuantity::Scalar(3.0);
        let b = ConservedQuantity::Scalar(4.0);
        let c = a.add(&b).unwrap();
        assert_eq!(c, ConservedQuantity::Scalar(7.0));
    }

    #[test]
    fn test_add_vectors() {
        let a = ConservedQuantity::Vector(DVector::from_vec(vec![1.0, 2.0]));
        let b = ConservedQuantity::Vector(DVector::from_vec(vec![3.0, 4.0]));
        let c = a.add(&b).unwrap();
        assert_eq!(c, ConservedQuantity::Vector(DVector::from_vec(vec![4.0, 6.0])));
    }

    #[test]
    fn test_add_mismatched_fails() {
        let a = ConservedQuantity::Scalar(1.0);
        let b = ConservedQuantity::Vector(DVector::from_vec(vec![1.0]));
        assert!(a.add(&b).is_none());
    }

    #[test]
    fn test_sub_scalars() {
        let a = ConservedQuantity::Scalar(10.0);
        let b = ConservedQuantity::Scalar(3.0);
        let c = a.sub(&b).unwrap();
        assert_eq!(c, ConservedQuantity::Scalar(7.0));
    }

    #[test]
    fn test_scale() {
        let q = ConservedQuantity::Scalar(5.0);
        let s = q.scale(3.0);
        assert_eq!(s, ConservedQuantity::Scalar(15.0));
    }

    #[test]
    fn test_norm_scalar() {
        let q = ConservedQuantity::Scalar(-3.0);
        assert!((q.norm() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_vector() {
        let q = ConservedQuantity::Vector(DVector::from_vec(vec![3.0, 4.0]));
        assert!((q.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_dof() {
        assert_eq!(ConservedQuantity::Scalar(1.0).dof(), 1);
        assert_eq!(ConservedQuantity::Vector(DVector::zeros(5)).dof(), 5);
        assert_eq!(ConservedQuantity::Tensor(DMatrix::zeros(3, 4)).dof(), 12);
    }

    #[test]
    fn test_zero_like() {
        let v = ConservedQuantity::Vector(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        let z = v.zero_like();
        assert_eq!(z, ConservedQuantity::Vector(DVector::zeros(3)));
    }

    #[test]
    fn test_conservation_law_verify() {
        let law = ConservationLaw::new("mass", ConservedQuantity::Scalar(0.0));
        let before = ConservedQuantity::Scalar(100.0);
        let after = ConservedQuantity::Scalar(100.0);
        assert!(law.verify(&before, &after, 1e-10));
    }

    #[test]
    fn test_conservation_law_error() {
        let law = ConservationLaw::new("energy", ConservedQuantity::Scalar(0.0));
        let before = ConservedQuantity::Scalar(100.0);
        let after = ConservedQuantity::Scalar(99.5);
        let err = law.error(&before, &after).unwrap();
        assert!((err - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_is_scalar_vector_tensor() {
        let s = ConservedQuantity::Scalar(1.0);
        let v = ConservedQuantity::Vector(DVector::zeros(2));
        let t = ConservedQuantity::Tensor(DMatrix::zeros(2, 2));
        assert!(s.is_scalar() && !s.is_vector() && !s.is_tensor());
        assert!(!v.is_scalar() && v.is_vector() && !v.is_tensor());
        assert!(!t.is_scalar() && !t.is_vector() && t.is_tensor());
    }
}

//! Finite volume schemes: flux computation, stencil assembly, conservative discretization.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// A numerical flux across a cell face.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flux {
    pub value: f64,
    pub face: usize,
    pub left_cell: usize,
    pub right_cell: usize,
}

impl Flux {
    pub fn new(value: f64, face: usize, left: usize, right: usize) -> Self {
        Flux { value, face, left_cell: left, right_cell: right }
    }
}

/// A stencil describing the neighborhood of a cell for flux computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stencil {
    /// Cell indices (center cell first).
    pub cells: Vec<usize>,
    /// Weights for each cell.
    pub weights: Vec<f64>,
    /// Center cell index.
    pub center: usize,
}

impl Stencil {
    pub fn new(cells: Vec<usize>, weights: Vec<f64>, center: usize) -> Self {
        assert_eq!(cells.len(), weights.len());
        Stencil { cells, weights, center }
    }

    /// Standard 3-point stencil for 1D FV.
    pub fn three_point(center: usize, left: usize, right: usize) -> Self {
        Stencil {
            cells: vec![left, center, right],
            weights: vec![-0.5, 0.0, 0.5],
            center,
        }
    }

    /// 5-point stencil for 1D higher-order.
    pub fn five_point(cells: [usize; 5], center: usize) -> Self {
        Stencil {
            cells: cells.to_vec(),
            weights: vec![1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0],
            center,
        }
    }

    /// Check if the stencil is conservative (weights sum to zero for flux).
    pub fn is_conservative(&self, tolerance: f64) -> bool {
        let sum: f64 = self.weights.iter().sum();
        sum.abs() < tolerance
    }

    /// Apply stencil to a state vector, returning weighted sum.
    pub fn apply(&self, state: &DVector<f64>) -> f64 {
        self.cells.iter().zip(self.weights.iter()).map(|(&c, &w)| w * state[c]).sum()
    }
}

/// A finite volume scheme on a 1D grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiniteVolumeScheme {
    /// Number of cells.
    pub n_cells: usize,
    /// Cell values.
    pub state: DVector<f64>,
    /// Stencils for each interior cell.
    pub stencils: Vec<Stencil>,
    /// Time step.
    pub dt: f64,
    /// Cell size.
    pub dx: f64,
    /// Boundary condition type.
    pub boundary: BoundaryCondition,
}

/// Boundary condition types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryCondition {
    /// Periodic boundaries.
    Periodic,
    /// Fixed (Dirichlet) boundaries.
    Dirichlet(f64, f64),
    /// Zero-flux (Neumann) boundaries.
    Neumann,
    /// Reflecting boundaries.
    Reflecting,
}

impl FiniteVolumeScheme {
    pub fn new(n_cells: usize, initial: DVector<f64>, dx: f64, dt: f64, boundary: BoundaryCondition) -> Self {
        assert_eq!(initial.nrows(), n_cells);
        let stencils = Self::build_stencils(n_cells, &boundary);
        FiniteVolumeScheme {
            n_cells,
            state: initial,
            stencils,
            dt,
            dx,
            boundary,
        }
    }

    /// Build stencils for all interior cells.
    fn build_stencils(n_cells: usize, boundary: &BoundaryCondition) -> Vec<Stencil> {
        let mut stencils = Vec::new();
        for i in 0..n_cells {
            let (left, right) = match boundary {
                BoundaryCondition::Periodic => {
                    let l = if i == 0 { n_cells - 1 } else { i - 1 };
                    let r = if i == n_cells - 1 { 0 } else { i + 1 };
                    (l, r)
                }
                BoundaryCondition::Neumann | BoundaryCondition::Reflecting => {
                    let l = if i == 0 { i } else { i - 1 };
                    let r = if i == n_cells - 1 { i } else { i + 1 };
                    (l, r)
                }
                BoundaryCondition::Dirichlet(_, _) => {
                    let l = if i == 0 { i } else { i - 1 };
                    let r = if i == n_cells - 1 { i } else { i + 1 };
                    (l, r)
                }
            };
            stencils.push(Stencil::three_point(i, left, right));
        }
        stencils
    }

    /// Compute all face fluxes.
    pub fn compute_fluxes(&self, velocity: f64) -> Vec<Flux> {
        let mut fluxes = Vec::new();
        // n_cells + 1 faces for n_cells cells
        for face in 0..=self.n_cells {
            let (left, right) = match self.boundary {
                BoundaryCondition::Periodic => {
                    let l = if face == 0 { self.n_cells - 1 } else { face - 1 };
                    let r = if face == self.n_cells { 0 } else { face };
                    (l, r)
                }
                BoundaryCondition::Neumann => {
                    let l = if face == 0 { 0 } else { face - 1 };
                    let r = if face == self.n_cells { self.n_cells - 1 } else { face };
                    (l, r)
                }
                BoundaryCondition::Reflecting => {
                    let l = if face == 0 { 0 } else { face - 1 };
                    let r = if face == self.n_cells { self.n_cells - 1 } else { face };
                    (l, r)
                }
                BoundaryCondition::Dirichlet(_, _) => {
                    let l = if face == 0 { 0 } else { face - 1 };
                    let r = if face == self.n_cells { self.n_cells - 1 } else { face };
                    (l, r)
                }
            };

            // Upwind flux
            let flux_value = if velocity >= 0.0 {
                velocity * self.state[left]
            } else {
                velocity * self.state[right]
            };
            fluxes.push(Flux::new(flux_value, face, left, right));
        }
        fluxes
    }

    /// Compute total conserved quantity (sum of all cell values).
    pub fn total(&self) -> f64 {
        self.state.iter().sum()
    }

    /// Advance one time step using the advection equation with upwind flux.
    /// Returns the new state.
    pub fn step(&self, velocity: f64) -> DVector<f64> {
        let fluxes = self.compute_fluxes(velocity);
        let mut new_state = self.state.clone();
        for i in 0..self.n_cells {
            let flux_left = fluxes[i].value;
            let flux_right = fluxes[i + 1].value;
            // dQ/dt = -(F_right - F_left) / dx
            new_state[i] = self.state[i] - self.dt / self.dx * (flux_right - flux_left);
        }
        new_state
    }

    /// Run for n_steps and return final state.
    pub fn evolve(&mut self, velocity: f64, n_steps: usize) {
        for _ in 0..n_steps {
            self.state = self.step(velocity);
        }
    }

    /// Assemble the flux matrix A such that dQ/dt = A*Q.
    pub fn flux_matrix(&self, velocity: f64) -> DMatrix<f64> {
        let n = self.n_cells;
        let alpha = velocity * self.dt / self.dx;
        let mut a = DMatrix::zeros(n, n);

        for i in 0..n {
            if velocity >= 0.0 {
                // Upwind: flux depends on left cell
                a[(i, i)] += 1.0 - alpha;
                if i > 0 || matches!(self.boundary, BoundaryCondition::Periodic) {
                    let left = if i == 0 { n - 1 } else { i - 1 };
                    a[(i, left)] += alpha;
                }
            } else {
                a[(i, i)] += 1.0 + alpha;
                if i < n - 1 || matches!(self.boundary, BoundaryCondition::Periodic) {
                    let right = if i == n - 1 { 0 } else { i + 1 };
                    a[(i, right)] -= alpha;
                }
            }
        }
        a
    }

    /// Check conservation: total before and after should match.
    pub fn check_conservation(&self, velocity: f64) -> bool {
        let total_before = self.total();
        let new_state = self.step(velocity);
        let total_after: f64 = new_state.iter().sum();
        (total_before - total_after).abs() < 1e-8
    }
}

/// Lax-Friedrichs flux: F_LF = 0.5*(F(UL) + F(UR)) - 0.5*λ*(UR - UL)
pub fn lax_friedrichs_flux(ul: f64, ur: f64, velocity: f64, max_wave_speed: f64) -> f64 {
    0.5 * (velocity * ul + velocity * ur) - 0.5 * max_wave_speed * (ur - ul)
}

/// Rusanov (local Lax-Friedrichs) flux.
pub fn rusanov_flux(ul: f64, ur: f64, velocity: f64) -> f64 {
    let max_speed = velocity.abs();
    lax_friedrichs_flux(ul, ur, velocity, max_speed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stencil_three_point() {
        let s = Stencil::three_point(1, 0, 2);
        assert_eq!(s.cells, vec![0, 1, 2]);
        assert_eq!(s.center, 1);
    }

    #[test]
    fn test_stencil_is_conservative() {
        let s = Stencil::three_point(1, 0, 2);
        assert!(s.is_conservative(1e-10));
    }

    #[test]
    fn test_stencil_apply() {
        let s = Stencil::three_point(1, 0, 2);
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        // -0.5*1 + 0*2 + 0.5*3 = 1.0
        let result = s.apply(&state);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_five_point_stencil_conservative() {
        let s = Stencil::five_point([0, 1, 2, 3, 4], 2);
        assert!(s.is_conservative(1e-10));
    }

    #[test]
    fn test_fv_periodic_conservation() {
        let n = 10;
        let initial = DVector::from_vec((0..n).map(|i| (i as f64 + 1.0).sin()).collect());
        let scheme = FiniteVolumeScheme::new(n, initial, 1.0, 0.01, BoundaryCondition::Periodic);
        assert!(scheme.check_conservation(1.0));
    }

    #[test]
    fn test_fv_total() {
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let scheme = FiniteVolumeScheme::new(4, state, 1.0, 0.1, BoundaryCondition::Periodic);
        assert!((scheme.total() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_fv_flux_matrix_size() {
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let scheme = FiniteVolumeScheme::new(3, state, 1.0, 0.1, BoundaryCondition::Periodic);
        let a = scheme.flux_matrix(1.0);
        assert_eq!(a.nrows(), 3);
        assert_eq!(a.ncols(), 3);
    }

    #[test]
    fn test_lax_friedrichs_flux() {
        let f = lax_friedrichs_flux(1.0, 2.0, 1.0, 1.0);
        // 0.5*(1*1 + 1*2) - 0.5*1*(2-1) = 1.5 - 0.5 = 1.0
        assert!((f - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rusanov_flux() {
        let f = rusanov_flux(1.0, 2.0, 1.0);
        assert!((f - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_evolve_preserves_total_periodic() {
        let n = 20;
        let initial = DVector::from_vec((0..n).map(|i| (i as f64 * 0.3).sin()).collect());
        let total_before: f64 = initial.iter().sum();
        let mut scheme = FiniteVolumeScheme::new(n, initial, 0.1, 0.01, BoundaryCondition::Periodic);
        scheme.evolve(0.5, 100);
        let total_after: f64 = scheme.state.iter().sum();
        assert!((total_before - total_after).abs() < 1e-6);
    }

    #[test]
    fn test_compute_fluxes_count() {
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let scheme = FiniteVolumeScheme::new(3, state, 1.0, 0.1, BoundaryCondition::Periodic);
        let fluxes = scheme.compute_fluxes(1.0);
        // n_cells + 1 = 4 faces
        assert_eq!(fluxes.len(), 4);
    }

    #[test]
    fn test_neumann_boundary() {
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let scheme = FiniteVolumeScheme::new(3, state, 1.0, 0.1, BoundaryCondition::Neumann);
        let fluxes = scheme.compute_fluxes(1.0);
        // At boundaries, left/right cell is the same → zero flux contribution
        assert_eq!(fluxes[0].left_cell, fluxes[0].right_cell);
    }
}

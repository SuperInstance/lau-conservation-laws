//! CUDAclaw integration: connecting conservation laws to cell-agent execution.
//!
//! This module shows that CUDAclaw's cell agents execute a finite-volume scheme,
//! linking the conservation law framework to GPU-accelerated agent simulations.

use serde::{Serialize, Deserialize};

/// A CUDAclaw cell agent that carries a conserved quantity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellAgent {
    pub id: usize,
    pub position: (f64, f64),
    pub quantity: f64,
    pub flux_in: f64,
    pub flux_out: f64,
}

impl CellAgent {
    pub fn new(id: usize, position: (f64, f64), quantity: f64) -> Self {
        CellAgent { id, position, quantity, flux_in: 0.0, flux_out: 0.0 }
    }

    /// Net flux change.
    pub fn net_flux(&self) -> f64 {
        self.flux_in - self.flux_out
    }

    /// Apply flux update.
    pub fn apply_flux(&mut self) {
        self.quantity += self.net_flux();
        self.flux_in = 0.0;
        self.flux_out = 0.0;
    }
}

/// A grid of cell agents executing a finite-volume scheme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellGrid {
    pub cells: Vec<CellAgent>,
    pub dx: f64,
    pub dy: f64,
}

impl CellGrid {
    pub fn new_1d(_n: usize, dx: f64, initial: &[f64]) -> Self {
        let cells: Vec<CellAgent> = initial.iter().enumerate().map(|(i, &q)| {
            CellAgent::new(i, (i as f64 * dx, 0.0), q)
        }).collect();
        CellGrid { cells, dx, dy: 0.0 }
    }

    /// Total conserved quantity.
    pub fn total(&self) -> f64 {
        self.cells.iter().map(|c| c.quantity).sum()
    }

    /// Perform one FV step with periodic boundaries.
    pub fn step_periodic(&mut self, velocity: f64, dt: f64) {
        let n = self.cells.len();
        // Compute fluxes first
        let fluxes: Vec<f64> = (0..n).map(|i| {
            let right = (i + 1) % n;
            if velocity >= 0.0 {
                velocity * self.cells[i].quantity
            } else {
                velocity * self.cells[right].quantity
            }
        }).collect();

        // Apply conservative update
        for i in 0..n {
            let left = if i == 0 { n - 1 } else { i - 1 };
            self.cells[i].quantity -= dt / self.dx * (fluxes[i] - fluxes[left]);
        }
    }

    /// Verify conservation.
    pub fn check_conservation(&self, initial_total: f64, tolerance: f64) -> bool {
        (self.total() - initial_total).abs() < tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_agent_create() {
        let c = CellAgent::new(0, (0.0, 0.0), 42.0);
        assert!((c.quantity - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_cell_agent_net_flux() {
        let mut c = CellAgent::new(0, (0.0, 0.0), 100.0);
        c.flux_in = 10.0;
        c.flux_out = 5.0;
        assert!((c.net_flux() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_cell_agent_apply_flux() {
        let mut c = CellAgent::new(0, (0.0, 0.0), 100.0);
        c.flux_in = 10.0;
        c.flux_out = 3.0;
        c.apply_flux();
        assert!((c.quantity - 107.0).abs() < 1e-10);
    }

    #[test]
    fn test_cell_grid_1d() {
        let grid = CellGrid::new_1d(5, 1.0, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((grid.total() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_cell_grid_periodic_conservation() {
        let initial: Vec<f64> = (0..10).map(|i| (i as f64 * 0.5).sin()).collect();
        let total: f64 = initial.iter().sum();
        let mut grid = CellGrid::new_1d(10, 0.1, &initial);
        for _ in 0..50 {
            grid.step_periodic(0.5, 0.01);
        }
        assert!(grid.check_conservation(total, 1e-6));
    }

    #[test]
    fn test_cell_grid_fluxes_conserve() {
        let initial = vec![10.0, 20.0, 30.0, 40.0];
        let total: f64 = initial.iter().sum();
        let mut grid = CellGrid::new_1d(4, 1.0, &initial);
        grid.step_periodic(1.0, 0.1);
        assert!((grid.total() - total).abs() < 1e-10);
    }
}

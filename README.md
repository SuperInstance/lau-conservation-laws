# lau-conservation-laws

> Conservation laws in discrete systems — Noether's theorem, finite-volume schemes, and CRDT conservation for agent fleets

## What This Does

Conservation laws in discrete systems — Noether's theorem, finite-volume schemes, and CRDT conservation for agent fleets. Part of the PLATO/LAU ecosystem — a mathematically rigorous framework for building educational agents that learn, teach, and evolve.

## The Key Idea

This crate implements the core abstractions needed for its domain, with a focus on correctness, composability, and conservation guarantees. Every public type is serializable (serde), every algorithm is tested, and every invariant is verified.

## Install

```bash
cargo add lau-conservation-laws
```

## Quick Start

See the API Reference below for complete usage. Key entry points:

```rust
use lau_conservation_laws::*;
// See types and methods below for complete usage
```

## API Reference

```rust
pub struct EntropyTracker 
    pub fn new(tolerance: f64) -> Self 
    pub fn shannon_entropy(probs: &[f64]) -> f64 
    pub fn boltzmann_entropy(n_microstates: u64) -> f64 
    pub fn record(&mut self, entropy: f64) 
    pub fn is_non_decreasing(&self) -> bool 
    pub fn total_production(&self) -> f64 
    pub fn production_rate(&self) -> f64 
    pub fn entropy_from_state(state: &[f64]) -> f64 
    pub fn is_production_positive(&self) -> bool 
    pub fn reset(&mut self) 
pub struct VerificationResult 
pub struct ConservationVerifier 
    pub fn new(law_name: &str, tolerance: f64) -> Self 
    pub fn record(&mut self, total: f64) 
    pub fn verify(&self) -> VerificationResult 
    pub fn verify_fv_scheme(
    pub fn max_deviation(&self) -> f64 
    pub fn drift_rate(&self) -> f64 
    pub fn reset(&mut self) 
pub struct Symmetry 
    pub fn new(name: &str, generator: DMatrix<f64>, conserved_name: &str) -> Self 
    pub fn transform(&self, state: &DVector<f64>, epsilon: f64) -> DVector<f64> 
    pub fn noether_charge(&self, state: &DVector<f64>) -> f64 
    pub fn is_antisymmetric(&self, tolerance: f64) -> bool 
    pub fn is_symmetric(&self, tolerance: f64) -> bool 
pub struct NoetherPair 
    pub fn new(symmetry: Symmetry, tolerance: f64) -> Self 
    pub fn verify_conservation(&self, state_before: &DVector<f64>, state_after: &DVector<f64>) -> bool 
    pub fn conservation_error(&self, state_before: &DVector<f64>, state_after: &DVector<f64>) -> f64 
pub fn discretize_symmetry(symmetry: &Symmetry, n_steps: usize) -> Vec<DMatrix<f64>> 
pub fn standard_2d_noether_pairs(tolerance: f64) -> Vec<NoetherPair> 
pub struct CellAgent 
    pub fn new(id: usize, position: (f64, f64), quantity: f64) -> Self 
    pub fn net_flux(&self) -> f64 
    pub fn apply_flux(&mut self) 
pub struct CellGrid 
    pub fn new_1d(_n: usize, dx: f64, initial: &[f64]) -> Self 
    pub fn total(&self) -> f64 
    pub fn step_periodic(&mut self, velocity: f64, dt: f64) 
    pub fn check_conservation(&self, initial_total: f64, tolerance: f64) -> bool 
pub enum ConservedQuantity 
    pub fn total(&self) -> f64 
    pub fn dof(&self) -> usize 
    pub fn add(&self, other: &ConservedQuantity) -> Option<ConservedQuantity> 
    pub fn sub(&self, other: &ConservedQuantity) -> Option<ConservedQuantity> 
    pub fn scale(&self, factor: f64) -> ConservedQuantity 
    pub fn norm(&self) -> f64 
    pub fn is_scalar(&self) -> bool 
    pub fn is_vector(&self) -> bool 
    pub fn is_tensor(&self) -> bool 
    pub fn zero_like(&self) -> ConservedQuantity 
pub struct ConservationLaw 
    pub fn new(name: &str, quantity: ConservedQuantity) -> Self 
    pub fn verify(&self, before: &ConservedQuantity, after: &ConservedQuantity, tolerance: f64) -> bool 
    pub fn error(&self, before: &ConservedQuantity, after: &ConservedQuantity) -> Option<f64> 
pub struct Agent 
    pub fn new(id: &str, charge: f64) -> Self 
pub struct AgentFleet 
    pub fn new(agents: Vec<Agent>) -> Self 
```

## How It Works

Read the source in `src/` for full implementation details. All algorithms are documented with inline comments explaining the mathematical foundations.

## The Math

This crate implements formal mathematical constructs. See the source documentation for theorem statements and proofs of correctness.

## Testing

**89 tests** covering construction, serialization, correctness properties, edge cases, and composability with other lau-* crates.

## License

MIT

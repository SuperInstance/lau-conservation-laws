# lau-conservation-laws

**Conservation laws in discrete systems — Noether's theorem, finite-volume schemes, and CRDT conservation for agent fleets.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

89 tests · Rust · `nalgebra` + `serde`

---

## What This Does

This crate proves that when autonomous agents move through state space, their collective behavior obeys conservation laws — the same kind that govern energy, momentum, and charge in physics. It provides:

1. **Noether's theorem, discretized** — map every continuous symmetry to a conserved quantity and verify it holds step-by-step
2. **Finite-volume schemes** — conservative numerical flux on 1-D grids with periodic boundaries
3. **CRDT merge conservation** — distributed CRDT operations that preserve a system-wide "charge" invariant
4. **Spectral (Parseval) conservation** — L²-norm tracking across physical and Fourier space
5. **Entropy production** — second-law compliance checking for discrete systems
6. **Agent fleet charge tracking** — transfer charge between agents and verify the total is invariant
7. **CUDAclaw cell grids** — finite-volume cell agents with upwind flux and conservation verification

## Key Idea

Every discrete system has symmetries. Every symmetry has a conserved current. If you can name the symmetry (time translation → energy, rotation → angular momentum, phase → charge), you can *prove* the quantity is conserved — and this crate checks that proof at runtime.

For agent fleets, the conserved quantity is *charge*: an abstract resource that can be created (fleet initialization), transferred (agent-to-agent), but never destroyed or created after initialization. CRDT merges between nodes respect this: the total system charge is invariant under merge.

## Install

```toml
[dependencies]
lau-conservation-laws = { git = "https://github.com/SuperInstance/lau-conservation-laws" }
```

Or clone and build locally:

```bash
git clone https://github.com/SuperInstance/lau-conservation-laws.git
cd lau-conservation-laws
cargo build
cargo test
```

## Quick Start

### Noether Conservation Check

```rust
use lau_conservation_laws::NoetherPair;
use nalgebra::{DVector, DMatrix};

// Rotation symmetry → angular momentum conservation
let n = 4;
let mut generator = DMatrix::zeros(n, n);
generator[(0,1)] = -1.0; generator[(1,0)] = 1.0;
generator[(2,3)] = -1.0; generator[(3,2)] = 1.0;

let symmetry = lau_conservation_laws::noether::Symmetry::new(
    "rotation",
    generator,
    "angular_momentum",
);

let pair = NoetherPair::new(symmetry, 1e-10);

let state_before = DVector::from_vec(vec![1.0, 0.0, 0.0, 1.0]);
let state_after  = DVector::from_vec(vec![0.0, 1.0, -1.0, 0.0]);

assert!(pair.verify_conservation(&state_before, &state_after));
```

### Agent Fleet Charge Transfer

```rust
use lau_conservation_laws::{Agent, AgentFleet};

let fleet = AgentFleet::new(vec![
    Agent::new("alice", 100.0),
    Agent::new("bob",   50.0),
]);
assert_eq!(fleet.total_charge(), 150.0);

fleet.transfer("alice", "bob", 25.0).unwrap();
assert_eq!(fleet.total_charge(), 150.0); // conserved!
assert!(fleet.is_conserved(1e-10));
```

### CUDAclaw Cell Grid (1-D FV)

```rust
use lau_conservation_laws::cudaclaw::CellGrid;

let initial = vec![1.0, 2.0, 3.0, 4.0];
let mut grid = CellGrid::new_1d(4, 0.1, &initial);
let total_0 = grid.total();

for _ in 0..100 {
    grid.step_periodic(1.0, 0.01);
}
assert!(grid.check_conservation(total_0, 1e-10));
```

## API Reference

| Module | Key Types | Tests | Purpose |
|--------|-----------|-------|---------|
| `conservation` | `ConservedQuantity` (Scalar/Vector/Tensor) | 15 | Algebra of conserved quantities |
| `noether` | `Symmetry`, `NoetherPair` | 9 | Discretized Noether's theorem |
| `finite_volume` | `FiniteVolumeScheme`, `Stencil`, `Flux` | 12 | 1-D conservative FV discretization |
| `verification` | `ConservationVerifier`, `VerificationResult` | 10 | Multi-step conservation tracking |
| `crdt` | `SmartCRDT`, `CrdtOp` | 8 | Distributed merge that preserves charge |
| `spectral` | `SpectralConservation` | 9 | Parseval / L² norm conservation |
| `agent` | `Agent`, `AgentFleet` | 8 | Charge-conserving agent transfers |
| `entropy` | `EntropyTracker` | 12 | Shannon/Boltzmann entropy, second-law checks |
| `cudaclaw` | `CellAgent`, `CellGrid` | 6 | GPU-ready cell agents with upwind flux |

All types derive `Serialize`/`Deserialize` and `Clone`.

## How It Works

### 1. Conserved Quantity Algebra

`ConservedQuantity` is an enum over `Scalar(f64)`, `Vector(DVector<f64>)`, and `Tensor(DMatrix<f64>)`. Every operation (add, sub, scale) preserves the variant shape. The `total()` method gives the single scalar invariant.

### 2. Noether's Theorem

For a state vector **Q** and a symmetry generator matrix **G**, the Noether charge is:

```
J = ½ Qᵀ(G + Gᵀ)Q
```

The `NoetherPair` stores a `Symmetry` (name + generator + conserved quantity name) and a tolerance. `verify_conservation(before, after)` computes J at both states and checks |ΔJ| < ε.

### 3. Finite-Volume Scheme

A 1-D grid of N cells with periodic boundaries. Each interior cell has a `Stencil` (3-point or 5-point). The upwind flux at face i is:

```
F_{i+½} = v · Q_i    if v ≥ 0
         = v · Q_{i+1} if v < 0
```

Conservative update:

```
Q_i^{n+1} = Q_i^n − (Δt/Δx)(F_{i+½} − F_{i−½})
```

Summing over all cells, the boundary fluxes telescope to zero → total is conserved exactly.

### 4. CRDT Conservation

Each `SmartCRDT` node has a local `charge` and an operation log. `apply_local()` adds charge; `merge()` replicates the log without double-counting; `transfer()` moves charge between nodes. The system-wide sum of all node charges is invariant under all three operations.

### 5. Spectral Conservation

Parseval's theorem: ‖f̂‖₂ = ‖f‖₂. The `SpectralConservation` tracker records both physical-space and spectral-space L² norms and verifies they agree. For linear PDEs with constant coefficients, spectral methods exactly preserve L².

### 6. Entropy Tracking

`EntropyTracker` records Shannon or Boltzmann entropy at each step. The second law requires non-decrease: `S(t₂) ≥ S(t₁)` for all t₂ > t₁. The `is_non_decreasing()` method checks this over the recorded history.

## The Math

### Noether's Theorem (Classical → Discrete)

In the continuous setting, for a Lagrangian system with action S = ∫ L(q, q̇, t) dt, every continuous symmetry δq = ε·G·q produces a conserved current:

```
J = ∂L/∂q̇ · δq = p · G · q
```

Discretized: the generator G becomes a matrix, the state q becomes a vector, and J becomes QᵀGQ/2. Conservation is checked to within numerical tolerance ε at each step.

### Finite-Volume Conservation

The fundamental theorem of conservation laws: ∂ₜu + ∇·F(u) = 0 integrated over a cell Ωᵢ gives:

```
d/dt ∫_{Ωᵢ} u dx = −∮_{∂Ωᵢ} F·n dS
```

Discretized: Qᵢⁿ⁺¹ = Qᵢⁿ − (Δt/Δx)(Fᵢ₊½ − Fᵢ₋½). Summing over all cells, the flux terms telescope. With periodic boundaries, the total Σᵢ Qᵢ is exactly conserved.

### CRDT Merge Invariant

For a fleet of N CRDT nodes, the system charge C = Σₙ chargeₙ is invariant under:
- **Local apply**: chargeₙ ← chargeₙ + δ (C increases by δ — only at initialization)
- **Merge**: log replication only, no charge change (ΔC = 0)
- **Transfer**: chargeₐ − δ, chargeᵦ + δ (ΔC = 0)

### Parseval's Theorem

For discrete Fourier transform F̂[k] = Σₙ f[n]e^{-2πikn/N}:

```
Σₖ |F̂[k]|² = N · Σₙ |f[n]|²
```

Spectral methods for linear PDEs preserve this exactly (each Fourier mode evolves independently with |growth| = 1).

## License

MIT

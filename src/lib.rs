//! # Conservation Laws in Discrete Systems
//!
//! Proves that CUDAclaw's cell agents execute a finite-volume scheme by establishing
//! conservation laws across discrete systems: Noether's theorem, CRDT merge as
//! conservative updates, spectral conservation, and entropy production tracking.

pub mod conservation;
pub mod noether;
pub mod finite_volume;
pub mod verification;
pub mod crdt;
pub mod spectral;
pub mod agent;
pub mod entropy;
pub mod cudaclaw;

pub use conservation::*;
pub use noether::NoetherPair;
pub use finite_volume::{FiniteVolumeScheme, Stencil, Flux};
pub use verification::ConservationVerifier;
pub use crdt::SmartCRDT as CrdtConservation;
pub use spectral::SpectralConservation;
pub use agent::AgentConservation;
pub use entropy::EntropyTracker;

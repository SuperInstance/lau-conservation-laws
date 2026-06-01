//! Agent fleet charge tracking: conservation of total charge across agent populations.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// An individual agent carrying a charge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    pub charge: f64,
    pub metadata: HashMap<String, String>,
}

impl Agent {
    pub fn new(id: &str, charge: f64) -> Self {
        Agent {
            id: id.to_string(),
            charge,
            metadata: HashMap::new(),
        }
    }
}

/// A fleet of agents whose total charge should be conserved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFleet {
    pub agents: HashMap<String, Agent>,
    /// Track total charge for conservation verification.
    pub initial_total: f64,
}

impl AgentFleet {
    pub fn new(agents: Vec<Agent>) -> Self {
        let total: f64 = agents.iter().map(|a| a.charge).sum();
        let map: HashMap<String, Agent> = agents.into_iter().map(|a| (a.id.clone(), a)).collect();
        AgentFleet {
            agents: map,
            initial_total: total,
        }
    }

    /// Total charge across all agents.
    pub fn total_charge(&self) -> f64 {
        self.agents.values().map(|a| a.charge).sum()
    }

    /// Transfer charge between two agents (conservative).
    pub fn transfer(&mut self, from: &str, to: &str, amount: f64) -> Result<(), String> {
        if !self.agents.contains_key(from) || !self.agents.contains_key(to) {
            return Err("Agent not found".into());
        }
        let from_charge = self.agents.get(from).map(|a| a.charge).unwrap_or(0.0);
        if from_charge < amount - 1e-10 {
            return Err("Insufficient charge".into());
        }
        self.agents.get_mut(from).unwrap().charge -= amount;
        self.agents.get_mut(to).unwrap().charge += amount;
        Ok(())
    }

    /// Check if total charge is conserved relative to initial.
    pub fn is_conserved(&self, tolerance: f64) -> bool {
        (self.total_charge() - self.initial_total).abs() < tolerance
    }

    /// Conservation error.
    pub fn conservation_error(&self) -> f64 {
        (self.total_charge() - self.initial_total).abs()
    }

    /// Number of agents in the fleet.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Check if fleet is empty.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }
}

/// Agent conservation tracker for time-series analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConservation {
    pub fleet: AgentFleet,
    pub history: Vec<f64>,
    pub tolerance: f64,
}

impl AgentConservation {
    pub fn new(fleet: AgentFleet, tolerance: f64) -> Self {
        let total = fleet.total_charge();
        AgentConservation {
            fleet,
            history: vec![total],
            tolerance,
        }
    }

    /// Record current total.
    pub fn record(&mut self) {
        self.history.push(self.fleet.total_charge());
    }

    /// Verify conservation across all recorded values.
    pub fn verify(&self) -> bool {
        if self.history.is_empty() { return true; }
        let first = self.history[0];
        self.history.iter().all(|v| (v - first).abs() < self.tolerance)
    }

    /// Perform a transfer and record.
    pub fn transfer_and_record(&mut self, from: &str, to: &str, amount: f64) -> Result<(), String> {
        self.fleet.transfer(from, to, amount)?;
        self.record();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_create() {
        let a = Agent::new("a1", 42.0);
        assert!((a.charge - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_fleet_total() {
        let fleet = AgentFleet::new(vec![
            Agent::new("a", 10.0),
            Agent::new("b", 20.0),
            Agent::new("c", 30.0),
        ]);
        assert!((fleet.total_charge() - 60.0).abs() < 1e-10);
    }

    #[test]
    fn test_fleet_transfer() {
        let mut fleet = AgentFleet::new(vec![
            Agent::new("a", 100.0),
            Agent::new("b", 50.0),
        ]);
        fleet.transfer("a", "b", 30.0).unwrap();
        assert!((fleet.agents["a"].charge - 70.0).abs() < 1e-10);
        assert!((fleet.agents["b"].charge - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_transfer_conserves() {
        let mut fleet = AgentFleet::new(vec![
            Agent::new("a", 100.0),
            Agent::new("b", 50.0),
        ]);
        let before = fleet.total_charge();
        fleet.transfer("a", "b", 30.0).unwrap();
        assert!((fleet.total_charge() - before).abs() < 1e-10);
    }

    #[test]
    fn test_transfer_missing_agent() {
        let mut fleet = AgentFleet::new(vec![Agent::new("a", 100.0)]);
        assert!(fleet.transfer("a", "ghost", 10.0).is_err());
    }

    #[test]
    fn test_is_conserved() {
        let mut fleet = AgentFleet::new(vec![
            Agent::new("a", 100.0),
            Agent::new("b", 50.0),
        ]);
        fleet.transfer("a", "b", 25.0).unwrap();
        assert!(fleet.is_conserved(1e-10));
    }

    #[test]
    fn test_agent_conservation_tracker() {
        let fleet = AgentFleet::new(vec![
            Agent::new("a", 100.0),
            Agent::new("b", 50.0),
        ]);
        let mut tracker = AgentConservation::new(fleet, 1e-10);
        tracker.transfer_and_record("a", "b", 10.0).unwrap();
        tracker.transfer_and_record("b", "a", 5.0).unwrap();
        assert!(tracker.verify());
    }

    #[test]
    fn test_fleet_len() {
        let fleet = AgentFleet::new(vec![
            Agent::new("a", 1.0),
            Agent::new("b", 2.0),
        ]);
        assert_eq!(fleet.len(), 2);
        assert!(!fleet.is_empty());
    }
}

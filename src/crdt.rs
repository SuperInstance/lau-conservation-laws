//! CRDT conservation: SmartCRDT merge as a conservative update.
//!
//! In a distributed system modeled by CRDTs, merge operations should preserve
//! invariants analogous to conservation laws. This module defines a SmartCRDT
//! whose merge operation conserves a "charge" quantity.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// A CRDT operation that carries a charge delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrdtOp {
    pub id: String,
    pub charge_delta: f64,
    pub timestamp: u64,
}

/// A SmartCRDT node that tracks local charge and applies merges conservatively.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartCRDT {
    /// Unique node identifier.
    pub node_id: String,
    /// Current local charge value.
    pub charge: f64,
    /// Operation log (id → applied delta).
    pub log: HashMap<String, f64>,
    /// Last seen timestamp per node.
    pub last_seen: HashMap<String, u64>,
}

impl SmartCRDT {
    pub fn new(node_id: &str, initial_charge: f64) -> Self {
        SmartCRDT {
            node_id: node_id.to_string(),
            charge: initial_charge,
            log: HashMap::new(),
            last_seen: HashMap::new(),
        }
    }

    /// Apply a local operation (adds to charge).
    pub fn apply_local(&mut self, op: CrdtOp) {
        if !self.log.contains_key(&op.id) {
            self.charge += op.charge_delta;
            self.log.insert(op.id, op.charge_delta);
            self.last_seen.insert(self.node_id.clone(), op.timestamp);
        }
    }

    /// Merge operations from another node. Total charge across all nodes is conserved.
    pub fn merge(&mut self, other: &SmartCRDT) {
        for (id, &delta) in &other.log {
            if !self.log.contains_key(id) {
                self.log.insert(id.clone(), delta);
                // We do NOT change local charge — the operation was already accounted for
                // on the other node. Total system charge is preserved.
            }
        }
        // Update last-seen timestamps
        for (node, &ts) in &other.last_seen {
            self.last_seen.insert(node.clone(), ts);
        }
    }

    /// Transfer charge to another node (conservative: sum is preserved).
    pub fn transfer(&mut self, target: &mut SmartCRDT, amount: f64, op_id: &str, ts: u64) {
        self.charge -= amount;
        target.charge += amount;
        let op = CrdtOp {
            id: op_id.to_string(),
            charge_delta: -amount,
            timestamp: ts,
        };
        let op_target = CrdtOp {
            id: format!("{}_recv", op_id),
            charge_delta: amount,
            timestamp: ts,
        };
        self.log.insert(op.id, -amount);
        target.log.insert(op_target.id, amount);
    }

    /// Total charge in this node.
    pub fn total_charge(&self) -> f64 {
        self.charge
    }
}

/// Compute total charge across a fleet of CRDT nodes.
pub fn fleet_total(nodes: &[SmartCRDT]) -> f64 {
    nodes.iter().map(|n| n.charge).sum()
}

/// Verify that a merge operation preserved total fleet charge.
pub fn verify_merge_conservation(before: &[SmartCRDT], after: &[SmartCRDT], tolerance: f64) -> bool {
    let total_before = fleet_total(before);
    let total_after = fleet_total(after);
    (total_before - total_after).abs() < tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crdt_create() {
        let crdt = SmartCRDT::new("node-1", 100.0);
        assert!((crdt.charge - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_crdt_local_op() {
        let mut crdt = SmartCRDT::new("node-1", 100.0);
        crdt.apply_local(CrdtOp {
            id: "op1".into(),
            charge_delta: 10.0,
            timestamp: 1,
        });
        assert!((crdt.charge - 110.0).abs() < 1e-10);
    }

    #[test]
    fn test_crdt_idempotent() {
        let mut crdt = SmartCRDT::new("node-1", 100.0);
        let op = CrdtOp {
            id: "op1".into(),
            charge_delta: 10.0,
            timestamp: 1,
        };
        crdt.apply_local(op.clone());
        crdt.apply_local(op.clone()); // duplicate — should be ignored
        assert!((crdt.charge - 110.0).abs() < 1e-10);
    }

    #[test]
    fn test_crdt_merge_syncs_log() {
        let mut a = SmartCRDT::new("a", 100.0);
        let mut b = SmartCRDT::new("b", 50.0);
        a.apply_local(CrdtOp { id: "op1".into(), charge_delta: 5.0, timestamp: 1 });
        b.merge(&a);
        assert!(b.log.contains_key("op1"));
    }

    #[test]
    fn test_transfer_conserves() {
        let mut a = SmartCRDT::new("a", 100.0);
        let mut b = SmartCRDT::new("b", 50.0);
        let before = a.charge + b.charge;
        a.transfer(&mut b, 30.0, "t1", 1);
        let after = a.charge + b.charge;
        assert!((before - after).abs() < 1e-10);
        assert!((a.charge - 70.0).abs() < 1e-10);
        assert!((b.charge - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_fleet_total() {
        let nodes = vec![
            SmartCRDT::new("a", 100.0),
            SmartCRDT::new("b", 200.0),
            SmartCRDT::new("c", 300.0),
        ];
        assert!((fleet_total(&nodes) - 600.0).abs() < 1e-10);
    }

    #[test]
    fn test_verify_merge_conservation() {
        let before = vec![
            SmartCRDT::new("a", 100.0),
            SmartCRDT::new("b", 200.0),
        ];
        let mut after = before.clone();
        after[0].charge -= 50.0;
        after[1].charge += 50.0;
        assert!(verify_merge_conservation(&before, &after, 1e-10));
    }

    #[test]
    fn test_verify_merge_violation() {
        let before = vec![
            SmartCRDT::new("a", 100.0),
            SmartCRDT::new("b", 200.0),
        ];
        let mut after = before.clone();
        after[0].charge -= 50.0;
        after[1].charge += 40.0; // 10 lost
        assert!(!verify_merge_conservation(&before, &after, 1e-10));
    }
}

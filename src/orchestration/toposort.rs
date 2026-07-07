//! Shared level-batched topological sort (Kahn's algorithm).
//!
//! Both [`Dag`](super::Dag) and [`Workflow`](super::Workflow) schedule nodes in
//! dependency order, running each level concurrently. The scheduling algorithm
//! is identical; only the wording of the cycle-detection error differs, so it
//! is passed in by the caller.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{DaimonError, Result};

/// Groups `all_nodes` into levels such that every node in level *n* depends only
/// on nodes in levels `< n`. Nodes within a level have no dependencies on each
/// other and may run concurrently.
///
/// Returns [`DaimonError::Orchestration`] with `cycle_message` if the graph
/// contains a cycle (i.e. not every node could be scheduled).
pub(crate) fn topological_levels(
    all_nodes: &HashSet<String>,
    successors: &HashMap<String, Vec<String>>,
    predecessors: &HashMap<String, Vec<String>>,
    cycle_message: &str,
) -> Result<Vec<Vec<String>>> {
    let mut in_degree: HashMap<String, usize> = HashMap::new();
    for node in all_nodes {
        in_degree.insert(
            node.clone(),
            predecessors.get(node).map(|p| p.len()).unwrap_or(0),
        );
    }

    let mut queue: VecDeque<String> = VecDeque::new();
    for (node, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(node.clone());
        }
    }

    let mut levels: Vec<Vec<String>> = Vec::new();
    let mut visited = 0usize;

    while !queue.is_empty() {
        let level: Vec<String> = queue.drain(..).collect();
        visited += level.len();

        let mut next: VecDeque<String> = VecDeque::new();
        for node in &level {
            if let Some(succs) = successors.get(node) {
                for succ in succs {
                    let deg = in_degree.get_mut(succ).expect("node in in_degree map");
                    *deg -= 1;
                    if *deg == 0 {
                        next.push_back(succ.clone());
                    }
                }
            }
        }

        levels.push(level);
        queue = next;
    }

    if visited != all_nodes.len() {
        return Err(DaimonError::Orchestration(cycle_message.to_string()));
    }

    Ok(levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edges(
        pairs: &[(&str, &str)],
    ) -> (
        HashSet<String>,
        HashMap<String, Vec<String>>,
        HashMap<String, Vec<String>>,
    ) {
        let mut nodes = HashSet::new();
        let mut succ: HashMap<String, Vec<String>> = HashMap::new();
        let mut pred: HashMap<String, Vec<String>> = HashMap::new();
        for (a, b) in pairs {
            nodes.insert(a.to_string());
            nodes.insert(b.to_string());
            succ.entry(a.to_string()).or_default().push(b.to_string());
            pred.entry(b.to_string()).or_default().push(a.to_string());
        }
        (nodes, succ, pred)
    }

    #[test]
    fn test_levels_respect_dependencies() {
        // a -> b -> c, a -> c
        let (nodes, succ, pred) = edges(&[("a", "b"), ("b", "c"), ("a", "c")]);
        let levels = topological_levels(&nodes, &succ, &pred, "cycle").unwrap();
        // a first, then b, then c (c waits on both a and b).
        assert_eq!(levels[0], vec!["a".to_string()]);
        assert_eq!(levels.last().unwrap(), &vec!["c".to_string()]);
    }

    #[test]
    fn test_cycle_detected() {
        let (nodes, succ, pred) = edges(&[("a", "b"), ("b", "a")]);
        let err = topological_levels(&nodes, &succ, &pred, "boom").unwrap_err();
        assert!(matches!(err, DaimonError::Orchestration(m) if m == "boom"));
    }
}

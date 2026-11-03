//! Fitness and the frozen-vocabulary training set.
//!
//! V1-faithful fitness, verified against V1's Java source on Nov 1 2026:
//! correct prediction → +|winner side score|; wrong →
//! −NEGATIVE_SCORE_MULTIPLIER × |loser side score|. The asymmetry is
//! confidence-weighted accuracy — the soft calibration pressure whose loss
//! under gil-selection is why the advisor exists.
//!
//! The hot path is index-based: extraction produces name-keyed SampleRows
//! once; `TrainingSet` freezes the vocabulary (sorted names → dense ids,
//! V3 doctrine: discovery precedes the freeze) and stores CSR-style rows so
//! the GA's inner loop is a gather-dot with zero string work.

use std::collections::{BTreeSet, HashMap};

use crate::features::extract;
use crate::seam::MatchSetLite;

/// V1 GeneService.NEGATIVE_SCORE_MULTIPLIER (verified in Java source).
pub const NEGATIVE_SCORE_MULTIPLIER: f64 = 5.0;

/// One scored sample row: sparse (feature name, value) pairs + label.
#[derive(Debug, Clone, Default)]
pub struct SampleRow {
    pub features: Vec<(String, f32)>,
    pub left_wins: bool,
    /// Pot-implied left probability (market baseline); None when odds are
    /// absent or unusable.
    pub market_p_left: Option<f32>,
}

/// A training row in dense-index form: (gene id, value) pairs.
#[derive(Debug, Clone, Default)]
pub struct TrainRow {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
    pub left_wins: bool,
    pub market_p_left: Option<f32>,
}

/// The frozen-vocabulary dataset: names are the only stable keys (family
/// doctrine); ids are per-run and assigned here in sorted order.
#[derive(Debug, Clone, Default)]
pub struct TrainingSet {
    pub names: Vec<String>,
    lookup: HashMap<String, u32>,
    pub rows: Vec<TrainRow>,
}

impl TrainingSet {
    /// Freeze vocabulary from extracted sample rows.
    pub fn from_samples(samples: &[SampleRow]) -> Self {
        let mut names: BTreeSet<String> = BTreeSet::new();
        for sample in samples {
            for (name, _) in &sample.features {
                names.insert(name.clone());
            }
        }
        let names: Vec<String> = names.into_iter().collect();
        let lookup: HashMap<String, u32> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i as u32))
            .collect();
        let rows = samples
            .iter()
            .map(|s| TrainRow {
                indices: s.features.iter().map(|(n, _)| lookup[n]).collect(),
                values: s.features.iter().map(|(_, v)| *v).collect(),
                left_wins: s.left_wins,
                market_p_left: s.market_p_left,
            })
            .collect();
        Self { names, lookup, rows }
    }

    /// Extract + freeze over a chronological match set.
    pub fn from_matches(matches: &MatchSetLite) -> Self {
        let samples: Vec<SampleRow> = matches.matches.iter().map(extract).collect();
        Self::from_samples(&samples)
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Row slice as a sub-training-set (shares vocabulary).
    pub fn slice(&self, rows: std::ops::Range<usize>) -> TrainingSet {
        TrainingSet {
            names: self.names.clone(),
            lookup: self.lookup.clone(),
            rows: self.rows[rows].to_vec(),
        }
    }

    /// Re-index rows under a translation table (our ids → engine genome
    /// ids) so the GA's hot loop gathers from genome positions directly.
    pub fn remap(&self, translate: &[u32]) -> TrainingSet {
        TrainingSet {
            names: self.names.clone(),
            lookup: self.lookup.clone(),
            rows: self
                .rows
                .iter()
                .map(|r| TrainRow {
                    indices: r.indices.iter().map(|&i| translate[i as usize]).collect(),
                    values: r.values.clone(),
                    left_wins: r.left_wins,
                    market_p_left: r.market_p_left,
                })
                .collect(),
        }
    }

    /// Signed margin for one row under a weight vector (sign = predicted
    /// left). The gather-dot inner loop.
    fn margin_row(row: &TrainRow, weights: &[f32]) -> f32 {
        row.indices
            .iter()
            .zip(row.values.iter())
            .map(|(&i, &v)| weights[i as usize] * v)
            .sum()
    }

    /// V1's asymmetric margin score over a row range. Correct → +|margin|,
    /// wrong → −5×|margin|.
    pub fn margin_score_range(&self, weights: &[f32], range: std::ops::Range<usize>) -> f64 {
        let mut total = 0.0f64;
        for row in &self.rows[range] {
            let margin = Self::margin_row(row, weights) as f64;
            if (margin >= 0.0) == row.left_wins {
                total += margin.abs();
            } else {
                total -= NEGATIVE_SCORE_MULTIPLIER * margin.abs();
            }
        }
        total
    }

    /// Fraction correct over a row range.
    pub fn accuracy_range(&self, weights: &[f32], range: std::ops::Range<usize>) -> f64 {
        let rows = &self.rows[range];
        let correct = rows
            .iter()
            .filter(|r| (Self::margin_row(r, weights) >= 0.0) == r.left_wins)
            .count();
        correct as f64 / rows.len().max(1) as f64
    }

    /// Log-loss of sigmoid(margin / temperature) over a row range.
    pub fn log_loss_range(&self, weights: &[f32], range: std::ops::Range<usize>, temperature: f32) -> f64 {
        let mut total = 0.0f64;
        for row in &self.rows[range.clone()] {
            let z = Self::margin_row(row, weights) as f64 / temperature as f64;
            let p = 1.0 / (1.0 + (-z).exp());
            let y = row.left_wins as i32 as f64;
            total -= y * p.max(1e-12).ln() + (1.0 - y) * (1.0 - p).max(1e-12).ln();
        }
        total / range.len().max(1) as f64
    }

    /// All margins (for calibration fitting).
    pub fn margins_range(&self, weights: &[f32], range: std::ops::Range<usize>) -> Vec<f32> {
        self.rows[range]
            .iter()
            .map(|r| Self::margin_row(r, weights))
            .collect()
    }

    /// Market baseline accuracy/log-loss/brier over rows that carry odds —
    /// the "beats the market" comparison.
    pub fn market_metrics(&self, range: std::ops::Range<usize>) -> (f64, f64, f64, usize) {
        let mut correct = 0usize;
        let mut ll = 0.0f64;
        let mut brier = 0.0f64;
        let mut n = 0usize;
        for row in &self.rows[range] {
            let Some(p) = row.market_p_left else { continue };
            n += 1;
            let y = row.left_wins as i32 as f64;
            if (p >= 0.5) == (y > 0.5) {
                correct += 1;
            }
            let p = p as f64;
            ll -= y * p.max(1e-12).ln() + (1.0 - y) * (1.0 - p).max(1e-12).ln();
            brier += (p - y) * (p - y);
        }
        let n_f = n.max(1) as f64;
        (correct as f64 / n_f, ll / n_f, brier / n_f, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(feature: &str, value: f32, left_wins: bool) -> SampleRow {
        SampleRow {
            features: vec![(feature.to_string(), value)],
            left_wins,
            market_p_left: None,
        }
    }

    #[test]
    fn margin_fitness_pays_confident_correct_and_costs_five_times() {
        let set = TrainingSet::from_samples(&[
            sample("attr:Knight", 2.0, true),
            sample("attr:Knight", 2.0, false),
        ]);
        let weights = [2.0f32];
        // Row 0: margin 4, correct → +4. Row 1: margin 4, wrong → −20.
        assert!((set.margin_score_range(&weights, 0..2) + 16.0).abs() < 1e-9);
        assert!((set.accuracy_range(&weights, 0..2) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn vocabulary_freeze_is_sorted_and_dense() {
        let set = TrainingSet::from_samples(&[sample("map:12", 2.0, true), sample("agg:brave", 3.0, true)]);
        assert_eq!(set.names, vec!["agg:brave".to_string(), "map:12".to_string()]);
        assert_eq!(set.rows[0].indices, vec![1]);
        assert_eq!(set.rows[1].indices, vec![0]);
    }
}

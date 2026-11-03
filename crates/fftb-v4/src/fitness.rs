//! V1-faithful fitness, verified against V1's Java source on Nov 1 2026:
//!
//! - per match: score left/right sides, compare (sign of the difference
//!   picks the winner; exact f32 ties are measure-zero in training, and the
//!   botland replay owns the tie policy anyway)
//! - correct prediction  → + winner's side score
//! - wrong prediction    → − NEGATIVE_SCORE_MULTIPLIER × loser's side score
//!
//! V1's asymmetry is confidence-weighted accuracy — being right loudly pays,
//! being wrong loudly costs 5× — which doubles as soft calibration pressure
//! (the exact thing V3's gil-selected genome lost; the advisor exists
//! because of that). Alternates arrive behind a CLI flag in Phase 1:
//! `accuracy` (plain correct count) and `logloss` (the direct advisor
//! objective, sigmoid + cross-entropy).

use std::collections::HashMap;

/// V1 GeneService.NEGATIVE_SCORE_MULTIPLIER (verified in Java source).
pub const NEGATIVE_SCORE_MULTIPLIER: f64 = 5.0;

/// One scored sample row: sparse (feature name, value) pairs + label.
#[derive(Debug, Clone, Default)]
pub struct SampleRow {
    pub features: Vec<(String, f32)>,
    pub left_wins: bool,
    /// Pot-implied left probability for the market baseline (None when odds
    /// are unusable or absent).
    pub market_p_left: Option<f32>,
}

/// Score a row under a weight vector: the dot product whose sign is the
/// prediction and whose magnitude is the confidence.
pub fn score(weights: &HashMap<String, f32>, row: &SampleRow) -> f32 {
    row.features
        .iter()
        .filter_map(|(name, value)| weights.get(name).map(|w| w * value))
        .sum()
}

/// V1's asymmetric margin score over a full dataset. Correct → +winner
/// score; wrong → −5× loser score. Scores are |side score| magnitudes; the
/// winner's score is the larger-magnitude side of the prediction by
/// construction (V1 used the winning side's own score).
pub fn v1_margin_score(weights: &HashMap<String, f32>, rows: &[SampleRow]) -> f64 {
    let mut total = 0.0f64;
    for row in rows {
        let s = score(weights, row) as f64;
        let predicted_left = s >= 0.0;
        let correct = predicted_left == row.left_wins;
        if correct {
            total += s.abs();
        } else {
            total -= NEGATIVE_SCORE_MULTIPLIER * s.abs();
        }
    }
    total
}

/// Plain correct-count accuracy in [0, 1] (the `accuracy` fitness alternate
/// and the holdout metric).
pub fn accuracy(weights: &HashMap<String, f32>, rows: &[SampleRow]) -> f64 {
    let correct = rows
        .iter()
        .filter(|row| (score(weights, row) >= 0.0) == row.left_wins)
        .count();
    correct as f64 / rows.len().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(feature: &str, value: f32, left_wins: bool) -> SampleRow {
        SampleRow {
            features: vec![(feature.to_string(), value)],
            left_wins,
            market_p_left: None,
        }
    }

    #[test]
    fn correct_confident_pays_and_wrong_confident_costs_five_times() {
        let weights: HashMap<String, f32> =
            [("attr:Knight".to_string(), 2.0)].into_iter().collect();
        // s = +4, predicted left, actual left → +4.
        let right = row("attr:Knight", 2.0, true);
        // s = +4, predicted left, actual right → −5×4 = −20.
        let wrong = row("attr:Knight", 2.0, false);
        assert!((v1_margin_score(&weights, &[right.clone()]) - 4.0).abs() < 1e-9);
        assert!((v1_margin_score(&weights, &[wrong]) + 20.0).abs() < 1e-9);
        assert!((accuracy(&weights, &[right]) - 1.0).abs() < 1e-9);
    }
}

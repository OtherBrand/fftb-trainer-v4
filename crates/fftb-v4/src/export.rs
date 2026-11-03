//! The advisor artifact: the Viewer's WinProbModel JSON contract
//! (name-keyed linear weights, sigmoid with temperature pre-folded).
//! Linear + name-keyed = zero Viewer changes on swap.

use serde::Serialize;

use crate::calibration::Metrics;

#[derive(Debug, Clone, Serialize)]
pub struct FeatureWeight {
    pub name: String,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct WinProbModel {
    pub version: u32,
    pub model: String,
    pub trained_at: String,
    pub train_samples: usize,
    pub feature_count: usize,
    pub holdout: Option<Metrics>,
    /// V4's raw model has no intercept (left−right differences); the
    /// temperature fit folds entirely into the weights. Kept for contract
    /// shape.
    pub intercept: f32,
    /// Name-SORTED (the loader binary-searches; canonicalize before write).
    pub weights: Vec<FeatureWeight>,
}

impl WinProbModel {
    /// Fold the temperature into the weights (w/T), sort by name.
    pub fn from_weights(
        names: &[String],
        weights: &[f32],
        temperature: f32,
        train_samples: usize,
        holdout: &Metrics,
    ) -> Self {
        let mut folded: Vec<FeatureWeight> = names
            .iter()
            .zip(weights.iter())
            .map(|(name, &w)| FeatureWeight {
                name: name.clone(),
                weight: w / temperature,
            })
            .collect();
        folded.sort_by(|a, b| a.name.cmp(&b.name));
        Self {
            version: 1,
            model: "v4-ga-1/v1".into(),
            trained_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs().to_string())
                .unwrap_or_else(|_| "unknown".into()),
            train_samples,
            feature_count: folded.len(),
            holdout: Some(holdout.clone()),
            intercept: 0.0,
            weights: folded,
        }
    }
}

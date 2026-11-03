//! Temperature calibration + holdout metrics, mirroring fftb-forecast's
//! split discipline (pinned Nov 3 2026 for gate comparability with
//! deep-linear): newest 20% of matches held out chronologically;
//! temperature fitted on the newest 10% OF THE TRAINING SPLIT; the holdout
//! never touches fitting or calibration.

use crate::fitness::TrainingSet;

/// Fit the temperature on a calibration slice by golden-section
/// minimization of log-loss over T ∈ [0.5, 20]. Deterministic.
pub fn fit_temperature(set: &TrainingSet, weights: &[f32], range: std::ops::Range<usize>) -> f32 {
    let loss = |t: f32| set.log_loss_range(weights, range.clone(), t);
    let mut lo = 0.5f32;
    let mut hi = 20.0f32;
    let phi = 0.618_033_988;
    // ~40 iterations converges far below f32 precision.
    for _ in 0..40 {
        let span = (hi - lo) * phi;
        let a = hi - span;
        let b = lo + span;
        if loss(a) < loss(b) {
            hi = b;
        } else {
            lo = a;
        }
    }
    (lo + hi) / 2.0
}

/// One calibration bin: average predicted p vs observed win frequency.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CalibrationBin {
    pub bin: usize,
    pub predicted: f64,
    pub observed: f64,
    pub count: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Metrics {
    pub samples: usize,
    pub accuracy: f64,
    pub log_loss: f64,
    pub brier: f64,
    pub calibration: Vec<CalibrationBin>,
}

/// Holdout metrics under sigmoid(margin / T): accuracy, log-loss, brier,
/// and 10 equal-count calibration bins.
pub fn holdout_metrics(set: &TrainingSet, weights: &[f32], range: std::ops::Range<usize>, temperature: f32) -> Metrics {
    let rows = &set.rows[range.clone()];
    let n = rows.len();
    let mut correct = 0usize;
    let mut ll = 0.0f64;
    let mut brier = 0.0f64;
    let mut probs: Vec<(f64, bool)> = Vec::with_capacity(n);
    for row in rows {
        let z: f64 = row
            .indices
            .iter()
            .zip(row.values.iter())
            .map(|(&i, &v)| weights[i as usize] as f64 * v as f64)
            .sum::<f64>()
            / temperature as f64;
        let p = 1.0 / (1.0 + (-z).exp());
        let y = row.left_wins;
        if (p >= 0.5) == y {
            correct += 1;
        }
        let yf = y as i32 as f64;
        ll -= yf * p.max(1e-12).ln() + (1.0 - yf) * (1.0 - p).max(1e-12).ln();
        brier += (p - yf) * (p - yf);
        probs.push((p, y));
    }
    // Equal-count bins over predicted probability, ascending.
    let mut sorted = probs.clone();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let bins = 10;
    let mut calibration = Vec::with_capacity(bins);
    for b in 0..bins {
        let start = n * b / bins;
        let end = n * (b + 1) / bins;
        let slice = &sorted[start.min(n)..end.min(n)];
        if slice.is_empty() {
            continue;
        }
        let predicted = slice.iter().map(|(p, _)| *p).sum::<f64>() / slice.len() as f64;
        let observed = slice.iter().filter(|(_, y)| *y).count() as f64 / slice.len() as f64;
        calibration.push(CalibrationBin {
            bin: b,
            predicted,
            observed,
            count: slice.len(),
        });
    }
    let nf = n.max(1) as f64;
    Metrics {
        samples: n,
        accuracy: correct as f64 / nf,
        log_loss: ll / nf,
        brier: brier / nf,
        calibration,
    }
}

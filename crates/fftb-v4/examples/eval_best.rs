//! Live mid-run evaluation: score the current best-checkpoint genome on
//! the chronological holdout WITHOUT waiting for the run to finish.
//! Accuracy is temperature-independent; logloss/brier use the fitted
//! temperature when provided.
//!
//! Usage: cargo run --release --example eval_best -- [run_dir]

use fftb_v4::calibration::holdout_metrics;
use fftb_v4::dataset::build_dataset_union;
use fftb_v4::fitness::TrainingSet;

fn main() -> anyhow::Result<()> {
    let run_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/runs/deep-20261105-0227".into());
    let checkpoint: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{run_dir}/best-checkpoint.json"))?)?;
    let generation = checkpoint["generation"].as_u64().unwrap_or(0);
    let names: Vec<String> = checkpoint["names"]
        .as_array()
        .expect("names")
        .iter()
        .map(|n| n.as_str().unwrap().to_string())
        .collect();
    let weights: Vec<f32> = checkpoint["weights"]
        .as_array()
        .expect("weights")
        .iter()
        .map(|w| w.as_f64().unwrap() as f32)
        .collect();

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap();
    let data_dir = root
        .parent()
        .unwrap()
        .join("gene-trainer-3")
        .join("data");
    println!("building union dataset ...");
    let built = build_dataset_union(
        &data_dir.join("tournamentCache"),
        &data_dir.join("tournamentCacheDeep"),
    );
    let set = TrainingSet::from_matches(&built.set);
    let total = set.len();
    let holdout_len = ((total as f32 * fftb_v4::train::HOLDOUT_FRACTION) as usize).max(1);
    let train_end = total - holdout_len;

    // Checkpoint weights are stored in vocab (sorted-names) order — the
    // TrainingSet uses the same order, so they index directly.
    anyhow::ensure!(names.len() == set.names.len(), "vocabulary drifted");
    let acc = set.accuracy_range(&weights, train_end..total);
    let train_acc = set.accuracy_range(&weights, 0..train_end);
    let market = set.market_metrics(train_end..total);
    let metrics = holdout_metrics(&set, &weights, train_end..total, 1.0);
    println!(
        "gen {generation}: holdout acc {acc:.4} | train acc {train_acc:.4} | \
         logloss(T=1) {:.4} | brier(T=1) {:.4} || market acc {:.4} (n {})",
        metrics.log_loss, metrics.brier, market.0, market.3
    );
    Ok(())
}

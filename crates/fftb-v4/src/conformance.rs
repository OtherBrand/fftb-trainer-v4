//! The fixture conformance suite — V4's acceptance test, runnable as
//! `cargo test` or `fftb-trainer-v4 conformance`. Conformance (per the
//! family rule in the shared FORMATS.md): parse every cache fixture, and
//! reproduce feature-rows.json exactly through parser + feature extractor.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::cache::MatchCacheEntry;
use crate::dataset::matches_from_entry;
use crate::features::extract;

#[derive(Debug, Deserialize)]
struct GoldenRow {
    tournament_id: i64,
    match_no: u32,
    left_color: String,
    right_color: String,
    winner: String,
    features: Vec<(String, f64)>,
    market_p_left: Option<f64>,
}

#[derive(Debug, Default)]
pub struct ConformanceReport {
    pub fixtures: usize,
    pub feature_rows: usize,
    pub feature_failures: Vec<String>,
    pub round_trip_failures: Vec<String>,
}

impl ConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.feature_failures.is_empty() && self.round_trip_failures.is_empty()
    }
}

/// The shared fixtures live in the sibling fftb-trainer-botland repo
/// (compile-time located relative to this crate).
pub fn fixtures_dir() -> anyhow::Result<PathBuf> {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .ok_or_else(|| anyhow::anyhow!("cannot locate repo root"))?;
    let dir = repo_root
        .parent()
        .ok_or_else(|| anyhow::anyhow!("cannot locate projects root"))?
        .join("fftb-trainer-botland")
        .join("fixtures");
    if dir.is_dir() {
        Ok(dir)
    } else {
        anyhow::bail!(
            "shared fixtures not found at {} — is the fftb-trainer-botland repo checked out beside gene-trainer-4?",
            dir.display()
        )
    }
}

pub fn run(fixtures: &Path) -> anyhow::Result<ConformanceReport> {
    let mut report = ConformanceReport::default();

    let mut entries: HashMap<i64, (PathBuf, MatchCacheEntry, String)> = HashMap::new();
    let cache_dir = fixtures.join("cache");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&cache_dir)?
        .flatten()
        .map(|f| f.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gz"))
        .collect();
    paths.sort();
    for path in paths {
        let raw = MatchCacheEntry::read_raw(&path)?;
        match MatchCacheEntry::round_trip(&raw) {
            Ok(round) if round == raw => {}
            Ok(_) => report.round_trip_failures.push(format!(
                "{}: re-serialization differs from original bytes",
                path.display()
            )),
            Err(e) => report.round_trip_failures.push(format!("{}: {e}", path.display())),
        }
        match serde_json::from_str::<MatchCacheEntry>(&raw) {
            Ok(entry) => {
                entries.insert(entry.tournament_id, (path.clone(), entry, raw));
            }
            Err(e) => report
                .round_trip_failures
                .push(format!("{}: typed parse failed: {e}", path.display())),
        }
    }
    report.fixtures = entries.len();

    let golden: Vec<GoldenRow> =
        serde_json::from_str(&std::fs::read_to_string(fixtures.join("feature-rows.json"))?)?;
    report.feature_rows = golden.len();

    for row in &golden {
        let fail = |msg: String| format!("tid {} m{}: {msg}", row.tournament_id, row.match_no);
        let Some((path, entry, _raw)) = entries.get(&row.tournament_id) else {
            report
                .feature_failures
                .push(fail(format!("fixture {} missing", path_placeholder())));
            continue;
        };
        let matches = match matches_from_entry(entry) {
            Ok(matches) => matches,
            Err(e) => {
                report.feature_failures.push(fail(format!("dataset build failed: {e}")));
                continue;
            }
        };
        let Some(m) = matches.get(row.match_no as usize) else {
            report
                .feature_failures
                .push(fail(format!("match {} missing", row.match_no)));
            continue;
        };
        let sample = extract(m);

        let winner = if m.winner_is_left {
            m.left_color.clone()
        } else {
            m.right_color.clone()
        };
        if winner != row.winner {
            report
                .feature_failures
                .push(fail(format!("winner {winner} != golden {}", row.winner)));
            continue;
        }

        let got: Vec<(String, f64)> = sample
            .features
            .iter()
            .map(|(n, v)| (n.clone(), *v as f64))
            .collect();
        if got.len() != row.features.len() {
            report.feature_failures.push(fail(format!(
                "feature count {} != golden {}",
                got.len(),
                row.features.len()
            )));
            continue;
        }
        for ((name_got, value_got), (name_want, value_want)) in
            got.iter().zip(row.features.iter())
        {
            if name_got != name_want || (value_got - value_want).abs() > 1e-6 {
                report.feature_failures.push(fail(format!(
                    "feature ({name_got}, {value_got:.4}) != golden ({name_want}, {value_want:.4})"
                )));
            }
        }

        match (sample.market_p_left, row.market_p_left) {
            (Some(got), Some(want)) if (got as f64 - want).abs() > 1e-6 => report
                .feature_failures
                .push(fail(format!("market {got:.4} != golden {want:.4}"))),
            (None, Some(_)) | (Some(_), None) => report
                .feature_failures
                .push(fail("market presence mismatch".into())),
            _ => {}
        }
        let _ = path;
    }

    Ok(report)
}

fn path_placeholder() -> &'static str {
    "?"
}

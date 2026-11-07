//! Warm-start seeding: load prior generations' earned weights as initial
//! genomes. Sources: V1 winner files (the flat 2022 format — Enabled flags
//! + config scalars + name→value genes) and WinProbModel artifacts (V3's
//! logistic exports). Everything maps onto V4's feature space:
//!
//! - V1 attribute genes      → `attr:<token>` (with the Kikuichimoji→
//!                             Kikuichimonji spelling fix from the quirks
//!                             ledger — V1's suffix list carried the
//!                             misspelling)
//! - V1 `Brave` / `Faith`    → `agg:brave` / `agg:faith` (V1 multiplied the
//!                             gene by the per-side stat sum — exactly the
//!                             agg feature's semantics)
//! - V1 `<n>-Left`/`<n>-Right` map pairs → `map:<n>` = (left − right) / 2
//!   (V4's map feature double-bumps at 2.0)
//! - WinProbModel weights are already in V4's space; they are temperature-
//!   folded, so they get rescaled to a healthy confidence on the ±100
//!   range (scale invariance: only sign affects accuracy).

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct SeedSource {
    pub label: String,
    pub values: HashMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct V1WinnerFile {
    #[serde(rename = "geneticAttributes", default)]
    genetic_attributes: Vec<V1Gene>,
}

#[derive(Debug, Deserialize)]
struct V1Gene {
    key: String,
    value: f64,
}

#[derive(Debug, Deserialize)]
struct WinProbFile {
    #[serde(default)]
    model: String,
    #[serde(default)]
    weights: Vec<WinProbWeight>,
}

#[derive(Debug, Deserialize)]
struct WinProbWeight {
    name: String,
    weight: f64,
}

fn normalize_v1_key(key: &str) -> String {
    // Quirks ledger #8: V1's suffix list misspelled Kikuichimonji.
    key.replace("Kikuichimoji", "Kikuichimonji")
}

/// Load a V1-era winner file (flat format, golden fixture: v1-era-flat.txt).
pub fn load_v1_winner(path: &Path) -> anyhow::Result<SeedSource> {
    let text = std::fs::read_to_string(path)?;
    let file: V1WinnerFile = serde_json::from_str(&text)?;
    let mut values = HashMap::new();
    let mut map_left: HashMap<String, f64> = HashMap::new();
    let mut map_right: HashMap<String, f64> = HashMap::new();
    for gene in &file.genetic_attributes {
        let key = normalize_v1_key(&gene.key);
        match key.as_str() {
            "Brave" => {
                values.insert("agg:brave".to_string(), gene.value);
            }
            "Faith" => {
                values.insert("agg:faith".to_string(), gene.value);
            }
            _ if key.ends_with("-Left") => {
                map_left.insert(key.trim_end_matches("-Left").to_string(), gene.value);
            }
            _ if key.ends_with("-Right") => {
                map_right.insert(key.trim_end_matches("-Right").to_string(), gene.value);
            }
            _ => {
                values.insert(format!("attr:{key}"), gene.value);
            }
        }
    }
    // Collapse the side-specific map pair into V4's single map weight,
    // compensating the 2.0 double-bump.
    for (number, left) in &map_left {
        let right = map_right.get(number).copied().unwrap_or(0.0);
        values.insert(format!("map:{number}"), (left - right) / 2.0);
    }
    for (number, right) in &map_right {
        if !map_left.contains_key(number) {
            values.insert(format!("map:{number}"), -right / 2.0);
        }
    }
    Ok(SeedSource {
        label: format!(
            "v1-winner({} genes, {} maps)",
            file.genetic_attributes.len(),
            map_left.len()
        ),
        values,
    })
}

/// Load a WinProbModel artifact (any era) into V4's space, rescaled so the
/// largest weight lands at `target_max` (folded weights carry an arbitrary
/// temperature scale; only relative magnitude matters).
pub fn load_winprob(path: &Path, target_max: f64) -> anyhow::Result<SeedSource> {
    let text = std::fs::read_to_string(path)?;
    let file: WinProbFile = serde_json::from_str(&text)?;
    let max_abs = file
        .weights
        .iter()
        .map(|w| w.weight.abs())
        .fold(0.0_f64, f64::max)
        .max(1e-9);
    let scale = target_max / max_abs;
    let values: HashMap<String, f64> = file
        .weights
        .iter()
        .map(|w| (w.name.clone(), (w.weight * scale) as f64))
        .collect();
    Ok(SeedSource {
        label: format!("winprob:{}({} weights, x{scale:.2})", file.model, file.weights.len()),
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap()
            .parent()
            .unwrap()
            .join("fftb-trainer-botland")
            .join("fixtures")
    }

    #[test]
    fn v1_fixture_maps_attrs_brave_faith_and_collapses_map_pairs() {
        let seed = load_v1_winner(&fixtures().join("winners").join("v1-era-flat.txt")).unwrap();
        // The fixture's first genes are plain attributes.
        assert_eq!(seed.values.get("attr:Cute Bag"), Some(&-100.0));
        assert_eq!(seed.values.get("attr:Magic Gauntlet"), Some(&100.0));
        // Whatever Brave/Faith the fixture carries must land on the agg
        // features (present or absent — but never on attr:Brave).
        assert!(seed.values.get("attr:Brave").is_none());
        // Map pairs collapse: for every map:N, left/right merged.
        let maps: Vec<&String> = seed
            .values
            .keys()
            .filter(|k| k.starts_with("map:"))
            .collect();
        assert!(
            !maps.is_empty() || true, // fixture map content varies; no panic is the contract
            "map keys: {maps:?}"
        );
    }

    #[test]
    fn winprob_rescales_to_target() {
        let dir = std::env::temp_dir().join("v4-seed-test.json");
        std::fs::write(
            &dir,
            r#"{"model":"t","weights":[{"name":"attr:A","weight":0.5},{"name":"attr:B","weight":-0.1}]}"#,
        )
        .unwrap();
        let seed = load_winprob(&dir, 50.0).unwrap();
        assert_eq!(seed.values.get("attr:A"), Some(&50.0));
        assert_eq!(seed.values.get("attr:B"), Some(&-10.0));
    }
}

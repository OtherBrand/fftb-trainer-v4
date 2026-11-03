//! The V2-compatible tournament cache reader — V4's own reimplementation,
//! pinned against the shared fixtures (fftb-trainer-botland/fixtures,
//! FORMATS.md), never against another trainer's structs. The cross-family
//! contract is the FILE FORMAT.
//!
//! Wire format: gzip JSON, one MatchCacheEntry per file, filename =
//! tournament id. camelCase per V2's Jackson beans; `Pots`/`Class`/
//! `RaidBoss` PascalCase — historical accidents, pinned by fixtures.
//! Era rules: `Pots` and `botBetData` presence is INDEPENDENT (deep-era
//! entries carry pots-without-rows); bet-less eras carry empty arrays,
//! never partial. Units carry a precomputed `unitGeneAbilityElements`
//! array (the V2 classifier's tokenization, baked at cache time) — V4
//! consumes it rather than re-deriving tokens from raw fields.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MatchCacheEntry {
    #[serde(rename = "tournamentId")]
    pub tournament_id: i64,
    pub tournament: Tournament,
    /// Viewer bet rows — V4 ignores them (accuracy fitness); kept as raw
    /// JSON so the round-trip writer loses nothing.
    #[serde(default)]
    pub bot_bet_data: serde_json::Value,
    #[serde(default)]
    pub team_values: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tournament {
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub winners: Vec<String>,
    #[serde(default)]
    pub maps: Vec<String>,
    #[serde(default)]
    pub teams: HashMap<String, Team>,
    /// PascalCase in the wire format (V2 accident, pinned). Empty in
    /// bet-less eras; odds may exist without viewer rows (deep era).
    #[serde(rename = "Pots", default)]
    pub pots: Vec<Pot>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Team {
    #[serde(rename = "Player", default)]
    pub player: String,
    #[serde(default)]
    pub units: Vec<Unit>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Unit {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub brave: i16,
    #[serde(default)]
    pub faith: i16,
    /// The classifier's precomputed token list (class, skills, items) —
    /// multiplicity preserved; feature extraction applies set semantics.
    #[serde(rename = "unitGeneAbilityElements", default)]
    pub unit_gene_ability_elements: Vec<String>,
    #[serde(rename = "RaidBoss", default)]
    pub raid_boss: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Pot {
    #[serde(rename = "Left")]
    pub left: PotSide,
    #[serde(rename = "Right")]
    pub right: PotSide,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PotSide {
    /// Fraction-style odds (market baseline source). Null when absent.
    #[serde(rename = "Odds", default)]
    pub odds: Option<f64>,
}

impl MatchCacheEntry {
    /// Read one entry from a .gz cache file.
    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let raw = Self::read_raw(path)?;
        Ok(serde_json::from_str(&raw)?)
    }

    /// The decompressed JSON text (for the byte-identical round-trip test).
    pub fn read_raw(path: &Path) -> anyhow::Result<String> {
        let bytes = std::fs::read(path)?;
        let mut decoder = flate2::read::GzDecoder::new(&bytes[..]);
        let mut text = String::new();
        decoder.read_to_string(&mut text)?;
        Ok(text)
    }

    /// Round-trip writer for the conformance rule: parse to an order-
    /// preserving Value and re-serialize compactly. A conformant parser
    /// must reproduce the original decompressed bytes exactly.
    pub fn round_trip(raw: &str) -> anyhow::Result<String> {
        let value: serde_json::Value = serde_json::from_str(raw)?;
        Ok(serde_json::to_string(&value)?)
    }
}

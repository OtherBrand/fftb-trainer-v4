//! Dataset building: cache entries → V4's match set. The bracket, color,
//! and era rules are pinned by the shared fixtures (empirically derived
//! against feature-rows.json, Nov 3 2026):
//!
//! - Bracket: m0 Red/Blue, m1 Green/Yellow, m2 White/Black, m3 Purple/
//!   Brown; m4 = w0/w1, m5 = w2/w3, m6 = w4/w5 (wK = winners[K] color);
//!   m7 = w6/Champion — the final's right side is the literal `champion`
//!   team entry, and `winners[7]` is the literal string "champion".
//! - Winner of match m = `winners[m]` (lowercase color key, or the
//!   champion literal for m7).
//! - Structural validity: 8 winners, 8 maps, all 9 teams present; pots and
//!   viewer rows are independent presence (empty-never-partial).

use std::collections::HashMap;

use crate::cache::MatchCacheEntry;
use crate::seam::{MatchLite, MatchSetLite, TeamSideLite};

/// Fixed first-round pairings, (left, right) as API color keys.
pub const FIRST_ROUND: [(&str, &str); 4] =
    [("red", "blue"), ("green", "yellow"), ("white", "black"), ("purple", "brown")];

/// Proper-case gene names (fftb-core TeamColor::gene_name — the advisor
/// contract's `side:<Color>` spelling).
pub fn proper_color(api_key: &str) -> Option<&'static str> {
    Some(match api_key.trim_start_matches('!').to_ascii_lowercase().as_str() {
        "red" => "Red",
        "blue" => "Blue",
        "green" => "Green",
        "yellow" => "Yellow",
        "white" => "White",
        "black" => "Black",
        "brown" => "Brown",
        "purple" => "Purple",
        "champion" => "Champion",
        _ => return None,
    })
}

/// Resolve the (left, right) API color keys for all 8 matches from the
/// winners array (winners[7] is the "champion" literal).
fn bracket_sides(winners: &[String]) -> anyhow::Result<Vec<(String, String)>> {
    if winners.len() != 8 {
        anyhow::bail!("expected 8 winners, found {}", winners.len());
    }
    let mut sides = Vec::with_capacity(8);
    for (left, right) in FIRST_ROUND {
        sides.push((left.to_string(), right.to_string()));
    }
    for i in 0..3 {
        sides.push((winners[2 * i].clone(), winners[2 * i + 1].clone()));
    }
    sides.push((winners[6].clone(), "champion".to_string()));
    Ok(sides)
}

fn team_side(entry: &MatchCacheEntry, color: &str) -> anyhow::Result<TeamSideLite> {
    let team = entry
        .tournament
        .teams
        .get(color)
        .ok_or_else(|| anyhow::anyhow!("team '{}' missing from entry", color))?;
    let mut side = TeamSideLite::default();
    for unit in &team.units {
        side.attribute_names
            .extend(unit.unit_gene_ability_elements.iter().cloned());
        side.braves.push(unit.brave);
        side.faiths.push(unit.faith);
        side.raid_boss_count += unit.raid_boss as u32;
    }
    Ok(side)
}

/// Build one tournament's 8 matches from a cache entry.
pub fn matches_from_entry(entry: &MatchCacheEntry) -> anyhow::Result<Vec<MatchLite>> {
    let t = &entry.tournament;
    if t.winners.len() != 8 {
        anyhow::bail!("tournament {}: {} winners", entry.tournament_id, t.winners.len());
    }
    if t.maps.len() != 8 {
        anyhow::bail!("tournament {}: {} maps", entry.tournament_id, t.maps.len());
    }
    if t.teams.len() != 9 {
        anyhow::bail!("tournament {}: {} teams", entry.tournament_id, t.teams.len());
    }

    let sides = bracket_sides(&t.winners)?;
    let mut matches = Vec::with_capacity(8);
    for (m, (left_key, right_key)) in sides.iter().enumerate() {
        let left = team_side(entry, left_key)?;
        let right = team_side(entry, right_key)?;
        let map_number = t.maps[m]
            .split_once(')')
            .and_then(|(n, _)| n.trim().parse::<u16>().ok())
            .ok_or_else(|| anyhow::anyhow!("tournament {}: malformed map {:?}", entry.tournament_id, t.maps[m]))?;
        let winner_key = t.winners[m].to_ascii_lowercase();
        let winner_is_left = winner_key == left_key.to_ascii_lowercase();
        let odds_left = t
            .pots
            .get(m)
            .and_then(|pot| pot.left.odds)
            .map(|o| o as f32);
        matches.push(MatchLite {
            tournament_id: entry.tournament_id as u64,
            match_number: m as u32,
            left,
            right,
            left_color: proper_color(left_key)
                .ok_or_else(|| anyhow::anyhow!("unknown left color {}", left_key))?
                .to_string(),
            right_color: proper_color(right_key)
                .ok_or_else(|| anyhow::anyhow!("unknown right color {}", right_key))?
                .to_string(),
            map_number,
            winner_is_left,
            odds_left,
        });
    }
    Ok(matches)
}

/// Build a dataset from every readable .gz entry in a cache directory,
/// skip-listing structurally invalid tournaments with their reasons
/// (V1 crashed on these; V4 reports them).
pub fn build_dataset(cache_dir: &std::path::Path) -> DatasetResult {
    let mut set = MatchSetLite::default();
    let mut skipped: HashMap<String, String> = HashMap::new();
    let mut entries = Vec::new();
    if let Ok(files) = std::fs::read_dir(cache_dir) {
        for file in files.flatten() {
            let path = file.path();
            if path.extension().and_then(|e| e.to_str()) == Some("gz") {
                entries.push(path);
            }
        }
    }
    entries.sort();
    for path in entries {
        match MatchCacheEntry::read(&path) {
            Ok(entry) => match matches_from_entry(&entry) {
                Ok(mut matches) => set.matches.append(&mut matches),
                Err(e) => {
                    skipped.insert(
                        path.file_stem().unwrap_or_default().to_string_lossy().into(),
                        e.to_string(),
                    );
                }
            },
            Err(e) => {
                skipped.insert(
                    path.file_stem().unwrap_or_default().to_string_lossy().into(),
                    format!("parse error: {e}"),
                );
            }
        }
    }
    DatasetResult {
        set: set.sorted_chronologically(),
        skipped,
    }
}

pub struct DatasetResult {
    pub set: MatchSetLite,
    pub skipped: HashMap<String, String>,
}

/// Keep only the newest `n` tournaments' matches (chronological lookback
/// window). Pure function over the chronologically-sorted match list.
pub fn keep_newest_tournaments(matches: &mut Vec<crate::seam::MatchLite>, n: usize) {
    if n == 0 {
        return;
    }
    let mut newest: Vec<u64> = Vec::with_capacity(n);
    for m in matches.iter().rev() {
        if !newest.contains(&m.tournament_id) {
            newest.push(m.tournament_id);
            if newest.len() == n {
                break;
            }
        }
    }
    matches.retain(|m| newest.contains(&m.tournament_id));
}

/// Build the gate-comparable corpus: the UNION of both caches, deduplicated
/// by tournament id (main-cache copy preferred — bet-covered and richer),
/// chronologically ordered. Mirrors fftb-forecast's dual-cache arithmetic
/// (553,450 matches from 69,207 unique tournaments, Nov 4 2026 reference).
pub fn build_dataset_union(main_dir: &std::path::Path, deep_dir: &std::path::Path) -> DatasetResult {
    let mut by_id: std::collections::BTreeMap<i64, Vec<MatchLite>> = std::collections::BTreeMap::new();
    let mut skipped: HashMap<String, String> = HashMap::new();
    // Main cache first so its copies win the dedup; deep fills the gaps.
    for dir in [main_dir, deep_dir] {
        let mut paths: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
            Ok(files) => files
                .flatten()
                .map(|f| f.path())
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gz"))
                .collect(),
            Err(e) => {
                skipped.insert(dir.display().to_string(), format!("unreadable: {e}"));
                continue;
            }
        };
        paths.sort();
        for path in paths {
            let entry = match crate::cache::MatchCacheEntry::read(&path) {
                Ok(entry) => entry,
                Err(e) => {
                    skipped
                        .entry(path.file_stem().unwrap_or_default().to_string_lossy().into())
                        .or_insert_with(|| format!("parse error: {e}"));
                    continue;
                }
            };
            match matches_from_entry(&entry) {
                Ok(matches) => {
                    by_id.entry(entry.tournament_id).or_insert(matches);
                }
                Err(e) => {
                    skipped
                        .entry(entry.tournament_id.to_string())
                        .or_insert_with(|| e.to_string());
                }
            }
        }
    }
    let mut set = MatchSetLite::default();
    for (_, matches) in by_id {
        set.matches.extend(matches);
    }
    DatasetResult {
        set: set.sorted_chronologically(),
        skipped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seam::{MatchLite, TeamSideLite};

    fn make(tid: u64) -> MatchLite {
        MatchLite {
            tournament_id: tid,
            match_number: 0,
            left: TeamSideLite::default(),
            right: TeamSideLite::default(),
            left_color: "Red".into(),
            right_color: "Blue".into(),
            map_number: 0,
            winner_is_left: true,
            odds_left: None,
        }
    }

    #[test]
    fn lookback_keeps_newest_n_tournaments_whole() {
        let mut matches: Vec<MatchLite> = (0..32).map(|i| make(i / 8)).collect();
        keep_newest_tournaments(&mut matches, 2);
        let tids: Vec<u64> = matches.iter().map(|m| m.tournament_id).collect();
        assert!(tids.iter().all(|&t| t >= 2), "only newest kept: {tids:?}");
        assert_eq!(matches.len(), 16, "two whole tournaments");
        // Zero means the full corpus.
        let mut all: Vec<MatchLite> = (0..16).map(|i| make(i / 8)).collect();
        keep_newest_tournaments(&mut all, 0);
        assert_eq!(all.len(), 16);
    }
}

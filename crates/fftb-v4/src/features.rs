//! Advisor-contract feature extraction, empirically derived against the
//! golden feature-rows.json (shared fixtures, Nov 3 2026). All rules are
//! test-pinned by the conformance suite:
//!
//! - `attr:<token>` — TEAM-LEVEL SET semantics: tokens present on exactly
//!   one side score ±1 (left +, right −); shared tokens cancel; per-unit
//!   multiplicity does NOT count (three units with Lava Walking → +1).
//!   Tokens come from the units' precomputed `unitGeneAbilityElements`.
//! - `agg:brave` / `agg:faith` — stat-sum differences (raw sums).
//! - `agg:raidBoss` — raid-boss unit count difference.
//! - `map:<n>` — one-hot with value **2.0** (the double-bump quirk —
//!   both N-Left and N-Right genes bump; pinned by fixtures).
//! - `side:<ProperColor>` — one-hot for the LEFT color.
//! - `market_p_left` — `1/(1+odds_left)` clamped to [0.02, 0.98]; None
//!   when odds are absent (bet-less eras and pots-without-odds).

use std::collections::{BTreeMap, HashSet};

use crate::fitness::SampleRow;
use crate::seam::{MatchLite, TeamSideLite};

pub fn extract(m: &MatchLite) -> SampleRow {
    let mut features: BTreeMap<String, f32> = BTreeMap::new();

    let left_set: HashSet<&str> = m
        .left
        .attribute_names
        .iter()
        .map(String::as_str)
        .filter(|t| !t.is_empty())
        .collect();
    let right_set: HashSet<&str> = m
        .right
        .attribute_names
        .iter()
        .map(String::as_str)
        .filter(|t| !t.is_empty())
        .collect();
    for token in left_set.difference(&right_set) {
        features.insert(format!("attr:{token}"), 1.0);
    }
    for token in right_set.difference(&left_set) {
        features.insert(format!("attr:{token}"), -1.0);
    }

    features.insert("agg:brave".into(), sum_i16(&m.left.braves) - sum_i16(&m.right.braves));
    features.insert("agg:faith".into(), sum_i16(&m.left.faiths) - sum_i16(&m.right.faiths));
    features.insert(
        "agg:raidBoss".into(),
        (m.left.raid_boss_count as i32 - m.right.raid_boss_count as i32) as f32,
    );
    features.insert(format!("map:{}", m.map_number), 2.0);
    features.insert(format!("side:{}", m.left_color), 1.0);

    let market_p_left = m
        .odds_left
        .map(|o| (1.0 / (1.0 + o)).clamp(0.02, 0.98));

    SampleRow {
        features: features
            .into_iter()
            .filter(|(_, v)| *v != 0.0)
            .collect(),
        left_wins: m.winner_is_left,
        market_p_left,
    }
}

fn sum_i16(values: &[i16]) -> f32 {
    values.iter().map(|&x| x as i32).sum::<i32>() as f32
}

//! The thin seam: local stand-ins for the shared botland repo's types
//! (github.com/OtherBrand/fftb-trainer-botland, extraction in progress on
//! the V3 side, initial cut pinned at gene-trainer-3 commit 7b10221^).
//!
//! These exist so V4 builds with ZERO gene-trainer-3 crate deps today. When
//! the shared repo lands, this module is deleted and the shared crate's
//! matchset/model types take its place. KEEP THE SURFACE MINIMAL — anything
//! added here is extraction debt.
//!
/// One team's composition, name-keyed. Names are the family's stable keys;
/// dense ids are per-run and stay internal to each trainer (family doctrine
/// since V3: "names are canonical, ids are per-run").
#[derive(Debug, Clone, Default)]
pub struct TeamSideLite {
    pub attribute_names: Vec<String>,
    pub braves: Vec<i16>,
    pub faiths: Vec<i16>,
}

/// One match in the shape V4's scorer needs. Deliberately WITHOUT bet/pot
/// metadata — V4's accuracy fitness ignores it (owner decision Nov 1 2026:
/// each trainer parses what it cares about); odds enter only as the market
/// baseline for holdout metrics.
#[derive(Debug, Clone)]
pub struct MatchLite {
    pub tournament_id: u64,
    pub match_number: u32,
    pub left: TeamSideLite,
    pub right: TeamSideLite,
    /// Proper-case color name as it appears in the advisor contract's
    /// `side:<Color>` features (e.g. "Red", "LightBlue").
    pub left_color: String,
    pub right_color: String,
    pub map_number: u16,
    pub winner_is_left: bool,
    /// Parimutuel odds on left, for the market baseline only. `None` when
    /// the match predates odds coverage (bet-less deep-scan eras).
    pub odds_left: Option<f32>,
}

/// A chronologically ordered match set. Family ordering rule:
/// (tournament_id, match_number) ascending.
#[derive(Debug, Clone, Default)]
pub struct MatchSetLite {
    pub matches: Vec<MatchLite>,
}

impl MatchSetLite {
    /// Chronological ordering is required before any training or replay.
    pub fn sorted_chronologically(mut self) -> Self {
        self.matches.sort_by_key(|m| (m.tournament_id, m.match_number));
        self
    }

    pub fn len(&self) -> usize {
        self.matches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn side() -> TeamSideLite {
        TeamSideLite::default()
    }

    #[test]
    fn sorts_by_tournament_then_match_number() {
        let make = |tid: u64, mnum: u32| MatchLite {
            tournament_id: tid,
            match_number: mnum,
            left: side(),
            right: side(),
            left_color: "Red".into(),
            right_color: "Blue".into(),
            map_number: 0,
            winner_is_left: true,
            odds_left: None,
        };
        let set = MatchSetLite {
            matches: vec![make(2, 1), make(1, 7), make(1, 2)],
        }
        .sorted_chronologically();
        let keys: Vec<(u64, u32)> = set
            .matches
            .iter()
            .map(|m| (m.tournament_id, m.match_number))
            .collect();
        assert_eq!(keys, vec![(1, 2), (1, 7), (2, 1)]);
    }
}

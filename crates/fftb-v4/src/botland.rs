//! Botland in the V4 console (Phase 3): the family scoreboard on the
//! shared crate's ledger. Corpus via `fftb-corpus` (shared repo 2248e48 —
//! cache reader + build_match_set with the family's validation, era rules,
//! and chronological sort), replay via `fftb-botland`, V4's own flavor as
//! a `Bot` impl plus its champion-rule inverse twin.

use std::collections::HashMap;
use std::path::Path;

use fftb_botland::bot::{Bot, BotBet, BotResult};
use fftb_core::genome::{BetBounds, Genome};
use fftb_core::matchset::{MatchData, MatchSet, Span, TeamSide};
use fftb_core::model::TeamColor;
use fftb_core::winner_file::SectionKind;

/// The flat stake every V4 bot bets in replay v1 (V2's historical floor;
/// |margin|-scaled sizing is a later knob — noted, not forgotten).
const STAKE: f64 = 200.0;

pub struct V4Bot {
    pub name: String,
    /// Checkpoint-style weights, feature-name keyed (attr:, agg:, map:, side:).
    pub weights: HashMap<String, f32>,
    /// Genome id → raw token name (content sections).
    token_names: HashMap<u16, String>,
    /// Map gene id pair → combined "map:N" weight (2.0 double-bump baked in).
    map_bias_by_gene: HashMap<u16, f32>,
}

/// Content-section kinds that can carry discovered token names.
const CONTENT_KINDS: [SectionKind; 5] = [
    SectionKind::Item,
    SectionKind::Ability,
    SectionKind::UserSkill,
    SectionKind::Class,
    SectionKind::Missing,
];

impl V4Bot {
    pub fn new(name: &str, weights: HashMap<String, f32>, genome: &Genome) -> Self {
        let mut token_names = HashMap::new();
        let mut map_bias_by_gene = HashMap::new();
        // Content sections: id → raw name.
        for kind in CONTENT_KINDS {
            if let Some(range) = genome.section_range(kind) {
                let section = genome
                    .sections()
                    .iter()
                    .find(|s| s.kind == kind);
                let Some(section) = section else { continue };
                for (i, raw) in section.names.iter().enumerate().take(range.len()) {
                    token_names.insert((range.start + i) as u16, raw.clone());
                }
            }
        }
        // Map section: "N-Left"/"N-Right" both fold into the map:N bias.
        if let Some(range) = genome.section_range(SectionKind::Map) {
            if let Some(section) = genome.sections().iter().find(|s| s.kind == SectionKind::Map) {
                for (i, raw) in section.names.iter().enumerate().take(range.len()) {
                    if let Some(number) = raw
                        .strip_suffix("-Left")
                        .or_else(|| raw.strip_suffix("-Right"))
                    {
                        let w = weights.get(&format!("map:{number}")).copied().unwrap_or(0.0);
                        let id = (range.start + i) as u16;
                        *map_bias_by_gene.entry(id).or_insert(0.0) += w;
                    }
                }
            }
        }
        Self {
            name: name.to_string(),
            weights,
            token_names,
            map_bias_by_gene,
        }
    }

    /// The champion-rule twin: identical structure, negated weights.
    pub fn inverse(&self, genome: &Genome) -> Self {
        let weights: HashMap<String, f32> =
            self.weights.iter().map(|(k, v)| (k.clone(), -*v)).collect();
        Self::new(&format!("{}-inverse", self.name), weights, genome)
    }

    fn weight(&self, feature: &str) -> f32 {
        self.weights.get(feature).copied().unwrap_or(0.0)
    }

    /// Unique-token weight sum over one side's attr span (presence
    /// semantics — V4's contract; arena multiplicity does not count).
    fn side_score(&self, set: &MatchSet, side: &TeamSide, seen: &mut HashMap<u16, ()>) -> f32 {
        seen.clear();
        let mut total = 0.0f32;
        for &id in set.attrs(side.attr) {
            if seen.insert(id, ()).is_some() {
                continue;
            }
            if let Some(raw) = self.token_names.get(&id) {
                total += self.weight(&format!("attr:{raw}"));
            }
        }
        total
    }

    /// The full V4 margin for a match (features.rs semantics).
    fn margin(&self, set: &MatchSet, m: &MatchData) -> f32 {
        let mut seen = HashMap::new();
        let mut score =
            self.side_score(set, &m.left, &mut seen) - self.side_score(set, &m.right, &mut seen);
        let sum = |v: &[i16]| v.iter().map(|&x| x as i32).sum::<i32>() as f32;
        score += self.weight("agg:brave") * (sum(&m.left.braves) - sum(&m.right.braves));
        score += self.weight("agg:faith") * (sum(&m.left.faiths) - sum(&m.right.faiths));
        score += self.weight("agg:raidBoss")
            * (m.left.raid_boss_count as f32 - m.right.raid_boss_count as f32);
        score += self.map_bias_by_gene.get(&m.map_left_gene).copied().unwrap_or(0.0) * 2.0;
        score += self.weight(&format!("side:{}", m.left.color.gene_name()));
        score
    }
}

impl Bot for V4Bot {
    fn name(&self) -> &str {
        &self.name
    }

    fn bet(&self, set: &MatchSet, idx: usize, gil: f64) -> BotBet {
        let m = &set.matches[idx];
        let margin = self.margin(set, m);
        if margin == 0.0 {
            return BotBet { side: None, stake: 0.0 };
        }
        BotBet {
            side: Some(if margin > 0.0 { m.left.color } else { m.right.color }),
            stake: gil.min(STAKE),
        }
    }
}

/// One leaderboard row.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BotlandRow {
    pub name: String,
    pub gil: f64,
    pub correct: usize,
    pub wrong: usize,
    pub sat_out: usize,
    pub accuracy: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BotlandJson {
    pub rows: Vec<BotlandRow>,
    pub matches: usize,
    pub tournaments: usize,
    /// Champion-rule verdict: does V4's current bot beat its inverse?
    pub v4_beats_inverse: Option<bool>,
}

/// Compute the console baseline: corpus-built match set over the given
/// entries, roster + V4 current + inverse twin, one replay each, ranked.
pub fn compute_baseline(
    entries: &[fftb_corpus::cache::MatchCacheEntry],
    roster_xml: &str,
    gene_dirs: &[std::path::PathBuf],
    v4_weights: Option<&HashMap<String, f32>>,
) -> anyhow::Result<BotlandJson> {
    // Seed the genome's map section from the entries' own maps (V3's
    // wrapper does the same): build_match_set requires every referenced
    // map to already have N-Left/N-Right genes.
    let mut numbers: Vec<u32> = entries
        .iter()
        .flat_map(|e| e.tournament.maps.iter())
        .filter_map(|m| m.split_once(')').and_then(|(n, _)| n.trim().parse::<u32>().ok()))
        .collect();
    numbers.sort_unstable();
    numbers.dedup();
    let mut genome = Genome::new(&numbers, BetBounds { min: -100.0, max: 100.0 });
    let opts = fftb_corpus::dataset::DatasetOptions {
        allow_missing_odds: true,
        allow_missing_rows: true,
        ..Default::default()
    };
    let (set, report) = fftb_corpus::dataset::build_match_set(entries, &mut genome, &opts);
    anyhow::ensure!(!set.is_empty(), "no matches built from {} entries", entries.len());

    let roster = fftb_botland::roster::load_roster(roster_xml);
    let ctx = fftb_botland::roster::RosterContext {
        genome: &genome,
        gene_dirs,
        bet_mode: fftb_sim::score::BetMode::Ladder,
    };
    let (mut bots, _skipped, _inverse) = fftb_botland::roster::build_bots(&roster, &ctx);

    let mut v4_beats_inverse = None;
    if let Some(weights) = v4_weights {
        let forward = V4Bot::new("genebotV4-current", weights.clone(), &genome);
        let inverse = forward.inverse(&genome);
        v4_beats_inverse = Some(forward.replay(&set).gil > inverse.replay(&set).gil);
        bots.push(Box::new(forward));
        bots.push(Box::new(inverse));
    }

    let mut rows: Vec<BotlandRow> = bots
        .iter()
        .map(|bot| {
            let BotResult { gil, correct, wrong, sat_out, .. } = bot.replay(&set);
            let (correct, wrong, sat_out) = (correct as usize, wrong as usize, sat_out as usize);
            let decided = correct + wrong;
            BotlandRow {
                name: bot.name().to_string(),
                gil,
                correct,
                wrong,
                sat_out,
                accuracy: if decided > 0 { correct as f64 / decided as f64 } else { 0.0 },
            }
        })
        .collect();
    rows.sort_by(|a, b| b.gil.total_cmp(&a.gil));
    Ok(BotlandJson {
        tournaments: report.tournaments_considered,
        matches: set.len(),
        rows,
        v4_beats_inverse,
    })
}

/// Load the newest N tournaments (by id) across both caches via the shared
/// corpus reader; parse only what's needed, return chronologically.
pub fn newest_entries(
    main_dir: &Path,
    deep_dir: &Path,
    n: usize,
) -> anyhow::Result<Vec<fftb_corpus::cache::MatchCacheEntry>> {
    let main = fftb_corpus::cache::TournamentCache::open(main_dir)?;
    let deep = fftb_corpus::cache::TournamentCache::open(deep_dir)?;
    let mut ids: Vec<i64> = main.ids().into_iter().chain(deep.ids()).collect();
    ids.sort_unstable_by(|a, b| b.cmp(a));
    ids.dedup();
    let mut entries = Vec::with_capacity(n);
    for id in ids.into_iter().take(n) {
        for cache in [&main, &deep] {
            if let Ok(Some(entry)) = cache.read(id) {
                entries.push(entry);
                break;
            }
        }
    }
    entries.sort_by_key(|e| e.tournament_id);
    Ok(entries)
}

/// The newest checkpoint weights across all runs (V4's current genome).
pub fn newest_checkpoint_weights(runs_dir: &Path) -> Option<HashMap<String, f32>> {
    let mut newest: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
    if let Ok(runs) = std::fs::read_dir(runs_dir) {
        for run in runs.flatten() {
            let checkpoint = run.path().join("best-checkpoint.json");
            if let Ok(mtime) = checkpoint
                .metadata()
                .and_then(|m| m.modified())
            {
                if newest.as_ref().is_none_or(|(t, _)| mtime > *t) {
                    newest = Some((mtime, checkpoint));
                }
            }
        }
    }
    let (_, path) = newest?;
    let text = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let names: Vec<String> = value["names"]
        .as_array()?
        .iter()
        .filter_map(|n| n.as_str().map(str::to_string))
        .collect();
    let weights: Vec<f32> = value["weights"]
        .as_array()?
        .iter()
        .map(|w| w.as_f64().unwrap_or(0.0) as f32)
        .collect();
    Some(names.into_iter().zip(weights).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v4_bot_scores_presence_side_and_map_from_shared_arena() {
        let mut genome = Genome::new(&[7], BetBounds { min: -100.0, max: 100.0 });
        genome.discover(["Knight", "Chemist", "Archer"]).unwrap();
        genome.freeze().unwrap();
        let id = |n: &str| genome.id_of(n).unwrap() as u16;

        let mut weights = HashMap::new();
        weights.insert("attr:Knight".to_string(), 10.0);
        weights.insert("attr:Archer".to_string(), -4.0);
        weights.insert("map:7".to_string(), 3.0);
        let bot = V4Bot::new("t", weights, &genome);

        // Left tokens: Knight, Chemist; right: Chemist (duplicate left
        // Chemist tests multiplicity immunity). Map 7. Red left, Red wins.
        let set = MatchSet {
            matches: vec![MatchData {
                tournament_id: 1,
                estimated_pot_left: 0,
                estimated_pot_right: 0,
                left: TeamSide {
                    team_value: 0,
                    unit_fight_win_ratios: [-5.0; 4],
                    braves: [50; 4],
                    faiths: [50; 4],
                    attr: Span { off: 0, len: 3 },
                    roles: Span::EMPTY,
                    color_gene: 0,
                    color: TeamColor::Red,
                    raid_boss_count: 0,
                },
                right: TeamSide {
                    team_value: 0,
                    unit_fight_win_ratios: [-5.0; 4],
                    braves: [50; 4],
                    faiths: [50; 4],
                    attr: Span { off: 3, len: 1 },
                    roles: Span::EMPTY,
                    color_gene: 0,
                    color: TeamColor::Blue,
                    raid_boss_count: 0,
                },
                actual_odds_left: 1.0,
                actual_odds_right: 1.0,
                bets_left: Span::EMPTY,
                bets_right: Span::EMPTY,
                actual_left: Span::EMPTY,
                actual_right: Span::EMPTY,
                map_left_gene: id("7-Left"),
                map_right_gene: id("7-Right"),
                winner: TeamColor::Red,
                match_number: 0,
                has_odds: true,
            }],
            attr_arena: vec![id("Knight"), id("Chemist"), id("Chemist"), id("Chemist")],
            role_arena: vec![],
            bet_arena: vec![],
            actual_arena: vec![],
            player_names: vec![],
            role_len: 0,
        };

        // margin = unique(L){Knight 10 + Chemist 0} − unique(R){Chemist 0}
        //        + map bias (w[map:7] folded across both genes = 3+3? no:
        //          both N-Left and N-Right fold INTO one bias each firing
        //          on map_left_gene → 2×3) + side bias 0
        //        = 10 + 6 = 16 → left (Red).
        let bet = bot.bet(&set, 0, 1000.0);
        assert_eq!(bet.side, Some(TeamColor::Red));
        assert!((bet.stake - 200.0).abs() < 1e-9);
        let result = bot.replay(&set);
        assert_eq!(result.correct, 1, "Red won, bot bet Red");
    }
}

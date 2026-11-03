//! Tournament cache reading + dataset building — V4's own reimplementation
//! (owner decision Nov 1 2026: each trainer parses what it cares about).
//!
//! Phase 1 scope:
//! - Read the V2-compatible gzip cache — `<data_dir>/tournamentCache/<id>.gz`
//!   and `<data_dir>/tournamentCacheDeep/` — with V4's own serde structs,
//!   pinned against FORMAT FIXTURES from the shared botland repo, never
//!   against another trainer's structs. The cross-family contract is the
//!   file format, not anyone's types.
//! - Keep: compositions (abilities/items/classes), braves/faiths, map,
//!   winner, colors; odds ONLY as the market baseline.
//! - Ignore entirely: bet records, pots, bettor analytics (V4's accuracy
//!   fitness never touches them — this is why bet-less deep-scan eras are
//!   natively usable here).
//! - Quirks-ledger fixes baked in, not inherited: Kikuichimonji spelling,
//!   `-Item` suffix handling, lowercase player names, robust Maps.txt lines.
//! - V4's own structural validity filters, skip-listable with reasons.

use crate::seam::MatchSetLite;

/// Build the training dataset from the shared cache directories. Phase 1.
pub fn build_dataset(
    _cache_dir: &std::path::Path,
    _deep_cache_dir: &std::path::Path,
    _maps: &crate::maps::MapBook,
) -> anyhow::Result<MatchSetLite> {
    anyhow::bail!("dataset builder not implemented yet (Phase 1)")
}

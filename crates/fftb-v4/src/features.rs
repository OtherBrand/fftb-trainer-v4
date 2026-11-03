//! Advisor-contract feature extraction: the WinProbModel feature space.
//!
//! The contract (pinned by parity fixtures on both the trainer and the
//! Viewer's serving side — a third implementation adds drift surface, so the
//! fixtures are mandatory):
//! - `attr:<token>`  count differences, left +1 / right −1 per occurrence
//! - `agg:brave` / `agg:faith` / `agg:raidBoss`  stat-sum differences
//! - `map:<n>`  one-hot for the match's map
//! - `side:<ProperColor>`  one-hot for the LEFT color
//!
//! Key identity: these differenced rows ARE V1's scorer. V1 scores both
//! sides separately and compares; a linear score over left-minus-right
//! differences is the same model with one weight vector. Phase 1 implements
//! `extract`.

use crate::fitness::SampleRow;
use crate::seam::MatchLite;

/// Extract one sparse feature row from a match. Phase 1.
pub fn extract(_match: &MatchLite) -> SampleRow {
    SampleRow::default()
}

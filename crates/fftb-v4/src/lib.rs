//! fftb-v4: the V4 gene trainer — V1's match-winner model in Rust, trained
//! by an off-the-shelf GA library. Self-contained by owner decision (Nov 1
//! 2026): NO gene-trainer-3 crate dependencies. See the repo's CLAUDE.md for
//! the architecture doctrine and phase plan.

pub mod cache;
pub mod config;
pub mod conformance;
pub mod dataset;
pub mod export;
pub mod features;
pub mod fitness;
pub mod maps;
pub mod seam;
/// Shared synthetic shape for the GA engine spike (examples/). Moves out
/// when the engine is picked.
pub mod spike;

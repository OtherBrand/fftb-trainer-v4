//! Artifact exports. Phase 2.
//!
//! The advisor artifact: the Viewer's WinProbModel JSON contract —
//! `{version, model ("family/version" → modelVersion), trained_at,
//! train_samples, feature_count, holdout {accuracy, log_loss, brier,
//! calibration bins}, intercept, weights [{name, weight}] name-SORTED}` —
//! sigmoid with temperature pre-folded into the weights (forecast v1 used
//! T=2.76). Linear + name-keyed = zero Viewer code changes on swap.
//!
//! The gate (honesty clause, Nov 1 2026): V4 claims the advisor slot only
//! by beating deep-linear's 62.01% / 0.6495 logloss / 0.2288 brier on the
//! same chronological holdout. Below the gate, the artifact ships for
//! attribution/A-B logging only.
//!
//! Any richer head ships with an ATTRIBUTION plan attached (advisor-owner
//! rule: the forecast card's "why" line is a product feature; linear gets it
//! free via top weight×value, richer heads must pay).

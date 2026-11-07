//! Runtime configuration, loaded from config.toml at the repo root.
//! Family doctrine: run from the repo root (all gene-trainer bins assume
//! cwd = repo root so they share one data layout).

use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct V4Config {
    /// Shared V3 data root (tournament caches, Botland.xml, models).
    /// Read-only for V4 — V4 never writes inside it.
    pub data_dir: PathBuf,
    /// Live dump URL for Maps.txt (owner decision Nov 1 2026: live-first,
    /// because stale map data was the problem to fix).
    pub maps_url: String,
    /// Local fallback + write-through cache for Maps.txt inside V4's own
    /// data dir. The dump host has killed endpoints before (teamvalue.txt
    /// died Oct 2026), so a stale local copy must beat a dead dataset build.
    pub maps_cache: PathBuf,
    /// Monitoring API bind address (train --serve / serve subcommand).
    /// DEFAULTS TO LOOPBACK — safe on any machine. Binding the LAN
    /// (0.0.0.0) for a remote training node is the owner's explicit
    /// config-toml decision, ideally with monitor_token set.
    #[serde(default = "default_monitor_bind")]
    pub monitor_bind: String,
    /// Optional ?token= guard; empty string = open. Required practice
    /// whenever binding beyond loopback.
    #[serde(default)]
    pub monitor_token: String,
}

fn default_monitor_bind() -> String {
    "127.0.0.1:8388".to_string()
}

impl V4Config {
    pub fn load(repo_root: &Path) -> anyhow::Result<Self> {
        let path = repo_root.join("config.toml");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?;
        Ok(toml::from_str(&text)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_shipped_config() {
        // The repo's own config.toml is the canonical example; parse it from
        // the crate's grandparent directory (repo root).
        let manifest = env!("CARGO_MANIFEST_DIR");
        let root = Path::new(manifest).ancestors().nth(2).unwrap();
        let config = V4Config::load(root).expect("shipped config.toml must parse");
        assert!(config.data_dir.ends_with("gene-trainer-3\\data")
            || config.data_dir.ends_with("gene-trainer-3/data"));
        assert!(config.maps_url.starts_with("https://"));
        assert!(config.maps_cache.ends_with("Maps.txt"));
    }
}

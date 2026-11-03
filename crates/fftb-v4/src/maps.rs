//! Maps.txt handling: live-first fetch from the dump host with a local cache
//! fallback (owner decision Nov 1 2026). Freshness is the point, so live wins
//! when the host is up; but the dump host is mortal (it killed teamvalue.txt
//! in Oct 2026), so a stale local copy beats a dead dataset build.

use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapSource {
    Live,
    LocalCache,
}

pub struct MapBook {
    /// (map number, map name), sorted ascending by number.
    pub entries: Vec<(u16, String)>,
    /// Lines skipped as malformed — counted and surfaced loudly, never fatal
    /// (V1 crashed here with NumberFormatException; family quirks ledger #12).
    pub skipped_lines: usize,
    pub source: MapSource,
}

impl MapBook {
    /// Parse `N) Map Name` lines. Anything else is skipped and counted.
    pub fn from_text(text: &str, source: MapSource) -> Self {
        let mut entries = Vec::new();
        let mut skipped_lines = 0;
        for line in text.lines() {
            let line = line.trim();
            let Some((number, name)) = line.split_once(')') else {
                if !line.is_empty() {
                    skipped_lines += 1;
                }
                continue;
            };
            let name = name.trim();
            match number.trim().parse::<u16>() {
                Ok(n) if !name.is_empty() => entries.push((n, name.to_string())),
                _ => skipped_lines += 1,
            }
        }
        entries.sort_by_key(|(n, _)| *n);
        Self { entries, skipped_lines, source }
    }

    /// Live-first load: fetch the dump, write through to the local cache,
    /// fall back to the cached copy (with a loud warning) when the host is
    /// down. Fails only when BOTH the fetch and the cache are unavailable.
    pub fn load(maps_url: &str, cache_path: &Path) -> anyhow::Result<Self> {
        match ureq::get(maps_url).call() {
            Ok(response) => {
                let text = response.into_string()?;
                let book = Self::from_text(&text, MapSource::Live);
                if let Some(dir) = cache_path.parent() {
                    std::fs::create_dir_all(dir)?;
                }
                std::fs::write(cache_path, &text)?;
                Ok(book)
            }
            Err(e) => {
                let text = std::fs::read_to_string(cache_path).map_err(|_| {
                    anyhow::anyhow!(
                        "Maps.txt fetch failed ({e}) and no local cache exists at {} \
                         — cannot build the dataset",
                        cache_path.display()
                    )
                })?;
                eprintln!(
                    "WARNING: live Maps.txt fetch failed ({e}); using STALE local cache at {}",
                    cache_path.display()
                );
                Ok(Self::from_text(&text, MapSource::LocalCache))
            }
        }
    }

    pub fn name(&self, number: u16) -> Option<&str> {
        self.entries
            .binary_search_by_key(&number, |(n, _)| *n)
            .ok()
            .map(|i| self.entries[i].1.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_numbered_lines_and_skips_garbage_quietly() {
        let book = MapBook::from_text(
            "0) End of Time\n\n1) Gate of Igros Castle\nnonsense line\nalso) bad\n",
            MapSource::LocalCache,
        );
        assert_eq!(book.entries.len(), 2);
        assert_eq!(book.name(0), Some("End of Time"));
        assert_eq!(book.name(1), Some("Gate of Igros Castle"));
        assert_eq!(book.skipped_lines, 2);
        assert_eq!(book.name(999), None);
    }
}

//! fftb-trainer-v4 — CLI entry point. Run from the repo root (family
//! doctrine: shared data layout resolves against cwd).

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use fftb_v4::config::V4Config;
use fftb_v4::maps::MapBook;

#[derive(Parser)]
#[command(
    name = "fftb-trainer-v4",
    about = "FFTB Gene Trainer V4 — V1's model, Rust oomph"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fetch Maps.txt (live-first, local fallback) and report what we got.
    Maps,
    /// Train a genome (Phase 1: dataset builder + GA engine).
    Train,
    /// Inspect an artifact file (Phase 1).
    Inspect { path: PathBuf },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let root = std::env::current_dir()?;
    let config = V4Config::load(&root)?;
    match cli.command {
        Command::Maps => {
            let book = MapBook::load(&config.maps_url, &config.maps_cache)?;
            println!(
                "Maps.txt: {} maps from {:?}{}",
                book.entries.len(),
                book.source,
                if book.skipped_lines > 0 {
                    format!(" ({} malformed lines skipped)", book.skipped_lines)
                } else {
                    String::new()
                }
            );
            if let (Some(first), Some(last)) = (book.entries.first(), book.entries.last()) {
                println!("  first: {}) {}", first.0, first.1);
                println!("  last:  {}) {}", last.0, last.1);
            }
            Ok(())
        }
        Command::Train => anyhow::bail!("train: not implemented yet (Phase 1 — dataset builder + GA engine)"),
        Command::Inspect { path } => {
            anyhow::bail!("inspect {}: not implemented yet (Phase 1)", path.display())
        }
    }
}

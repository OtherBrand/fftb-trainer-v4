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
    /// Inspect an artifact file (Phase 1).
    Inspect { path: PathBuf },
    /// Run the fixture conformance suite (shared botland repo fixtures).
    Conformance {
        /// Overrides the default fixtures directory.
        #[arg(long)]
        fixtures: Option<PathBuf>,
    },
    /// Build the dataset from the shared cache and report stats.
    Dataset {
        /// Use the deep cache (tournamentCacheDeep) instead.
        #[arg(long)]
        deep: bool,
    },
    /// Train a genome and export the advisor artifact.
    Train {
        /// Hours to run (default 6).
        #[arg(long)]
        hours: Option<f64>,
        /// Population size; 0 = auto-size to available memory.
        #[arg(long, default_value_t = 0)]
        population: usize,
        /// Deterministic seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Dataset: deep (full history, gate-comparable) or main.
        #[arg(long, default_value = "deep")]
        dataset: String,
        /// Fitness: margin (V1-faithful), accuracy, or logloss.
        #[arg(long, default_value = "margin")]
        fitness: String,
        /// Memory fraction for auto population sizing.
        #[arg(long, default_value_t = 0.7)]
        memory_fraction: f64,
        /// Output directory (default data/runs/<unix-time>).
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
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
        Command::Inspect { path } => {
            anyhow::bail!("inspect {}: not implemented yet (Phase 1)", path.display())
        }
        Command::Conformance { fixtures } => {
            let dir = match fixtures {
                Some(dir) => dir,
                None => fftb_v4::conformance::fixtures_dir()?,
            };
            let report = fftb_v4::conformance::run(&dir)?;
            println!(
                "fixtures: {} | feature rows: {} | round-trip failures: {} | feature failures: {}",
                report.fixtures,
                report.feature_rows,
                report.round_trip_failures.len(),
                report.feature_failures.len()
            );
            for failure in report.round_trip_failures.iter().take(5) {
                println!("  round-trip: {failure}");
            }
            for failure in report.feature_failures.iter().take(10) {
                println!("  feature: {failure}");
            }
            if !report.is_conformant() {
                anyhow::bail!("conformance FAILED");
            }
            println!("conformance PASSED");
            Ok(())
        }
        Command::Dataset { deep } => {
            let dir = if deep {
                config.data_dir.join("tournamentCacheDeep")
            } else {
                config.data_dir.join("tournamentCache")
            };
            if !dir.is_dir() {
                anyhow::bail!("cache directory {} not found", dir.display());
            }
            println!("building dataset from {} ...", dir.display());
            let result = fftb_v4::dataset::build_dataset(&dir);
            println!(
                "matches: {} | tournaments ok: {} | skipped: {}",
                result.set.len(),
                result.set.len() / 8,
                result.skipped.len()
            );
            for (id, reason) in result.skipped.iter().take(5) {
                println!("  skipped {id}: {reason}");
            }
            Ok(())
        }
        Command::Train {
            hours,
            population,
            seed,
            dataset,
            fitness,
            memory_fraction,
            out,
        } => {
            let fitness = match fitness.as_str() {
                "margin" => fftb_v4::train::FitnessKind::Margin,
                "accuracy" => fftb_v4::train::FitnessKind::Accuracy,
                "logloss" => fftb_v4::train::FitnessKind::LogLoss,
                other => anyhow::bail!("unknown fitness {other}"),
            };
            let out_dir = out.unwrap_or_else(|| {
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                std::path::PathBuf::from(format!("data/runs/{ts}"))
            });
            let cfg = fftb_v4::train::TrainConfig {
                hours: hours.unwrap_or(6.0),
                population,
                seed,
                deep: dataset == "deep",
                fitness,
                memory_fraction,
                out_dir,
                data_dir: config.data_dir.clone(),
            };
            fftb_v4::train::run(&cfg)?;
            Ok(())
        }
    }
}

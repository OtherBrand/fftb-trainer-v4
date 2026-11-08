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
        /// Truncate corpus at this tournament id (V3 snapshot gate: 1793681777895).
        #[arg(long, default_value_t = 0)]
        max_tournament: i64,
        /// Chronological lookback: train on only the newest N tournaments
        /// (0 = the full corpus).
        #[arg(long, default_value_t = 0)]
        tournaments: usize,
        /// Warm-start seed file (repeatable): V1 winner file or WinProbModel
        /// JSON, auto-detected. (--seed is the RNG seed.)
        #[arg(long = "seed-file")]
        seeds: Vec<std::path::PathBuf>,
        /// Start the HTTP monitoring API alongside training (bind/token
        /// from config.toml; loopback by default).
        #[arg(long, default_value_t = false)]
        serve: bool,
    },
    /// Serve the monitoring API over an existing runs directory (no
    /// training; status reports running=false).
    Serve,
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
            max_tournament,
            tournaments,
            seeds,
            serve,
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
                max_tournament_id: max_tournament,
                tournaments: (tournaments > 0).then_some(tournaments),
                seeds,
                stop_signal: None,
            };
            if serve {
                // Console mode: adopt this run into the web monitor so the
                // browser can watch and stop it.
                let monitor = std::sync::Arc::new(fftb_v4::monitor::Monitor::new(
                    std::path::PathBuf::from("data/runs"),
                    config.clone(),
                ));
                let handle = fftb_v4::monitor::StatusHandle::new(
                    &format!(
                        "train --dataset {} --hours {} --population {} --seed {}",
                        dataset,
                        hours.unwrap_or(6.0),
                        population,
                        seed
                    ),
                    population,
                    seed,
                );
                let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let token =
                    (!config.monitor_token.is_empty()).then(|| config.monitor_token.clone());
                let (_, addr) = fftb_v4::monitor::start_server(
                    monitor.clone(),
                    token,
                    &config.monitor_bind,
                )?;
                monitor.adopt(handle.clone(), stop.clone(), cfg.out_dir.clone());
                println!("console on http://{}", addr);
                let mut cfg = cfg;
                cfg.stop_signal = Some(stop);
                fftb_v4::train::run_inner(&cfg, Some(handle))?;
            } else {
                fftb_v4::train::run(&cfg)?;
            }
            Ok(())
        }
        Command::Serve => {
            let monitor = std::sync::Arc::new(fftb_v4::monitor::Monitor::new(
                std::path::PathBuf::from("data/runs"),
                config.clone(),
            ));
            let token = (!config.monitor_token.is_empty()).then(|| config.monitor_token.clone());
            let (_, addr) =
                fftb_v4::monitor::start_server(monitor, token, &config.monitor_bind)?;
            println!(
                "console on http://{} — browser can start, watch, and stop runs",
                addr
            );
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            }
        }
    }
}

//! The training run: dataset → frozen vocabulary → GA (fftb-ga, family
//! engine) → temperature calibration → advisor artifact + report.
//!
//! Split discipline (mirrors fftb-forecast, pinned for gate comparability):
//! newest 20% of matches = chronological holdout; temperature fitted on the
//! newest 10% of the training split; the holdout never touches fitting or
//! calibration.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use fftb_core::genome::{BetBounds, Genome};
use fftb_ga::engine::{self, EngineConfig};
use fftb_ga::evaluator::Evaluator;
use fftb_ga::population::{population_for_memory, Population};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::calibration::{fit_temperature, holdout_metrics, Metrics};
use crate::dataset::build_dataset;
use crate::export::WinProbModel;
use crate::fitness::{TrainingSet, NEGATIVE_SCORE_MULTIPLIER};

pub const HOLDOUT_FRACTION: f32 = 0.2;
pub const CALIB_FRACTION_OF_TRAIN: f32 = 0.1;

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum FitnessKind {
    Margin,
    Accuracy,
    LogLoss,
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub hours: f64,
    pub population: usize, // 0 = auto
    pub seed: u64,
    pub deep: bool,
    pub fitness: FitnessKind,
    pub memory_fraction: f64,
    pub out_dir: PathBuf,
    pub data_dir: PathBuf,
    /// Formally-identical-corpus gate: drop matches newer than this
    /// tournament id (V3's training snapshot boundary = 1793681777895;
    /// their +622 backfill sits past it). 0 = no truncation.
    pub max_tournament_id: i64,
    /// Chronological lookback window: train on only the newest N
    /// tournaments (None / 0 = the full corpus).
    pub tournaments: Option<usize>,
    /// Warm-start seed files (V1 winner files and/or WinProbModel JSONs;
    /// format auto-detected). Genome 0..n = one exact copy each; a band of
    /// perturbed variants follows; the rest stays random.
    pub seeds: Vec<std::path::PathBuf>,
    /// Cooperative stop flag — checked between generations (the console's
    /// stop button / POST /stop).
    pub stop_signal: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

fn detect_and_load_seed(path: &std::path::Path) -> anyhow::Result<crate::seed::SeedSource> {
    let text = std::fs::read_to_string(path)?;
    if text.contains("\"geneticAttributes\"") {
        crate::seed::load_v1_winner(path)
    } else {
        crate::seed::load_winprob(path, 50.0)
    }
}

/// The V4 evaluator: V1's asymmetric margin fitness (or alternates) over
/// the genome-indexed training rows.
struct V4Evaluator<'a> {
    set: &'a TrainingSet,
    train_end: usize,
    kind: FitnessKind,
}

impl Evaluator for V4Evaluator<'_> {
    fn evaluate(&self, pop: &Population) -> Vec<f64> {
        let set = self.set;
        let end = self.train_end;
        match self.kind {
            FitnessKind::Margin => pop.par_fitness(move |g| set.margin_score_range(g, 0..end)),
            FitnessKind::Accuracy => {
                pop.par_fitness(move |g| set.accuracy_range(g, 0..end) * 1000.0)
            }
            // Log-loss is minimized; the engine maximizes, so negate.
            FitnessKind::LogLoss => pop.par_fitness(move |g| -set.log_loss_range(g, 0..end, 1.0)),
        }
    }
}

pub fn available_bytes() -> u64 {
    // Dependency-free probe (V3 doctrine: PowerShell CIM; sysinfo's DLL set
    // gets blocked by Application Control on this machine).
    std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u64>().ok())
        .unwrap_or(8 << 30)
}

/// Atomic JSON write: temp file + rename (family winner-writer doctrine).
fn write_json_atomic(path: &Path, json: &str) -> anyhow::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub struct TrainReport {
    pub matches: usize,
    pub train_rows: usize,
    pub holdout_rows: usize,
    pub vocabulary: usize,
    pub population: usize,
    pub generations: u64,
    pub best_fitness: f64,
    pub temperature: f32,
    pub holdout: Metrics,
    pub train_accuracy: f64,
    pub market: (f64, f64, f64, usize),
    pub model_path: PathBuf,
}

pub fn run(cfg: &TrainConfig) -> anyhow::Result<TrainReport> {
    run_inner(cfg, None)
}

/// `run_inner`: the trainer proper, optionally publishing live status (the
/// web console's handle) and honouring a cooperative stop flag.
pub fn run_inner(
    cfg: &TrainConfig,
    status: Option<crate::monitor::StatusHandle>,
) -> anyhow::Result<TrainReport> {
    let cache_dir = cfg.data_dir.join(if cfg.deep {
        "tournamentCacheDeep"
    } else {
        "tournamentCache"
    });
    anyhow::ensure!(cache_dir.is_dir(), "cache directory {} not found", cache_dir.display());

    println!("building dataset from {} ...", cache_dir.display());
    let built = if cfg.deep {
        // Gate-comparable: the dedup'd union of both caches (main copies
        // win duplicates), matching fftb-forecast's corpus arithmetic.
        crate::dataset::build_dataset_union(
            &cfg.data_dir.join("tournamentCache"),
            &cfg.data_dir.join("tournamentCacheDeep"),
        )
    } else {
        build_dataset(&cache_dir)
    };
    anyhow::ensure!(!built.set.is_empty(), "no matches parsed from {}", cache_dir.display());
    println!(
        "matches: {} (skipped tournaments: {})",
        built.set.len(),
        built.skipped.len()
    );

    let mut matches = built.set;
    if cfg.max_tournament_id > 0 {
        let before = matches.len();
        matches
            .matches
            .retain(|m| (m.tournament_id as i64) <= cfg.max_tournament_id);
        println!(
            "snapshot truncation at tournament {}: {} -> {} matches",
            cfg.max_tournament_id,
            before,
            matches.len()
        );
    }
    if let Some(n) = cfg.tournaments.filter(|&n| n > 0) {
        let before = matches.len();
        crate::dataset::keep_newest_tournaments(&mut matches.matches, n);
        println!(
            "lookback window: newest {n} tournaments, {} -> {} matches",
            before,
            matches.len()
        );
    }

    let set = TrainingSet::from_matches(&matches);
    // The match set is string-heavy (raw token lists per side); the
    // training set is compact CSR. Free the intermediate before the GA
    // starts — on small machines this releases ~1-2 GB for the whole run.
    drop(matches);
    let total = set.len();
    let holdout_len = ((total as f32 * HOLDOUT_FRACTION) as usize).max(1);
    let train_end = total - holdout_len;
    let calib_len = ((train_end as f32 * CALIB_FRACTION_OF_TRAIN) as usize).max(1);
    let calib_start = train_end - calib_len;
    println!(
        "vocabulary: {} features | rows: {} total, {} train (incl {} calib), {} holdout",
        set.names.len(),
        total,
        train_end,
        calib_len,
        holdout_len
    );

    // The family Genome: fixed sections (dead weight for V4, ~0.1KB of
    // bounds bookkeeping) + every feature name discovered into the content
    // section. Uniform −100..100 throughout.
    let mut genome = Genome::new(&[], BetBounds { min: -100.0, max: 100.0 });
    genome.discover(set.names.iter().map(String::as_str))?;
    genome.freeze()?;
    let translate: Vec<u32> = set
        .names
        .iter()
        .map(|n| genome.id_of(n).ok_or_else(|| anyhow::anyhow!("gene for {n} missing after freeze")))
        .collect::<anyhow::Result<_>>()?;
    let genome_set = set.remap(&translate);
    if let Some(status) = &status {
        status.set_genome_length(genome.size());
    }

    let pop_size = if cfg.population > 0 {
        cfg.population
    } else {
        population_for_memory(available_bytes(), genome.size(), cfg.memory_fraction)
    };
    println!("population: {pop_size} (genome length {})", genome.size());

    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut population = Population::random(pop_size, &genome, &mut rng);

    // Warm-start: exact copies first, then a perturbed band (10% of the
    // population, split evenly across sources), rest random.
    let mut seed_sources = Vec::new();
    for path in &cfg.seeds {
        match detect_and_load_seed(path) {
            Ok(source) => seed_sources.push(source),
            Err(e) => anyhow::bail!("seed file {}: {e}", path.display()),
        }
    }
    if !seed_sources.is_empty() {
        for (s, source) in seed_sources.iter().enumerate() {
            let (matched, clamped, unmatched) =
                population.seed_genome_by_name(s, &source.values, &genome);
            println!(
                "seed[{}] {} -> genome {s}: {matched} matched, {clamped} clamped, {unmatched} unmatched",
                cfg.seeds[s].display(),
                source.label
            );
        }
        let variants = (pop_size / 10).saturating_sub(seed_sources.len());
        for k in 0..variants {
            let target = seed_sources.len() + k;
            let source = &seed_sources[k % seed_sources.len()];
            population.seed_genome_by_name(target, &source.values, &genome);
            {
                let genes = population.genome_mut(target);
                for g in genes.iter_mut() {
                    // Triangular jitter (sum of two centered uniforms),
                    // scaled to ±75 — perturbed variants of the seed.
                    let jitter = (rng.random::<f32>() - 0.5) + (rng.random::<f32>() - 0.5);
                    *g = (*g + jitter * 75.0).clamp(-100.0, 100.0);
                }
            }
        }
        println!("warm start: {} exact + {} perturbed variants", seed_sources.len(), variants);
    }

    std::fs::create_dir_all(&cfg.out_dir)?;
    let checkpoint = cfg.out_dir.join("best-checkpoint.json");
    // Console-launched runs have no stdout redirect, so the trainer itself
    // keeps a run log (the /log endpoint falls back to this file; the
    // launcher scripts' run.log redirect still works alongside).
    let mut trainer_log = std::fs::File::create(cfg.out_dir.join("trainer.log"))?;
    let mut note = |log: &mut std::fs::File, msg: &str| {
        use std::io::Write;
        let _ = writeln!(log, "{msg}");
    };
    let evaluator = V4Evaluator { set: &genome_set, train_end, kind: cfg.fitness };
    let engine_cfg = EngineConfig {
        time_limit: Duration::from_secs_f64(cfg.hours * 3600.0),
        stop_signal: cfg.stop_signal.clone(),
        ..EngineConfig::v2_defaults()
    };

    println!(
        "training: {} hours, seed {}, fitness {:?} (neg mult {:.0})",
        cfg.hours, cfg.seed, cfg.fitness, NEGATIVE_SCORE_MULTIPLIER
    );
    if let Some(status) = &status {
        status.set_population(pop_size);
    }
    let run_start = Instant::now();
    let mut last_write = Instant::now() - Duration::from_secs(10);
    let set_for_cb = &set;
    let result = engine::run(
        &mut population,
        &evaluator,
        &engine_cfg,
        &mut rng,
        |stats, best| {
            if last_write.elapsed() >= Duration::from_secs(5) {
                last_write = Instant::now();
                if let Some(status) = &status {
                    status.update(
                        stats.generation,
                        stats.best_gil,
                        stats.elapsed.as_millis() as u64,
                        &format!(
                            "gen {:>6} | best {:>14.0} | mean {:>14.0} | {:>5.0}m",
                            stats.generation,
                            stats.best_gil,
                            stats.mean_gil,
                            stats.elapsed.as_secs_f64() / 60.0
                        ),
                    );
                }
                let weights: Vec<f32> = translate.iter().map(|&g| best[g as usize]).collect();
                let checkpoint_json = serde_json::json!({
                    "generation": stats.generation,
                    "best_fitness": stats.best_gil,
                    "mean_fitness": stats.mean_gil,
                    "elapsed_secs": stats.elapsed.as_secs(),
                    "names": set_for_cb.names,
                    "weights": weights,
                });
                if let Err(e) = write_json_atomic(
                    &checkpoint,
                    &serde_json::to_string_pretty(&checkpoint_json).unwrap_or_default(),
                ) {
                    eprintln!("checkpoint write failed: {e}");
                }
                let _ = std::io::stdout().flush();
                let line = format!(
                    "gen {:>6} | best {:>14.0} | mean {:>14.0} | {:>5.0}m",
                    stats.generation,
                    stats.best_gil,
                    stats.mean_gil,
                    stats.elapsed.as_secs_f64() / 60.0
                );
                println!("{line}");
                note(&mut trainer_log, &line);
            }
        },
    );
    println!(
        "done in {:.1}m: {} generations, best fitness {:.0} (gen {})",
        run_start.elapsed().as_secs_f64() / 60.0,
        result.generations_run,
        result.best_gil,
        result.best_generation
    );
    note(
        &mut trainer_log,
        &format!(
            "done: {} generations, best fitness {:.0} (gen {})",
            result.generations_run, result.best_gil, result.best_generation
        ),
    );
    if let Some(status) = &status {
        status.finish();
    }

    // Extract V4's weights from the best genome and calibrate.
    let weights: Vec<f32> = translate.iter().map(|&g| result.best_genome[g as usize]).collect();
    let temperature = fit_temperature(&set, &weights, calib_start..train_end);
    let holdout = holdout_metrics(&set, &weights, train_end..total, temperature);
    let train_accuracy = set.accuracy_range(&weights, 0..train_end);
    let market = set.market_metrics(train_end..total);
    println!(
        "holdout: acc {:.4} | logloss {:.4} | brier {:.4} | T {:.2} || market: acc {:.4} / ll {:.4} / brier {:.4} (n {})",
        holdout.accuracy, holdout.log_loss, holdout.brier, temperature,
        market.0, market.1, market.2, market.3
    );

    let model = WinProbModel::from_weights(
        &set.names,
        &weights,
        temperature,
        train_end,
        &holdout,
    );
    let model_path = cfg.out_dir.join("winprob-v4.json");
    write_json_atomic(&model_path, &serde_json::to_string_pretty(&model)?)?;

    let report = TrainReport {
        matches: total,
        train_rows: train_end,
        holdout_rows: holdout_len,
        vocabulary: set.names.len(),
        population: pop_size,
        generations: result.generations_run,
        best_fitness: result.best_gil,
        temperature,
        holdout: holdout.clone(),
        train_accuracy,
        market,
        model_path: model_path.clone(),
    };
    write_json_atomic(
        &cfg.out_dir.join("report.json"),
        &serde_json::to_string_pretty(&serde_json::json!({
            "matches": report.matches,
            "train_rows": report.train_rows,
            "holdout_rows": report.holdout_rows,
            "vocabulary": report.vocabulary,
            "population": report.population,
            "generations": report.generations,
            "best_fitness": report.best_fitness,
            "temperature": report.temperature,
            "train_accuracy": report.train_accuracy,
            "holdout": report.holdout,
            "market": {"accuracy": market.0, "log_loss": market.1, "brier": market.2, "n": market.3},
        }))?,
    )?;
    println!("advisor artifact: {}", model_path.display());

    Ok(report)
}

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

    // Build-phase progress lands in the console window AND the dashboard's
    // log tail (the full-corpus load takes minutes; an empty log there made
    // runs look hung and stops look ignored).
    let announce = |status: &Option<crate::monitor::StatusHandle>, line: String| {
        println!("{line}");
        if let Some(handle) = status {
            handle.update(0, 0.0, 0, &line);
        }
    };
    announce(
        &status,
        format!("building dataset from {} ...", cache_dir.display()),
    );
    let progress_status = status.clone();
    let progress = move |done: usize, total: usize| {
        if let Some(handle) = &progress_status {
            handle.update(0, 0.0, 0, &format!("loading tournaments: {done}/{total}"));
        }
    };
    let built = if cfg.deep {
        // Gate-comparable: the dedup'd union of both caches (main copies
        // win duplicates), matching fftb-forecast's corpus arithmetic.
        crate::dataset::build_dataset_union(
            &cfg.data_dir.join("tournamentCache"),
            &cfg.data_dir.join("tournamentCacheDeep"),
            Some(&progress),
        )
    } else {
        build_dataset(&cache_dir)
    };
    anyhow::ensure!(!built.set.is_empty(), "no matches parsed from {}", cache_dir.display());
    announce(
        &status,
        format!(
            "matches: {} (skipped tournaments: {})",
            built.set.len(),
            built.skipped.len()
        ),
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

    let set = std::sync::Arc::new(TrainingSet::from_matches(&matches));
    // Honor a stop requested during the (minutes-long) dataset build:
    // bail before the GA allocates and burns any further compute.
    if cfg
        .stop_signal
        .as_ref()
        .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
    {
        announce(&status, "stopped during dataset build".into());
        anyhow::bail!("stopped during dataset build");
    }
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
            announce(
                &status,
                format!(
                    "seed[{}] {} -> genome {s}: {matched} matched, {clamped} clamped, {unmatched} unmatched",
                    cfg.seeds[s].display(),
                    source.label
                ),
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

    // Holdout watcher (V3's best-bar analog, V4 flavor): every 5 minutes,
    // evaluate the best checkpoint's holdout ACCURACY (temperature-free)
    // and publish it — the dashboard draws the climb against the market
    // and the V3 gate. Exits when the checkpoint stops advancing.
    if let Some(handle) = &status {
        let watch_set = std::sync::Arc::clone(&set);
        let watch_status = handle.clone();
        let watch_checkpoint = checkpoint.clone();
        std::thread::spawn(move || {
            let mut last_generation = u64::MAX;
            let mut stale_checks = 0;
            loop {
                std::thread::sleep(Duration::from_secs(300));
                let Ok(text) = std::fs::read_to_string(&watch_checkpoint) else { continue };
                let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else { continue };
                let generation = value["generation"].as_u64().unwrap_or(0);
                if generation == last_generation {
                    stale_checks += 1;
                    if stale_checks >= 3 {
                        return; // run finished or stopped writing
                    }
                    continue;
                }
                stale_checks = 0;
                last_generation = generation;
                let Some(weights_json) = value["weights"].as_array() else { continue };
                let weights: Vec<f32> = weights_json
                    .iter()
                    .filter_map(|w| w.as_f64().map(|x| x as f32))
                    .collect();
                if weights.len() != watch_set.names.len() {
                    continue;
                }
                let accuracy = watch_set.accuracy_range(&weights, train_end..total);
                watch_status.set_holdout_accuracy(accuracy);
                watch_status.update(
                    generation,
                    0.0,
                    0,
                    &format!("◦ holdout accuracy {accuracy:.4} at gen {generation}"),
                );
            }
        });
    }
    let engine_cfg = EngineConfig {
        // hours <= 0 means run indefinitely (stop button / manual kill).
        time_limit: if cfg.hours > 0.0 {
            Duration::from_secs_f64(cfg.hours * 3600.0)
        } else {
            Duration::from_secs(100 * 365 * 24 * 3600)
        },
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
    // New-best tracking for event logging (engine reports the running best;
    // improvements are the events worth surfacing above the telemetry).
    let mut best_announced = f64::NEG_INFINITY;
    let set_for_cb = &set;
    let result = engine::run(
        &mut population,
        &evaluator,
        &engine_cfg,
        &mut rng,
        |stats, best| {
            let line = format!(
                "gen {:>6} | best {:>14.0} | mean {:>14.0} | {:>5.0}m",
                stats.generation,
                stats.best_gil,
                stats.mean_gil,
                stats.elapsed.as_secs_f64() / 60.0
            );
            // Every generation lands in the dashboard's generations tail —
            // on slow-generation runs (laptop, big populations) each line
            // is minutes apart and users watch for exactly this.
            if let Some(handle) = &status {
                handle.update_generation(
                    stats.generation,
                    stats.best_gil,
                    stats.mean_gil,
                    stats.elapsed.as_millis() as u64,
                    &line,
                );
            }
            // Events (unthrottled, above the telemetry): new bests and
            // every-1000-generation milestones.
            let new_best = stats.best_gil > best_announced;
            let milestone = stats.generation > 0 && stats.generation % 1000 == 0;
            if new_best {
                best_announced = stats.best_gil;
                if let Some(handle) = &status {
                    handle.update(
                        stats.generation,
                        stats.best_gil,
                        stats.elapsed.as_millis() as u64,
                        &format!("★ new best {:.0} at generation {}", stats.best_gil, stats.generation),
                    );
                }
            } else if milestone {
                if let Some(handle) = &status {
                    handle.update(
                        stats.generation,
                        stats.best_gil,
                        stats.elapsed.as_millis() as u64,
                        &format!("◆ generation {} | best {:.0}", stats.generation, stats.best_gil),
                    );
                }
            }
            if last_write.elapsed() >= Duration::from_secs(5) {
                last_write = Instant::now();
                println!("{}{}", if new_best { "★ " } else { "" }, line);
                note(&mut trainer_log, &format!("{}{}", if new_best { "★ " } else { "" }, line));
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

//! The web console (owner feature request, Nov 7-8 2026 — product bar:
//! V3's fftb-server experience): the browser can START a run (params),
//! WATCH it (live status + log tail + dashboard), and STOP it (the
//! engine's cooperative stop flag). tiny_http on a dedicated thread; the
//! trainer itself stays synchronous — no async runtime, per V4's doctrine
//! of small boring pieces.
//!
//! Endpoints:
//!   GET  /                        dashboard (self-contained HTML)
//!   GET  /status                  live run state
//!   GET  /runs                    list of runs
//!   GET  /runs/{name}/report      report.json
//!   GET  /runs/{name}/artifact    winprob-v4.json
//!   GET  /runs/{name}/log?tail=N  run.log tail (default 200)
//!   POST /start                   {dataset, hours, population, seed,
//!                                 fitness, seed_files[]} → spawns a run
//!   POST /stop                    sets the active run's stop flag
//!
//! Optional ?token= guard (config monitor_token; empty = open; loopback
//! bind by default — LAN binding is an owner config decision).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::Deserialize;
use serde::Serialize;

pub const DEFAULT_LOG_TAIL: usize = 200;
const MAX_LOG_LINES: usize = 50;

#[derive(Debug, Clone, Serialize)]
pub struct RunStatus {
    pub running: bool,
    /// True when a stop has been requested but the run is still finishing
    /// (dataset build or current generation).
    pub stop_requested: bool,
    pub command: String,
    pub generation: u64,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub elapsed_ms: u64,
    pub population: usize,
    pub seed: u64,
    /// Genome length of the current/last run (0 = none yet) — feeds the
    /// memory estimator.
    pub genome_length: usize,
    /// Free system RAM right now (30s cache) — drives the estimator.
    pub available_bytes: u64,
    /// Trainer process CPU, normalized to total machine capacity
    /// (0-100 = idle..all cores saturated) — V1/V2 parity.
    pub cpu_percent: f64,
    /// Trainer process working set.
    pub process_memory_bytes: u64,
    /// Live holdout accuracy of the best checkpoint (temperature-free),
    /// evaluated periodically by the trainer's watcher thread. None until
    /// the first evaluation lands.
    pub holdout_accuracy: Option<f64>,
    /// Meaningful events: run config, dataset summary, ★ new bests,
    /// milestones, completion. The dashboard's primary log.
    pub last_log_lines: Vec<String>,
    /// Per-generation telemetry — every generation, no throttle (on
    /// slow-generation runs each line is minutes apart and precious).
    pub last_gen_lines: Vec<String>,
}

impl Default for RunStatus {
    fn default() -> Self {
        Self {
            running: false,
            stop_requested: false,
            command: String::new(),
            generation: 0,
            best_fitness: 0.0,
            mean_fitness: 0.0,
            elapsed_ms: 0,
            population: 0,
            seed: 0,
            genome_length: 0,
            available_bytes: 0,
            cpu_percent: 0.0,
            process_memory_bytes: 0,
            holdout_accuracy: None,
            last_log_lines: Vec::new(),
            last_gen_lines: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct StatusHandle {
    inner: Arc<Mutex<RunStatus>>,
}

impl StatusHandle {
    pub fn new(command: &str, population: usize, seed: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RunStatus {
                running: true,
                command: command.to_string(),
                population,
                seed,
                ..RunStatus::default()
            })),
        }
    }

    /// A not-running handle for the idle monitor.
    pub fn idle() -> Self {
        Self {
            inner: Arc::new(Mutex::new(RunStatus::default())),
        }
    }

    pub fn update(&self, generation: u64, best_fitness: f64, elapsed_ms: u64, line: &str) {
        let mut status = self.inner.lock().unwrap();
        status.generation = generation;
        status.best_fitness = best_fitness;
        status.elapsed_ms = elapsed_ms;
        let mut trimmed: VecDeque<String> =
            status.last_log_lines.clone().into_iter().collect();
        trimmed.push_back(line.to_string());
        while trimmed.len() > MAX_LOG_LINES {
            trimmed.pop_front();
        }
        status.last_log_lines = trimmed.into_iter().collect();
    }

    /// Per-generation telemetry feed: refreshes the metrics and records the
    /// line in the generations tail WITHOUT touching the events log.
    pub fn update_generation(
        &self,
        generation: u64,
        best_fitness: f64,
        mean_fitness: f64,
        elapsed_ms: u64,
        line: &str,
    ) {
        let mut status = self.inner.lock().unwrap();
        status.generation = generation;
        status.best_fitness = best_fitness;
        status.mean_fitness = mean_fitness;
        status.elapsed_ms = elapsed_ms;
        let mut trimmed: VecDeque<String> =
            status.last_gen_lines.clone().into_iter().collect();
        trimmed.push_back(line.to_string());
        while trimmed.len() > MAX_LOG_LINES {
            trimmed.pop_front();
        }
        status.last_gen_lines = trimmed.into_iter().collect();
    }

    pub fn set_population(&self, population: usize) {
        self.inner.lock().unwrap().population = population;
    }

    pub fn set_genome_length(&self, genome_length: usize) {
        self.inner.lock().unwrap().genome_length = genome_length;
    }

    pub fn set_holdout_accuracy(&self, accuracy: f64) {
        self.inner.lock().unwrap().holdout_accuracy = Some(accuracy);
    }

    pub fn finish(&self) {
        self.inner.lock().unwrap().running = false;
    }

    pub fn snapshot(&self) -> RunStatus {
        self.inner.lock().unwrap().clone()
    }
}

/// Parameters accepted by POST /start (JSON body).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct StartParams {
    #[serde(default = "default_dataset")]
    pub dataset: String,
    #[serde(default = "default_hours")]
    pub hours: f64,
    #[serde(default)]
    pub population: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_fitness")]
    pub fitness: String,
    #[serde(default)]
    pub seed_files: Vec<String>,
    #[serde(default)]
    pub max_tournament: i64,
    /// Chronological lookback: train on only the newest N tournaments
    /// (None / 0 = full corpus).
    #[serde(default)]
    pub tournaments: Option<usize>,
    /// Memory fraction for population auto-sizing (estimator slider).
    #[serde(default = "default_fraction")]
    pub memory_fraction: f64,
}

fn default_fraction() -> f64 {
    0.7
}

fn default_dataset() -> String { "deep".into() }
fn default_hours() -> f64 { 6.0 }
fn default_seed() -> u64 { 42 }
fn default_fitness() -> String { "margin".into() }

struct ActiveRun {
    stop: Arc<AtomicBool>,
    /// Set by the run thread when it completes; the monitor reaps the slot
    /// lazily (start/stop/status check it).
    finished: Arc<AtomicBool>,
    out_dir: PathBuf,
}

/// Owns run management: one run at a time, startable/stoppable from HTTP
/// or the CLI (`train --serve` adopts its in-process run).
pub struct Monitor {
    runs_dir: PathBuf,
    config: crate::config::V4Config,
    status: Mutex<StatusHandle>,
    active: Mutex<Option<ActiveRun>>,
    /// Cached console botland leaderboard (computed once; ?refresh=1).
    botland: Mutex<Option<crate::botland::BotlandJson>>,
    botland_computing: std::sync::atomic::AtomicBool,
    /// Available-RAM cache (the probe spawns PowerShell; polled status
    /// should not). TTL 30s.
    memory_cache: Mutex<(std::time::Instant, u64)>,
    /// Rolling process-CPU samples (wall instant, cumulative cpu seconds,
    /// working set bytes) from a background refresher — V1/V2 parity: the
    /// dashboard shows how hard the trainer is actually working, which
    /// distinguishes a grinding dataset build from a hang.
    cpu_samples: Arc<Mutex<VecDeque<(std::time::Instant, f64, u64)>>>,
}

/// Parse "cpu_seconds workingset_bytes" from the probe.
fn parse_probe(text: &str) -> Option<(f64, u64)> {
    let mut parts = text.split_whitespace();
    let cpu: f64 = parts.next()?.parse().ok()?;
    let ws: u64 = parts.next()?.parse().ok()?;
    Some((cpu, ws))
}

/// Machine-normalized CPU % (0-100 across ALL cores) from the sample
/// window. Pure function for testing.
fn cpu_percent_from(
    samples: &VecDeque<(std::time::Instant, f64, u64)>,
    logical_cores: usize,
) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let (t0, c0, _) = samples.front().unwrap();
    let (t1, c1, _) = samples.back().unwrap();
    let wall = t1.duration_since(*t0).as_secs_f64();
    if wall <= 0.0 || logical_cores == 0 {
        return 0.0;
    }
    let delta_cpu = (c1 - c0).max(0.0);
    (delta_cpu / wall / logical_cores as f64 * 100.0).clamp(0.0, 100.0)
}

impl Monitor {
    pub fn new(runs_dir: PathBuf, config: crate::config::V4Config) -> Self {
        let monitor = Self {
            runs_dir,
            config,
            status: Mutex::new(StatusHandle::idle()),
            active: Mutex::new(None),
            botland: Mutex::new(None),
            botland_computing: std::sync::atomic::AtomicBool::new(false),
            memory_cache: Mutex::new((
                std::time::Instant::now() - std::time::Duration::from_secs(3600),
                0,
            )),
            cpu_samples: Arc::new(Mutex::new(VecDeque::new())),
        };
        // Background refresher: probe process CPU + working set every 5s so
        // status polls never pay PowerShell's latency.
        let samples = Arc::clone(&monitor.cpu_samples);
        let pid = std::process::id();
        std::thread::spawn(move || loop {
            if let Ok(output) = std::process::Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    &format!(
                        "$p = Get-Process -Id {pid}; \"$($p.TotalProcessorTime.TotalSeconds) $($p.WorkingSet64)\""
                    ),
                ])
                .output()
            {
                if let Some((cpu, ws)) =
                    parse_probe(&String::from_utf8_lossy(&output.stdout))
                {
                    let mut queue = samples.lock().unwrap();
                    queue.push_back((std::time::Instant::now(), cpu, ws));
                    while queue.len() > 12 {
                        queue.pop_front();
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_secs(5));
        });
        monitor
    }

    fn available_bytes_cached(&self) -> u64 {
        let mut cache = self.memory_cache.lock().unwrap();
        if cache.0.elapsed() > std::time::Duration::from_secs(30) {
            cache.0 = std::time::Instant::now();
            cache.1 = crate::train::available_bytes();
        }
        cache.1
    }

    pub fn status(&self) -> RunStatus {
        let mut snapshot = self.status.lock().unwrap().snapshot();
        snapshot.available_bytes = self.available_bytes_cached();
        snapshot.stop_requested = self
            .active
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|run| run.stop.load(Ordering::Relaxed));
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        snapshot.cpu_percent = cpu_percent_from(&self.cpu_samples.lock().unwrap(), cores);
        snapshot.process_memory_bytes = self
            .cpu_samples
            .lock()
            .unwrap()
            .back()
            .map(|(_, _, ws)| *ws)
            .unwrap_or(0);
        snapshot
    }

    /// `train --serve`: the CLI's in-process run registers itself so the
    /// console can watch and stop it. The finished flag never fires here —
    /// the CLI exits with the run — so reap() leaves the slot alone; only
    /// long-lived servers (start_run) need slot recycling.
    pub fn adopt(&self, handle: StatusHandle, stop: Arc<AtomicBool>, out_dir: PathBuf) {
        *self.status.lock().unwrap() = handle;
        *self.active.lock().unwrap() = Some(ActiveRun {
            stop,
            finished: Arc::new(AtomicBool::new(false)),
            out_dir,
        });
    }

    /// Clear the active slot if its run thread has finished.
    fn reap(&self) {
        let mut active = self.active.lock().unwrap();
        if active
            .as_ref()
            .is_some_and(|run| run.finished.load(Ordering::SeqCst))
        {
            *active = None;
        }
    }

    pub fn start_run(&self, params: StartParams) -> anyhow::Result<String> {
        self.reap();
        let mut active = self.active.lock().unwrap();
        if let Some(run) = active.as_ref() {
            anyhow::bail!("a run is already active ({})", run.out_dir.display());
        }
        let fitness = match params.fitness.as_str() {
            "margin" => crate::train::FitnessKind::Margin,
            "accuracy" => crate::train::FitnessKind::Accuracy,
            "logloss" => crate::train::FitnessKind::LogLoss,
            other => anyhow::bail!("unknown fitness {other}"),
        };
        let deep = params.dataset == "deep";
        anyhow::ensure!(
            params.dataset == "deep" || params.dataset == "main",
            "dataset must be deep or main"
        );
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let out_dir = self
            .runs_dir
            .join(format!("{}-{ts}", if deep { "deep" } else { "main" }));
        let stop = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let command = format!(
            "train --dataset {} --hours {} --population {} --seed {} --fitness {}",
            params.dataset, params.hours, params.population, params.seed, params.fitness
        );
        let handle = StatusHandle::new(&command, params.population, params.seed);
        let cfg = crate::train::TrainConfig {
            hours: params.hours,
            population: params.population,
            seed: params.seed,
            deep,
            fitness,
            memory_fraction: params.memory_fraction.clamp(0.05, 0.95),
            out_dir: out_dir.clone(),
            data_dir: self.config.data_dir.clone(),
            max_tournament_id: params.max_tournament,
            tournaments: params.tournaments,
            seeds: params.seed_files.iter().map(PathBuf::from).collect(),
            stop_signal: Some(Arc::clone(&stop)),
        };
        let run_handle = handle.clone();
        let finished_flag = Arc::clone(&finished);
        std::thread::spawn(move || {
            if let Err(e) = crate::train::run_inner(&cfg, Some(run_handle.clone())) {
                eprintln!("run failed: {e:#}");
                run_handle.update(0, 0.0, 0, &format!("RUN FAILED: {e:#}"));
                run_handle.finish();
            }
            finished_flag.store(true, Ordering::SeqCst);
        });
        *self.status.lock().unwrap() = handle;
        let name = out_dir
            .file_name()
            .map(|n| n.to_string_lossy().into())
            .unwrap_or_default();
        *active = Some(ActiveRun {
            stop,
            finished,
            out_dir,
        });
        Ok(name)
    }

    pub fn stop_run(&self) -> bool {
        self.reap();
        let active = self.active.lock().unwrap();
        match active.as_ref() {
            Some(run) => {
                run.stop.store(true, Ordering::SeqCst);
                true
            }
            None => false,
        }
    }

    fn runs_list(&self) -> Vec<String> {
        let mut runs: Vec<String> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.runs_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    runs.push(entry.file_name().to_string_lossy().into());
                }
            }
        }
        runs.sort();
        runs
    }
}

/// Reject path traversal: run names are single path segments.
fn safe_name(name: &str) -> bool {
    !name.is_empty() && !name.contains('/') && !name.contains('\\') && !name.contains("..")
}

pub fn start_server(
    monitor: Arc<Monitor>,
    token: Option<String>,
    bind: &str,
) -> anyhow::Result<(std::thread::JoinHandle<()>, String)> {
    let server = tiny_http::Server::http(bind)
        .map_err(|e| anyhow::anyhow!("cannot bind {bind}: {e}"))?;
    let addr = server.server_addr().to_string();
    let handle = std::thread::spawn(move || {
        for mut request in server.incoming_requests() {
            let url = request.url().to_string();
            let (path, query) = match url.split_once('?') {
                Some((p, q)) => (p.to_string(), q.to_string()),
                None => (url.clone(), String::new()),
            };
            if let Some(expected) = &token {
                // Search ALL pairs — the token may sit anywhere in the
                // query string (regression: ?population=N&token=… 403'd
                // when the extractor only inspected the first pair).
                let supplied = query
                    .split('&')
                    .filter_map(|pair| pair.split_once('='))
                    .find(|(k, _)| *k == "token")
                    .map(|(_, v)| v.to_string());
                if supplied.as_deref() != Some(expected.as_str()) {
                    let response = tiny_http::Response::from_string("forbidden".to_string())
                        .with_status_code(403);
                    request.respond(response).ok();
                    continue;
                }
            }
            let response = match (request.method(), path.trim_matches('/')) {
                (tiny_http::Method::Get, "") => tiny_http::Response::from_string(
                    include_str!("dashboard.html"),
                )
                .with_header(
                    tiny_http::Header::from_bytes(
                        &b"Content-Type"[..],
                        &b"text/html; charset=utf-8"[..],
                    )
                    .unwrap(),
                ),
                (tiny_http::Method::Get, "status") => json_response(&monitor.status()),
                (tiny_http::Method::Get, "estimate") => {
                    // Memory calculator (V3-console parity): live available
                    // RAM, the last run's genome length, the suggested
                    // population at a fraction, and — when ?population=N is
                    // supplied — the double-buffered footprint for that
                    // population. Footprint per genome: 2×genes×4B + 8B.
                    let query_param = |key: &str| {
                        query
                            .split('&')
                            .filter_map(|pair| pair.split_once('='))
                            .find(|(k, _)| *k == key)
                            .and_then(|(_, v)| v.parse::<f64>().ok())
                    };
                    let available = crate::train::available_bytes();
                    let genome_length = if query_param("genes").is_some() {
                        query_param("genes").unwrap_or(0.0) as usize
                    } else {
                        let known = monitor.status().genome_length;
                        if known > 0 { known } else { 1260 }
                    };
                    let fraction = query_param("fraction").unwrap_or(0.7).clamp(0.05, 0.95);
                    // Tournament window: rows and dataset memory scale with
                    // the lookback; the genome length stays the full-corpus
                    // vocabulary (conservative — windowed genomes are
                    // shorter). 8 matches per tournament.
                    const FULL_CORPUS_MATCHES: usize = 554_072;
                    const ROW_BYTES: u64 = 480;
                    const BUILD_TRANSIENT_BYTES_PER_MATCH: u64 = 3_000;
                    let tournaments = query_param("tournaments").unwrap_or(0.0).max(0.0) as usize;
                    let matches = if tournaments > 0 {
                        tournaments.saturating_mul(8)
                    } else {
                        FULL_CORPUS_MATCHES
                    };
                    let dataset_bytes = matches as u64 * ROW_BYTES;
                    let build_transient = matches as u64 * BUILD_TRANSIENT_BYTES_PER_MATCH;
                    let suggested = fftb_ga::population::population_for_memory(
                        available.saturating_sub(dataset_bytes),
                        genome_length,
                        fraction,
                    );
                    let mut payload = serde_json::json!({
                        "available_bytes": available,
                        "genome_length": genome_length,
                        "fraction": fraction,
                        "tournaments": tournaments,
                        "matches": matches,
                        "dataset_bytes": dataset_bytes,
                        "suggested_population": suggested,
                        "footprint_per_genome_bytes": genome_length * 8 + 8,
                    });
                    if let Some(pop) = query_param("population") {
                        let pop = pop.max(0.0) as usize;
                        let population_bytes = pop as u64 * (genome_length as u64 * 8 + 8);
                        // Two lifecycle peaks: build (dataset + transient,
                        // before the population allocates) and training
                        // (dataset + population, after the transient drops).
                        let build_peak = dataset_bytes + build_transient;
                        let training_peak = dataset_bytes + population_bytes;
                        let peak = build_peak.max(training_peak);
                        payload["population"] = serde_json::json!(pop);
                        payload["population_bytes"] = serde_json::json!(population_bytes);
                        payload["peak_bytes"] = serde_json::json!(peak);
                        payload["fits"] =
                            serde_json::json!(peak as f64 <= available as f64 * fraction);
                    }
                    json_response(&payload)
                }
                (tiny_http::Method::Get, "runs") => {
                    json_response(&serde_json::json!({ "runs": monitor.runs_list() }))
                }
                (tiny_http::Method::Get, "botland") => {
                    let refresh = query.contains("refresh=1");
                    match monitor.botland_begin(refresh) {
                        Some(json) => json_response(&json),
                        None => tiny_http::Response::from_string(
                            "{\"computing\": true}".to_string(),
                        )
                        .with_status_code(503)
                        .with_header(
                            tiny_http::Header::from_bytes(
                                &b"Content-Type"[..],
                                &b"application/json"[..],
                            )
                            .unwrap(),
                        ),
                    }
                }
                (tiny_http::Method::Post, "stop") => {
                    let stopped = monitor.stop_run();
                    json_response(&serde_json::json!({ "stop_requested": stopped }))
                }
                (tiny_http::Method::Post, "start") => {
                    let mut body = String::new();
                    use std::io::Read;
                    if let Err(e) = request.as_reader().read_to_string(&mut body) {
                        let response = tiny_http::Response::from_string(format!(
                            "cannot read body: {e}"
                        ))
                        .with_status_code(400);
                        request.respond(response).ok();
                        continue;
                    }
                    match serde_json::from_str::<StartParams>(&body)
                        .map_err(|e| anyhow::anyhow!("bad params: {e}"))
                        .and_then(|params| monitor.start_run(params))
                    {
                        Ok(name) => {
                            json_response(&serde_json::json!({ "started": true, "run": name }))
                        }
                        Err(e) => tiny_http::Response::from_string(format!("{e:#}"))
                            .with_status_code(409),
                    }
                }
                (tiny_http::Method::Get, path) => {
                    let segments: Vec<&str> = path.split('/').collect();
                    match segments.as_slice() {
                        ["runs", name, "report"] if safe_name(name) => {
                            file_or_404(monitor.runs_dir_path().join(name).join("report.json"))
                        }
                        ["runs", name, "artifact"] if safe_name(name) => {
                            file_or_404(monitor.runs_dir_path().join(name).join("winprob-v4.json"))
                        }
                        ["runs", name, "log"] if safe_name(name) => {
                            let tail = query
                                .split('&')
                                .filter_map(|pair| pair.split_once('='))
                                .find(|(k, _)| *k == "tail")
                                .and_then(|(_, v)| v.parse::<usize>().ok())
                                .unwrap_or(DEFAULT_LOG_TAIL);
                            // run.log = launcher-redirected stdout; trainer.log
                            // = the trainer's own log (console-launched runs
                            // have no redirect — verification caught this).
                            let base = monitor.runs_dir_path().join(name);
                            let text = std::fs::read_to_string(base.join("run.log"))
                                .or_else(|_| std::fs::read_to_string(base.join("trainer.log")));
                            match text {
                                Ok(text) => {
                                    let lines: Vec<&str> = text.lines().collect();
                                    let start = lines.len().saturating_sub(tail);
                                    text_response(lines[start..].join("\n"))
                                }
                                Err(_) => not_found("run or log not found"),
                            }
                        }
                        _ => not_found("unknown endpoint"),
                    }
                }
                _ => not_found("method not allowed"),
            };
            request.respond(response).ok();
        }
    });
    Ok((handle, addr))
}

impl Monitor {
    fn runs_dir_path(&self) -> &Path {
        &self.runs_dir
    }

    /// Cached leaderboard if computed.
    pub fn botland_cached(&self) -> Option<crate::botland::BotlandJson> {
        self.botland.lock().unwrap().clone()
    }

    /// GET /botland handler (V3's api_botland pattern): serve the cache,
    /// or kick a background compute over the newest 1000 tournaments.
    /// Returns None while the first compute runs.
    pub fn botland_begin(self: &Arc<Self>, refresh: bool) -> Option<crate::botland::BotlandJson> {
        if refresh {
            *self.botland.lock().unwrap() = None;
        }
        if let Some(json) = self.botland_cached() {
            return Some(json);
        }
        if !self.botland_computing.swap(true, Ordering::SeqCst) {
            let this = Arc::clone(self);
            std::thread::spawn(move || {
                let result = (|| {
                    let entries = crate::botland::newest_entries(
                        &this.config.data_dir.join("tournamentCache"),
                        &this.config.data_dir.join("tournamentCacheDeep"),
                        1000,
                    )?;
                    let roster_xml =
                        std::fs::read_to_string(this.config.data_dir.join("Botland.xml"))?;
                    let gene_dirs = vec![this.config.data_dir.join("GeneFiles")];
                    let weights =
                        crate::botland::newest_checkpoint_weights(&this.runs_dir);
                    crate::botland::compute_baseline(
                        &entries,
                        &roster_xml,
                        &gene_dirs,
                        weights.as_ref(),
                    )
                })();
                match result {
                    Ok(json) => *this.botland.lock().unwrap() = Some(json),
                    Err(e) => {
                        eprintln!("botland compute failed: {e:#}");
                        // Surface the failure to callers via an empty table.
                        *this.botland.lock().unwrap() = Some(crate::botland::BotlandJson {
                            rows: Vec::new(),
                            matches: 0,
                            tournaments: 0,
                            v4_beats_inverse: None,
                        });
                    }
                }
                this.botland_computing.store(false, Ordering::SeqCst);
            });
        }
        None
    }
}

fn json_response<T: Serialize>(value: &T) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let body = serde_json::to_string(value).unwrap_or_else(|_| "{}".into());
    tiny_http::Response::from_string(body).with_header(
        tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
    )
}

fn text_response(body: String) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    tiny_http::Response::from_string(body).with_header(
        tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/plain; charset=utf-8"[..])
            .unwrap(),
    )
}

fn file_or_404(path: PathBuf) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    match std::fs::read_to_string(&path) {
        Ok(text) => text_response(text),
        Err(_) => not_found("run or file not found"),
    }
}

fn not_found(message: &str) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    tiny_http::Response::from_string(message.to_string()).with_status_code(404)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_updates_and_tails_logs() {
        let status = StatusHandle::new("train --dataset deep", 20_000, 42);
        status.update(7, -100.5, 60_000, "gen 7 | best -100");
        status.update(8, -99.0, 70_000, "gen 8 | best -99");
        let snapshot = status.snapshot();
        assert!(snapshot.running);
        assert_eq!(snapshot.generation, 8);
        assert_eq!(snapshot.last_log_lines.len(), 2);
        status.finish();
        assert!(!status.snapshot().running);
    }

    #[test]
    fn token_accepted_in_any_query_position() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap();
        let config = crate::config::V4Config::load(repo_root).expect("shipped config");
        let monitor = Arc::new(Monitor::new(
            std::env::temp_dir().join("v4-monitor-token-test"),
            config,
        ));
        let (_, addr) = start_server(monitor, Some("sekret".into()), "127.0.0.1:0").unwrap();
        let base = format!("http://{addr}");
        // Token FIRST and token NOT-first must both authenticate…
        assert_eq!(
            ureq::get(&format!("{base}/status"))
                .query("token", "sekret")
                .call()
                .unwrap()
                .status(),
            200
        );
        assert_eq!(
            ureq::get(&format!("{base}/status"))
                .query("population", "5")
                .query("token", "sekret")
                .call()
                .unwrap()
                .status(),
            200
        );
        // …and absence must be refused (ureq surfaces non-2xx as Err).
        let refused = ureq::get(&format!("{base}/status")).call();
        assert!(
            matches!(&refused, Err(ureq::Error::Status(403, _))),
            "expected 403 without token, got {refused:?}"
        );
    }

    #[test]
    fn cpu_percent_math_normalizes_to_all_cores() {
        let now = std::time::Instant::now();
        let mut samples: VecDeque<(std::time::Instant, f64, u64)> = VecDeque::new();
        // 8 cores, 10s window, 40 cpu-seconds consumed = 50% of the machine.
        samples.push_back((now, 100.0, 0));
        samples.push_back((now + std::time::Duration::from_secs(10), 140.0, 0));
        assert!((cpu_percent_from(&samples, 8) - 50.0).abs() < 0.01);
        // Degenerate windows are safe.
        samples.clear();
        samples.push_back((now, 100.0, 0));
        assert_eq!(cpu_percent_from(&samples, 8), 0.0);
    }

    #[test]
    fn start_run_spawns_watches_and_stops() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap();
        let config = crate::config::V4Config::load(repo_root).expect("shipped config");
        let runs_dir = std::env::temp_dir().join("v4-monitor-run-test");
        std::fs::create_dir_all(&runs_dir).unwrap();
        let monitor = Arc::new(Monitor::new(runs_dir.clone(), config));

        // One at a time: second start while the first is active is refused.
        let name = monitor
            .start_run(StartParams {
                dataset: "main".into(),
                hours: 0.02,
                population: 50,
                seed: 7,
                fitness: "margin".into(),
                seed_files: vec![],
                max_tournament: 0,
                tournaments: Some(40),
                memory_fraction: 0.7,
            })
            .expect("first start");
        assert!(monitor.start_run(StartParams::default()).is_err());

        // The run finishes on its own (72s time cap + calibration/export);
        // poll until idle with headroom well past the run's own budget.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
        while monitor.status().running && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        assert!(!monitor.status().running, "run should finish");
        assert!(runs_dir.join(&name).join("report.json").exists(), "report written");
        assert!(runs_dir.join(&name).join("winprob-v4.json").exists());
        assert!(
            runs_dir.join(&name).join("trainer.log").exists(),
            "console-launched runs keep their own log"
        );

        // After completion a new run is allowed again, and stop on idle
        // reports false.
        assert!(!monitor.stop_run());
    }
}

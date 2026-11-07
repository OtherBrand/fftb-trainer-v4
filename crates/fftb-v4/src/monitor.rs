//! The HTTP monitoring API (owner feature request, Nov 7 2026): any machine
//! running V4 becomes watchable by any session with curl — built for the
//! laptop training node joining over the LAN. tiny_http on a dedicated
//! thread; the training loop publishes status through an Arc<Mutex<>>.
//!
//! Endpoints (GET only):
//!   /status                    live run state
//!   /runs                      list of data/runs/*
//!   /runs/{name}/report        report.json
//!   /runs/{name}/artifact      winprob-v4.json
//!   /runs/{name}/log[?tail=N]  run.log, last N lines (default 200)
//!
//! Optional ?token= guard (config monitor_token; empty = open, home LAN).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::Serialize;

pub const DEFAULT_LOG_TAIL: usize = 200;
const MAX_LOG_LINES: usize = 50;

#[derive(Debug, Clone, Serialize)]
pub struct RunStatus {
    pub running: bool,
    pub command: String,
    pub generation: u64,
    pub best_fitness: f64,
    pub elapsed_ms: u64,
    pub population: usize,
    pub seed: u64,
    pub last_log_lines: Vec<String>,
}

impl Default for RunStatus {
    fn default() -> Self {
        Self {
            running: false,
            command: String::new(),
            generation: 0,
            best_fitness: 0.0,
            elapsed_ms: 0,
            population: 0,
            seed: 0,
            last_log_lines: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct StatusHandle {
    inner: Arc<Mutex<RunStatus>>,
}

impl StatusHandle {
    pub fn new(command: &str, population: usize, seed: u64) -> Self {
        let handle = Self {
            inner: Arc::new(Mutex::new(RunStatus {
                running: true,
                command: command.to_string(),
                population,
                seed,
                ..RunStatus::default()
            })),
        };
        handle
    }

    /// A not-running handle for the standalone `serve` subcommand.
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

    pub fn finish(&self) {
        self.inner.lock().unwrap().running = false;
    }

    pub fn snapshot(&self) -> RunStatus {
        self.inner.lock().unwrap().clone()
    }
}

/// Reject path traversal: run names are single path segments.
fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && !name.contains('/')
        && !name.contains('\\')
        && !name.contains("..")
}

/// Serve forever on its own thread. Returns the JoinHandle (never joined in
/// practice — the thread lives with the process).
pub fn start_server(
    status: StatusHandle,
    runs_dir: PathBuf,
    token: Option<String>,
    bind: &str,
) -> anyhow::Result<std::thread::JoinHandle<()>> {
    let server = tiny_http::Server::http(bind)
        .map_err(|e| anyhow::anyhow!("cannot bind {bind}: {e}"))?;
    let handle = std::thread::spawn(move || {
        for mut request in server.incoming_requests() {
            let url = request.url().to_string();
            let (path, query) = match url.split_once('?') {
                Some((p, q)) => (p.to_string(), q.to_string()),
                None => (url.clone(), String::new()),
            };
            // Token guard: present-and-matching, or open when unconfigured.
            if let Some(expected) = &token {
                let supplied = query
                    .split('&')
                    .find_map(|pair| pair.split_once('='))
                    .filter(|(k, _)| *k == "token")
                    .map(|(_, v)| v.to_string());
                if supplied.as_deref() != Some(expected.as_str()) {
                    let mut response =
                        tiny_http::Response::from_string("forbidden".to_string())
                            .with_status_code(403);
                    request.respond(response).ok();
                    continue;
                }
            }
            let response = route(&status, &runs_dir, &path, &query);
            request.respond(response).ok();
        }
    });
    Ok(handle)
}

fn route(status: &StatusHandle, runs_dir: &Path, path: &str, query: &str) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let segments: Vec<&str> = path.trim_matches('/').split('/').collect();
    match segments.as_slice() {
        ["status"] => json_response(&status.snapshot()),
        ["runs"] => {
            let mut runs: Vec<String> = Vec::new();
            if let Ok(entries) = std::fs::read_dir(runs_dir) {
                for entry in entries.flatten() {
                    if entry.path().is_dir() {
                        runs.push(entry.file_name().to_string_lossy().into());
                    }
                }
                runs.sort();
            }
            json_response(&serde_json::json!({ "runs": runs }))
        }
        ["runs", name, "report"] if safe_name(name) => {
            file_or_404(runs_dir.join(name).join("report.json"))
        }
        ["runs", name, "artifact"] if safe_name(name) => {
            file_or_404(runs_dir.join(name).join("winprob-v4.json"))
        }
        ["runs", name, "log"] if safe_name(name) => {
            let tail = query
                .split('&')
                .find_map(|pair| pair.split_once('='))
                .filter(|(k, _)| *k == "tail")
                .and_then(|(_, v)| v.parse::<usize>().ok())
                .unwrap_or(DEFAULT_LOG_TAIL);
            match std::fs::read_to_string(runs_dir.join(name).join("run.log")) {
                Ok(text) => {
                    let lines: Vec<&str> = text.lines().collect();
                    let start = lines.len().saturating_sub(tail);
                    let tail_text = lines[start..].join("\n");
                    text_response(tail_text)
                }
                Err(_) => not_found("run or log not found"),
            }
        }
        _ => not_found("unknown endpoint"),
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
    fn router_serves_runs_reports_and_404s() {
        let dir = std::env::temp_dir().join("v4-monitor-test");
        std::fs::create_dir_all(dir.join("run-a")).unwrap();
        std::fs::write(dir.join("run-a").join("report.json"), r#"{"ok":1}"#).unwrap();
        std::fs::write(dir.join("run-a").join("run.log"), "l1\nl2\nl3\n").unwrap();

        let status = StatusHandle::idle();
        let response = route(&status, &dir, "/runs/run-a/report", "");
        assert_eq!(response.status_code(), 200);
        let log = route(&status, &dir, "/runs/run-a/log", "tail=2");
        assert_eq!(log.status_code(), 200);
        assert_eq!(
            route(&status, &dir, "/runs/run-a/../../etc/passwd", "").status_code(),
            404
        );
        assert_eq!(route(&status, &dir, "/nope", "").status_code(), 404);
    }
}

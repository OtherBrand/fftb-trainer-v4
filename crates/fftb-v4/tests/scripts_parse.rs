//! Script gate (owner feedback, Nov 8 2026): every PowerShell script in
//! the repo must parse under the REAL Windows PowerShell 5.1 parser before
//! anything ships. This gate exists because a shipped Open-Monitor.ps1
//! contained a UTF-8 em dash that PS 5.1 misreads as ANSI — its 0x94 byte
//! is a smart quote, which PowerShell accepts as a string delimiter — and
//! the script was never executed before being called verified.
//! Reading a script is not testing it.

use std::path::Path;
use std::process::Command;

#[test]
fn repo_powershell_scripts_parse_clean() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("repo root");
    let mut scripts: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(repo_root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("ps1") {
                scripts.push(path);
            }
        }
    }
    // The gate is meaningful even when the list is empty only if we know
    // scripts exist; assert we found at least the trainer launcher.
    assert!(
        scripts.iter().any(|p| p.file_name().is_some_and(|n| n == "start-trainer-v4.ps1")),
        "expected start-trainer-v4.ps1 in the repo root"
    );
    for script in &scripts {
        let output = Command::new("powershell")
            .env("SCRIPT_PATH", script)
            .args([
                "-NoProfile",
                "-Command",
                "$t=$null; $e=$null; [System.Management.Automation.Language.Parser]::ParseFile($env:SCRIPT_PATH, [ref]$t, [ref]$e) | Out-Null; if ($e.Count -gt 0) { $e | ForEach-Object { Write-Output ($_.Message + ' at line ' + $_.Extent.StartLineNumber) }; exit 1 } else { exit 0 }",
            ])
            .output()
            .expect("powershell runs");
        let name = script.file_name().unwrap_or_default().to_string_lossy();
        assert!(
            output.status.success(),
            "{name} failed to parse:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }
}

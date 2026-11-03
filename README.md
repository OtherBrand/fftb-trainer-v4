# FFTB Gene Trainer V4

V1's proven match-winner model — linear attribute weights, pick the side with
the higher score — rebuilt in Rust and trained by an off-the-shelf genetic
algorithm library. Self-contained: no V3 crate dependencies; tournament data
is read from the shared V3 cache directory, and the botland scoreboard lives
in the shared [fftb-trainer-botland](https://github.com/OtherBrand/fftb-trainer-botland) repo.

## Status

Phase 0 — scaffold. `fftb-trainer-v4 maps` works (live Maps.txt fetch with
local fallback). Training, advisor export, and botland land in Phases 1–3;
see CLAUDE.md for the plan and the architecture doctrine.

## Build

```
cargo build --release
target\release\fftb-trainer-v4.exe maps
```

Requires Rust 1.81+. Run from the repo root.

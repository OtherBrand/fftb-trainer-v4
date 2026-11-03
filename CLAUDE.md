# FFTB Gene Trainer V4

V1's match-winner model in Rust, trained by an off-the-shelf GA library.
The workhorse that made positive gil on stream, rebuilt with oomph.

- V3 (Rust, gil-economics GA): `C:\Projects\gene-trainer-3` — github.com/OtherBrand/fftb-trainer-v3
- V2 (Java, frozen Oct 17 2026): `C:\Projects\gene-trainer-2\FFTB-Gene-Trainer-2\Gene-Trainer-V2`
- V1 (Java, original): `C:\Projects\gene-trainer`
- Shared botland sim core: github.com/OtherBrand/fftb-trainer-botland (extraction in progress; initial cut pinned at gene-trainer-3 `7b10221^`)
- Viewer (consumes advisor artifacts + fields botland bots): `C:\Projects\FFTBViewer`

## Quick start

```
cargo build --release
target\release\fftb-trainer-v4.exe maps    # live Maps.txt fetch + report (works today)
```

Run all commands from the repo root (family doctrine: data resolves against cwd).

## Architecture doctrine (owner decisions, Nov 1 2026 — do not undo)

1. **Self-contained.** ZERO gene-trainer-3 crate dependencies. V4 reimplements
   its own cache parsing (each trainer cares about different aspects of
   tournament data). The cross-family contracts are FILE FORMATS — cache .gz
   JSON, winner files, Botland.xml, WinProbModel JSON — pinned by FIXTURES
   from the shared botland repo, never by another trainer's structs.
2. **The seam.** `seam.rs` holds thin local stand-ins for the shared botland
   types. When fftb-trainer-botland lands, delete it and depend on the shared
   crate (git rev pin). Keep its surface minimal — everything added there is
   extraction debt.
3. **Data.** Tournament caches are READ from `../gene-trainer-3/data`
   (`config.toml: data_dir`) — never written. V4's own outputs go in `./data`
   (gitignored). Maps.txt is fetched LIVE first (stale map data was the bug),
   falling back to the local cache with a loud warning when the dump host is
   down (that host has killed endpoints before).
4. **Model = V1.** Linear weights over the differenced feature space
   (`attr:` counts, `agg:brave/faith/raidBoss` sums, `map:` one-hot,
   `side:` one-hot — the WinProbModel contract names, exactly). Signed f32
   genes −100..100. Fitness is V1's asymmetric margin score
   (correct → +winner score; wrong → −5× loser score;
   `fitness.rs: NEGATIVE_SCORE_MULTIPLIER`), verified against V1's Java
   source. Alternates (`accuracy`, `logloss`) ship behind flags.
5. **GA engine: off-the-shelf library** (owner's explicit choice over bespoke
   SoA — memory is not the constraint in Rust). Candidates vetted Nov 1:
   `genetic_algorithms` 3.0.0 and `radiate` 1.3.2; spike before committing
   (ergonomics at 20k pop, genomes/s at 500k+). Population ambitions stay
   ≤~500k on a library (boxed per-genome storage hits the memory-bandwidth
   wall beyond that; accepted trade).
6. **Advisor export.** Calibrated WinProbModel JSON (linear, name-keyed,
   weights sorted, temperature pre-folded) = zero Viewer changes on swap.
   The gate: V4 claims the advisor slot only by beating deep-linear's
   **62.01% / 0.6495 logloss / 0.2288 brier** on the same chronological
   holdout. Viewer-side accommodation for richer models is owner-sanctioned,
   but any richer head ships with an ATTRIBUTION plan (the "why" line is a
   product feature).
7. **Governance.** Frozen replay-semantics changes in the shared botland repo
   (scorer/settle/winner-file format) require cross-review by the other
   trainer's session before merge. The gene-trainer-2 session owns the parity
   fixtures and conformance tests; this session holds consumer-with-standing
   commit rights.

## Code style (owner preference, family-wide)

Organize code class-like: one type per concern, `impl` blocks with `new()`
constructors, traits as interfaces. (V3's SoA exception does NOT apply here —
V4 uses a library engine, not a hand-rolled hot core.)

## Phase status

- Phase 0 (scaffold + Maps.txt live fetch): **done** (this commit).
- Phase 1: cache reader + dataset builder + feature extraction + GA engine
  (after the spike) + incremental winner files + train CLI.
- Phase 2: calibration + WinProbModel export + holdout gate vs deep-linear.
- Phase 3: botland via the shared repo (swap `seam.rs`) + roster entry.
- Phase 4 (optional): GA-selected pair features (`attr:A×attr:B` — the one
  accuracy path where evolution beats gradient; must beat both deep-linear
  AND deep-combo's negative result); Viewer-side Java wiring if V4 goes live.

Push target: github.com/OtherBrand/fftb-trainer-v4 (owner-created Nov 2 2026).
`data/` is gitignored — caches, models, and winner files stay local, only
code+docs go up (family doctrine).

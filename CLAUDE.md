# FFTB Gene Trainer V4

V1's match-winner model in Rust, trained by an off-the-shelf GA library.
The workhorse that made positive gil on stream, rebuilt with oomph.

- V3 (Rust, gil-economics GA): `C:\Projects\gene-trainer-3` — github.com/OtherBrand/fftb-trainer-v3
- V2 (Java, frozen Oct 17 2026): `C:\Projects\gene-trainer-2\FFTB-Gene-Trainer-2\Gene-Trainer-V2`
- V1 (Java, original): `C:\Projects\gene-trainer`
- Shared botland sim core: github.com/OtherBrand/fftb-trainer-botland — cut Nov 3
  from gene-trainer-3 baseline `ef3325c` (pre-archetype; the earlier `7b10221^`
  reference still contained the archetype commit), local head `6eb330a`
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
5. **GA engine: fftb-ga from the shared fftb-trainer-botland repo** (owner
   decision Nov 3, after the spike). The off-the-shelf premise ("Rust saves
   memory so a normal library suffices") did not survive measurement: on our
   shape (1,000-gene real-coded genomes) genetic_algorithms 3.0.0 and
   radiate 1.3.2 both run at 3.3–5.2k genomes/s vs fftb-ga's 165k — a
   32–49× machinery wall (per-gene struct ops in selection/crossover/
   mutation), proven fitness-independent by the trivial-fitness control.
   Evidence + regression harnesses: examples/bench_ga_lib (modes per/
   trivial/batch), bench_radiate, bench_kernel (commit 20b0248). Until the
   shared-repo cut lands, V4's fitness interface stays a pure function
   (`fitness.rs`) so the engine swap is mechanical. Engine governance:
   determinism gate — same seed + data + rev = same winner file, always.
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
8. **Sets-vs-multiplicity adjudication (open family Tier 1 item, Nov 3
   2026).** The trainer lineage (V3 + V4 + the golden) uses team-level SET
   semantics (±1 presence); the Java Viewer's scorer uses per-unit
   multiplicity. A parity harness on the V3 side quantifies before any
   family decision. V4 is adjudication-proof by construction:
   `attribute_names` preserves raw multiplicity at parse; semantics live
   only in `features.rs` extract() (set-diff → count-diff is a one-function
   change); the hardened golden (per-side token counts) flags any flip.

## Code style (owner preference, family-wide)

Organize code class-like: one type per concern, `impl` blocks with `new()`
constructors, traits as interfaces. (V3's SoA exception does NOT apply here —
V4 uses a library engine, not a hand-rolled hot core.)

## Operational notes

- **Windows Application Control (error 4551)** blocks freshly-built DLLs/test
  binaries intermittently (family-known). Fixes that worked here: for the
  `zerofrom`/`zerofrom-derive` proc-macro (pulled via `ureq → url → idna →
  ICU4X`), `cargo clean -p zerofrom-derive` then rebuild; for a blocked test
  binary, `touch crates/fftb-v4/src/lib.rs` and rerun. Flaky, not fatal.
- First live Maps.txt fetch (Nov 2 2026): **128 maps** — the dump has grown
  past V1-era 117, which is the staleness problem the live-first policy
  exists to fix.

## Phase status

- Phase 0 (scaffold + Maps.txt live fetch): **done, pushed**.
- Phase 0.5 (GA engine spike): **done, verdict: fftb-ga shared** — off-the-shelf
  libraries measured 32–49× slow (machinery-bound, fitness-independent);
  engine arrives with the shared-repo cut, fitness stays engine-agnostic
  until then.
- Phase 1: **DONE** — train CLI live; engine = fftb-ga from the shared repo
  (git-pinned to the local checkout at a294229 until the GitHub push
  confirms; then the URL at the same rev). Split discipline mirrors
  fftb-forecast (newest 20% holdout, temperature on newest 10% of train).
  Smoke: 814 gens / 90s at pop 2k; market baseline sanity-checks against
  deep-linear's reference. First real run: ~2:30 AM Nov 5 2026, detached
  via start-trainer-v4.ps1 (deep dataset, 6h, seed 42, auto pop).
  Winner files: NO local writer — the sectioned winner-file format is family
  Tier-1 infrastructure living in the shared repo; V4 proposes its section
  kind as a cross-reviewed PR when the repo lands (cheap while V4 has zero
  artifacts in the wild). The advisor WinProbModel export landed WITH
  Phase 1 — calibration + artifact + gate numbers ship with every run.
- Phase 2: (folded into Phase 1 runs) holdout gate vs deep-linear and vs
  the market baseline, reported per run.
- Phase 3: botland via the shared repo (swap `seam.rs`) + roster entry.
- Phase 4 (optional): GA-selected pair features (`attr:A×attr:B` — the one
  accuracy path where evolution beats gradient; must beat both deep-linear
  AND deep-combo's negative result); Viewer-side Java wiring if V4 goes live.

Push target: github.com/OtherBrand/fftb-trainer-v4 (owner-created Nov 2 2026).
`data/` is gitignored — caches, models, and winner files stay local, only
code+docs go up (family doctrine).

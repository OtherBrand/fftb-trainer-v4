//! Spike control: the fitness kernel ALONE (no GA library) over 20k random
//! genomes, single-threaded and rayon-parallel. Separates "machine is
//! contended" from "library evaluation path is slow".
//!
//! Usage: cargo run --release --example bench_kernel

use std::time::Instant;

use rayon::prelude::*;

use fftb_v4::spike::{SparseRows, GENES};

fn main() {
    let dataset = SparseRows::deterministic();

    // Deterministic random genomes.
    let mut state: u64 = 0xA5A5_5A5A_DEAD_BEEF;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let genomes: Vec<Vec<f32>> = (0..20_000)
        .map(|_| (0..GENES).map(|_| (next() % 200) as f32 - 100.0).collect())
        .collect();

    let start = Instant::now();
    let mut checksum = 0.0f32;
    for genome in &genomes {
        checksum += dataset.fitness(genome);
    }
    let serial = start.elapsed().as_secs_f64();
    println!(
        "kernel serial: 20k genomes in {serial:.2}s = {:.0} genomes/s (checksum {checksum:.0})",
        20_000.0 / serial
    );

    let start = Instant::now();
    let checksum: f32 = genomes
        .par_iter()
        .map(|g| dataset.fitness(g))
        .sum();
    let parallel = start.elapsed().as_secs_f64();
    println!(
        "kernel rayon:  20k genomes in {parallel:.2}s = {:.0} genomes/s (checksum {checksum:.0})",
        20_000.0 / parallel
    );
}

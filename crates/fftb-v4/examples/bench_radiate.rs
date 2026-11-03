//! Spike Part B for `radiate` 1.3.2: genomes/s at a given population size
//! over the shared synthetic shape. API assembled from registry sources
//! (docs.rs is 0% documented — itself a Part A finding).
//!
//! Usage: cargo run --release --example bench_radiate -- [pop] [gens]

use std::time::Instant;

use radiate::prelude::*;

use fftb_v4::spike::{SparseRows, GENES};

fn main() {
    let pop: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(20_000);
    let gens: usize = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(10);

    let dataset = std::sync::Arc::new(SparseRows::deterministic());
    let dataset_for_fn = std::sync::Arc::clone(&dataset);
    // FloatCodec::vector decodes to Vec<f32>; fitness returns anything
    // Into<Score>.
    let fitness_fn = move |geno: Vec<f32>| -> f32 { dataset_for_fn.fitness(&geno) };

    let engine = GeneticEngine::builder()
        .codec(FloatCodec::vector(GENES, -100.0_f32..100.0_f32))
        .fitness_fn(fitness_fn)
        .population_size(pop)
        .build();

    let start = Instant::now();
    let mut iterations = engine.iter();
    for _ in 0..gens {
        iterations.next();
    }
    let elapsed = start.elapsed().as_secs_f64();

    let total_genomes = pop as f64 * gens as f64;
    println!(
        "radiate: pop={pop} gens={gens} elapsed={elapsed:.2}s \
         throughput={:.0} genomes/s",
        total_genomes / elapsed,
    );
}

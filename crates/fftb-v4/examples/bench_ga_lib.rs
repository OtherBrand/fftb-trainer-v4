//! Spike Part B for `genetic_algorithms` 3.0.0: genomes/s at a given
//! population size over the shared synthetic shape.
//!
//! Usage: cargo run --release --example bench_ga_lib -- [pop] [gens]

use std::time::Instant;

use genetic_algorithms::chromosomes::Range as RangeChromosome;
use genetic_algorithms::configuration::ProblemSolving;
use genetic_algorithms::ga::Ga;
use genetic_algorithms::genotypes::Range as RangeGenotype;
use genetic_algorithms::initializers::range_random_initialization;
use genetic_algorithms::operations::{Crossover, GaussianParams, Mutation, Selection, Survivor};
use genetic_algorithms::fitness::BatchFitnessEvaluator;
use genetic_algorithms::traits::{
    ConfigurationT, CrossoverConfig, MutationConfig, SelectionConfig, StoppingConfig,
    SurvivorConfig,
};
use genetic_algorithms::ChromosomeLength;

use fftb_v4::spike::{SparseRows, GENES};

/// Batch evaluator: the whole population evaluated by OUR rayon kernel in
/// one call — the escape hatch for per-individual dispatch overhead.
struct BatchKernel {
    dataset: std::sync::Arc<SparseRows>,
}

impl BatchFitnessEvaluator<RangeChromosome<f64>> for BatchKernel {
    fn evaluate_batch(&self, chromosomes: &[RangeChromosome<f64>]) -> Vec<f64> {
        use rayon::prelude::*;
        chromosomes
            .par_iter()
            .map(|c| {
                let genome: Vec<f32> = c.dna.iter().map(|g| g.value as f32).collect();
                self.dataset.fitness(&genome) as f64
            })
            .collect()
    }
}

fn main() {
    let pop: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(20_000);
    let gens: usize = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(10);
    let mode = std::env::args().nth(3).unwrap_or_else(|| "per".into());

    let dataset = std::sync::Arc::new(SparseRows::deterministic());
    let alleles = vec![RangeGenotype::new(0, vec![(-100.0_f64, 100.0_f64)], 0.0_f64)];
    let alleles_clone = alleles.clone();

    let mut builder = Ga::new()
        .with_chromosome_length(ChromosomeLength::Fixed(GENES))
        .with_population_size(pop)
        // CRITICAL: the engine's default thread count is 1 (configuration.rs
        // Default impl) — the "parallel" feature only compiles the rayon
        // machinery in. Without this the engine runs at serial speed.
        .with_threads(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8))
        .with_initialization_fn(move |n, _| range_random_initialization(n, Some(&alleles_clone)))
        .with_selection_method(Selection::Tournament)
        .with_crossover_method(Crossover::Uniform)
        .with_mutation_method(Mutation::Gaussian(GaussianParams { sigma: None }))
        .with_survivor_method(Survivor::Fitness)
        .with_problem_solving(ProblemSolving::Maximization)
        .with_max_generations(gens);

    let mut ga = match mode.as_str() {
        "trivial" => {
            let dataset_for_fn = std::sync::Arc::clone(&dataset);
            builder
                .with_fitness_fn(move |dna: &[RangeGenotype<f64>]| -> f64 {
                    // Same access pattern as the real kernel, near-zero math:
                    // isolates engine machinery cost from kernel cost.
                    let _keep = &dataset_for_fn;
                    dna.iter().map(|g| g.value).sum()
                })
                .build()
        }
        "batch" => builder
            .with_batch_evaluator(std::sync::Arc::new(BatchKernel {
                dataset: std::sync::Arc::clone(&dataset),
            }))
            .build(),
        _ => {
            let dataset_for_fn = std::sync::Arc::clone(&dataset);
            builder
                .with_fitness_fn(move |dna: &[RangeGenotype<f64>]| -> f64 {
                    let genome: Vec<f32> = dna.iter().map(|g| g.value as f32).collect();
                    dataset_for_fn.fitness(&genome) as f64
                })
                .build()
        }
    }
    .expect("Invalid configuration");

    let start = Instant::now();
    let population = ga.run().expect("GA run failed");
    let elapsed = start.elapsed().as_secs_f64();

    let total_genomes = pop as f64 * gens as f64;
    println!(
        "genetic_algorithms: pop={pop} gens={gens} elapsed={elapsed:.2}s \
         throughput={:.0} genomes/s (best fitness {:.2})",
        total_genomes / elapsed,
        population.best_chromosome.fitness as f64
    );
}

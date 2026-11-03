//! Shared synthetic shape for the GA engine spike: 1,000-gene genomes,
//! fitness = dot-product sum over 8,000 sparse rows (~10 nnz each),
//! deterministic. Mirrors gene-trainer-2-47's fftb-ga bench_engine shape so
//! genomes/s compare apples-to-apples (their SoA bar: 165k @ 20k pop,
//! 118k @ 500k, 114k @ 1M).

pub const GENES: usize = 1_000;
pub const ROWS: usize = 8_000;
pub const NNZ_MIN: usize = 8;
pub const NNZ_MAX: usize = 12;

pub struct SparseRows {
    /// Flat (gene_index, value) pairs per row.
    pub rows: Vec<Vec<(usize, f32)>>,
}

impl SparseRows {
    /// Deterministic xorshift64 dataset — identical across every engine.
    pub fn deterministic() -> Self {
        let mut state: u64 = 0x5EED_CAFE_F00D_D00D;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut rows = Vec::with_capacity(ROWS);
        for _ in 0..ROWS {
            let nnz = NNZ_MIN + (next() as usize % (NNZ_MAX - NNZ_MIN + 1));
            let mut row: Vec<(usize, f32)> = Vec::with_capacity(nnz);
            let mut seen = [false; GENES];
            while row.len() < nnz {
                let idx = next() as usize % GENES;
                if !seen[idx] {
                    seen[idx] = true;
                    let value = if next() & 1 == 0 { 1.0 } else { -1.0 };
                    row.push((idx, value));
                }
            }
            rows.push(row);
        }
        Self { rows }
    }

    /// The fitness kernel both engines share: sum of per-row dot products.
    pub fn fitness(&self, genome: &[f32]) -> f32 {
        let mut total = 0.0f32;
        for row in &self.rows {
            let mut dot = 0.0f32;
            for &(idx, value) in row {
                dot += genome[idx] * value;
            }
            total += dot;
        }
        total
    }
}

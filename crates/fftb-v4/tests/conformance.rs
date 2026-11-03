//! Fixture conformance as a cargo test: parser + feature extractor must
//! reproduce the shared golden feature-rows.json exactly, and cache
//! entries must round-trip byte-identically.

#[test]
fn fixtures_conform() {
    let dir = fftb_v4::conformance::fixtures_dir().expect("shared fixtures present");
    let report = fftb_v4::conformance::run(&dir).expect("conformance run");
    assert!(
        report.is_conformant(),
        "conformance failures: {:?}",
        report
            .feature_failures
            .iter()
            .chain(report.round_trip_failures.iter())
            .take(10)
            .collect::<Vec<_>>()
    );
}

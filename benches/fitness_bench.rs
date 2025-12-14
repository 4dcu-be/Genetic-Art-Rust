// Benchmark suite for fitness calculation performance
//
// This benchmark compares scalar vs SIMD implementations of fitness calculation
// across different image sizes to measure performance improvements.
//
// Run with: cargo bench --bench fitness_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use genetic_art::fitness::{image_diff, image_diff_parallel};
#[cfg(feature = "simd")]
use genetic_art::fitness::image_diff_simd;
use image::{Rgba, RgbaImage};

/// Benchmark fitness calculation across different implementations
fn benchmark_fitness(c: &mut Criterion) {
    let mut group = c.benchmark_group("fitness_calculation");

    // Test three representative image sizes:
    // - 100×75 (7,500 pixels) - small image
    // - 400×300 (120,000 pixels) - medium image
    // - 800×600 (480,000 pixels) - large image (typical use case)
    for size in [100, 400, 800].iter() {
        // Use 3:4 aspect ratio (height = size × 3/4)
        let width = *size;
        let height = size * 3 / 4;

        // Create two different images to compare
        // Using different RGB values to ensure non-zero fitness calculation
        let img1 = RgbaImage::from_pixel(width, height, Rgba([100, 150, 200, 255]));
        let img2 = RgbaImage::from_pixel(width, height, Rgba([110, 140, 190, 128]));

        // Benchmark scalar implementation (baseline)
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}", width, height)),
            &size,
            |b, _| b.iter(|| image_diff(black_box(&img1), black_box(&img2))),
        );

        // Benchmark parallel implementation
        group.bench_with_input(
            BenchmarkId::new("parallel", format!("{}x{}", width, height)),
            &size,
            |b, _| b.iter(|| image_diff_parallel(black_box(&img1), black_box(&img2))),
        );

        // Benchmark SIMD implementation (if feature enabled)
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}", width, height)),
            &size,
            |b, _| b.iter(|| image_diff_simd(black_box(&img1), black_box(&img2))),
        );
    }

    group.finish();
}

/// Benchmark edge cases: very small and very large images
fn benchmark_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("fitness_edge_cases");

    // Tiny image: 4×4 (exactly 64 bytes = 1 SIMD chunk)
    let tiny1 = RgbaImage::from_pixel(4, 4, Rgba([100, 150, 200, 255]));
    let tiny2 = RgbaImage::from_pixel(4, 4, Rgba([110, 140, 190, 255]));

    group.bench_function("scalar_4x4", |b| {
        b.iter(|| image_diff(black_box(&tiny1), black_box(&tiny2)))
    });

    #[cfg(feature = "simd")]
    group.bench_function("simd_4x4", |b| {
        b.iter(|| image_diff_simd(black_box(&tiny1), black_box(&tiny2)))
    });

    // Non-aligned image: 17×1 (68 bytes = 1 SIMD chunk + 4 byte remainder)
    let non_aligned1 = RgbaImage::from_pixel(17, 1, Rgba([128, 128, 128, 255]));
    let non_aligned2 = RgbaImage::from_pixel(17, 1, Rgba([138, 138, 138, 255]));

    group.bench_function("scalar_17x1_non_aligned", |b| {
        b.iter(|| image_diff(black_box(&non_aligned1), black_box(&non_aligned2)))
    });

    #[cfg(feature = "simd")]
    group.bench_function("simd_17x1_non_aligned", |b| {
        b.iter(|| image_diff_simd(black_box(&non_aligned1), black_box(&non_aligned2)))
    });

    // Very large image: 1920×1080 (Full HD resolution)
    let large1 = RgbaImage::from_pixel(1920, 1080, Rgba([100, 150, 200, 255]));
    let large2 = RgbaImage::from_pixel(1920, 1080, Rgba([110, 140, 190, 255]));

    group.bench_function("scalar_1920x1080", |b| {
        b.iter(|| image_diff(black_box(&large1), black_box(&large2)))
    });

    #[cfg(feature = "simd")]
    group.bench_function("simd_1920x1080", |b| {
        b.iter(|| image_diff_simd(black_box(&large1), black_box(&large2)))
    });

    group.finish();
}

criterion_group!(benches, benchmark_fitness, benchmark_edge_cases);
criterion_main!(benches);

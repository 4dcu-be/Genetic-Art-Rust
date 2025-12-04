// Image type for comparing pixels
use image::RgbaImage;

// Rayon provides parallel iterators for free performance!
// Just change .iter() to .par_iter() and it uses all CPU cores
use rayon::prelude::*;

/// Calculate the difference between two images (lower is better)
///
/// Uses mean absolute difference of RGB channels.
/// Alpha channel is ignored in comparison.
///
/// # Returns
/// Average pixel difference across all pixels (0.0 = identical, 255.0 = maximum difference)
///
/// # Panics
/// Panics if images have different dimensions
///
/// **Rust Concept: Generic Programming with Monomorphization**
/// This function is generic in the sense that it works with any image type,
/// but Rust compiles specialized versions for each use case (zero runtime cost!)
pub fn image_diff(source: &RgbaImage, target: &RgbaImage) -> f64 {
    // Verify images have same dimensions
    // `assert_eq!` panics with a helpful message if the condition fails
    //
    // **Why panic?**
    // In Rust, panics are for "impossible" situations - bugs in the program
    // Comparing different-sized images is a programming error, not a runtime condition
    assert_eq!(
        source.dimensions(),
        target.dimensions(),
        "Images must have same dimensions"
    );

    // Calculate total pixel difference
    // **Rust Concept: Iterator Chains**
    // This entire calculation is done in a single pass with no intermediate allocations
    //
    // Let's break down this iterator chain:
    let total_diff: u64 = source
        .pixels() // Iterator over source pixels
        .zip(target.pixels()) // Pair with target pixels: (source_pixel, target_pixel)
        .map(|(s, t)| {
            // For each pixel pair, calculate color difference
            //
            // **Why `as i32` then `abs()`?**
            // - Pixels are `u8` (0-255)
            // - Subtraction could underflow: 10 - 200 = ???
            // - Cast to `i32` first, then abs() gives correct distance
            // - Cast to `u64` for accumulation (prevent overflow in sum)
            let dr = (s[0] as i32 - t[0] as i32).abs() as u64; // Red channel
            let dg = (s[1] as i32 - t[1] as i32).abs() as u64; // Green channel
            let db = (s[2] as i32 - t[2] as i32).abs() as u64; // Blue channel
            // s[3] and t[3] are alpha - we ignore them

            dr + dg + db // Total difference for this pixel
        })
        .sum(); // Sum all pixel differences
    // **Note:** This compiles to very efficient code - often auto-vectorized by LLVM!

    // Normalize by total number of color values
    // We have (width * height) pixels, each with 3 channels (RGB)
    // Result is average difference per color channel
    //
    // **Why f64?**
    // - Provides enough precision for large images
    // - Standard for scientific computing in Rust
    // - as f64 casts integer to floating point
    total_diff as f64 / (source.width() * source.height() * 3) as f64
}

/// Parallel version of image_diff (faster for large images)
///
/// **Rust Concept: Fearless Concurrency**
/// This function uses ALL your CPU cores automatically, safely!
///
/// How does Rust make parallel programming safe?
/// 1. The borrow checker prevents data races at compile time
/// 2. Only immutable references (&) are used, so parallel access is safe
/// 3. No locks, no mutexes needed - the type system guarantees safety
///
/// **When to use this?**
/// - Large images (>1000x1000 pixels) - overhead is worth it
/// - When you have multiple CPU cores (which you probably do!)
/// - The population evaluation (comparing many images) benefits hugely
///
/// **Performance Note:**
/// On a 4-core CPU, this can be 3-4x faster than sequential version!
/// On an 8-core CPU, 6-7x faster!
pub fn image_diff_parallel(source: &RgbaImage, target: &RgbaImage) -> f64 {
    assert_eq!(
        source.dimensions(),
        target.dimensions(),
        "Images must have same dimensions"
    );

    // The ONLY difference: collect into vectors and use par_iter!
    //
    // **What's .par_iter()?**
    // - Creates a parallel iterator over a collection
    // - Rayon automatically splits work across CPU cores
    // - Uses work-stealing: idle cores help busy cores
    // - No manual thread management needed!
    //
    // We need to collect both iterators first since rayon's zip needs both sides parallel
    let source_pixels: Vec<_> = source.pixels().collect();
    let target_pixels: Vec<_> = target.pixels().collect();

    let total_diff: u64 = source_pixels
        .par_iter() // <-- Parallel iterator over source pixels
        .zip(target_pixels.par_iter()) // <-- Zip with parallel iterator over target pixels
        .map(|(s, t)| {
            let dr = (s[0] as i32 - t[0] as i32).abs() as u64;
            let dg = (s[1] as i32 - t[1] as i32).abs() as u64;
            let db = (s[2] as i32 - t[2] as i32).abs() as u64;
            dr + dg + db
        })
        .sum(); // Rayon's sum() automatically combines results from all threads

    total_diff as f64 / (source.width() * source.height() * 3) as f64
}

// **Learning Note: Why two versions?**
// For small images, the parallel version has overhead (thread scheduling, work splitting)
// For large images, the speedup is worth it
// In practice, we'll always use parallel for the genetic algorithm since images are large
// But having both lets us test that they give identical results!

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn test_identical_images() {
        // Create two identical gray images
        let img1 = RgbaImage::from_pixel(100, 100, Rgba([128, 128, 128, 255]));
        let img2 = RgbaImage::from_pixel(100, 100, Rgba([128, 128, 128, 255]));

        let diff = image_diff(&img1, &img2);
        assert_eq!(diff, 0.0, "Identical images should have zero difference");
    }

    #[test]
    fn test_different_images() {
        // Black vs white - maximum possible difference
        let img1 = RgbaImage::from_pixel(100, 100, Rgba([0, 0, 0, 255]));
        let img2 = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));

        let diff = image_diff(&img1, &img2);

        // Each pixel differs by 255 in each of RGB channels
        // Average: (255 + 255 + 255) / 3 = 255
        assert_eq!(diff, 255.0, "Black vs white should be maximum difference");
    }

    #[test]
    fn test_partial_difference() {
        // Gray (128, 128, 128) vs slightly different gray (138, 138, 138)
        let img1 = RgbaImage::from_pixel(100, 100, Rgba([128, 128, 128, 255]));
        let img2 = RgbaImage::from_pixel(100, 100, Rgba([138, 138, 138, 255]));

        let diff = image_diff(&img1, &img2);

        // Difference: 10 in each channel
        // Average: (10 + 10 + 10) / 3 = 10.0
        assert_eq!(diff, 10.0);
    }

    #[test]
    fn test_alpha_ignored() {
        // Same RGB, different alpha - should be identical
        let img1 = RgbaImage::from_pixel(100, 100, Rgba([128, 128, 128, 255]));
        let img2 = RgbaImage::from_pixel(100, 100, Rgba([128, 128, 128, 0]));

        let diff = image_diff(&img1, &img2);
        assert_eq!(diff, 0.0, "Alpha channel should be ignored");
    }

    #[test]
    fn test_parallel_equals_sequential() {
        // Verify parallel version gives same result as sequential
        let img1 = RgbaImage::from_pixel(100, 100, Rgba([100, 150, 200, 255]));
        let img2 = RgbaImage::from_pixel(100, 100, Rgba([110, 140, 190, 255]));

        let diff_seq = image_diff(&img1, &img2);
        let diff_par = image_diff_parallel(&img1, &img2);

        assert_eq!(
            diff_seq, diff_par,
            "Sequential and parallel should give identical results"
        );
    }

    #[test]
    fn test_parallel_with_larger_image() {
        // Test parallel version with a larger image where parallelism helps
        let img1 = RgbaImage::from_pixel(500, 500, Rgba([50, 100, 150, 255]));
        let img2 = RgbaImage::from_pixel(500, 500, Rgba([60, 110, 160, 255]));

        let diff = image_diff_parallel(&img1, &img2);

        // Expected: (10 + 10 + 10) / 3 = 10.0
        assert_eq!(diff, 10.0);
    }

    #[test]
    #[should_panic(expected = "Images must have same dimensions")]
    fn test_different_sizes_panics() {
        // This test verifies that comparing different-sized images panics
        let img1 = RgbaImage::from_pixel(100, 100, Rgba([128, 128, 128, 255]));
        let img2 = RgbaImage::from_pixel(200, 200, Rgba([128, 128, 128, 255]));

        // Should panic here
        image_diff(&img1, &img2);
    }
}

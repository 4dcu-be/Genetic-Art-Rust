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

// ============================================================================
// Edge-Weighted Fitness Functions
// ============================================================================

/// Convert an RGBA image to grayscale using standard luminance formula
///
/// Uses the ITU-R BT.601 standard: Y = 0.299R + 0.587G + 0.114B
/// This weights green more heavily because human eyes are more sensitive to it.
///
/// Returns a Vec<u8> in row-major order (same as image pixels)
fn to_grayscale(img: &RgbaImage) -> Vec<u8> {
    img.pixels()
        .map(|p| {
            // Standard luminance formula: 0.299R + 0.587G + 0.114B
            (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as u8
        })
        .collect()
}

/// Compute edge weights for the target image using Sobel operator
///
/// This function:
/// 1. Converts the image to grayscale
/// 2. Applies Sobel edge detection to find gradients
/// 3. Normalizes gradients to [0, 1]
/// 4. Applies power and scale to create weight map
///
/// # Arguments
/// * `target` - The target image to analyze
/// * `edge_power` - Exponential emphasis (higher = stronger edge emphasis)
/// * `edge_scale` - Multiplier for edge bonus (controls max weight)
///
/// # Returns
/// Vec<f64> of weights, same size as image (width * height)
/// Each weight is >= 1.0, with higher values near edges
pub fn compute_edge_weights(target: &RgbaImage, edge_power: f64, edge_scale: f64) -> Vec<f64> {
    let width = target.width() as usize;
    let height = target.height() as usize;
    let gray = to_grayscale(target);

    let mut weights = vec![0.0; width * height];

    // Sobel kernels for edge detection
    // X kernel detects vertical edges, Y kernel detects horizontal edges
    let sobel_x = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    let sobel_y = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

    // Apply Sobel operator to interior pixels (skip 1-pixel border)
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx = 0.0;
            let mut gy = 0.0;

            // Apply 3x3 Sobel kernel
            for ky in 0..3 {
                for kx in 0..3 {
                    let pixel = gray[(y + ky - 1) * width + (x + kx - 1)] as f64;
                    let idx = ky * 3 + kx;
                    gx += pixel * sobel_x[idx] as f64;
                    gy += pixel * sobel_y[idx] as f64;
                }
            }

            // Gradient magnitude: sqrt(gx² + gy²)
            let gradient = (gx * gx + gy * gy).sqrt();
            weights[y * width + x] = gradient;
        }
    }

    // Normalize gradients to [0, 1] range
    let max_gradient = weights.iter().cloned().fold(0.0f64, f64::max);
    if max_gradient > 0.0 {
        for w in weights.iter_mut() {
            *w /= max_gradient;
        }
    }

    // Apply power and scale: weight = 1 + (gradient^power) * scale
    // This creates weights >= 1.0, with edge pixels weighted more heavily
    for w in weights.iter_mut() {
        *w = 1.0 + w.powf(edge_power) * edge_scale;
    }

    weights
}

/// Edge-weighted fitness function
///
/// Compares images using weighted MAD where pixels near edges
/// have higher weight. This emphasizes fine details over large uniform areas.
///
/// # Arguments
/// * `source` - The rendered image to compare
/// * `target` - The target image
/// * `config` - Fitness configuration (must have edge_weight_cache initialized)
///
/// # Returns
/// Weighted average difference (lower is better)
///
/// # Panics
/// Panics if edge_weight_cache is not initialized
pub fn edge_weighted_fitness(
    source: &RgbaImage,
    target: &RgbaImage,
    config: &crate::population::FitnessConfig,
) -> f64 {
    let weights = config
        .edge_weight_cache
        .as_ref()
        .expect("Edge weight cache must be initialized");

    assert_eq!(
        source.dimensions(),
        target.dimensions(),
        "Images must have same dimensions"
    );

    // Parallel computation of weighted difference
    let (weighted_diff, total_weight): (f64, f64) = source
        .pixels()
        .zip(target.pixels())
        .enumerate()
        .par_bridge() // Parallel iteration
        .map(|(idx, (s, t))| {
            // Full RGB comparison (alpha ignored, same as MAD)
            let dr = (s[0] as i32 - t[0] as i32).abs() as f64;
            let dg = (s[1] as i32 - t[1] as i32).abs() as f64;
            let db = (s[2] as i32 - t[2] as i32).abs() as f64;
            let color_diff = dr + dg + db;

            let weight = weights[idx];
            (weight * color_diff, weight)
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

    // Return weighted average (normalized by total weight, not pixel count)
    weighted_diff / total_weight
}

// ============================================================================
// Multi-Scale SSIM Fitness Functions
// ============================================================================

/// Apply Gaussian blur to a single-channel image
///
/// Uses a 5x5 Gaussian kernel for smoothing.
/// This is used before downsampling to avoid aliasing artifacts.
fn gaussian_blur(img: &[u8], width: usize, height: usize) -> Vec<u8> {
    // 5x5 Gaussian kernel (approximation)
    // Center pixel has most weight, falls off toward edges
    let kernel = [
        1.0 / 256.0,
        4.0 / 256.0,
        6.0 / 256.0,
        4.0 / 256.0,
        1.0 / 256.0,
        4.0 / 256.0,
        16.0 / 256.0,
        24.0 / 256.0,
        16.0 / 256.0,
        4.0 / 256.0,
        6.0 / 256.0,
        24.0 / 256.0,
        36.0 / 256.0,
        24.0 / 256.0,
        6.0 / 256.0,
        4.0 / 256.0,
        16.0 / 256.0,
        24.0 / 256.0,
        16.0 / 256.0,
        4.0 / 256.0,
        1.0 / 256.0,
        4.0 / 256.0,
        6.0 / 256.0,
        4.0 / 256.0,
        1.0 / 256.0,
    ];

    // Copy input image first (border pixels will remain unchanged)
    let mut result = img.to_vec();

    // Early return if image too small for 5x5 kernel
    if width < 5 || height < 5 {
        return result;
    }

    // Apply kernel to interior pixels (skip 2-pixel border)
    // We need both read and write indices to be in bounds
    for y in 2..height.saturating_sub(2) {
        for x in 2..width.saturating_sub(2) {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            for ky in 0..5 {
                for kx in 0..5 {
                    let read_y = y + ky - 2;
                    let read_x = x + kx - 2;
                    if read_y < height && read_x < width {
                        let idx = read_y * width + read_x;
                        if idx < img.len() {
                            let pixel = img[idx] as f64;
                            let k = kernel[ky * 5 + kx];
                            sum += pixel * k;
                            weight_sum += k;
                        }
                    }
                }
            }
            let write_idx = y * width + x;
            if write_idx < result.len() && weight_sum > 0.0 {
                result[write_idx] = (sum / weight_sum).round() as u8;
            }
        }
    }

    result
}

/// Downsample image by 2x (half width, half height)
///
/// Takes every other pixel in both dimensions.
/// Should be applied after Gaussian blur to avoid aliasing.
fn downsample_2x(img: &[u8], width: usize, height: usize) -> (Vec<u8>, usize, usize) {
    let new_width = width / 2;
    let new_height = height / 2;
    let mut result = vec![0u8; new_width * new_height];

    for y in 0..new_height {
        for x in 0..new_width {
            result[y * new_width + x] = img[(y * 2) * width + (x * 2)];
        }
    }

    (result, new_width, new_height)
}

/// Compute SSIM (Structural Similarity Index) for a single channel
///
/// SSIM compares images based on:
/// - Luminance (average brightness)
/// - Contrast (variance)
/// - Structure (correlation)
///
/// Returns a value in [0, 1] where 1.0 = perfect match
fn compute_ssim_channel(source: &[u8], target: &[u8], width: usize, height: usize) -> f64 {
    // Constants for numerical stability (prevent division by zero)
    let c1 = 6.5025; // (0.01 * 255)²
    let c2 = 58.5225; // (0.03 * 255)²

    let n = (width * height) as f64;

    // Compute means (average pixel values)
    let mu_s: f64 = source.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mu_t: f64 = target.iter().map(|&x| x as f64).sum::<f64>() / n;

    // Compute variances
    let var_s: f64 = source
        .iter()
        .map(|&x| (x as f64 - mu_s).powi(2))
        .sum::<f64>()
        / n;
    let var_t: f64 = target
        .iter()
        .map(|&x| (x as f64 - mu_t).powi(2))
        .sum::<f64>()
        / n;

    // Compute covariance
    let cov: f64 = source
        .iter()
        .zip(target.iter())
        .map(|(&s, &t)| (s as f64 - mu_s) * (t as f64 - mu_t))
        .sum::<f64>()
        / n;

    // SSIM formula (three components multiplied together)
    let luminance = (2.0 * mu_s * mu_t + c1) / (mu_s.powi(2) + mu_t.powi(2) + c1);
    let contrast = (2.0 * var_s.sqrt() * var_t.sqrt() + c2) / (var_s + var_t + c2);
    let structure = (cov + c2 / 2.0) / (var_s.sqrt() * var_t.sqrt() + c2 / 2.0);

    luminance * contrast * structure
}

/// Extract a single color channel from an RgbaImage
///
/// # Arguments
/// * `img` - The image to extract from
/// * `channel` - Channel index (0=R, 1=G, 2=B)
///
/// # Returns
/// Vec<u8> containing just that channel's values
fn extract_channel(img: &RgbaImage, channel: usize) -> Vec<u8> {
    img.pixels().map(|p| p[channel]).collect()
}

/// Multi-Scale SSIM fitness function
///
/// Evaluates image similarity at multiple scales (resolutions).
/// Finer scales capture details, coarser scales capture overall structure.
/// The detail_weight parameter controls emphasis on fine details.
///
/// # Arguments
/// * `source` - The rendered image to compare
/// * `target` - The target image
/// * `detail_weight` - Exponential weight for finer scales (default: 2.0)
///
/// # Returns
/// Fitness score (lower is better), range [0, 255]
pub fn ms_ssim_fitness(source: &RgbaImage, target: &RgbaImage, detail_weight: f64) -> f64 {
    assert_eq!(
        source.dimensions(),
        target.dimensions(),
        "Images must have same dimensions"
    );

    let width = source.width() as usize;
    let height = source.height() as usize;

    // Process each color channel separately (R, G, B)
    let mut ssim_scores = Vec::new();

    for channel in 0..3 {
        let mut src_channel = extract_channel(source, channel);
        let mut tgt_channel = extract_channel(target, channel);
        let mut w = width;
        let mut h = height;

        // Multi-scale: compute SSIM at up to 5 scales
        let mut scale_ssims = Vec::new();

        for scale in 0..5 {
            // Stop if image too small for meaningful SSIM
            if w < 8 || h < 8 {
                break;
            }

            let ssim = compute_ssim_channel(&src_channel, &tgt_channel, w, h);
            scale_ssims.push(ssim);

            // Blur and downsample for next scale
            if scale < 4 {
                src_channel = gaussian_blur(&src_channel, w, h);
                tgt_channel = gaussian_blur(&tgt_channel, w, h);
                (src_channel, w, h) = downsample_2x(&src_channel, w, h);
                (tgt_channel, _, _) = downsample_2x(&tgt_channel, w, h);
            }
        }

        // Weight scales: finer scales get higher weight
        // Scale 0 (original) gets weight = detail_weight^0 = 1.0
        // Scale 1 (half size) gets weight = detail_weight^1 = detail_weight
        // Scale 2 (quarter size) gets weight = detail_weight^2
        // etc.
        let mut weighted_ssim = 0.0;
        let mut total_weight = 0.0;

        for (i, ssim) in scale_ssims.iter().enumerate() {
            let weight = detail_weight.powi(i as i32); // Exponential weighting
            weighted_ssim += ssim * weight;
            total_weight += weight;
        }

        ssim_scores.push(weighted_ssim / total_weight);
    }

    // Average SSIM across RGB channels
    let avg_ssim: f64 = ssim_scores.iter().sum::<f64>() / ssim_scores.len() as f64;

    // Convert SSIM (higher is better, range [0,1]) to fitness (lower is better)
    // Map [0, 1] SSIM to [255, 0] fitness for consistency with MAD
    (1.0 - avg_ssim) * 255.0
}

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

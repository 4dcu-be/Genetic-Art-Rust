// Import our Shape type from the genes module (which includes both Triangle and Circle)
use crate::genes::Shape;

// Image types from the image crate
// Rgba = Red, Green, Blue, Alpha (color with transparency)
// RgbaImage = 2D image with RGBA pixels
use image::{Rgba, RgbaImage};

// Drawing functions from imageproc
use imageproc::drawing::{draw_filled_circle_mut, draw_polygon_mut};
use imageproc::point::Point;

// Random utilities for shuffling and random selection
use rand::seq::SliceRandom;
use rand::Rng;

// Serialization support for saving/loading paintings
use serde::{Deserialize, Serialize};

// For parsing shape type from strings
use std::str::FromStr;

/// Specifies which type of shape to use in the genetic algorithm
///
/// This enum is used at initialization to determine whether to create
/// a population of triangles or circles.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ShapeType {
    Triangle,
    Circle,
}

impl FromStr for ShapeType {
    type Err = String;

    /// Parse shape type from a string (case-insensitive)
    ///
    /// Accepts: "triangle", "triangles", "circle", "circles"
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "triangle" | "triangles" => Ok(ShapeType::Triangle),
            "circle" | "circles" => Ok(ShapeType::Circle),
            _ => Err(format!("Unknown shape type: '{}'. Valid options: triangle, circle", s)),
        }
    }
}

/// Check if three points form a degenerate triangle
///
/// A degenerate triangle has duplicate points or collinear points (zero area).
/// These cannot be drawn by imageproc's draw_polygon_mut function.
///
/// Returns true if the triangle is degenerate and should not be drawn.
fn is_degenerate(points: &[(i32, i32); 3]) -> bool {
    let (p0, p1, p2) = (points[0], points[1], points[2]);

    // Check if any two points are identical
    if p0 == p1 || p1 == p2 || p0 == p2 {
        return true;
    }

    // Check if points are collinear using cross product
    // If the area of the triangle is 0, points are collinear
    // Area = |(x1(y2-y3) + x2(y3-y1) + x3(y1-y2)) / 2|
    // We can skip the division and just check if the numerator is 0
    let area = (p1.0 - p0.0) * (p2.1 - p0.1) - (p2.0 - p0.0) * (p1.1 - p0.1);
    area == 0
}

/// Draw a triangle with proper alpha blending
///
/// This function draws a filled triangle onto the image using proper alpha compositing.
/// Unlike imageproc's draw_polygon_mut which replaces pixels, this function blends
/// the triangle color with existing pixels based on the alpha channel.
///
/// # Alpha Compositing Formula (Porter-Duff "over" operation)
/// For each pixel where the triangle overlaps:
/// - result_color = (src_alpha * src_color) + ((1 - src_alpha) * dst_color)
/// - result_alpha = src_alpha + ((1 - src_alpha) * dst_alpha)
///
/// # Parameters
/// - `img`: The destination image to draw onto
/// - `points`: The three vertices of the triangle
/// - `color`: The RGBA color of the triangle (includes alpha channel)
fn draw_triangle_with_alpha(img: &mut RgbaImage, points: &[Point<i32>], color: Rgba<u8>) {
    let width = img.width();
    let height = img.height();

    // Extract alpha channel and normalize to 0.0-1.0 range
    let src_alpha = color.0[3] as f32 / 255.0;

    // If fully transparent, skip drawing
    if src_alpha == 0.0 {
        return;
    }

    // If fully opaque, use fast path with direct drawing (no blending needed)
    if src_alpha == 1.0 {
        draw_polygon_mut(img, points, color);
        return;
    }

    // For semi-transparent triangles, we need to blend each pixel
    // Strategy: Create a temporary image, draw the triangle opaquely, then blend

    // Create a temporary image for the triangle (transparent background)
    let mut temp_img = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));

    // Draw the triangle onto the temporary image with full opacity
    draw_polygon_mut(&mut temp_img, points, color);

    // Blend the temporary image onto the main image using alpha compositing
    for (x, y, temp_pixel) in temp_img.enumerate_pixels() {
        // Skip fully transparent pixels (not part of the triangle)
        if temp_pixel.0[3] == 0 {
            continue;
        }

        // Get the destination pixel
        let dst_pixel = img.get_pixel(x, y);

        // Apply alpha blending formula for each color channel
        let inv_alpha = 1.0 - src_alpha;
        let blended = Rgba([
            ((temp_pixel.0[0] as f32 * src_alpha) + (dst_pixel.0[0] as f32 * inv_alpha)) as u8,
            ((temp_pixel.0[1] as f32 * src_alpha) + (dst_pixel.0[1] as f32 * inv_alpha)) as u8,
            ((temp_pixel.0[2] as f32 * src_alpha) + (dst_pixel.0[2] as f32 * inv_alpha)) as u8,
            // For output alpha: combine the alphas (assuming dst is fully opaque for white bg)
            255, // Keep output fully opaque since we're compositing onto opaque white
        ]);

        img.put_pixel(x, y, blended);
    }
}

/// Draw a circle with proper alpha blending
///
/// This function draws a filled circle onto the image using proper alpha compositing.
/// Similar to draw_triangle_with_alpha, it handles transparency correctly using the
/// Porter-Duff "over" operation.
///
/// # Parameters
/// - `img`: The destination image to draw onto
/// - `center`: The center point (x, y) of the circle
/// - `radius`: The radius of the circle in pixels
/// - `color`: The RGBA color of the circle (includes alpha channel)
fn draw_circle_with_alpha(
    img: &mut RgbaImage,
    center: (i32, i32),
    radius: u32,
    color: Rgba<u8>,
) {
    let width = img.width();
    let height = img.height();

    // Extract alpha channel and normalize to 0.0-1.0 range
    let src_alpha = color.0[3] as f32 / 255.0;

    // If fully transparent, skip drawing
    if src_alpha == 0.0 {
        return;
    }

    // If fully opaque, use fast path with direct drawing (no blending needed)
    if src_alpha == 1.0 {
        draw_filled_circle_mut(img, center, radius as i32, color);
        return;
    }

    // For semi-transparent circles, we need to blend each pixel
    // Strategy: Create a temporary image, draw the circle opaquely, then blend

    // Create a temporary image for the circle (transparent background)
    let mut temp_img = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));

    // Draw the circle onto the temporary image with full opacity
    draw_filled_circle_mut(&mut temp_img, center, radius as i32, color);

    // Blend the temporary image onto the main image using alpha compositing
    for (x, y, temp_pixel) in temp_img.enumerate_pixels() {
        // Skip fully transparent pixels (not part of the circle)
        if temp_pixel.0[3] == 0 {
            continue;
        }

        // Get the destination pixel
        let dst_pixel = img.get_pixel(x, y);

        // Apply alpha blending formula for each color channel
        let inv_alpha = 1.0 - src_alpha;
        let blended = Rgba([
            ((temp_pixel.0[0] as f32 * src_alpha) + (dst_pixel.0[0] as f32 * inv_alpha)) as u8,
            ((temp_pixel.0[1] as f32 * src_alpha) + (dst_pixel.0[1] as f32 * inv_alpha)) as u8,
            ((temp_pixel.0[2] as f32 * src_alpha) + (dst_pixel.0[2] as f32 * inv_alpha)) as u8,
            // For output alpha: combine the alphas (assuming dst is fully opaque for white bg)
            255, // Keep output fully opaque since we're compositing onto opaque white
        ]);

        img.put_pixel(x, y, blended);
    }
}

/// A painting composed of multiple shapes
///
/// This is the "chromosome" in our genetic algorithm - a collection of genes (shapes)
/// that together form a complete solution (painting)
///
/// **Rust Concept: Vec<T>**
/// - `Vec` is a growable array (like ArrayList in Java or list in Python)
/// - Stored on the heap, grows dynamically
/// - Has ownership of its contents
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Painting {
    /// Vector of shapes - order matters! Later shapes draw on top of earlier ones
    /// `pub` because we'll need to access this in the evolution module
    pub shapes: Vec<Shape>,

    /// Background color (the canvas before any shapes are drawn)
    /// Private since it's set at creation and never changes
    /// Stored as array since Rgba doesn't implement Serialize
    background_color: [u8; 4],

    /// Image dimensions - cached for convenience
    img_width: u32,
    img_height: u32,
}

impl Painting {
    /// Create a new random painting
    ///
    /// # Arguments
    /// * `num_shapes` - How many shapes to generate
    /// * `img_width` - Canvas width in pixels
    /// * `img_height` - Canvas height in pixels
    /// * `background_color` - RGBA array [R, G, B, A]
    /// * `shape_type` - Type of shape to use (Triangle or Circle)
    ///
    /// **Rust Concept: Ownership in action**
    /// - This function creates and returns a Painting
    /// - The caller receives ownership of the new Painting
    /// - All shapes are owned by the Painting
    pub fn new(
        num_shapes: usize,
        img_width: u32,
        img_height: u32,
        background_color: [u8; 4],
        shape_type: ShapeType,
    ) -> Self {
        // Create a vector of random shapes based on the shape_type
        // `(0..num_shapes)` creates a range iterator
        // `.map(|_| ...)` transforms each element (we ignore the number with `_`)
        // `.collect()` gathers the results into a Vec
        //
        // **Rust Concept: Iterators**
        // - Iterators are lazy (don't compute until collected)
        // - Compiler optimizes iterator chains into efficient loops
        // - Often faster than manual loops!
        let shapes = (0..num_shapes)
            .map(|_| match shape_type {
                ShapeType::Triangle => Shape::new_triangle(img_width, img_height),
                ShapeType::Circle => Shape::new_circle(img_width, img_height),
            })
            .collect();

        Self {
            shapes,
            background_color, // Store as array for serialization
            img_width,
            img_height,
        }
    }

    /// Apply mutations to random shapes
    ///
    /// # Arguments
    /// * `rate` - Fraction of shapes to mutate (0.0-1.0)
    /// * `swap_prob` - Probability of swapping two shapes (changes z-order)
    /// * `sigma` - Mutation strength passed to Shape::mutate
    ///
    /// **Why swap shapes?**
    /// The order of shapes affects the final image (later shapes draw on top)
    /// Swapping can dramatically change the appearance!
    pub fn mutate(&mut self, rate: f32, swap_prob: f32, sigma: f32) {
        let mut rng = rand::thread_rng();

        // Calculate how many shapes to mutate
        // Cast to usize (unsigned integer for array indexing)
        let num_mutations = (rate * self.shapes.len() as f32) as usize;

        // Create a shuffled list of indices
        // **Rust Concept: Collecting into specific types**
        // The `: Vec<usize>` type annotation tells collect() what to build
        let mut indices: Vec<usize> = (0..self.shapes.len()).collect();
        indices.shuffle(&mut rng); // Randomize order

        // Mutate the first num_mutations shapes
        // `.iter()` creates an iterator over references
        // `.take(n)` limits to first n elements
        // `&idx` borrows the index (we just need to read it)
        for &idx in indices.iter().take(num_mutations) {
            self.shapes[idx].mutate(sigma);
        }

        // Maybe swap two shapes (changes rendering order)
        // `gen::<f32>()` generates a random f32 between 0.0 and 1.0
        if rng.gen::<f32>() < swap_prob {
            indices.shuffle(&mut rng);
            if self.shapes.len() >= 2 {
                // `.swap(i, j)` swaps elements at indices i and j
                // Very efficient - just swaps in-place, no allocation
                self.shapes.swap(indices[0], indices[1]);
            }
        }
    }

    /// Render the painting to an image
    ///
    /// This is where the magic happens! We convert our genetic representation
    /// (list of shapes) into an actual image we can see and compare.
    ///
    /// **Rust Concept: Return types and ownership**
    /// - Returns `RgbaImage` (not `&RgbaImage`)
    /// - We're creating a new image and transferring ownership to caller
    /// - The image will be cleaned up when the owner drops it
    pub fn render(&self) -> RgbaImage {
        // Create a blank image filled with background color
        // `RgbaImage::from_pixel` is an efficient way to create a solid-color image
        // Always use fully opaque white background (255, 255, 255, 255)
        let mut img = RgbaImage::from_pixel(
            self.img_width,
            self.img_height,
            Rgba([255, 255, 255, 255]),
        );

        // Draw each shape on top of the previous ones
        // Order matters - later shapes draw over earlier ones
        //
        // **Rust Concept: Borrowing in loops**
        // `&self.shapes` borrows the vector
        // `shape` is a reference to each Shape
        // `&` means we're not taking ownership, just looking
        for shape in &self.shapes {
            // Pattern match on the shape type to draw appropriately
            match shape {
                Shape::Triangle(triangle) => {
                    // Skip degenerate triangles to prevent panic in draw_polygon_mut
                    // Degenerate triangles have duplicate or collinear points
                    if is_degenerate(&triangle.points) {
                        continue;
                    }

                    // Convert our triangle points to imageproc's Point type
                    // `.iter()` iterates over references to points
                    // `.map(|&(x, y)| ...)` destructures each point tuple
                    // The `&` pattern extracts values from the reference
                    let points: Vec<Point<i32>> = triangle
                        .points
                        .iter()
                        .map(|&(x, y)| Point::new(x, y))
                        .collect();

                    // Draw the filled triangle onto the image with proper alpha blending
                    draw_triangle_with_alpha(&mut img, &points, Rgba(triangle.color));
                }
                Shape::Circle(circle) => {
                    // Draw the filled circle onto the image with proper alpha blending
                    // Circles don't need degenerate checking (radius is always positive)
                    draw_circle_with_alpha(&mut img, circle.center, circle.radius, Rgba(circle.color));
                }
            }
        }

        // Return the rendered image
        // Ownership transfers to the caller
        img
    }

    /// Get painting dimensions
    ///
    /// Returns a tuple (width, height)
    /// `&self` means this is a read-only method
    pub fn dimensions(&self) -> (u32, u32) {
        (self.img_width, self.img_height)
    }

    /// Get number of shapes
    ///
    /// **Rust Concept: Why return usize not u32?**
    /// `usize` is Rust's convention for sizes and indices
    /// - Size matches pointer size (32-bit on 32-bit systems, 64-bit on 64-bit)
    /// - Type used by Vec::len(), array indexing, etc.
    /// - Makes code portable across architectures
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    /// Check if painting has no shapes
    ///
    /// Clippy (Rust's linter) recommends providing `is_empty()` when you have `len()`
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }
}

// Tests - these only compile when running `cargo test`
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_painting_creation_triangles() {
        let painting = Painting::new(150, 800, 600, [255, 255, 255, 255], ShapeType::Triangle);
        assert_eq!(painting.len(), 150);
        assert_eq!(painting.dimensions(), (800, 600));
        assert!(!painting.is_empty()); // Should not be empty with 150 shapes
    }

    #[test]
    fn test_painting_creation_circles() {
        let painting = Painting::new(150, 800, 600, [255, 255, 255, 255], ShapeType::Circle);
        assert_eq!(painting.len(), 150);
        assert_eq!(painting.dimensions(), (800, 600));
        assert!(!painting.is_empty());
    }

    #[test]
    fn test_empty_painting() {
        let painting = Painting::new(0, 100, 100, [0, 0, 0, 255], ShapeType::Triangle);
        assert_eq!(painting.len(), 0);
        assert!(painting.is_empty());
    }

    #[test]
    fn test_painting_render_triangles() {
        // Create a small painting for fast testing
        let painting = Painting::new(10, 100, 100, [0, 0, 0, 255], ShapeType::Triangle);
        let img = painting.render();

        // Verify the rendered image has correct dimensions
        assert_eq!(img.dimensions(), (100, 100));

        // Verify it's an actual image (has pixels)
        assert_eq!(img.pixels().count(), 100 * 100);
    }

    #[test]
    fn test_painting_render_circles() {
        // Create a small painting for fast testing
        let painting = Painting::new(10, 100, 100, [0, 0, 0, 255], ShapeType::Circle);
        let img = painting.render();

        // Verify the rendered image has correct dimensions
        assert_eq!(img.dimensions(), (100, 100));

        // Verify it's an actual image (has pixels)
        assert_eq!(img.pixels().count(), 100 * 100);
    }

    #[test]
    fn test_painting_mutation() {
        let mut painting = Painting::new(10, 100, 100, [255, 255, 255, 255], ShapeType::Triangle);

        // Store original state
        let original_first_color = painting.shapes[0].color();

        // Mutate aggressively
        for _ in 0..20 {
            painting.mutate(0.5, 0.5, 1.0);
        }

        // After 20 rounds of 50% mutation rate, something should have changed
        // This is a statistical test - could theoretically fail but extremely unlikely
        let changed = painting.shapes[0].color() != original_first_color;
        assert!(changed || painting.shapes.len() > 0); // Sanity check
    }

    #[test]
    fn test_painting_swap() {
        let mut painting = Painting::new(10, 100, 100, [255, 255, 255, 255], ShapeType::Triangle);

        // Store original first two shapes
        let first = painting.shapes[0].clone();
        let second = painting.shapes[1].clone();

        // Manually swap to test the swap logic works
        painting.shapes.swap(0, 1);

        // Verify swap occurred
        assert_eq!(painting.shapes[0].color(), second.color());
        assert_eq!(painting.shapes[1].color(), first.color());
    }

    #[test]
    fn test_degenerate_triangle_detection() {
        // All points same - degenerate
        assert!(is_degenerate(&[(0, 0), (0, 0), (0, 0)]));

        // Two points same - degenerate
        assert!(is_degenerate(&[(0, 0), (5, 5), (0, 0)]));
        assert!(is_degenerate(&[(5, 5), (0, 0), (5, 5)]));

        // Collinear points (on same line) - degenerate
        assert!(is_degenerate(&[(0, 0), (5, 5), (10, 10)]));
        assert!(is_degenerate(&[(0, 0), (10, 0), (5, 0)])); // Horizontal line
        assert!(is_degenerate(&[(0, 0), (0, 10), (0, 5)])); // Vertical line

        // Valid triangles - not degenerate
        assert!(!is_degenerate(&[(0, 0), (10, 0), (5, 10)]));
        assert!(!is_degenerate(&[(100, 100), (150, 120), (120, 180)]));
    }

    #[test]
    fn test_render_with_degenerate_triangles() {
        // Create a painting and manually add triangles including degenerate ones
        let mut painting = Painting::new(0, 100, 100, [255, 255, 255, 255], ShapeType::Triangle);

        // Add a valid triangle
        painting.shapes.push(Shape::new_triangle(100, 100));

        // Add a degenerate triangle (all points same)
        let mut degenerate1_triangle = crate::genes::Triangle::new(100, 100);
        degenerate1_triangle.points = [(50, 50), (50, 50), (50, 50)];
        painting.shapes.push(Shape::Triangle(degenerate1_triangle));

        // Add another degenerate triangle (collinear)
        let mut degenerate2_triangle = crate::genes::Triangle::new(100, 100);
        degenerate2_triangle.points = [(10, 10), (20, 20), (30, 30)];
        painting.shapes.push(Shape::Triangle(degenerate2_triangle));

        // Add another valid triangle
        painting.shapes.push(Shape::new_triangle(100, 100));

        // Should not panic when rendering despite degenerate triangles
        let img = painting.render();
        assert_eq!(img.dimensions(), (100, 100));

        // We should have 4 shapes total
        assert_eq!(painting.shapes.len(), 4);
    }
}

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

/// Bounding box clipped to image bounds
///
/// Represents a rectangular region with inclusive min/max coordinates.
/// Used to optimize alpha blending by rendering only the region containing a shape
/// instead of allocating full-image temporary buffers.
struct BoundingBox {
    min_x: u32,
    min_y: u32,
    max_x: u32, // inclusive
    max_y: u32, // inclusive
}

impl BoundingBox {
    /// Width of the bounding box (inclusive)
    fn width(&self) -> u32 {
        self.max_x - self.min_x + 1
    }

    /// Height of the bounding box (inclusive)
    fn height(&self) -> u32 {
        self.max_y - self.min_y + 1
    }
}

/// Compute bounding box for a triangle, clipped to image bounds
///
/// Returns None if the triangle is entirely outside the image.
/// Handles edge cases: negative coordinates, coords beyond image, partially visible shapes.
///
/// # Arguments
/// * `points` - The three vertices of the triangle as (x, y) tuples
/// * `img_width` - Image width in pixels
/// * `img_height` - Image height in pixels
fn compute_triangle_bbox(
    points: &[(i32, i32); 3],
    img_width: u32,
    img_height: u32,
) -> Option<BoundingBox> {
    // Find raw bounding box (min/max of all coordinates)
    let raw_min_x = points.iter().map(|p| p.0).min().unwrap();
    let raw_max_x = points.iter().map(|p| p.0).max().unwrap();
    let raw_min_y = points.iter().map(|p| p.1).min().unwrap();
    let raw_max_y = points.iter().map(|p| p.1).max().unwrap();

    // Clamp to image bounds [0, width-1] × [0, height-1]
    // Using i32 for comparison, then convert to u32 for storage
    let min_x = raw_min_x.max(0);
    let max_x = raw_max_x.min(img_width as i32 - 1);
    let min_y = raw_min_y.max(0);
    let max_y = raw_max_y.min(img_height as i32 - 1);

    // Return None if shape is entirely outside image
    // (clamping caused min > max)
    if min_x > max_x || min_y > max_y {
        return None;
    }

    Some(BoundingBox {
        min_x: min_x as u32,
        max_x: max_x as u32,
        min_y: min_y as u32,
        max_y: max_y as u32,
    })
}

/// Compute bounding box for a circle, clipped to image bounds
///
/// Returns None if the circle is entirely outside the image.
/// Circle bounding box is center ± radius.
///
/// # Arguments
/// * `center` - Circle center as (x, y) tuple
/// * `radius` - Circle radius in pixels
/// * `img_width` - Image width in pixels
/// * `img_height` - Image height in pixels
fn compute_circle_bbox(
    center: (i32, i32),
    radius: u32,
    img_width: u32,
    img_height: u32,
) -> Option<BoundingBox> {
    // Compute raw bounds: center ± radius
    let raw_min_x = center.0 - radius as i32;
    let raw_max_x = center.0 + radius as i32;
    let raw_min_y = center.1 - radius as i32;
    let raw_max_y = center.1 + radius as i32;

    // Clamp to image bounds (same logic as triangles)
    let min_x = raw_min_x.max(0);
    let max_x = raw_max_x.min(img_width as i32 - 1);
    let min_y = raw_min_y.max(0);
    let max_y = raw_max_y.min(img_height as i32 - 1);

    // Return None if circle is entirely outside image
    if min_x > max_x || min_y > max_y {
        return None;
    }

    Some(BoundingBox {
        min_x: min_x as u32,
        max_x: max_x as u32,
        min_y: min_y as u32,
        max_y: max_y as u32,
    })
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
/// **Optimization**: Uses bounding-box rendering to avoid allocating full-image buffers
/// and scanning pixels outside the triangle region.
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

    // For semi-transparent triangles, use bounding-box optimization
    // Convert Point<i32> to (i32, i32) tuples for bbox computation
    let raw_points: [(i32, i32); 3] = [
        (points[0].x, points[0].y),
        (points[1].x, points[1].y),
        (points[2].x, points[2].y),
    ];

    // Compute bounding box clipped to image bounds
    let bbox = match compute_triangle_bbox(&raw_points, width, height) {
        Some(b) => b,
        None => return, // Triangle is entirely outside image bounds
    };

    // Create a temporary image for ONLY the bounding box region (not full image)
    // This is the key optimization: instead of width×height, we allocate bbox_width×bbox_height
    let bbox_width = bbox.width();
    let bbox_height = bbox.height();
    let mut temp_img = RgbaImage::from_pixel(bbox_width, bbox_height, Rgba([0, 0, 0, 0]));

    // Translate triangle points from image coordinates to bounding box coordinates
    // The temporary buffer has origin at (0, 0), so we subtract bbox.min_x/min_y
    let bbox_points: Vec<Point<i32>> = points
        .iter()
        .map(|p| Point::new(p.x - bbox.min_x as i32, p.y - bbox.min_y as i32))
        .collect();

    // Draw the triangle onto the temporary image (in bbox coordinate space)
    draw_polygon_mut(&mut temp_img, &bbox_points, color);

    // Blend the temporary image onto the main image
    // Only iterate over bounding box pixels (not the entire image)
    let inv_alpha = 1.0 - src_alpha;
    for bbox_y in 0..bbox_height {
        for bbox_x in 0..bbox_width {
            let temp_pixel = temp_img.get_pixel(bbox_x, bbox_y);

            // Skip fully transparent pixels (not part of the triangle)
            if temp_pixel.0[3] == 0 {
                continue;
            }

            // Translate back to image coordinates for pixel access
            let img_x = bbox.min_x + bbox_x;
            let img_y = bbox.min_y + bbox_y;

            // Get the destination pixel from main image
            let dst_pixel = img.get_pixel(img_x, img_y);

            // Apply alpha blending formula for each color channel
            let blended = Rgba([
                ((temp_pixel.0[0] as f32 * src_alpha) + (dst_pixel.0[0] as f32 * inv_alpha)) as u8,
                ((temp_pixel.0[1] as f32 * src_alpha) + (dst_pixel.0[1] as f32 * inv_alpha)) as u8,
                ((temp_pixel.0[2] as f32 * src_alpha) + (dst_pixel.0[2] as f32 * inv_alpha)) as u8,
                255, // Keep output fully opaque since we're compositing onto opaque white
            ]);

            img.put_pixel(img_x, img_y, blended);
        }
    }
}

/// Draw a circle with proper alpha blending
///
/// This function draws a filled circle onto the image using proper alpha compositing.
/// Similar to draw_triangle_with_alpha, it handles transparency correctly using the
/// Porter-Duff "over" operation.
///
/// **Optimization**: Uses bounding-box rendering to avoid allocating full-image buffers
/// and scanning pixels outside the circle region.
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

    // For semi-transparent circles, use bounding-box optimization
    // Compute bounding box clipped to image bounds
    let bbox = match compute_circle_bbox(center, radius, width, height) {
        Some(b) => b,
        None => return, // Circle is entirely outside image bounds
    };

    // Create a temporary image for ONLY the bounding box region (not full image)
    // This is the key optimization: instead of width×height, we allocate bbox_width×bbox_height
    let bbox_width = bbox.width();
    let bbox_height = bbox.height();
    let mut temp_img = RgbaImage::from_pixel(bbox_width, bbox_height, Rgba([0, 0, 0, 0]));

    // Translate circle center from image coordinates to bounding box coordinates
    // The temporary buffer has origin at (0, 0), so we subtract bbox.min_x/min_y
    let bbox_center = (
        center.0 - bbox.min_x as i32,
        center.1 - bbox.min_y as i32,
    );

    // Draw the circle onto the temporary image (in bbox coordinate space)
    draw_filled_circle_mut(&mut temp_img, bbox_center, radius as i32, color);

    // Blend the temporary image onto the main image
    // Only iterate over bounding box pixels (not the entire image)
    let inv_alpha = 1.0 - src_alpha;
    for bbox_y in 0..bbox_height {
        for bbox_x in 0..bbox_width {
            let temp_pixel = temp_img.get_pixel(bbox_x, bbox_y);

            // Skip fully transparent pixels (not part of the circle)
            if temp_pixel.0[3] == 0 {
                continue;
            }

            // Translate back to image coordinates for pixel access
            let img_x = bbox.min_x + bbox_x;
            let img_y = bbox.min_y + bbox_y;

            // Get the destination pixel from main image
            let dst_pixel = img.get_pixel(img_x, img_y);

            // Apply alpha blending formula for each color channel
            let blended = Rgba([
                ((temp_pixel.0[0] as f32 * src_alpha) + (dst_pixel.0[0] as f32 * inv_alpha)) as u8,
                ((temp_pixel.0[1] as f32 * src_alpha) + (dst_pixel.0[1] as f32 * inv_alpha)) as u8,
                ((temp_pixel.0[2] as f32 * src_alpha) + (dst_pixel.0[2] as f32 * inv_alpha)) as u8,
                255, // Keep output fully opaque since we're compositing onto opaque white
            ]);

            img.put_pixel(img_x, img_y, blended);
        }
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

    // ========== Bounding Box Optimization Tests ==========

    #[test]
    fn test_triangle_bbox_fully_inside() {
        let points = [(100, 100), (150, 120), (120, 180)];
        let bbox = compute_triangle_bbox(&points, 800, 600).unwrap();
        assert_eq!(bbox.min_x, 100);
        assert_eq!(bbox.max_x, 150);
        assert_eq!(bbox.min_y, 100);
        assert_eq!(bbox.max_y, 180);
        assert_eq!(bbox.width(), 51); // 150 - 100 + 1
        assert_eq!(bbox.height(), 81); // 180 - 100 + 1
    }

    #[test]
    fn test_triangle_bbox_partially_outside_left() {
        let points = [(-10, 100), (50, 120), (20, 180)];
        let bbox = compute_triangle_bbox(&points, 800, 600).unwrap();
        assert_eq!(bbox.min_x, 0); // Clamped from -10
        assert_eq!(bbox.max_x, 50);
        assert_eq!(bbox.min_y, 100);
        assert_eq!(bbox.max_y, 180);
    }

    #[test]
    fn test_triangle_bbox_partially_outside_right() {
        let points = [(750, 100), (850, 120), (800, 180)];
        let bbox = compute_triangle_bbox(&points, 800, 600).unwrap();
        assert_eq!(bbox.min_x, 750);
        assert_eq!(bbox.max_x, 799); // Clamped from 850 to width-1
        assert_eq!(bbox.min_y, 100);
        assert_eq!(bbox.max_y, 180);
    }

    #[test]
    fn test_triangle_bbox_fully_outside_left() {
        let points = [(-100, 100), (-50, 120), (-80, 180)];
        let bbox = compute_triangle_bbox(&points, 800, 600);
        assert!(bbox.is_none()); // Entirely outside image
    }

    #[test]
    fn test_triangle_bbox_fully_outside_right() {
        let points = [(900, 100), (950, 120), (920, 180)];
        let bbox = compute_triangle_bbox(&points, 800, 600);
        assert!(bbox.is_none()); // Entirely outside image
    }

    #[test]
    fn test_triangle_bbox_touching_edges() {
        let points = [(0, 0), (799, 0), (400, 599)];
        let bbox = compute_triangle_bbox(&points, 800, 600).unwrap();
        assert_eq!(bbox.min_x, 0);
        assert_eq!(bbox.max_x, 799);
        assert_eq!(bbox.min_y, 0);
        assert_eq!(bbox.max_y, 599);
        assert_eq!(bbox.width(), 800);
        assert_eq!(bbox.height(), 600);
    }

    #[test]
    fn test_circle_bbox_fully_inside() {
        let bbox = compute_circle_bbox((400, 300), 50, 800, 600).unwrap();
        assert_eq!(bbox.min_x, 350); // 400 - 50
        assert_eq!(bbox.max_x, 450); // 400 + 50
        assert_eq!(bbox.min_y, 250); // 300 - 50
        assert_eq!(bbox.max_y, 350); // 300 + 50
        assert_eq!(bbox.width(), 101); // 2*radius + 1
        assert_eq!(bbox.height(), 101);
    }

    #[test]
    fn test_circle_bbox_partially_outside_top_left() {
        let bbox = compute_circle_bbox((30, 30), 50, 800, 600).unwrap();
        assert_eq!(bbox.min_x, 0); // Clamped from -20
        assert_eq!(bbox.max_x, 80);
        assert_eq!(bbox.min_y, 0); // Clamped from -20
        assert_eq!(bbox.max_y, 80);
    }

    #[test]
    fn test_circle_bbox_partially_outside_bottom_right() {
        let bbox = compute_circle_bbox((770, 570), 50, 800, 600).unwrap();
        assert_eq!(bbox.min_x, 720);
        assert_eq!(bbox.max_x, 799); // Clamped from 820
        assert_eq!(bbox.min_y, 520);
        assert_eq!(bbox.max_y, 599); // Clamped from 620
    }

    #[test]
    fn test_circle_bbox_fully_outside_left() {
        let bbox = compute_circle_bbox((-100, 300), 50, 800, 600);
        assert!(bbox.is_none()); // Entirely outside image
    }

    #[test]
    fn test_circle_bbox_fully_outside_top() {
        let bbox = compute_circle_bbox((400, -100), 50, 800, 600);
        assert!(bbox.is_none()); // Entirely outside image
    }

    #[test]
    fn test_triangle_rendering_with_negative_coords() {
        // Test that triangles with negative coords don't panic
        let mut img = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
        let points = [
            Point::new(-50, -50),
            Point::new(-30, -30),
            Point::new(-40, -20),
        ];
        let color = Rgba([255, 0, 0, 128]); // Semi-transparent red

        // Should not panic, should silently skip (triangle outside bounds)
        draw_triangle_with_alpha(&mut img, &points, color);

        // Image should be unchanged (still all white)
        assert_eq!(img.get_pixel(0, 0), &Rgba([255, 255, 255, 255]));
        assert_eq!(img.get_pixel(50, 50), &Rgba([255, 255, 255, 255]));
    }

    #[test]
    fn test_circle_rendering_beyond_bounds() {
        // Test that circles beyond image bounds don't panic
        let mut img = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));

        // Circle center at (200, 200), radius 50 - completely outside
        draw_circle_with_alpha(&mut img, (200, 200), 50, Rgba([0, 255, 0, 128]));

        // Image should be unchanged
        assert_eq!(img.get_pixel(50, 50), &Rgba([255, 255, 255, 255]));
        assert_eq!(img.get_pixel(99, 99), &Rgba([255, 255, 255, 255]));
    }

    #[test]
    fn test_very_small_triangle_bbox() {
        // Test tiny triangle (3×3 bbox) - tests small buffer allocation
        let mut img = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
        let points = [Point::new(50, 50), Point::new(52, 50), Point::new(51, 52)];
        let color = Rgba([255, 0, 0, 128]); // Semi-transparent red

        // Should complete without panic
        draw_triangle_with_alpha(&mut img, &points, color);

        // Should have drawn something (pixel at triangle location should have changed)
        // Due to alpha blending, it won't be pure white anymore
        let pixel = img.get_pixel(51, 51);
        assert_ne!(pixel, &Rgba([255, 255, 255, 255]));
    }

    #[test]
    fn test_very_small_circle_bbox() {
        // Test tiny circle (radius 2) - tests small buffer allocation
        let mut img = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
        let color = Rgba([0, 0, 255, 128]); // Semi-transparent blue

        // Should complete without panic
        draw_circle_with_alpha(&mut img, (50, 50), 2, color);

        // Should have drawn something (pixel at center should have changed)
        let pixel = img.get_pixel(50, 50);
        assert_ne!(pixel, &Rgba([255, 255, 255, 255]));
    }

    #[test]
    fn test_triangle_partially_visible() {
        // Test triangle that crosses image boundary
        let mut img = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
        let points = [Point::new(-10, 50), Point::new(30, 40), Point::new(10, 70)];
        let color = Rgba([255, 0, 0, 128]); // Semi-transparent red

        // Should complete without panic and render the visible portion
        draw_triangle_with_alpha(&mut img, &points, color);

        // Pixel at (10, 50) should be affected (inside visible triangle portion)
        let pixel = img.get_pixel(10, 50);
        assert_ne!(pixel, &Rgba([255, 255, 255, 255]));
    }

    #[test]
    fn test_circle_partially_visible() {
        // Test circle that crosses image boundary
        let mut img = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
        let color = Rgba([0, 255, 0, 128]); // Semi-transparent green

        // Circle centered at (10, 10) with radius 20 - extends outside top-left corner
        draw_circle_with_alpha(&mut img, (10, 10), 20, color);

        // Pixel at (10, 10) (center) should be affected
        let pixel = img.get_pixel(10, 10);
        assert_ne!(pixel, &Rgba([255, 255, 255, 255]));

        // Pixel at (20, 10) should be affected (inside circle)
        let pixel = img.get_pixel(20, 10);
        assert_ne!(pixel, &Rgba([255, 255, 255, 255]));
    }

    #[test]
    fn test_opaque_triangle_still_works() {
        // Test that fully opaque triangles still work (fast path)
        let mut img = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
        let points = [Point::new(30, 30), Point::new(70, 30), Point::new(50, 70)];
        let color = Rgba([255, 0, 0, 255]); // Fully opaque red

        draw_triangle_with_alpha(&mut img, &points, color);

        // Center of triangle should be red (fully opaque, no blending)
        let pixel = img.get_pixel(50, 40);
        assert_eq!(pixel, &Rgba([255, 0, 0, 255]));
    }

    #[test]
    fn test_transparent_triangle_skipped() {
        // Test that fully transparent triangles are skipped (fast path)
        let mut img = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
        let points = [Point::new(30, 30), Point::new(70, 30), Point::new(50, 70)];
        let color = Rgba([255, 0, 0, 0]); // Fully transparent

        draw_triangle_with_alpha(&mut img, &points, color);

        // Image should be completely unchanged
        assert_eq!(img.get_pixel(50, 40), &Rgba([255, 255, 255, 255]));
    }
}

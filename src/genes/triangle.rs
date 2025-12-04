// Import the random number generator trait from the rand crate
use rand::Rng;
// Import serialization traits - these allow converting our Triangle to/from JSON
use serde::{Deserialize, Serialize};

/// Represents a single triangle gene with position, shape, and color
///
/// The `#[derive(...)]` macro automatically generates implementations for common traits:
/// - Clone: Allows creating copies with .clone()
/// - Debug: Allows printing with {:?} for debugging
/// - Serialize/Deserialize: Allows converting to/from JSON
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Triangle {
    /// Three points defining the triangle vertices
    /// `pub` makes this accessible from outside the module
    /// `[(i32, i32); 3]` is a fixed-size array of 3 tuples, stored on the stack (fast!)
    pub points: [(i32, i32); 3],

    /// RGBA color values (0-255)
    /// `[u8; 4]` is an array of 4 unsigned 8-bit integers
    pub color: [u8; 4],

    /// Image dimensions for boundary checking
    /// These are private (no `pub`) - implementation details hidden from users
    img_width: u32,
    img_height: u32,
}

/// Types of mutations that can occur
///
/// Rust enums are more powerful than C enums - they can hold data (though these don't)
/// This is private to the module (no `pub`) since it's just an implementation detail
enum MutationType {
    Shift,  // Move entire triangle
    Point,  // Move single point
    Color,  // Change color
    Reset,  // Complete reset to random
}

// Implementation block - this is where we define methods for Triangle
// Like a class in other languages, but without inheritance
impl Triangle {
    /// Create a new random triangle within image boundaries
    ///
    /// `pub fn new(...)` is a public associated function (like a static method)
    /// Returns `Self` (which means Triangle in this context)
    pub fn new(img_width: u32, img_height: u32) -> Self {
        // Get a thread-local random number generator
        // `mut` means the variable can be modified (mutability must be explicit in Rust)
        let mut rng = rand::thread_rng();

        // Pick a random center point for the triangle
        // `as i32` converts u32 to i32 (signed integer, can be negative)
        let x = rng.gen_range(0..img_width as i32);
        let y = rng.gen_range(0..img_height as i32);

        // Generate three points around the center
        // `Self { ... }` constructs a new Triangle instance
        Self {
            // Array syntax: create 3 points with random offsets from center
            points: [
                (x + rng.gen_range(-50..=50), y + rng.gen_range(-50..=50)),
                (x + rng.gen_range(-50..=50), y + rng.gen_range(-50..=50)),
                (x + rng.gen_range(-50..=50), y + rng.gen_range(-50..=50)),
            ],
            // Random RGBA color (RGB + Alpha for transparency)
            color: [
                rng.gen_range(0..=255),  // Red
                rng.gen_range(0..=255),  // Green
                rng.gen_range(0..=255),  // Blue
                rng.gen_range(0..=255),  // Alpha (transparency)
            ],
            img_width,
            img_height,
        }
    }

    /// Apply a random mutation to this triangle
    ///
    /// # Arguments
    /// * `sigma` - Mutation strength (0.0-2.0, default 1.0)
    ///
    /// `&mut self` means:
    /// - `&` = borrowed (we don't take ownership)
    /// - `mut` = mutable borrow (we can modify the triangle)
    /// - `self` = this is a method that operates on an instance
    pub fn mutate(&mut self, sigma: f32) {
        // Import the weighted index distribution for selecting mutation types
        use rand::distributions::WeightedIndex;
        use rand::prelude::*;

        let mut rng = rand::thread_rng();

        // Weighted selection of mutation type
        // Higher weights = more likely to be selected
        // [30, 35, 30, 5] means Shift and Color are less common, Point is most common, Reset is rare
        let weights = [30, 35, 30, 5];
        let dist = WeightedIndex::new(&weights).unwrap();

        // Sample from the distribution to pick a mutation type
        // `match` is Rust's pattern matching - like switch but exhaustive (compiler checks all cases)
        let mutation_type = match dist.sample(&mut rng) {
            0 => MutationType::Shift,
            1 => MutationType::Point,
            2 => MutationType::Color,
            _ => MutationType::Reset,  // `_` is a catch-all pattern
        };

        // Apply the selected mutation
        // Note how each mutation method borrows the RNG mutably
        match mutation_type {
            MutationType::Shift => self.mutate_shift(sigma, &mut rng),
            MutationType::Point => self.mutate_point(sigma, &mut rng),
            MutationType::Color => self.mutate_color(sigma, &mut rng),
            // For reset, we create a new random triangle and replace self's data with `*self = ...`
            // The `*` dereferences the mutable reference to assign to the actual value
            MutationType::Reset => *self = Triangle::new(self.img_width, self.img_height),
        }
    }

    /// Shift entire triangle by a random amount
    ///
    /// Private method (no `pub`) - internal implementation detail
    /// `&mut self` - we need to modify the triangle's points
    /// `rng: &mut impl Rng` - accepts ANY type that implements the Rng trait
    ///   This is a "trait bound" - enables zero-cost polymorphism
    fn mutate_shift(&mut self, sigma: f32, rng: &mut impl Rng) {
        // Calculate random x and y shifts, scaled by sigma
        // `as f32` converts i32 to f32, then multiply by sigma, then `as i32` converts back
        let x_shift = (rng.gen_range(-50..=50) as f32 * sigma) as i32;
        let y_shift = (rng.gen_range(-50..=50) as f32 * sigma) as i32;

        // Iterate through all points and shift them
        // `&mut self.points` borrows the array mutably
        // `point` is a mutable reference to each element
        for point in &mut self.points {
            point.0 += x_shift;  // .0 accesses first tuple element (x)
            point.1 += y_shift;  // .1 accesses second tuple element (y)
        }
    }

    /// Move a single point of the triangle
    fn mutate_point(&mut self, sigma: f32, rng: &mut impl Rng) {
        // Pick a random point index (0, 1, or 2)
        let index = rng.gen_range(0..3);
        // Modify that point's coordinates
        self.points[index].0 += (rng.gen_range(-50..=50) as f32 * sigma) as i32;
        self.points[index].1 += (rng.gen_range(-50..=50) as f32 * sigma) as i32;
    }

    /// Change the color of the triangle
    fn mutate_color(&mut self, sigma: f32, rng: &mut impl Rng) {
        // Iterate through each color channel (R, G, B, A)
        for channel in &mut self.color {
            let change = (rng.gen_range(-50..=50) as f32 * sigma) as i32;
            // Important: clamp the value to 0-255 range to prevent overflow
            // `*channel` dereferences to get the u8 value
            // `.clamp(0, 255)` ensures value stays in valid range
            *channel = (*channel as i32 + change).clamp(0, 255) as u8;
        }
    }
}

// Conditional compilation: this module only exists in test builds
// Keeps test code out of release binaries
#[cfg(test)]
mod tests {
    // Import everything from parent module (Triangle and its types)
    use super::*;

    /// Test that triangle creation works correctly
    ///
    /// `#[test]` marks this function as a test - cargo test will run it
    #[test]
    fn test_triangle_creation() {
        let tri = Triangle::new(800, 600);
        // `assert_eq!` checks equality and panics with helpful message if they differ
        assert_eq!(tri.img_width, 800);
        assert_eq!(tri.img_height, 600);
        assert_eq!(tri.points.len(), 3);
        assert_eq!(tri.color.len(), 4);
    }

    /// Test that mutation actually changes the triangle
    #[test]
    fn test_triangle_mutation() {
        let mut tri = Triangle::new(800, 600);
        let original_color = tri.color;

        // Mutate multiple times - should eventually change something
        // `_` means we don't use the loop variable
        for _ in 0..10 {
            tri.mutate(1.0);
        }

        // Very unlikely to be identical after 10 mutations
        // `||` is logical OR - check if EITHER color OR points changed
        // `!=` works because we derived PartialEq (via Debug)
        assert!(tri.color != original_color || tri.points != [(0, 0); 3]);
    }
}

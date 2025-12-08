// Import the random number generator trait from the rand crate
use rand::Rng;
// Import serialization traits - these allow converting our Circle to/from JSON
use serde::{Deserialize, Serialize};

/// Represents a single circle gene with position, size, and color
///
/// The `#[derive(...)]` macro automatically generates implementations for common traits:
/// - Clone: Allows creating copies with .clone()
/// - Debug: Allows printing with {:?} for debugging
/// - Serialize/Deserialize: Allows converting to/from JSON
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Circle {
    /// Center point of the circle
    /// `pub` makes this accessible from outside the module
    pub center: (i32, i32),

    /// Radius of the circle in pixels
    /// `u32` ensures radius is always positive
    pub radius: u32,

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
/// This is private to the module (no `pub`) since it's just an implementation detail
enum MutationType {
    Shift,   // Move center point
    Resize,  // Change radius
    Color,   // Change color
    Reset,   // Complete reset to random
}

// Implementation block - this is where we define methods for Circle
impl Circle {
    /// Create a new random circle within image boundaries
    ///
    /// `pub fn new(...)` is a public associated function (like a static method)
    /// Returns `Self` (which means Circle in this context)
    pub fn new(img_width: u32, img_height: u32) -> Self {
        // Get a thread-local random number generator
        let mut rng = rand::thread_rng();

        // Pick a random center point within the image bounds
        let center_x = rng.gen_range(0..img_width as i32);
        let center_y = rng.gen_range(0..img_height as i32);

        // Generate a random radius between 20 and 100 pixels
        let radius = rng.gen_range(20..=100);

        // Create and return the circle
        Self {
            center: (center_x, center_y),
            radius,
            // Random RGBA color (RGB + Alpha for transparency)
            color: [
                rng.gen_range(0..=255), // Red
                rng.gen_range(0..=255), // Green
                rng.gen_range(0..=255), // Blue
                rng.gen_range(0..=255), // Alpha (transparency)
            ],
            img_width,
            img_height,
        }
    }

    /// Apply a random mutation to this circle
    ///
    /// # Arguments
    /// * `sigma` - Mutation strength (0.0-2.0, default 1.0)
    ///
    /// `&mut self` means:
    /// - `&` = borrowed (we don't take ownership)
    /// - `mut` = mutable borrow (we can modify the circle)
    /// - `self` = this is a method that operates on an instance
    pub fn mutate(&mut self, sigma: f32) {
        // Import the weighted index distribution for selecting mutation types
        use rand::distributions::WeightedIndex;
        use rand::prelude::*;

        let mut rng = rand::thread_rng();

        // Weighted selection of mutation type
        // Higher weights = more likely to be selected
        // [30, 35, 30, 5] means Shift and Color are less common, Resize is most common, Reset is rare
        let weights = [30, 35, 30, 5];
        let dist = WeightedIndex::new(&weights).unwrap();

        // Sample from the distribution to pick a mutation type
        let mutation_type = match dist.sample(&mut rng) {
            0 => MutationType::Shift,
            1 => MutationType::Resize,
            2 => MutationType::Color,
            _ => MutationType::Reset,
        };

        // Apply the selected mutation
        match mutation_type {
            MutationType::Shift => self.mutate_shift(sigma, &mut rng),
            MutationType::Resize => self.mutate_resize(sigma, &mut rng),
            MutationType::Color => self.mutate_color(sigma, &mut rng),
            // For reset, we create a new random circle and replace self's data
            MutationType::Reset => *self = Circle::new(self.img_width, self.img_height),
        }
    }

    /// Shift circle center by a random amount
    ///
    /// Private method (no `pub`) - internal implementation detail
    /// `&mut self` - we need to modify the circle's center
    /// `rng: &mut impl Rng` - accepts ANY type that implements the Rng trait
    fn mutate_shift(&mut self, sigma: f32, rng: &mut impl Rng) {
        // Calculate random x and y shifts, scaled by sigma
        let x_shift = (rng.gen_range(-50..=50) as f32 * sigma) as i32;
        let y_shift = (rng.gen_range(-50..=50) as f32 * sigma) as i32;

        // Apply the shift to the center
        self.center.0 += x_shift;
        self.center.1 += y_shift;
    }

    /// Change the radius of the circle
    fn mutate_resize(&mut self, sigma: f32, rng: &mut impl Rng) {
        // Calculate random radius change, scaled by sigma
        let radius_change = (rng.gen_range(-20..=20) as f32 * sigma) as i32;

        // Apply the change and clamp to reasonable range (5-150 pixels)
        // We use i32 for the calculation to handle negative values, then convert back to u32
        self.radius = ((self.radius as i32 + radius_change).clamp(5, 150)) as u32;
    }

    /// Change the color of the circle
    fn mutate_color(&mut self, sigma: f32, rng: &mut impl Rng) {
        // Iterate through each color channel (R, G, B, A)
        for channel in &mut self.color {
            let change = (rng.gen_range(-50..=50) as f32 * sigma) as i32;
            // Clamp the value to 0-255 range to prevent overflow
            *channel = (*channel as i32 + change).clamp(0, 255) as u8;
        }
    }
}

// Conditional compilation: this module only exists in test builds
#[cfg(test)]
mod tests {
    // Import everything from parent module (Circle and its types)
    use super::*;

    /// Test that circle creation works correctly
    #[test]
    fn test_circle_creation() {
        let circle = Circle::new(800, 600);
        assert_eq!(circle.img_width, 800);
        assert_eq!(circle.img_height, 600);
        assert_eq!(circle.color.len(), 4);
        // Radius should be in the initial range
        assert!(circle.radius >= 20 && circle.radius <= 100);
    }

    /// Test that mutation actually changes the circle
    #[test]
    fn test_circle_mutation() {
        let mut circle = Circle::new(800, 600);
        let original_color = circle.color;
        let original_center = circle.center;
        let original_radius = circle.radius;

        // Mutate multiple times - should eventually change something
        for _ in 0..10 {
            circle.mutate(1.0);
        }

        // Very unlikely to be completely identical after 10 mutations
        assert!(
            circle.color != original_color
                || circle.center != original_center
                || circle.radius != original_radius
        );
    }

    /// Test that radius stays within reasonable bounds after mutations
    #[test]
    fn test_radius_clamping() {
        let mut circle = Circle::new(800, 600);

        // Mutate many times with high sigma to try to break bounds
        for _ in 0..100 {
            circle.mutate(2.0);
        }

        // Radius should stay clamped between 5 and 150
        assert!(circle.radius >= 5 && circle.radius <= 150);
    }

    /// Test that center point mutation doesn't panic
    #[test]
    fn test_center_mutation() {
        let mut circle = Circle::new(800, 600);

        // Mutate many times - should not panic
        for _ in 0..50 {
            circle.mutate(1.5);
        }

        // Test passes if we get here without panicking
        assert!(true);
    }
}

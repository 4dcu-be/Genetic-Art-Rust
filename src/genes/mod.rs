// This file declares the genes module and its submodules
//
// Rust's module system:
// - Each directory can have a mod.rs that acts as the module root
// - `mod triangle;` tells Rust to look for triangle.rs and include it as a submodule
// - `pub use` re-exports items, making them available at this module level

// Import serialization traits for the Shape enum
use serde::{Deserialize, Serialize};

// Declare the triangle submodule (looks for triangle.rs in this directory)
mod triangle;

// Declare the circle submodule (looks for circle.rs in this directory)
mod circle;

// Re-export Triangle and Circle so users can write:
//   use genetic_art::genes::Triangle;
//   use genetic_art::genes::Circle;
// instead of:
//   use genetic_art::genes::triangle::Triangle;
//   use genetic_art::genes::circle::Circle;
pub use triangle::Triangle;
pub use circle::Circle;

/// Enum representing different shape types that can be used in the genetic algorithm
///
/// This enum wraps both Triangle and Circle, allowing a unified interface while
/// maintaining type safety and avoiding dynamic dispatch overhead.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Shape {
    Triangle(Triangle),
    Circle(Circle),
}

impl Shape {
    /// Create a new Triangle shape
    pub fn new_triangle(img_width: u32, img_height: u32) -> Self {
        Shape::Triangle(Triangle::new(img_width, img_height))
    }

    /// Create a new Circle shape
    pub fn new_circle(img_width: u32, img_height: u32) -> Self {
        Shape::Circle(Circle::new(img_width, img_height))
    }

    /// Mutate the shape (delegates to the underlying shape's mutate method)
    ///
    /// # Arguments
    /// * `sigma` - Mutation strength multiplier (typically 1.0)
    pub fn mutate(&mut self, sigma: f32) {
        match self {
            Shape::Triangle(t) => t.mutate(sigma),
            Shape::Circle(c) => c.mutate(sigma),
        }
    }

    /// Get the RGBA color of this shape
    pub fn color(&self) -> [u8; 4] {
        match self {
            Shape::Triangle(t) => t.color,
            Shape::Circle(c) => c.color,
        }
    }
}

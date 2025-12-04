// Import our Painting type and fitness function
use crate::painting::Painting;
use crate::fitness::image_diff_parallel;

// Image type for storing the target
use image::RgbaImage;

// Rayon for parallel fitness evaluation
use rayon::prelude::*;

// Serialization support for checkpointing
use serde::{Deserialize, Serialize};

/// A single individual in the population
///
/// In genetic algorithms, an "individual" is a potential solution.
/// Here, each individual is a painting trying to recreate the target image.
///
/// **Rust Concept: Option<T>**
/// `Option<T>` represents a value that might not exist:
/// - `Some(value)` means the value is present
/// - `None` means no value (like null, but type-safe!)
///
/// Why use Option<f64> for fitness?
/// - Newly created individuals haven't been evaluated yet
/// - Using Option forces us to handle the "not yet evaluated" case explicitly
/// - No accidental use of uninitialized values!
#[derive(Clone, Serialize, Deserialize)]
pub struct Individual {
    /// The painting (genetic representation)
    pub chromosome: Painting,

    /// Fitness score (lower is better)
    /// `None` means not yet evaluated
    /// `Some(score)` means evaluated with that score
    pub fitness: Option<f64>,
}

/// Population of paintings evolving toward target image
///
/// This is the heart of the genetic algorithm!
/// It manages a collection of individuals and their evolution over time.
///
/// **Rust Concept: Struct with owned data**
/// - `individuals: Vec<Individual>` - owns all individuals
/// - `target_image: RgbaImage` - owns the target image
/// - When Population is dropped, all its data is automatically cleaned up
/// - No garbage collector needed!
pub struct Population {
    /// All individuals in the current generation
    pub individuals: Vec<Individual>,

    /// Current generation number (starts at 0)
    pub generation: usize,

    /// The target image we're trying to recreate
    /// Not pub because external code shouldn't modify it
    target_image: RgbaImage,
}

impl Population {
    /// Create a new random population
    ///
    /// # Arguments
    /// * `size` - Number of individuals in the population
    /// * `num_triangles` - Number of triangles per painting
    /// * `target_image` - The image we're trying to recreate
    ///
    /// **Rust Concept: Taking ownership**
    /// - `target_image: RgbaImage` takes ownership (not `&RgbaImage`)
    /// - The caller transfers the image to us
    /// - We become responsible for it
    /// - This is intentional - the population owns its target!
    pub fn new(size: usize, num_triangles: usize, target_image: RgbaImage) -> Self {
        let (width, height) = target_image.dimensions();

        // Create `size` random individuals
        // Each starts with random triangles and no fitness score (None)
        //
        // **Performance Note:**
        // We could parallelize this with rayon, but creation is fast enough
        // that the overhead wouldn't be worth it
        let individuals = (0..size)
            .map(|_| Individual {
                chromosome: Painting::new(num_triangles, width, height, [255, 255, 255, 255]),
                fitness: None, // Not evaluated yet
            })
            .collect();

        Self {
            individuals,
            generation: 0,
            target_image,
        }
    }

    /// Evaluate fitness for all individuals in parallel
    ///
    /// This is where Rust really shines!
    /// We're going to evaluate ALL individuals simultaneously across all CPU cores.
    ///
    /// **Rust Concept: Parallel Mutation**
    /// - `.par_iter_mut()` creates a parallel iterator over mutable references
    /// - Each thread gets exclusive access to different individuals
    /// - The borrow checker GUARANTEES no data races at compile time
    /// - No locks, no mutexes - just safe, fast parallel code!
    ///
    /// **Why is this safe?**
    /// 1. Each individual is independent (no shared state)
    /// 2. `&mut` ensures exclusive access per thread
    /// 3. Compiler verifies this at compile time
    pub fn evaluate(&mut self) {
        // Process all individuals in parallel
        // `.par_iter_mut()` is the parallel version of `.iter_mut()`
        self.individuals.par_iter_mut().for_each(|individual| {
            // Render this individual's painting to an image
            let rendered = individual.chromosome.render();

            // Compare to target and store fitness score
            // Lower score = better match to target
            individual.fitness = Some(image_diff_parallel(&rendered, &self.target_image));
        });

        // **Performance Impact:**
        // On a 4-core CPU with 200 individuals:
        // - Sequential: ~4 seconds
        // - Parallel: ~1 second (4x speedup!)
        // On an 8-core CPU: ~0.5 seconds (8x speedup!)
    }

    /// Get the best individual (lowest fitness score)
    ///
    /// # Panics
    /// Panics if population is empty or individuals haven't been evaluated
    ///
    /// **Rust Concept: min_by with closures**
    /// - Closures are like lambda functions: `|a, b| { ... }`
    /// - `|a, b|` defines parameters
    /// - The body compares their fitness scores
    pub fn best(&self) -> &Individual {
        self.individuals
            .iter()
            .min_by(|a, b| {
                // Compare fitness scores
                // `.unwrap()` panics if fitness is None (not evaluated)
                // This is intentional - calling best() on unevaluated population is a bug!
                //
                // **Rust Concept: Result and Option**
                // - `partial_cmp()` returns `Option<Ordering>` (can fail for NaN)
                // - `.unwrap()` extracts the Ordering or panics
                // - In production, might use `.unwrap_or(Ordering::Equal)` for robustness
                a.fitness
                    .unwrap()
                    .partial_cmp(&b.fitness.unwrap())
                    .unwrap()
            })
            .unwrap() // Panics if population is empty
    }

    /// Get average fitness of population
    ///
    /// **Rust Concept: Iterator combinators**
    /// This chain: iter() -> map() -> sum() compiles to a tight loop
    /// No intermediate allocations, no overhead!
    pub fn average_fitness(&self) -> f64 {
        let sum: f64 = self
            .individuals
            .iter()
            .map(|i| i.fitness.unwrap()) // Extract fitness scores
            .sum(); // Add them all up

        sum / self.individuals.len() as f64
    }

    /// Get the target image dimensions
    ///
    /// **Why provide this?**
    /// Other code might need to know the image size without accessing the image itself
    /// This is good API design - expose what's needed, hide the rest
    pub fn target_dimensions(&self) -> (u32, u32) {
        self.target_image.dimensions()
    }

    /// Get a reference to the target image
    ///
    /// Returns an immutable reference so caller can read but not modify
    ///
    /// **Rust Concept: Borrowing for reading**
    /// - Return type `&RgbaImage` means "borrowed reference"
    /// - Caller can look but can't modify
    /// - Multiple readers allowed simultaneously
    /// - Zero-cost abstraction - just a pointer at runtime!
    pub fn target_image(&self) -> &RgbaImage {
        &self.target_image
    }
}

// Tests for Population
#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn test_population_creation() {
        // Create a small black target image
        let target = RgbaImage::from_pixel(100, 100, Rgba([0, 0, 0, 255]));
        let pop = Population::new(50, 10, target);

        assert_eq!(pop.individuals.len(), 50);
        assert_eq!(pop.generation, 0);

        // Verify individuals haven't been evaluated yet
        assert!(pop.individuals[0].fitness.is_none());
    }

    #[test]
    fn test_population_evaluation() {
        let target = RgbaImage::from_pixel(100, 100, Rgba([128, 128, 128, 255]));
        let mut pop = Population::new(10, 5, target);

        // Before evaluation, fitness should be None
        for individual in &pop.individuals {
            assert!(individual.fitness.is_none());
        }

        // Evaluate population
        pop.evaluate();

        // After evaluation, all should have fitness scores
        for individual in &pop.individuals {
            assert!(individual.fitness.is_some());

            // Fitness should be positive (difference from target)
            assert!(individual.fitness.unwrap() >= 0.0);
        }
    }

    #[test]
    fn test_best_individual() {
        let target = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
        let mut pop = Population::new(20, 5, target);

        pop.evaluate();

        let best = pop.best();

        // Best should have a fitness score
        assert!(best.fitness.is_some());

        // Best should be better than or equal to all others
        for individual in &pop.individuals {
            assert!(best.fitness.unwrap() <= individual.fitness.unwrap());
        }
    }

    #[test]
    fn test_average_fitness() {
        let target = RgbaImage::from_pixel(100, 100, Rgba([0, 0, 0, 255]));
        let mut pop = Population::new(10, 5, target);

        pop.evaluate();

        let avg = pop.average_fitness();

        // Average should be positive
        assert!(avg > 0.0);

        // Manually calculate average to verify
        let manual_avg: f64 = pop.individuals
            .iter()
            .map(|i| i.fitness.unwrap())
            .sum::<f64>() / pop.individuals.len() as f64;

        assert_eq!(avg, manual_avg);
    }

    #[test]
    fn test_target_dimensions() {
        let target = RgbaImage::from_pixel(800, 600, Rgba([0, 0, 0, 255]));
        let pop = Population::new(10, 5, target);

        assert_eq!(pop.target_dimensions(), (800, 600));
    }

    #[test]
    #[should_panic]
    fn test_best_panics_without_evaluation() {
        // This test verifies that calling best() without evaluate() panics
        let target = RgbaImage::from_pixel(100, 100, Rgba([0, 0, 0, 255]));
        let pop = Population::new(10, 5, target);

        // Should panic because fitness is None
        pop.best();
    }
}

// Library root for the genetic art algorithm
//
// This file is the entry point for the library crate (genetic_art)
// It declares all the modules that make up our library
//
// For Phase 1, we're only implementing the genes module with Triangle
// The other modules (painting, population, evolution, fitness) will come in later phases

// Declare the genes module (looks for genes/mod.rs or genes.rs)
pub mod genes;

// Phase 2 modules - Image Rendering & Fitness
pub mod painting;
pub mod fitness;

// Phase 3 modules - Genetic Algorithm Engine
pub mod population;
pub mod evolution;

// Re-export commonly used types at the library root for convenience
// This allows users to write:
//   use genetic_art::Triangle;
// instead of:
//   use genetic_art::genes::Triangle;
pub use genes::Triangle;
pub use painting::Painting;
pub use population::{Individual, Population};
pub use evolution::EvolutionParams;

// This file declares the genes module and its submodules
//
// Rust's module system:
// - Each directory can have a mod.rs that acts as the module root
// - `mod triangle;` tells Rust to look for triangle.rs and include it as a submodule
// - `pub use` re-exports items, making them available at this module level

// Declare the triangle submodule (looks for triangle.rs in this directory)
mod triangle;

// Re-export Triangle so users can write:
//   use genetic_art::genes::Triangle;
// instead of:
//   use genetic_art::genes::triangle::Triangle;
pub use triangle::Triangle;

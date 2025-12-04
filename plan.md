# Genetic Art Algorithm - Rust Implementation Plan

## Project Overview

Port of the Python genetic art algorithm to Rust. This implementation evolves 150 random triangles to recreate famous paintings using genetic algorithms.

**Expected Performance:** 20-50x faster than Python implementation
**Target:** Complete working implementation with CLI in 3-4 weeks

---

## Project Structure
```
genetic-art-rust/
├── Cargo.toml
├── README.md
├── .gitignore
├── input/
│   └── starry_night.jpg        # Target image (you provide)
├── output/                      # Generated images and checkpoints
│   ├── generation_0000.png
│   ├── generation_0050.png
│   └── ...
└── src/
    ├── main.rs                  # CLI entry point
    ├── lib.rs                   # Library root
    ├── genes/
    │   ├── mod.rs               # Gene module exports
    │   └── triangle.rs          # Triangle gene implementation
    ├── painting.rs              # Painting chromosome
    ├── population.rs            # Population management
    ├── evolution.rs             # Genetic operators
    └── fitness.rs               # Image comparison
```

---

## Phase 1: Core Data Structures

### Step 1.1: Initialize Project
```bash
# Create new Rust project
cargo new genetic-art-rust
cd genetic-art-rust

# Create directory structure
mkdir -p src/genes
mkdir -p input
mkdir -p output

# Create module files
touch src/lib.rs
touch src/genes/mod.rs
touch src/genes/triangle.rs
touch src/painting.rs
touch src/population.rs
touch src/evolution.rs
touch src/fitness.rs
```

### Step 1.2: Setup Cargo.toml

**File:** `Cargo.toml`
```toml
[package]
name = "genetic-art-rust"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "genetic-art"
path = "src/main.rs"

[lib]
name = "genetic_art"
path = "src/lib.rs"

[dependencies]
# Core dependencies
rand = "0.8"
rayon = "1.8"

# Image processing
image = "0.24"
imageproc = "0.23"

# CLI
clap = { version = "4.4", features = ["derive"] }
indicatif = "0.17"

# Serialization (for checkpoints)
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

### Step 1.3: Setup .gitignore

**File:** `.gitignore`
```gitignore
/target
/output
Cargo.lock
*.png
*.jpg
!input/starry_night.jpg
.DS_Store
```

### Step 1.4: Implement Triangle Gene

**File:** `src/genes/triangle.rs`
```rust
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Represents a single triangle gene with position, shape, and color
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Triangle {
    /// Three points defining the triangle vertices
    pub points: [(i32, i32); 3],
    /// RGBA color values (0-255)
    pub color: [u8; 4],
    /// Image dimensions for boundary checking
    img_width: u32,
    img_height: u32,
}

/// Types of mutations that can occur
enum MutationType {
    Shift,  // Move entire triangle
    Point,  // Move single point
    Color,  // Change color
    Reset,  // Complete reset to random
}

impl Triangle {
    /// Create a new random triangle within image boundaries
    pub fn new(img_width: u32, img_height: u32) -> Self {
        let mut rng = rand::thread_rng();
        
        // Pick a random center point
        let x = rng.gen_range(0..img_width as i32);
        let y = rng.gen_range(0..img_height as i32);
        
        // Generate three points around the center
        Self {
            points: [
                (x + rng.gen_range(-50..=50), y + rng.gen_range(-50..=50)),
                (x + rng.gen_range(-50..=50), y + rng.gen_range(-50..=50)),
                (x + rng.gen_range(-50..=50), y + rng.gen_range(-50..=50)),
            ],
            color: [
                rng.gen_range(0..=255),
                rng.gen_range(0..=255),
                rng.gen_range(0..=255),
                rng.gen_range(0..=255),
            ],
            img_width,
            img_height,
        }
    }
    
    /// Apply a random mutation to this triangle
    /// 
    /// # Arguments
    /// * `sigma` - Mutation strength (0.0-2.0, default 1.0)
    pub fn mutate(&mut self, sigma: f32) {
        use rand::distributions::WeightedIndex;
        use rand::prelude::*;
        
        let mut rng = rand::thread_rng();
        
        // Weighted selection of mutation type
        let weights = [30, 35, 30, 5];
        let dist = WeightedIndex::new(&weights).unwrap();
        
        let mutation_type = match dist.sample(&mut rng) {
            0 => MutationType::Shift,
            1 => MutationType::Point,
            2 => MutationType::Color,
            _ => MutationType::Reset,
        };
        
        match mutation_type {
            MutationType::Shift => self.mutate_shift(sigma, &mut rng),
            MutationType::Point => self.mutate_point(sigma, &mut rng),
            MutationType::Color => self.mutate_color(sigma, &mut rng),
            MutationType::Reset => *self = Triangle::new(self.img_width, self.img_height),
        }
    }
    
    /// Shift entire triangle by a random amount
    fn mutate_shift(&mut self, sigma: f32, rng: &mut impl Rng) {
        let x_shift = (rng.gen_range(-50..=50) as f32 * sigma) as i32;
        let y_shift = (rng.gen_range(-50..=50) as f32 * sigma) as i32;
        
        for point in &mut self.points {
            point.0 += x_shift;
            point.1 += y_shift;
        }
    }
    
    /// Move a single point of the triangle
    fn mutate_point(&mut self, sigma: f32, rng: &mut impl Rng) {
        let index = rng.gen_range(0..3);
        self.points[index].0 += (rng.gen_range(-50..=50) as f32 * sigma) as i32;
        self.points[index].1 += (rng.gen_range(-50..=50) as f32 * sigma) as i32;
    }
    
    /// Change the color of the triangle
    fn mutate_color(&mut self, sigma: f32, rng: &mut impl Rng) {
        for channel in &mut self.color {
            let change = (rng.gen_range(-50..=50) as f32 * sigma) as i32;
            *channel = (*channel as i32 + change).clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_creation() {
        let tri = Triangle::new(800, 600);
        assert_eq!(tri.img_width, 800);
        assert_eq!(tri.img_height, 600);
        assert_eq!(tri.points.len(), 3);
        assert_eq!(tri.color.len(), 4);
    }

    #[test]
    fn test_triangle_mutation() {
        let mut tri = Triangle::new(800, 600);
        let original_color = tri.color;
        
        // Mutate multiple times - should eventually change something
        for _ in 0..10 {
            tri.mutate(1.0);
        }
        
        // Very unlikely to be identical after 10 mutations
        assert!(tri.color != original_color || tri.points != tri.points);
    }
}
```

### Step 1.5: Setup Gene Module

**File:** `src/genes/mod.rs`
```rust
mod triangle;

pub use triangle::Triangle;
```

---

## Phase 2: Image Rendering & Fitness

### Step 2.1: Implement Painting Chromosome

**File:** `src/painting.rs`
```rust
use crate::genes::Triangle;
use image::{Rgba, RgbaImage};
use imageproc::drawing::draw_polygon_mut;
use imageproc::point::Point;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// A painting composed of multiple triangles
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Painting {
    pub triangles: Vec<Triangle>,
    background_color: Rgba<u8>,
    img_width: u32,
    img_height: u32,
}

impl Painting {
    /// Create a new random painting
    pub fn new(
        num_triangles: usize,
        img_width: u32,
        img_height: u32,
        background_color: [u8; 4],
    ) -> Self {
        let triangles = (0..num_triangles)
            .map(|_| Triangle::new(img_width, img_height))
            .collect();

        Self {
            triangles,
            background_color: Rgba(background_color),
            img_width,
            img_height,
        }
    }

    /// Apply mutations to random triangles
    pub fn mutate(&mut self, rate: f32, swap_prob: f32, sigma: f32) {
        let mut rng = rand::thread_rng();
        let num_mutations = (rate * self.triangles.len() as f32) as usize;

        // Get random indices for mutation
        let mut indices: Vec<usize> = (0..self.triangles.len()).collect();
        indices.shuffle(&mut rng);

        // Mutate selected triangles
        for &idx in indices.iter().take(num_mutations) {
            self.triangles[idx].mutate(sigma);
        }

        // Maybe swap two triangles (changes rendering order)
        if rng.gen::<f32>() < swap_prob {
            indices.shuffle(&mut rng);
            if self.triangles.len() >= 2 {
                self.triangles.swap(indices[0], indices[1]);
            }
        }
    }

    /// Render the painting to an image
    pub fn render(&self) -> RgbaImage {
        // Create blank image with background color
        let mut img = RgbaImage::from_pixel(
            self.img_width,
            self.img_height,
            self.background_color,
        );

        // Draw each triangle
        for triangle in &self.triangles {
            let points: Vec<Point<i32>> = triangle
                .points
                .iter()
                .map(|&(x, y)| Point::new(x, y))
                .collect();

            draw_polygon_mut(&mut img, &points, Rgba(triangle.color));
        }

        img
    }

    /// Get painting dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.img_width, self.img_height)
    }

    /// Get number of triangles
    pub fn len(&self) -> usize {
        self.triangles.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_painting_creation() {
        let painting = Painting::new(150, 800, 600, [255, 255, 255, 255]);
        assert_eq!(painting.len(), 150);
        assert_eq!(painting.dimensions(), (800, 600));
    }

    #[test]
    fn test_painting_render() {
        let painting = Painting::new(10, 100, 100, [0, 0, 0, 255]);
        let img = painting.render();
        assert_eq!(img.dimensions(), (100, 100));
    }

    #[test]
    fn test_painting_mutation() {
        let mut painting = Painting::new(10, 100, 100, [255, 255, 255, 255]);
        painting.mutate(0.5, 0.5, 1.0);
        // Test passes if no panic occurs
    }
}
```

### Step 2.2: Implement Fitness Function

**File:** `src/fitness.rs`
```rust
use image::RgbaImage;
use rayon::prelude::*;

/// Calculate the difference between two images (lower is better)
/// 
/// Uses mean absolute difference of RGB channels.
/// Alpha channel is ignored.
pub fn image_diff(source: &RgbaImage, target: &RgbaImage) -> f64 {
    assert_eq!(
        source.dimensions(),
        target.dimensions(),
        "Images must have same dimensions"
    );

    let total_diff: u64 = source
        .pixels()
        .zip(target.pixels())
        .map(|(s, t)| {
            let dr = (s[0] as i32 - t[0] as i32).abs() as u64;
            let dg = (s[1] as i32 - t[1] as i32).abs() as u64;
            let db = (s[2] as i32 - t[2] as i32).abs() as u64;
            dr + dg + db
        })
        .sum();

    total_diff as f64 / (source.width() * source.height() * 3) as f64
}

/// Parallel version of image_diff (faster for large images)
pub fn image_diff_parallel(source: &RgbaImage, target: &RgbaImage) -> f64 {
    assert_eq!(
        source.dimensions(),
        target.dimensions(),
        "Images must have same dimensions"
    );

    let total_diff: u64 = source
        .pixels()
        .par_bridge()
        .zip(target.pixels())
        .map(|(s, t)| {
            let dr = (s[0] as i32 - t[0] as i32).abs() as u64;
            let dg = (s[1] as i32 - t[1] as i32).abs() as u64;
            let db = (s[2] as i32 - t[2] as i32).abs() as u64;
            dr + dg + db
        })
        .sum();

    total_diff as f64 / (source.width() * source.height() * 3) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn test_identical_images() {
        let img1 = RgbaImage::from_pixel(100, 100, Rgba([128, 128, 128, 255]));
        let img2 = RgbaImage::from_pixel(100, 100, Rgba([128, 128, 128, 255]));
        
        let diff = image_diff(&img1, &img2);
        assert_eq!(diff, 0.0);
    }

    #[test]
    fn test_different_images() {
        let img1 = RgbaImage::from_pixel(100, 100, Rgba([0, 0, 0, 255]));
        let img2 = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
        
        let diff = image_diff(&img1, &img2);
        assert!(diff > 0.0);
        assert_eq!(diff, 255.0); // Maximum difference
    }

    #[test]
    fn test_parallel_equals_sequential() {
        let img1 = RgbaImage::from_pixel(100, 100, Rgba([100, 150, 200, 255]));
        let img2 = RgbaImage::from_pixel(100, 100, Rgba([110, 140, 190, 255]));
        
        let diff_seq = image_diff(&img1, &img2);
        let diff_par = image_diff_parallel(&img1, &img2);
        
        assert_eq!(diff_seq, diff_par);
    }
}
```

---

## Phase 3: Genetic Algorithm Engine

### Step 3.1: Implement Population Management

**File:** `src/population.rs`
```rust
use crate::painting::Painting;
use crate::fitness::image_diff_parallel;
use image::RgbaImage;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A single individual in the population
#[derive(Clone, Serialize, Deserialize)]
pub struct Individual {
    pub chromosome: Painting,
    pub fitness: Option<f64>,
}

/// Population of paintings evolving toward target image
pub struct Population {
    pub individuals: Vec<Individual>,
    pub generation: usize,
    target_image: RgbaImage,
}

impl Population {
    /// Create a new random population
    pub fn new(size: usize, num_triangles: usize, target_image: RgbaImage) -> Self {
        let (width, height) = target_image.dimensions();

        let individuals = (0..size)
            .map(|_| Individual {
                chromosome: Painting::new(num_triangles, width, height, [255, 255, 255, 255]),
                fitness: None,
            })
            .collect();

        Self {
            individuals,
            generation: 0,
            target_image,
        }
    }

    /// Evaluate fitness for all individuals in parallel
    pub fn evaluate(&mut self) {
        self.individuals.par_iter_mut().for_each(|individual| {
            let rendered = individual.chromosome.render();
            individual.fitness = Some(image_diff_parallel(&rendered, &self.target_image));
        });
    }

    /// Get the best individual (lowest fitness score)
    pub fn best(&self) -> &Individual {
        self.individuals
            .iter()
            .min_by(|a, b| {
                a.fitness
                    .unwrap()
                    .partial_cmp(&b.fitness.unwrap())
                    .unwrap()
            })
            .unwrap()
    }

    /// Get average fitness of population
    pub fn average_fitness(&self) -> f64 {
        let sum: f64 = self
            .individuals
            .iter()
            .map(|i| i.fitness.unwrap())
            .sum();
        sum / self.individuals.len() as f64
    }

    /// Get the target image dimensions
    pub fn target_dimensions(&self) -> (u32, u32) {
        self.target_image.dimensions()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_population_creation() {
        let target = RgbaImage::new(100, 100);
        let pop = Population::new(50, 10, target);
        
        assert_eq!(pop.individuals.len(), 50);
        assert_eq!(pop.generation, 0);
    }

    #[test]
    fn test_population_evaluation() {
        let target = RgbaImage::new(100, 100);
        let mut pop = Population::new(10, 5, target);
        
        pop.evaluate();
        
        for individual in &pop.individuals {
            assert!(individual.fitness.is_some());
        }
    }
}
```

### Step 3.2: Implement Evolution Operators

**File:** `src/evolution.rs`
```rust
use crate::painting::Painting;
use crate::population::{Individual, Population};
use rand::Rng;

/// Parameters controlling the evolution process
#[derive(Clone, Debug)]
pub struct EvolutionParams {
    pub population_size: usize,
    pub survival_rate: f32,
    pub mutation_rate: f32,
    pub swap_prob: f32,
    pub sigma: f32,
}

impl Default for EvolutionParams {
    fn default() -> Self {
        Self {
            population_size: 200,
            survival_rate: 0.05,
            mutation_rate: 0.04,
            swap_prob: 0.5,
            sigma: 1.0,
        }
    }
}

impl Population {
    /// Select two parents from the population
    /// Returns (best_individual, random_individual)
    pub fn select_parents(&self) -> (&Individual, &Individual) {
        let mut rng = rand::thread_rng();

        // Pick best individual
        let mom = self.best();

        // Pick random individual
        let dad_idx = rng.gen_range(0..self.individuals.len());
        let dad = &self.individuals[dad_idx];

        (mom, dad)
    }

    /// Breed two paintings to create offspring
    pub fn breed(parent_a: &Painting, parent_b: &Painting) -> Painting {
        let mut rng = rand::thread_rng();
        let (width, height) = parent_a.dimensions();

        let mut child = Painting::new(0, width, height, [255, 255, 255, 255]);

        // Crossover: randomly take triangles from either parent
        for (tri_a, tri_b) in parent_a.triangles.iter().zip(&parent_b.triangles) {
            if rng.gen_bool(0.5) {
                child.triangles.push(tri_a.clone());
            } else {
                child.triangles.push(tri_b.clone());
            }
        }

        child
    }

    /// Evolve the population for one generation
    pub fn evolve_generation(&mut self, params: &EvolutionParams) {
        // 1. Evaluate fitness for all individuals
        self.evaluate();

        // 2. Sort by fitness (best first)
        self.individuals.sort_by(|a, b| {
            a.fitness
                .unwrap()
                .partial_cmp(&b.fitness.unwrap())
                .unwrap()
        });

        // 3. Keep only the best individuals (survival of the fittest)
        let survivors = (self.individuals.len() as f32 * params.survival_rate) as usize;
        self.individuals.truncate(survivors.max(2)); // Keep at least 2

        // 4. Breed to repopulate
        while self.individuals.len() < params.population_size {
            let (mom, dad) = self.select_parents();
            let mut child_chromosome = Self::breed(&mom.chromosome, &dad.chromosome);
            child_chromosome.mutate(params.mutation_rate, params.swap_prob, params.sigma);

            self.individuals.push(Individual {
                chromosome: child_chromosome,
                fitness: None,
            });
        }

        self.generation += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbaImage;

    #[test]
    fn test_evolution_params_default() {
        let params = EvolutionParams::default();
        assert_eq!(params.population_size, 200);
        assert_eq!(params.survival_rate, 0.05);
    }

    #[test]
    fn test_breeding() {
        let parent_a = Painting::new(10, 100, 100, [255, 255, 255, 255]);
        let parent_b = Painting::new(10, 100, 100, [255, 255, 255, 255]);
        
        let child = Population::breed(&parent_a, &parent_b);
        assert_eq!(child.len(), 10);
    }

    #[test]
    fn test_evolution_cycle() {
        let target = RgbaImage::new(100, 100);
        let mut pop = Population::new(20, 5, target);
        let params = EvolutionParams::default();
        
        let initial_gen = pop.generation;
        pop.evolve_generation(&params);
        
        assert_eq!(pop.generation, initial_gen + 1);
        assert_eq!(pop.individuals.len(), params.population_size);
    }
}
```

### Step 3.3: Setup Library Root

**File:** `src/lib.rs`
```rust
pub mod genes;
pub mod painting;
pub mod population;
pub mod evolution;
pub mod fitness;

// Re-export main types for convenience
pub use genes::Triangle;
pub use painting::Painting;
pub use population::{Individual, Population};
pub use evolution::EvolutionParams;
```

---

## Phase 4: CLI Application

### Step 4.1: Implement Main CLI

**File:** `src/main.rs`
```rust
use clap::Parser;
use genetic_art::{EvolutionParams, Population};
use image::RgbaImage;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::path::Path;

#[derive(Parser)]
#[command(name = "genetic-art")]
#[command(about = "Generate art using genetic algorithms", long_about = None)]
struct Args {
    /// Path to target image
    #[arg(short, long)]
    input: String,

    /// Number of triangles per painting
    #[arg(short, long, default_value_t = 150)]
    triangles: usize,

    /// Population size
    #[arg(short, long, default_value_t = 200)]
    population: usize,

    /// Number of generations to evolve
    #[arg(short, long, default_value_t = 5000)]
    generations: usize,

    /// Output directory for generated images
    #[arg(short, long, default_value = "./output")]
    output: String,

    /// Save image every N generations
    #[arg(long, default_value_t = 50)]
    save_interval: usize,

    /// Mutation rate (0.0-1.0)
    #[arg(long, default_value_t = 0.04)]
    mutation_rate: f32,

    /// Survival rate (0.0-1.0)
    #[arg(long, default_value_t = 0.05)]
    survival_rate: f32,
}

fn main() {
    let args = Args::parse();

    // Create output directory if it doesn't exist
    fs::create_dir_all(&args.output).expect("Failed to create output directory");

    // Load target image
    println!("Loading target image: {}", args.input);
    let target = image::open(&args.input)
        .expect("Failed to open target image")
        .to_rgba8();

    let (width, height) = target.dimensions();
    println!("Image dimensions: {}x{}", width, height);

    // Create initial population
    println!("Creating population of {} individuals...", args.population);
    println!("Each with {} triangles", args.triangles);
    let mut pop = Population::new(args.population, args.triangles, target);

    // Evolution parameters
    let params = EvolutionParams {
        population_size: args.population,
        survival_rate: args.survival_rate,
        mutation_rate: args.mutation_rate,
        swap_prob: 0.5,
        sigma: 1.0,
    };

    // Setup progress bar
    let pb = ProgressBar::new(args.generations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} | Gen: {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    println!("\nStarting evolution...\n");

    // Evolution loop
    for gen in 0..args.generations {
        pop.evolve_generation(&params);

        let best_fitness = pop.best().fitness.unwrap();
        let avg_fitness = pop.average_fitness();

        // Update progress bar
        pb.set_message(format!("Best: {:.2}, Avg: {:.2}", best_fitness, avg_fitness));
        pb.inc(1);

        // Save image at intervals
        if gen % args.save_interval == 0 {
            save_generation(&pop, &args.output, gen);
        }
    }

    pb.finish_with_message("Evolution complete!");

    // Save final result
    save_generation(&pop, &args.output, args.generations);
    
    let best = pop.best();
    println!("\n✓ Evolution complete!");
    println!("  Final fitness: {:.2}", best.fitness.unwrap());
    println!("  Total generations: {}", pop.generation);
    println!("  Output directory: {}", args.output);
}

fn save_generation(pop: &Population, output_dir: &str, generation: usize) {
    let best = pop.best();
    let img = best.chromosome.render();
    
    let filename = format!("{}/generation_{:05}.png", output_dir, generation);
    img.save(&filename).expect("Failed to save image");
    
    // Also save as "latest.png" for easy viewing
    let latest = format!("{}/latest.png", output_dir);
    img.save(&latest).ok();
}
```

---

## Building and Running

### Step 1: Build the project
```bash
# Debug build (faster compilation)
cargo build

# Release build (optimized, much faster execution)
cargo build --release
```

### Step 2: Run tests
```bash
cargo test
```

### Step 3: Get a target image

Download a target image (e.g., Van Gogh's Starry Night) and place it in the `input/` directory:
```bash
# Example: download Starry Night
curl -o input/starry_night.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
```

### Step 4: Run the algorithm
```bash
# Quick test (10 generations)
cargo run --release -- --input input/starry_night.jpg --generations 10

# Full run (5000 generations, ~10-15 minutes on modern hardware)
cargo run --release -- \
    --input input/starry_night.jpg \
    --triangles 150 \
    --population 200 \
    --generations 5000 \
    --output ./output \
    --save-interval 50

# Custom parameters
cargo run --release -- \
    --input input/starry_night.jpg \
    --triangles 100 \
    --population 100 \
    --generations 1000 \
    --mutation-rate 0.06 \
    --survival-rate 0.10
```

### Step 5: View results

Results are saved in the `output/` directory:
- `generation_00000.png` - Initial random image
- `generation_00050.png` - After 50 generations
- `generation_00100.png` - After 100 generations
- ...
- `generation_05000.png` - Final result
- `latest.png` - Always contains the most recent result

---

## Expected Output

### Console Output
```
Loading target image: input/starry_night.jpg
Image dimensions: 1280x960
Creating population of 200 individuals...
Each with 150 triangles

Starting evolution...

[00:00:45] ========================================> 50/5000 | Gen: Best: 145.23, Avg: 167.89
[00:01:30] ========================================> 100/5000 | Gen: Best: 132.45, Avg: 154.32
...
[00:12:15] ========================================> 5000/5000 | Gen: Best: 45.67, Avg: 52.34

✓ Evolution complete!
  Final fitness: 45.67
  Total generations: 5000
  Output directory: ./output
```

### Performance Expectations

On a modern 6-core CPU:
- **Generation 0:** ~1 second (initial evaluation)
- **Generations 1-100:** ~2-3 generations/second
- **Generations 100-1000:** ~3-4 generations/second
- **Total time (5000 gens):** 10-15 minutes

Compare to Python: ~6-8 hours for same result!

---

## Troubleshooting

### Problem: "Failed to open target image"

**Solution:** Ensure the input image exists and is a valid image format (PNG, JPEG, etc.)
```bash
ls -lh input/starry_night.jpg
```

### Problem: Compilation errors with image crates

**Solution:** Update dependencies
```bash
cargo update
cargo clean
cargo build --release
```

### Problem: Out of memory

**Solution:** Reduce population size or image dimensions
```bash
# Reduce population
cargo run --release -- --input input/starry_night.jpg --population 100

# Or resize input image first
convert input/starry_night.jpg -resize 800x600 input/starry_night_small.jpg
```

### Problem: Very slow performance

**Solution:** Make sure you're using `--release` flag
```bash
# Wrong (10-50x slower)
cargo run -- --input input/starry_night.jpg

# Correct
cargo run --release -- --input input/starry_night.jpg
```

---

## Next Steps

After completing this implementation:

1. **Profile performance:** Use `cargo flamegraph` to identify bottlenecks
2. **Add benchmarks:** Compare against Python implementation
3. **Write blog post:** Document the porting experience and performance gains
4. **Implement Part 2:** Add Voronoi diagram support (see Phase 5 in full plan)
5. **Add features:**
   - Resume from checkpoint
   - Real-time preview window
   - Export evolution video
   - Multi-image batch processing

---

## Success Criteria

✅ Project compiles without warnings
✅ All tests pass
✅ Can process target image
✅ Generates improving results over generations
✅ Fitness score decreases over time
✅ Final image resembles target
✅ Performance is 20-50x faster than Python

---

## Resources

- **Rust Book:** https://doc.rust-lang.org/book/
- **Image Crate Docs:** https://docs.rs/image/
- **Rayon Docs:** https://docs.rs/rayon/
- **Clap Docs:** https://docs.rs/clap/
- **Original Python Repo:** https://github.com/4dcu-be/Genetic-Art-Algorithm

---

## Estimated Timeline

- **Day 1-2:** Project setup, dependencies, triangle implementation
- **Day 3-4:** Painting and fitness functions
- **Day 5-7:** Population and evolution logic
- **Day 8-10:** CLI, testing, debugging
- **Day 11-14:** Optimization, benchmarking, documentation

**Total:** 2-3 weeks for complete, polished implementation
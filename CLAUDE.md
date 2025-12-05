# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A high-performance genetic algorithm in Rust that evolves random triangles to recreate images. The algorithm uses natural selection, crossover breeding, and mutation to transform collections of semi-transparent triangles into approximations of target images.

**Performance:** 20-50x faster than equivalent Python implementations due to Rust's zero-cost abstractions and automatic parallelization via Rayon.

## Build & Test Commands

```bash
# Release build (ALWAYS use for actual evolution - 10-100x faster)
cargo build --release

# Development build (fast compile, slow runtime)
cargo build

# Run all tests
cargo test

# Run library tests only (excludes integration tests)
cargo test --lib

# Run specific test by name
cargo test test_triangle_creation

# Run tests with output visible
cargo test -- --nocapture

# Run the CLI binary
./target/release/genetic-art --input input/image.jpg --generations 100 --triangles 50
```

## Architecture Overview

### Dual Crate Structure

The project is both a **library** (`genetic_art`) and a **binary** (`genetic-art`):

- **Library** (`src/lib.rs`): Core genetic algorithm implementation, reusable
- **Binary** (`src/main.rs`): CLI application that uses the library

### Core Data Flow

```
1. Load target image (main.rs)
   ↓
2. Initialize Population with random Paintings (population.rs)
   ↓
3. For each generation:
   a. Evaluate fitness: Compare each Painting's rendering to target (fitness.rs)
   b. Select survivors: Keep best individuals (evolution.rs)
   c. Breed: Cross-breed survivors to create offspring (evolution.rs)
   d. Mutate: Apply random changes to offspring (painting.rs, genes/triangle.rs)
   e. Repeat
```

### Module Responsibilities

**`genes/triangle.rs`** - Individual gene (triangle)
- Struct: `Triangle` with 3 points `[(i32, i32); 3]` and RGBA color `[u8; 4]`
- Random generation with `Triangle::new(rng, width, height)`
- Mutation operations: `mutate_shift`, `mutate_point`, `mutate_color`, `mutate_reset`

**`painting.rs`** - Chromosome (collection of triangles)
- Struct: `Painting` with `Vec<Triangle>`, dimensions, and background color
- Rendering: `render() -> RgbaImage` draws triangles with **alpha blending**
- Mutation: `mutate(rng, params)` applies random changes to triangles
- Crossover: `breed(parent1, parent2, rng)` creates offspring from two parents
- **Critical implementation detail**: Uses custom `draw_triangle_with_alpha()` for proper Porter-Duff alpha compositing

**`population.rs`** - Collection of individuals
- Struct: `Individual` wraps a `Painting` with an `Option<f64>` fitness score
- Struct: `Population` manages `Vec<Individual>` and tracks target image
- Parallel fitness evaluation using Rayon's `par_iter_mut()`

**`evolution.rs`** - Genetic operators
- Struct: `EvolutionParams` controls mutation rate, survival rate, sigma, etc.
- Selection strategy: **Elitist + Random** (best individual + random second parent)
- Main loop: `evolve_generation(&mut self, params)` performs one complete generation

**`fitness.rs`** - Image comparison
- Function: `calculate_fitness(source, target) -> f64`
- Uses Mean Absolute Difference (MAD) across RGB channels
- **Alpha channels are ignored** in fitness calculation
- Parallel pixel-by-pixel comparison via Rayon

**`main.rs`** - CLI application
- Uses `clap` for argument parsing with derive macros
- Progress bar via `indicatif` showing elapsed time, ETA, and fitness stats
- Saves images at intervals to `output/` directory

## Critical Implementation Details

### Alpha Blending (painting.rs)

The rendering system uses **proper alpha compositing** (Porter-Duff "over" operation):

```rust
// Located in painting.rs:42-106
fn draw_triangle_with_alpha(img: &mut RgbaImage, points: &[Point<i32>], color: Rgba<u8>)
```

**Why this exists:** The `imageproc` library's `draw_polygon_mut()` does **opaque pixel replacement**, ignoring alpha channels. This custom function implements proper blending:

- **Fast path**: Fully opaque triangles (alpha=255) use direct `draw_polygon_mut()`
- **Skip**: Fully transparent triangles (alpha=0) are skipped
- **Blend**: Semi-transparent triangles (0 < alpha < 255) use pixel-by-pixel compositing

**Formula:** `result = (src_alpha × src_color) + ((1 - src_alpha) × dst_color)`

**Performance impact:** Semi-transparent triangles are slower due to temporary buffer creation and per-pixel blending.

### Degenerate Triangle Handling (painting.rs)

```rust
// Located in painting.rs:25-40
fn is_degenerate(points: &[(i32, i32); 3]) -> bool
```

Random triangle generation can create degenerate triangles (duplicate/collinear points), which cause `draw_polygon_mut()` to panic. The rendering loop checks for and skips these triangles using cross-product area calculation.

### Two-Stage Evaluation in Main Loop (main.rs:187-193)

```rust
pop.evolve_generation(&params);  // Creates unevaluated offspring
pop.evaluate();                  // MUST evaluate before accessing fitness
```

**Critical:** After breeding, new individuals have `fitness: None`. You must call `evaluate()` before calling `best()` or `average_fitness()`, or the code will panic on `.unwrap()`.

### Background Color Hardcoded to White (painting.rs:163-167)

```rust
let mut img = RgbaImage::from_pixel(
    self.img_width,
    self.img_height,
    Rgba([255, 255, 255, 255]),  // Always fully opaque white
);
```

All output images use white backgrounds regardless of the `background_color` field stored in `Painting`. This ensures consistent output for the genetic algorithm.

## Parallelization Strategy

The codebase uses **Rayon** for data parallelism with zero manual thread management:

1. **Fitness evaluation** (`population.rs:130-132`): `individuals.par_iter_mut()` evaluates all individuals in parallel
2. **Pixel comparison** (`fitness.rs:47-49`): `par_bridge()` parallelizes image pixel iteration
3. **Performance scaling**: Near-linear with CPU core count

**No locks or mutexes needed** - Rayon handles work stealing and synchronization automatically.

## Common Pitfalls

1. **Running debug builds**: Debug builds are 10-100x slower. Always use `cargo build --release` for evolution runs.

2. **Forgetting to evaluate after breeding**: The main loop must call `pop.evaluate()` after `pop.evolve_generation()` or fitness access will panic.

3. **Assuming alpha channels affect fitness**: The fitness function ignores alpha - only RGB values are compared (see `fitness.rs:51-56`).

4. **Modifying triangle generation without degenerate checks**: If you change `Triangle::new()`, ensure the rendering loop can still handle degenerate cases.

5. **Large images**: Image dimensions directly affect performance. Images are not resized - a 4000×3000px image will be much slower than 800×600px.

## Testing Strategy

- **28 unit tests** covering all modules
- Tests use small dimensions (20×20 to 100×100) for speed
- Key test patterns:
  - Property tests (e.g., `test_triangle_creation` verifies points within bounds)
  - Invariant tests (e.g., `test_population_evaluation` checks all fitness scores are set)
  - Parallel equivalence tests (e.g., `test_parallel_equals_sequential` ensures Rayon correctness)
  - Panic tests (e.g., `test_best_panics_without_evaluation` verifies defensive programming)

## Genetic Algorithm Parameters

Located in `evolution.rs` as `EvolutionParams`:

- **population_size**: Number of individuals (default: 200)
- **survival_rate**: Fraction that survives (default: 0.05 = top 5%)
- **mutation_rate**: Fraction of triangles to mutate (default: 0.04 = 4%)
- **swap_prob**: Probability of z-order swap (default: 0.5 = 50%)
- **sigma**: Mutation strength multiplier (default: 1.0)

**Tuning guidance:**
- Higher mutation_rate = more exploration, less stability
- Lower survival_rate = stronger selection pressure, faster convergence
- Higher sigma = larger mutations, can escape local optima but disrupts good solutions

## Image I/O

- **Input**: Supports PNG, JPEG, GIF, BMP via `image` crate
- **Output**: Always PNG format with `.png` extension
- **Naming**: `generation_{gen:05}.png` (zero-padded to 5 digits) and `latest.png`
- **Location**: `args.output` directory (default: `./output/`)

## Dependencies

- **rand 0.8**: RNG for mutation and selection
- **rayon 1.8**: Data parallelism (par_iter, par_bridge)
- **image 0.24**: Image loading/saving/manipulation
- **imageproc 0.23**: Drawing primitives (draw_polygon_mut)
- **clap 4.4**: CLI parsing with derive macros
- **indicatif 0.17**: Progress bars with ETA
- **serde 1.0 + serde_json 1.0**: Serialization (currently unused in CLI but available)

## Performance Optimization Settings

In `Cargo.toml`:

```toml
[profile.release]
opt-level = 3       # Maximum optimization
lto = true          # Link-time optimization
codegen-units = 1   # Single compilation unit for better optimization
```

These settings prioritize runtime speed over compile time. Release builds take ~30 seconds but produce highly optimized binaries.

## Development Workflow

1. Make changes to library code (`src/lib.rs`, `src/*.rs` except `main.rs`)
2. Run `cargo test --lib` to verify tests pass (fast, ~1-2 seconds)
3. Build release: `cargo build --release` (~30 seconds)
4. Test with small run: `./target/release/genetic-art --input test.jpg --generations 10 --triangles 50`
5. Verify output in `./output/latest.png`

## Code Style Notes

- Extensive inline comments explaining Rust concepts for learning purposes
- Each struct/function has doc comments (`///`)
- Liberal use of `derive` macros (Clone, Debug, Serialize, Deserialize)
- Prefers iterator chains over explicit loops where idiomatic
- Uses `pub` fields on structs for simplicity (this is a learning project, not a library with API stability concerns)

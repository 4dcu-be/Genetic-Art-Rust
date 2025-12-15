# Genetic Art Generator 🎨

A high-performance genetic algorithm implementation in Rust that evolves random shapes (triangles or circles) to recreate famous paintings. Watch as natural selection, crossover breeding, and mutation combine to transform chaos into art!

![Rust Version](https://img.shields.io/badge/rust-1.91+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Quick Start

### Prerequisites

- Rust 1.91+ ([install from rustup.rs](https://rustup.rs/))
- A target image (PNG, JPEG, etc.)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd rust_genetic_algorithm

# Build the release version (optimized for speed)
cargo build --release

# The binary will be at: ./target/release/genetic-art
```

### Your First Evolution

```bash
# Run a quick test with triangles (100 generations)
./target/release/genetic-art \
  --input input/starry_night.jpg \
  --generations 100 \
  --shapes 100

# Or try with circles!
./target/release/genetic-art \
  --input input/starry_night.jpg \
  --shape circle \
  --generations 100 \
  --shapes 100

# Check the result
open output/latest.png  # macOS
# or
xdg-open output/latest.png  # Linux
```

---

## 📖 How It Works

The algorithm uses a genetic approach inspired by natural evolution:

1. **Initialize**: Create 200 random "paintings" (each made of 150 shapes - triangles or circles)
2. **Evaluate**: Compare each painting to the target image (fitness score)
3. **Select**: Keep only the best 5% (natural selection)
4. **Breed**: Cross-breed survivors to create offspring
5. **Mutate**: Apply random changes to introduce variation
6. **Repeat**: Continue for thousands of generations

Over time, the paintings evolve to increasingly resemble the target!

### Why Rust?

This implementation is **10-20x faster** than equivalent Python code:

- **Zero-cost abstractions**: High-level code compiles to efficient machine code
- **Parallel processing**: Automatically uses all CPU cores (via Rayon)
- **Memory safety**: No garbage collection pauses, no memory leaks
- **Optimized rendering**: Fast triangle rasterization

---

## 🎮 Usage

### Basic Usage

```bash
genetic-art --input <PATH_TO_IMAGE>
```

This uses sensible defaults for all parameters. The algorithm will run for 5000 generations and save progress images every 50 generations to `./output/`.

### Full Command Syntax

```bash
genetic-art [OPTIONS] --input <INPUT>
```

---

## ⚙️ Command Line Options

### Required Arguments

| Flag | Description |
|------|-------------|
| `-i, --input <PATH>` | Path to target image (PNG, JPEG, GIF, BMP, etc.) |

### Optional Arguments

#### Core Parameters

| Flag | Default | Range | Description |
|------|---------|-------|-------------|
| `-s, --shape <TYPE>` | triangle | triangle, circle | Type of shape to use (triangle or circle) |
| `-n, --shapes <NUM>` | 150 | 10-500 | Number of shapes per painting. More = more detail possible but slower evolution |
| `-p, --population <NUM>` | 200 | 50-1000 | Population size. Larger = more diversity but slower per generation |
| `-g, --generations <NUM>` | 5000 | 100-50000 | Number of generations. More = better results but takes longer |

#### Evolution Parameters

| Flag | Default | Range | Description |
|------|---------|-------|-------------|
| `--mutation-rate <RATE>` | 0.04 | 0.0-1.0 | Fraction of shapes to mutate. Higher = more variation |
| `--survival-rate <RATE>` | 0.05 | 0.0-1.0 | Fraction that survives. Lower = stronger selection pressure |
| `--sigma <STRENGTH>` | 1.0 | 0.0-2.0 | Mutation strength. Higher = larger random changes |

#### Fitness Function Parameters

| Flag | Default | Options | Description |
|------|---------|---------|-------------|
| `--fitness-function <FUNC>` | mad | mad, edge-weighted, ms-ssim | Fitness function for image comparison |
| `--edge-power <POWER>` | 2.0 | 0.5-3.0 | Edge emphasis power (edge-weighted only) |
| `--edge-scale <SCALE>` | 4.0 | 1.0-10.0 | Edge weight scale factor (edge-weighted only) |
| `--detail-weight <WEIGHT>` | 2.0 | 1.0-5.0 | Detail scale weight (ms-ssim only) |

#### Output Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output <DIR>` | ./output | Directory for generated images |
| `--save-interval <NUM>` | 50 | Save image every N generations |

#### Performance Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `-t, --threads <NUM>` | All cores | Number of threads for parallel processing. Set lower to reduce CPU usage |

---

## 💡 Usage Examples

### Quick Test (1 minute)
```bash
# Fast test with fewer generations and shapes
./target/release/genetic-art \
  --input input/image.jpg \
  --shapes 50 \
  --generations 500
```

### High Quality (10-15 minutes)
```bash
# Full quality evolution
./target/release/genetic-art \
  --input input/image.jpg \
  --shapes 150 \
  --generations 5000
```

### Maximum Detail (30-60 minutes)
```bash
# Very high detail (slow but beautiful)
./target/release/genetic-art \
  --input input/image.jpg \
  --shapes 250 \
  --population 300 \
  --generations 10000 \
  --save-interval 100
```

### Limited CPU Usage
```bash
# Run with only 4 threads (useful when multitasking)
./target/release/genetic-art \
  --input input/image.jpg \
  --threads 4 \
  --generations 5000
```

### Using Edge-Weighted Fitness (Emphasize Details)
```bash
# Prioritizes edges and fine details over uniform areas
./target/release/genetic-art \
  --input input/image.jpg \
  --fitness-function edge-weighted \
  --edge-power 2.0 \
  --edge-scale 4.0 \
  --generations 5000
```

### Using MS-SSIM Fitness (Best Quality)
```bash
# Multi-scale structural similarity (slower but best quality)
./target/release/genetic-art \
  --input input/image.jpg \
  --fitness-function ms-ssim \
  --detail-weight 2.0 \
  --generations 5000
```

### Aggressive Evolution
```bash
# Strong selection pressure, higher mutation
./target/release/genetic-art \
  --input input/image.jpg \
  --survival-rate 0.02 \
  --mutation-rate 0.08 \
  --sigma 1.5
```

### Conservative Evolution
```bash
# Gentle selection, lower mutation
./target/release/genetic-art \
  --input input/image.jpg \
  --survival-rate 0.10 \
  --mutation-rate 0.02 \
  --sigma 0.5
```

---

## 📊 Understanding the Parameters

### Shapes (`-n, --shapes`)

**What it does:** Sets how many shapes (triangles or circles) compose each painting.

- **50-100**: Abstract style, fast evolution
- **150-200**: Good balance (recommended)
- **250-500**: High detail, slower but more accurate

**Trade-off:** More shapes = more detail possible, but each generation takes longer to compute.

### Population Size (`-p, --population`)

**What it does:** How many paintings evolve simultaneously.

- **50-100**: Fast, less diversity
- **200-300**: Good balance (recommended)
- **400-1000**: More diversity, slower

**Trade-off:** Larger population explores more solutions but takes longer per generation.

### Generations (`-g, --generations`)

**What it does:** How many evolution cycles to run.

- **100-500**: Quick preview
- **1000-5000**: Good results (recommended)
- **10000+**: Refinement stage

**Trade-off:** More generations = better results, but diminishing returns after a point.

### Mutation Rate (`--mutation-rate`)

**What it does:** Fraction of shapes that mutate in each offspring (0.0-1.0).

- **0.02-0.03**: Conservative (small changes)
- **0.04-0.06**: Balanced (recommended)
- **0.08-0.10**: Aggressive (large changes)

**Trade-off:** Higher = more exploration but less stability. Lower = more refinement but slower progress.

### Survival Rate (`--survival-rate`)

**What it does:** Fraction of population that survives each generation (0.0-1.0).

- **0.02-0.03**: Very strong selection
- **0.05**: Strong selection (recommended)
- **0.10-0.20**: Weak selection, maintains diversity

**Trade-off:** Lower = faster improvement but less diversity. Higher = more diversity but slower improvement.

### Sigma (`--sigma`)

**What it does:** Controls the magnitude of mutations (0.0-2.0).

- **0.5**: Small adjustments (fine-tuning)
- **1.0**: Standard changes (recommended)
- **1.5-2.0**: Large jumps (exploration)

**Trade-off:** Higher values make bigger random changes, good for escaping local optima but can disrupt good solutions.

### Fitness Functions (`--fitness-function`)

**What it does:** Determines how the algorithm compares rendered images to the target.

#### MAD (Mean Absolute Difference) - Default
- **Speed:** Fastest (baseline)
- **Behavior:** Treats all pixels uniformly
- **Best for:** General use, quick iterations
- **Formula:** `sum(|pixel_source - pixel_target|) / (width × height × 3)`

```bash
--fitness-function mad
```

#### Edge-Weighted MAD
- **Speed:** ~2x slower than MAD
- **Behavior:** Emphasizes pixels near edges and details
- **Best for:** Images with important fine details (portraits, text, intricate patterns)
- **How it works:**
  - Detects edges in target image using Sobel operator
  - Assigns higher weights to pixels near edges
  - Getting edge details wrong costs more than uniform areas
- **Parameters:**
  - `--edge-power`: Controls emphasis strength (1.0 = linear, 2.0 = quadratic)
  - `--edge-scale`: Maximum weight multiplier (4.0 = edges 5x more important)

```bash
--fitness-function edge-weighted --edge-power 2.0 --edge-scale 4.0
```

**Example:** For a portrait, getting the eyes and facial features right matters more than the background sky.

#### MS-SSIM (Multi-Scale Structural Similarity)
- **Speed:** ~5-10x slower than MAD
- **Behavior:** Evaluates similarity at multiple scales (resolutions)
- **Best for:** Highest quality results when time permits
- **How it works:**
  - Creates image pyramid (5 scales: 1x, 0.5x, 0.25x, etc.)
  - Compares structure, contrast, and luminance at each scale
  - Finer scales capture details, coarser scales capture overall composition
  - Weights can emphasize fine details
- **Parameters:**
  - `--detail-weight`: Exponential weight for finer scales (2.0 = each finer scale 2x more important)

```bash
--fitness-function ms-ssim --detail-weight 2.0
```

**Example:** Better at capturing both overall composition AND fine details simultaneously.

#### Choosing a Fitness Function

| Use Case | Recommended Function | Reason |
|----------|---------------------|--------|
| Quick experiments | MAD | Fastest, good general results |
| Portraits/faces | Edge-Weighted | Emphasizes facial features |
| Text/logos | Edge-Weighted | Sharp edges are critical |
| Landscapes | MAD or MS-SSIM | Balance of detail and speed |
| Artistic quality | MS-SSIM | Best perceptual quality |
| Limited time | MAD | Best speed/quality ratio |

**Color Handling:** All fitness functions compare full RGB colors correctly. Edge-weighted uses grayscale ONLY to detect edges, but still compares colors when computing fitness.

---

## 📁 Output Files

The algorithm saves images to the output directory:

```
output/
├── generation_00000.png    # Initial random state
├── generation_00050.png    # After 50 generations
├── generation_00100.png    # After 100 generations
├── ...
├── generation_05000.png    # Final result
└── latest.png              # Always the most recent (for quick viewing)
```

---

## 🧬 Algorithm Details

### Genetic Operators

**Selection:**
- Elitist strategy: Best individual always survives
- Plus random selection for second parent (maintains diversity)

**Crossover:**
- Uniform crossover: Each shape randomly chosen from either parent
- 50/50 chance for each gene

**Mutation:**
- Shape shift: Move entire shape
- Point mutation: Move single vertex (triangles) or center (circles)
- Color mutation: Change RGB and alpha values
- Reset: Complete randomization (rare)
- Z-order swap: Change shape rendering order
- Radius mutation: Change circle size (circles only)

### Fitness Functions

Three fitness functions available via `--fitness-function`:

**MAD (Mean Absolute Difference)** - Default:
```
fitness = sum(|pixel_source - pixel_target|) / (width × height × 3)
```

**Edge-Weighted MAD:**
```
fitness = sum(weight_pixel × |pixel_source - pixel_target|) / sum(weights)
```
Where weights are higher near edges detected in the target image.

**MS-SSIM (Multi-Scale Structural Similarity):**
Compares images at multiple resolutions using structural similarity metrics.

Lower score = better match. Perfect match = 0.0 (for MAD and Edge-Weighted) or 255.0 (for MS-SSIM converted to same scale).

### Parallel Processing

The algorithm automatically uses all your CPU cores:

- **Fitness evaluation**: All individuals evaluated in parallel
- **Image comparison**: Parallel pixel-by-pixel comparison
- **Performance scaling**: Near-linear with core count (4 cores ≈ 4x faster)

---

## 🔧 Development

### Building from Source

```bash
# Development build (fast compile, slow runtime)
cargo build

# Release build (slow compile, fast runtime)
cargo build --release

# Always use --release for actual evolution!
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_triangle_creation

# Run with output
cargo test -- --nocapture
```

### Project Structure

```
src/
├── main.rs           # CLI application entry point
├── lib.rs            # Library root
├── genes/
│   ├── mod.rs        # Gene module
│   └── triangle.rs   # Triangle gene implementation
├── painting.rs       # Painting (collection of triangles)
├── population.rs     # Population management
├── evolution.rs      # Genetic operators
└── fitness.rs        # Image comparison
```

---

## 📈 Performance Tips

### For Fastest Evolution

1. **Use release mode**: `cargo build --release` (10-100x faster than debug!)
2. **Start small**: Test with `--generations 100` first
3. **Reduce shapes**: Start with `--shapes 50` for quick iteration
4. **Use MAD fitness**: `--fitness-function mad` (fastest option)
5. **Increase save interval**: `--save-interval 100` reduces I/O overhead
6. **Use smaller images**: Resize target to 400-800px width

### For Best Quality

1. **More generations**: 5000-10000 generations
2. **More shapes**: 150-250 shapes
3. **Larger population**: 200-300 individuals
4. **Try MS-SSIM fitness**: `--fitness-function ms-ssim` for perceptually better results
5. **Use edge-weighted for portraits**: `--fitness-function edge-weighted` for detailed faces
6. **Let it run overnight**: Best results take time!

### Expected Runtime

On a modern 6-core CPU (2.5 GHz):

| Configuration | Time |
|---------------|------|
| 100 gen, 50 tri, pop 100 | ~30 seconds |
| 1000 gen, 150 tri, pop 200 | ~5 minutes |
| 5000 gen, 150 tri, pop 200 | ~20 minutes |
| 10000 gen, 250 tri, pop 300 | ~90 minutes |

**Note:** Times vary significantly based on CPU, image size, and parameters.

---

## 🎨 Tips for Best Results

### Image Selection

**Good targets:**
- High contrast images
- Clear subjects
- Not too much fine detail
- Medium size (400-1000px)

**Avoid:**
- Very noisy/grainy images
- Extremely detailed photographs
- Very large images (resize first)

### Parameter Tuning

**For portraits:**
```bash
--shapes 200 --sigma 0.8 --mutation-rate 0.05 \
--fitness-function edge-weighted --edge-power 2.0
```

**For landscapes:**
```bash
--shapes 150 --sigma 1.2 --mutation-rate 0.06 \
--fitness-function mad
```

**For abstract art:**
```bash
--shapes 100 --sigma 1.5 --survival-rate 0.03 \
--fitness-function mad
```

**For maximum detail:**
```bash
--shapes 250 --fitness-function ms-ssim \
--detail-weight 2.5 --generations 10000
```

---

## 📚 Learning Resources

### About Genetic Algorithms

- [Introduction to Genetic Algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm)
- [Evolutionary Art](https://en.wikipedia.org/wiki/Evolutionary_art)

### About Rust

- [The Rust Book](https://doc.rust-lang.org/book/) - Official Rust learning resource
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) - Learn by doing

### Libraries Used

- [image](https://docs.rs/image/) - Image loading and processing
- [imageproc](https://docs.rs/imageproc/) - Image processing primitives
- [rayon](https://docs.rs/rayon/) - Data parallelism
- [clap](https://docs.rs/clap/) - Command line parsing
- [indicatif](https://docs.rs/indicatif/) - Progress bars

---

## 🤝 Contributing

This project was created as a learning exercise. Feel free to:

- Report bugs
- Suggest improvements
- Submit pull requests
- Share your evolved art!

---

## 📝 License

MIT License - see LICENSE file for details.


## 🙏 Acknowledgments

- Inspired by Roger Johansson's [genetic art algorithm](https://rogerjohansson.blog/2008/12/07/genetic-programming-evolution-of-mona-lisa/)
- Based on the [Python implementation](https://github.com/4dcu-be/Genetic-Art-Algorithm)


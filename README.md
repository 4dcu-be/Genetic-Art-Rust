# Genetic Art Generator 🎨

A high-performance genetic algorithm implementation in Rust that evolves random triangles to recreate famous paintings. Watch as natural selection, crossover breeding, and mutation combine to transform chaos into art!

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
# Run a quick test (100 generations)
./target/release/genetic-art \
  --input input/starry_night.jpg \
  --generations 100 \
  --triangles 100

# Check the result
open output/latest.png  # macOS
# or
xdg-open output/latest.png  # Linux
```

---

## 📖 How It Works

The algorithm uses a genetic approach inspired by natural evolution:

1. **Initialize**: Create 200 random "paintings" (each made of 150 triangles)
2. **Evaluate**: Compare each painting to the target image (fitness score)
3. **Select**: Keep only the best 5% (natural selection)
4. **Breed**: Cross-breed survivors to create offspring
5. **Mutate**: Apply random changes to introduce variation
6. **Repeat**: Continue for thousands of generations

Over time, the paintings evolve to increasingly resemble the target!

### Why Rust?

This implementation is **20-50x faster** than equivalent Python code:

- **Zero-cost abstractions**: High-level code compiles to efficient machine code
- **Parallel processing**: Automatically uses all CPU cores (via Rayon)
- **Memory safety**: No garbage collection pauses, no memory leaks
- **Optimized rendering**: Fast triangle rasterization

**Performance:** On a modern 6-core CPU, ~3-4 generations per second with 200 individuals.

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
| `-t, --triangles <NUM>` | 150 | 10-500 | Number of triangles per painting. More = more detail possible but slower evolution |
| `-p, --population <NUM>` | 200 | 50-1000 | Population size. Larger = more diversity but slower per generation |
| `-g, --generations <NUM>` | 5000 | 100-50000 | Number of generations. More = better results but takes longer |

#### Evolution Parameters

| Flag | Default | Range | Description |
|------|---------|-------|-------------|
| `--mutation-rate <RATE>` | 0.04 | 0.0-1.0 | Fraction of triangles to mutate. Higher = more variation |
| `--survival-rate <RATE>` | 0.05 | 0.0-1.0 | Fraction that survives. Lower = stronger selection pressure |
| `--sigma <STRENGTH>` | 1.0 | 0.0-2.0 | Mutation strength. Higher = larger random changes |

#### Output Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output <DIR>` | ./output | Directory for generated images |
| `--save-interval <NUM>` | 50 | Save image every N generations |

---

## 💡 Usage Examples

### Quick Test (1 minute)
```bash
# Fast test with fewer generations and triangles
./target/release/genetic-art \
  --input input/image.jpg \
  --triangles 50 \
  --generations 500
```

### High Quality (10-15 minutes)
```bash
# Full quality evolution
./target/release/genetic-art \
  --input input/image.jpg \
  --triangles 150 \
  --generations 5000
```

### Maximum Detail (30-60 minutes)
```bash
# Very high detail (slow but beautiful)
./target/release/genetic-art \
  --input input/image.jpg \
  --triangles 250 \
  --population 300 \
  --generations 10000 \
  --save-interval 100
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

### Triangles (`-t, --triangles`)

**What it does:** Sets how many triangles compose each painting.

- **50-100**: Abstract style, fast evolution
- **150-200**: Good balance (recommended)
- **250-500**: High detail, slower but more accurate

**Trade-off:** More triangles = more detail possible, but each generation takes longer to compute.

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

**What it does:** Fraction of triangles that mutate in each offspring (0.0-1.0).

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

**Tip:** Create an animation from the images:
```bash
# Using ffmpeg
ffmpeg -framerate 10 -pattern_type glob -i 'output/generation_*.png' \
  -c:v libx264 -pix_fmt yuv420p evolution.mp4
```

---

## 🧬 Algorithm Details

### Genetic Operators

**Selection:**
- Elitist strategy: Best individual always survives
- Plus random selection for second parent (maintains diversity)

**Crossover:**
- Uniform crossover: Each triangle randomly chosen from either parent
- 50/50 chance for each gene

**Mutation:**
- Triangle shift: Move entire triangle
- Point mutation: Move single vertex
- Color mutation: Change RGB values
- Reset: Complete randomization (rare)
- Z-order swap: Change triangle rendering order

### Fitness Function

Uses mean absolute difference (MAD) between images:

```
fitness = sum(|pixel_source - pixel_target|) / (width × height × 3)
```

Lower score = better match. Perfect match = 0.0.

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
3. **Reduce triangles**: Start with `--triangles 50` for quick iteration
4. **Increase save interval**: `--save-interval 100` reduces I/O overhead
5. **Use smaller images**: Resize target to 400-800px width

### For Best Quality

1. **More generations**: 5000-10000 generations
2. **More triangles**: 150-250 triangles
3. **Larger population**: 200-300 individuals
4. **Let it run overnight**: Best results take time!

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
--triangles 200 --sigma 0.8 --mutation-rate 0.05
```

**For landscapes:**
```bash
--triangles 150 --sigma 1.2 --mutation-rate 0.06
```

**For abstract art:**
```bash
--triangles 100 --sigma 1.5 --survival-rate 0.03
```

---

## 🐛 Troubleshooting

### "Input file not found"

**Solution:** Check the path is correct. Use absolute paths or `./` for relative paths.
```bash
./target/release/genetic-art --input ./input/image.jpg
```

### Out of Memory

**Solution:** Reduce population size or triangle count.
```bash
--population 100 --triangles 100
```

### Very Slow Performance

**Problem:** Running debug build instead of release.

**Solution:** Always use `cargo build --release` and run `./target/release/genetic-art`.

### Fitness Not Improving

**Possible causes:**
- Not enough generations (try 2x-5x more)
- Population too small (increase `--population`)
- Mutation rate too high (try lower `--mutation-rate`)

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

---

## 🎯 Future Enhancements

Potential improvements:

- [ ] Resume from checkpoint
- [ ] Real-time preview window
- [ ] Multiple shape types (circles, polygons)
- [ ] Adaptive mutation rates
- [ ] Multi-objective optimization
- [ ] GPU acceleration
- [ ] WebAssembly port

---

## 🙏 Acknowledgments

- Inspired by Roger Johansson's [genetic art algorithm](https://rogerjohansson.blog/2008/12/07/genetic-programming-evolution-of-mona-lisa/)
- Based on the [Python implementation](https://github.com/4dcu-be/Genetic-Art-Algorithm)
- Built with Rust 🦀 - a language empowering everyone to build reliable and efficient software

---

**Enjoy evolving art!** 🎨✨

If you create something cool, share it! Tag your creations with #GeneticArtRust

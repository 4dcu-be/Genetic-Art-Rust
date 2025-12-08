// Import the clap Parser derive macro for CLI argument parsing
use clap::Parser;

// Import our library types
use genetic_art::{EvolutionParams, FitnessConfig, FitnessFunction, Population, ShapeType};

// Import indicatif for progress bars
use indicatif::{ProgressBar, ProgressStyle};

// Import standard library modules
use std::fs;
use std::path::Path;

/// Genetic Art Generator - Evolve shapes to recreate famous paintings
///
/// This application uses a genetic algorithm to evolve a population of paintings
/// (made of random shapes) to recreate a target image. Over generations, the
/// paintings become increasingly similar to the target through selection, breeding,
/// and mutation.
///
/// **Rust Concept: Derive macros for CLI parsing**
/// The `#[derive(Parser)]` macro automatically generates argument parsing code
/// The doc comments become help text!
#[derive(Parser)]
#[command(name = "genetic-art")]
#[command(about = "Generate art using genetic algorithms", long_about = None)]
#[command(version)]
struct Args {
    /// Path to target image (PNG, JPEG, etc.)
    ///
    /// This is the image we're trying to recreate with shapes.
    /// The algorithm will evolve paintings to match this as closely as possible.
    #[arg(short, long)]
    input: String,

    /// Type of shape to use (triangle or circle)
    ///
    /// Determines which shape primitive will be evolved
    #[arg(short = 's', long, default_value = "triangle")]
    shape: String,

    /// Number of shapes per painting
    ///
    /// More shapes = more detail possible, but slower evolution
    /// Recommended: 50-200
    #[arg(short = 'n', long, default_value_t = 150)]
    shapes: usize,

    /// Population size (number of paintings evolving simultaneously)
    ///
    /// Larger population = more diversity, but slower per generation
    /// Recommended: 100-300
    #[arg(short, long, default_value_t = 200)]
    population: usize,

    /// Number of generations to evolve
    ///
    /// More generations = better results (but takes longer)
    /// 1000-5000 gives good results
    #[arg(short, long, default_value_t = 5000)]
    generations: usize,

    /// Output directory for generated images
    ///
    /// Images will be saved here at regular intervals
    #[arg(short, long, default_value = "./output")]
    output: String,

    /// Save image every N generations
    ///
    /// Lower = more saved images (larger disk space)
    /// Higher = fewer saved images (saves faster)
    #[arg(long, default_value_t = 50)]
    save_interval: usize,

    /// Mutation rate (fraction of shapes to mutate, 0.0-1.0)
    ///
    /// Higher = more variation per generation
    /// Lower = more stable evolution
    #[arg(long, default_value_t = 0.04)]
    mutation_rate: f32,

    /// Survival rate (fraction of population that survives, 0.0-1.0)
    ///
    /// Lower = stronger selection pressure
    /// Higher = more diversity maintained
    #[arg(long, default_value_t = 0.05)]
    survival_rate: f32,

    /// Mutation strength (0.0-2.0, typically 1.0)
    ///
    /// Higher = larger random changes
    /// Lower = smaller adjustments
    #[arg(long, default_value_t = 1.0)]
    sigma: f32,

    /// Fitness function to use for image comparison
    ///
    /// - mad: Mean Absolute Difference (fast, uniform pixel weighting)
    /// - edge-weighted: Emphasizes edges and details (2x slower)
    /// - ms-ssim: Multi-scale Structural Similarity (best quality, 5-10x slower)
    #[arg(long, default_value = "mad")]
    fitness_function: FitnessFunction,

    /// Edge emphasis power (edge-weighted only)
    ///
    /// Controls how strongly edges are weighted:
    /// - 1.0 = linear weighting
    /// - 2.0 = quadratic (default, strong emphasis)
    /// - 0.5 = sublinear (gentle emphasis)
    #[arg(long, default_value_t = 2.0)]
    edge_power: f64,

    /// Edge weight scale factor (edge-weighted only)
    ///
    /// Maximum weight multiplier for high-gradient regions
    /// Higher = stronger emphasis on edges vs uniform areas
    #[arg(long, default_value_t = 4.0)]
    edge_scale: f64,

    /// Detail scale weight (ms-ssim only)
    ///
    /// Exponential weight applied to finer scales:
    /// - 1.0 = equal weighting across all scales
    /// - 2.0 = each finer scale gets 2x more weight (default)
    /// - 3.0 = aggressive fine-detail emphasis
    #[arg(long, default_value_t = 2.0)]
    detail_weight: f64,

    /// Number of threads for parallel processing
    ///
    /// Limits Rayon's thread pool size. By default, uses all available CPU cores.
    /// Set lower to reduce CPU usage or when running multiple instances.
    #[arg(short = 't', long)]
    threads: Option<usize>,
}

/// Main entry point for the CLI application
///
/// **Rust Concept: Result and error handling**
/// Rust doesn't have exceptions. Instead, functions return `Result<T, E>`:
/// - `Ok(value)` for success
/// - `Err(error)` for failure
///
/// The `?` operator automatically propagates errors up the call stack.
/// This is like `try-catch` but enforced by the type system!
fn main() {
    // Parse command line arguments
    // If parsing fails (invalid args), clap automatically prints help and exits
    let args = Args::parse();

    // Run the algorithm and handle any errors
    // Using a separate function keeps main() clean
    if let Err(e) = run(args) {
        // If something went wrong, print the error and exit with error code
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Run the genetic algorithm with given arguments
///
/// **Rust Concept: Box<dyn Error>**
/// - `Box` is a smart pointer to heap-allocated data
/// - `dyn Error` means "any type that implements the Error trait"
/// - This lets us return different error types from one function
/// - Dynamic dispatch (slight runtime cost, but more flexible)
///
/// **Why use this?**
/// Different operations can fail in different ways:
/// - File I/O errors (io::Error)
/// - Image loading errors (image::Error)
/// - etc.
fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    // Step 0: Configure Rayon thread pool if thread limit is specified
    if let Some(num_threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .map_err(|e| format!("Failed to configure thread pool: {}", e))?;
        println!("Using {} thread(s) for parallel processing", num_threads);
    } else {
        println!("Using all available CPU cores for parallel processing");
    }

    // Step 1: Create output directory if it doesn't exist
    println!("Setting up output directory...");
    fs::create_dir_all(&args.output)?;
    // **Note:** The `?` operator returns early if this fails

    // Step 2: Load target image
    println!("Loading target image: {}", args.input);

    // Check if file exists first (better error message)
    if !Path::new(&args.input).exists() {
        return Err(format!("Input file not found: {}", args.input).into());
    }

    // Load and convert to RGBA
    // `image::open()` returns Result<DynamicImage, ImageError>
    // `.to_rgba8()` converts to RGBA format (needed for our algorithm)
    let target = image::open(&args.input)?.to_rgba8();

    let (width, height) = target.dimensions();
    println!("Image dimensions: {}x{}", width, height);

    // Parse shape type
    let shape_type: ShapeType = args.shape.parse()
        .map_err(|e| format!("Invalid shape type: {}", e))?;

    // Step 3: Create fitness configuration
    let fitness_config = FitnessConfig::new(
        args.fitness_function,
        args.edge_power,
        args.edge_scale,
        args.detail_weight,
    );

    // Step 4: Create initial population
    println!("\nInitializing genetic algorithm...");
    println!("  Shape type: {:?}", shape_type);
    println!("  Population size: {}", args.population);
    println!("  Shapes per painting: {}", args.shapes);
    println!("  Generations: {}", args.generations);
    println!("  Mutation rate: {:.1}%", args.mutation_rate * 100.0);
    println!("  Survival rate: {:.1}%", args.survival_rate * 100.0);
    println!("  Fitness function: {:?}", args.fitness_function);

    let mut pop = Population::new(args.population, args.shapes, target, shape_type, fitness_config);

    // Step 5: Configure evolution parameters
    let params = EvolutionParams {
        population_size: args.population,
        survival_rate: args.survival_rate,
        mutation_rate: args.mutation_rate,
        swap_prob: 0.5, // 50% chance of z-order swap
        sigma: args.sigma,
    };

    // Step 6: Setup progress bar
    // `indicatif` creates beautiful terminal progress bars
    let pb = ProgressBar::new(args.generations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            // Template string defines how the bar looks
            // {elapsed_precise} = time elapsed
            // {bar} = the actual progress bar
            // {pos}/{len} = current/total
            // {eta} = estimated time remaining
            // {msg} = custom message (fitness stats)
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} (ETA: {eta}) | {msg}")?
            .progress_chars("=>-"),
    );

    println!("\n🧬 Starting evolution...\n");

    // Step 7: Evolution loop!
    // This is where the magic happens
    for gen in 0..args.generations {
        // Evolve one generation
        // This evaluates fitness, selects survivors, breeds, and mutates
        pop.evolve_generation(&params);

        // Evaluate the new generation (offspring were just created and need fitness scores)
        pop.evaluate();

        // Get fitness statistics
        let best_fitness = pop.best().fitness.unwrap();
        let avg_fitness = pop.average_fitness();

        // Update progress bar with current stats
        pb.set_message(format!(
            "Best: {:.2}, Avg: {:.2}",
            best_fitness, avg_fitness
        ));
        pb.inc(1);

        // Save image at intervals
        if gen % args.save_interval == 0 || gen == args.generations - 1 {
            save_generation(&pop, &args.output, gen)?;
        }
    }

    // Finish progress bar with success message
    pb.finish_with_message("Evolution complete! 🎉");

    // Step 8: Print final statistics
    let best = pop.best();
    println!("\n✨ Results:");
    println!("  Final fitness: {:.2}", best.fitness.unwrap());
    println!("  Total generations: {}", pop.generation);
    println!("  Output directory: {}", args.output);
    println!("\nCheck {}/latest.png for the final result!", args.output);

    Ok(())
}

/// Save the best individual from a generation as an image
///
/// **Rust Concept: Error propagation with ?**
/// This function returns Result, and uses `?` to propagate errors
/// If any save operation fails, the error bubbles up to the caller
fn save_generation(
    pop: &Population,
    output_dir: &str,
    generation: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get the best individual from the population
    let best = pop.best();

    // Render the painting to an image
    let img = best.chromosome.render();

    // Save with generation number in filename
    // Format: generation_00000.png, generation_00050.png, etc.
    let filename = format!("{}/generation_{:05}.png", output_dir, generation);
    img.save(&filename)?;

    // Also save as "latest.png" for easy viewing
    // This always contains the most recent result
    let latest = format!("{}/latest.png", output_dir);
    img.save(&latest)?;

    Ok(())
}

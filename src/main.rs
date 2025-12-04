// Import the clap Parser derive macro for CLI argument parsing
use clap::Parser;

// Import our library types
use genetic_art::{EvolutionParams, Population};

// Import indicatif for progress bars
use indicatif::{ProgressBar, ProgressStyle};

// Import standard library modules
use std::fs;
use std::path::Path;

/// Genetic Art Generator - Evolve triangles to recreate famous paintings
///
/// This application uses a genetic algorithm to evolve a population of paintings
/// (made of random triangles) to recreate a target image. Over generations, the
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
    /// This is the image we're trying to recreate with triangles.
    /// The algorithm will evolve paintings to match this as closely as possible.
    #[arg(short, long)]
    input: String,

    /// Number of triangles per painting
    ///
    /// More triangles = more detail possible, but slower evolution
    /// Recommended: 50-200
    #[arg(short, long, default_value_t = 150)]
    triangles: usize,

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

    /// Mutation rate (fraction of triangles to mutate, 0.0-1.0)
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

    // Step 3: Create initial population
    println!("\nInitializing genetic algorithm...");
    println!("  Population size: {}", args.population);
    println!("  Triangles per painting: {}", args.triangles);
    println!("  Generations: {}", args.generations);
    println!("  Mutation rate: {:.1}%", args.mutation_rate * 100.0);
    println!("  Survival rate: {:.1}%", args.survival_rate * 100.0);

    let mut pop = Population::new(args.population, args.triangles, target);

    // Step 4: Configure evolution parameters
    let params = EvolutionParams {
        population_size: args.population,
        survival_rate: args.survival_rate,
        mutation_rate: args.mutation_rate,
        swap_prob: 0.5, // 50% chance of z-order swap
        sigma: args.sigma,
    };

    // Step 5: Setup progress bar
    // `indicatif` creates beautiful terminal progress bars
    let pb = ProgressBar::new(args.generations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            // Template string defines how the bar looks
            // {elapsed_precise} = time elapsed
            // {bar} = the actual progress bar
            // {pos}/{len} = current/total
            // {msg} = custom message
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} | {msg}")?
            .progress_chars("=>-"),
    );

    println!("\n🧬 Starting evolution...\n");

    // Step 6: Evolution loop!
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

    // Step 7: Print final statistics
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

// Import types we need
use crate::painting::Painting;
use crate::population::{Individual, Population};
use rand::Rng;

/// Parameters controlling the evolution process
///
/// These are the "knobs" you can turn to control how evolution works.
/// Different values lead to different evolutionary dynamics!
///
/// **Rust Concept: Derive macros**
/// - `Clone` lets us copy the params
/// - `Debug` lets us print them with {:?}
/// - These are "auto-derived" by the compiler
#[derive(Clone, Debug)]
pub struct EvolutionParams {
    /// Size of the population (how many individuals)
    /// Larger = more diversity, but slower
    pub population_size: usize,

    /// Fraction of population that survives each generation (0.0-1.0)
    /// Lower = more selective pressure, faster evolution
    /// Higher = more diversity, slower evolution
    pub survival_rate: f32,

    /// Fraction of triangles to mutate in each offspring (0.0-1.0)
    /// Lower = more stable, children similar to parents
    /// Higher = more exploration, children can differ a lot
    pub mutation_rate: f32,

    /// Probability of swapping two triangles (changes z-order)
    /// This can dramatically change appearance!
    pub swap_prob: f32,

    /// Mutation strength (0.0-2.0, typically 1.0)
    /// Lower = small changes
    /// Higher = large jumps
    pub sigma: f32,
}

/// Default parameters based on experimentation
///
/// **Rust Concept: Default trait**
/// Implementing Default lets us create params with `EvolutionParams::default()`
/// This is a common Rust pattern for "sensible defaults"
impl Default for EvolutionParams {
    fn default() -> Self {
        Self {
            population_size: 200,    // Medium-large population
            survival_rate: 0.05,     // Only top 5% survive (strong selection)
            mutation_rate: 0.04,     // Mutate 4% of triangles per child
            swap_prob: 0.5,          // 50% chance of z-order swap
            sigma: 1.0,              // Standard mutation strength
        }
    }
}

// Additional methods for Population related to evolution
//
// **Rust Concept: Multiple impl blocks**
// We can have multiple `impl Population` blocks
// This one is in evolution.rs, the other in population.rs
// They both add methods to the same type!
impl Population {
    /// Select two parents from the population
    ///
    /// **Selection Strategy: Elitist + Random**
    /// - Mom: Always the best individual (elitism)
    /// - Dad: Random individual (maintains diversity)
    ///
    /// Why this strategy?
    /// - Elitism ensures best traits are preserved
    /// - Random second parent maintains genetic diversity
    /// - Balance between exploitation (best) and exploration (random)
    ///
    /// **Rust Concept: Returning references**
    /// Returns `(&Individual, &Individual)` - two borrowed references
    /// - We're not taking ownership or copying
    /// - Just pointing to individuals in the population
    /// - Lifetime is tied to the Population (managed by compiler)
    pub fn select_parents(&self) -> (&Individual, &Individual) {
        let mut rng = rand::thread_rng();

        // Pick best individual as mom
        let mom = self.best();

        // Pick random individual as dad
        let dad_idx = rng.gen_range(0..self.individuals.len());
        let dad = &self.individuals[dad_idx];

        (mom, dad)
    }

    /// Breed two paintings to create offspring
    ///
    /// **Genetic Operator: Uniform Crossover**
    /// For each triangle, randomly choose from either parent
    /// This is called "uniform crossover" in genetic algorithms
    ///
    /// **Why static method?**
    /// This doesn't need access to Population data, just the two parents
    /// Making it static (no `&self`) makes the design clearer
    ///
    /// **Rust Concept: Borrowing vs Ownership**
    /// - Takes `&Painting` (borrowed references) not `Painting` (ownership)
    /// - We're just reading the parents, not consuming them
    /// - They can be used again after this function returns
    pub fn breed(parent_a: &Painting, parent_b: &Painting) -> Painting {
        let mut rng = rand::thread_rng();
        let (width, height) = parent_a.dimensions();

        // Start with empty painting (0 triangles)
        let mut child = Painting::new(0, width, height, [255, 255, 255, 255]);

        // For each triangle position, randomly pick from either parent
        // `.zip()` pairs up triangles from both parents
        // Both parents have same number of triangles (by design)
        for (tri_a, tri_b) in parent_a.triangles.iter().zip(&parent_b.triangles) {
            // Flip a coin: 50% chance of each parent
            if rng.gen_bool(0.5) {
                child.triangles.push(tri_a.clone()); // Clone from parent A
            } else {
                child.triangles.push(tri_b.clone()); // Clone from parent B
            }
        }

        // **Genetic Diversity Note:**
        // This creates a new unique combination of triangles
        // Some from mom, some from dad
        // Like genetic recombination in biology!

        child
    }

    /// Evolve the population for one generation
    ///
    /// This is the heart of the genetic algorithm!
    ///
    /// **Genetic Algorithm Steps:**
    /// 1. Evaluate fitness (who's good?)
    /// 2. Select survivors (natural selection)
    /// 3. Breed to create offspring (reproduction)
    /// 4. Mutate offspring (genetic variation)
    /// 5. Repeat!
    ///
    /// **Rust Concept: Mutable self**
    /// `&mut self` means this method modifies the Population
    /// After calling this, the population will be completely different!
    pub fn evolve_generation(&mut self, params: &EvolutionParams) {
        // Step 1: Evaluate fitness for all individuals
        // This happens in parallel across all CPU cores!
        self.evaluate();

        // Step 2: Sort by fitness (best first)
        // After this, individuals[0] is the best, individuals[last] is worst
        //
        // **Performance Note:**
        // Rust's sort is very fast (Timsort variant)
        // For 200 individuals, this takes <1ms
        self.individuals.sort_by(|a, b| {
            a.fitness
                .unwrap()
                .partial_cmp(&b.fitness.unwrap())
                .unwrap()
        });

        // Step 3: Keep only the best individuals (survival of the fittest!)
        // Calculate how many survive
        let survivors = (self.individuals.len() as f32 * params.survival_rate) as usize;
        // Keep at least 2 (need 2 for breeding)
        let survivors = survivors.max(2);

        // `.truncate()` removes all elements after index `survivors`
        // This is very efficient - no reallocation, just sets length
        //
        // **What happens to the removed individuals?**
        // Rust automatically drops them (frees memory)
        // No memory leaks, no manual cleanup!
        self.individuals.truncate(survivors);

        // Step 4: Breed to repopulate
        // Keep breeding until we're back to full population size
        //
        // **Why this loop?**
        // After selection, we have few individuals (e.g., 10 from 200)
        // We need to breed them to get back to 200
        // Each iteration creates one child
        //
        // **Important:** We cache the number of survivors to avoid mixing
        // evaluated survivors with unevaluated offspring during selection
        let num_survivors = self.individuals.len();

        while self.individuals.len() < params.population_size {
            let mut rng = rand::thread_rng();

            // Select parents from the survivors only (first num_survivors individuals)
            // Mom: Always the best (individuals[0] after sorting)
            // Dad: Random from survivors
            let mom = &self.individuals[0];
            let dad_idx = rng.gen_range(0..num_survivors);
            let dad = &self.individuals[dad_idx];

            // Create child by breeding
            // **Rust Concept: References vs Values**
            // `&mom.chromosome` - we're borrowing, not taking ownership
            // This is important because we might breed from the same parents multiple times!
            let mut child_chromosome = Self::breed(&mom.chromosome, &dad.chromosome);

            // Mutate the child
            // This introduces variation - the key to evolution!
            child_chromosome.mutate(params.mutation_rate, params.swap_prob, params.sigma);

            // Add child to population
            // It hasn't been evaluated yet (fitness = None)
            self.individuals.push(Individual {
                chromosome: child_chromosome,
                fitness: None, // Will be evaluated next generation
            });
        }

        // Increment generation counter
        self.generation += 1;

        // **End of Generation!**
        // The population is now ready for the next cycle
        // It contains:
        // - A few survivors from the previous generation (the elite)
        // - Many new children bred from survivors
        // - All with fresh mutations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    #[test]
    fn test_evolution_params_default() {
        let params = EvolutionParams::default();
        assert_eq!(params.population_size, 200);
        assert_eq!(params.survival_rate, 0.05);
        assert_eq!(params.mutation_rate, 0.04);
        assert_eq!(params.swap_prob, 0.5);
        assert_eq!(params.sigma, 1.0);
    }

    #[test]
    fn test_breeding() {
        // Create two different paintings
        let parent_a = Painting::new(10, 100, 100, [255, 255, 255, 255]);
        let parent_b = Painting::new(10, 100, 100, [255, 255, 255, 255]);

        let child = Population::breed(&parent_a, &parent_b);

        // Child should have same number of triangles
        assert_eq!(child.len(), 10);

        // Child should have same dimensions
        assert_eq!(child.dimensions(), (100, 100));
    }

    #[test]
    fn test_parent_selection() {
        let target = RgbaImage::from_pixel(100, 100, Rgba([128, 128, 128, 255]));
        let mut pop = Population::new(20, 5, target);

        // Evaluate so we can select
        pop.evaluate();

        let (mom, dad) = pop.select_parents();

        // Mom should be the best
        let best = pop.best();
        assert_eq!(mom.fitness, best.fitness);

        // Both should have fitness scores
        assert!(mom.fitness.is_some());
        assert!(dad.fitness.is_some());
    }

    #[test]
    fn test_evolution_cycle() {
        let target = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
        let mut pop = Population::new(20, 5, target);

        let params = EvolutionParams::default();

        let initial_gen = pop.generation;
        let _initial_best = {
            pop.evaluate();
            pop.best().fitness.unwrap()
        };

        // Evolve one generation
        pop.evolve_generation(&params);

        // Verify generation counter incremented
        assert_eq!(pop.generation, initial_gen + 1);

        // Verify population size maintained
        assert_eq!(pop.individuals.len(), params.population_size);

        // Most individuals should not be evaluated yet (newly bred)
        // Only the survivors (top 5%) were evaluated before breeding
        let unevaluated = pop.individuals.iter().filter(|i| i.fitness.is_none()).count();
        let evaluated = pop.individuals.len() - unevaluated;

        // Survivors should be much less than total population (we keep ~5%)
        assert!(evaluated < pop.individuals.len() / 2,
                "Most individuals should be unevaluated after breeding");
    }

    #[test]
    fn test_multiple_generations() {
        // Run multiple generations and verify it works correctly
        let target = RgbaImage::from_pixel(50, 50, Rgba([100, 100, 100, 255]));
        let mut pop = Population::new(50, 10, target);

        let params = EvolutionParams {
            population_size: 50,
            survival_rate: 0.1,
            mutation_rate: 0.1,
            swap_prob: 0.5,
            sigma: 1.0,
        };

        // Run 10 generations
        for _ in 0..10 {
            pop.evolve_generation(&params);
        }

        // After 10 generations, we should be at generation 10
        assert_eq!(pop.generation, 10);

        // Population should still be correct size
        assert_eq!(pop.individuals.len(), params.population_size);

        // Evaluate current generation before checking fitness
        pop.evaluate();
        let best = pop.best();
        assert!(best.fitness.is_some());

        // Fitness should be reasonable (not NaN, not infinite)
        let fitness = best.fitness.unwrap();
        assert!(fitness.is_finite());
        assert!(fitness >= 0.0);
    }

    #[test]
    fn test_survival_maintains_minimum() {
        // Test that we keep at least 2 survivors even with very low survival rate
        let target = RgbaImage::from_pixel(100, 100, Rgba([0, 0, 0, 255]));
        let mut pop = Population::new(100, 5, target);

        let params = EvolutionParams {
            population_size: 100,
            survival_rate: 0.001, // Only 0.1% survival = 0 individuals, but should be clamped to 2
            mutation_rate: 0.04,
            swap_prob: 0.5,
            sigma: 1.0,
        };

        pop.evolve_generation(&params);

        // Should have full population after breeding
        assert_eq!(pop.individuals.len(), 100);

        // Evaluate and verify we can get best individual
        pop.evaluate();
        let best = pop.best();
        assert!(best.fitness.is_some());
    }
}

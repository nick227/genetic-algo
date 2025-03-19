"""Module implementing the genetic algorithm for idea generation."""

import csv
import datetime
import random
import signal
import sys
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

from ..config.settings import (
    POPULATION_SIZE,
    DEFAULT_MIN_POPULATION,
    CSV_BASE_NAME,
    CSV_HEADERS,
    MUTATION_RATE,
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_RANDOM_RATE,
    DEFAULT_MAX_MUTATION,
    DEFAULT_MIN_CROSSOVER,
    DEFAULT_MAX_GENERATIONS_WITHOUT_IMPROVEMENT,
    DEFAULT_MIN_DIVERSITY_THRESHOLD,
    DEFAULT_TARGET_FITNESS,
    DEFAULT_FITNESS_IMPROVEMENT_THRESHOLD,
    DEFAULT_MAX_GENERATIONS,
    COST_PER_1K_TOKENS,
    DEFAULT_COST_MODEL
)
from ..models.idea import Idea
from .evaluator import create_evaluator

@dataclass
class Population:
    """Class representing a population of ideas."""
    ideas: List[Idea]
    generation: int = 0
    diversity_score: float = 1.0
    min_population: int = DEFAULT_MIN_POPULATION
    
    @property
    def quartile_size(self) -> int:
        """Get the size of one quartile of the population."""
        min_quartile = max(5, self.min_population // 4)  # Ensure at least 5 ideas per quartile
        calculated = len(self.ideas) // 4
        return max(min_quartile, calculated)
    
    def calculate_diversity(self) -> float:
        """Calculate diversity score of the population."""
        if not self.ideas:
            return 0.0
            
        try:
            components = {
                'technical': set(),
                'industry': set(),
                'action': set(),
                'object': set()
            }
            
            # Collect unique components
            for idea in self.ideas:
                components['technical'].add(idea.technical)
                components['industry'].add(idea.industry)
                components['action'].add(idea.action)
                components['object'].add(idea.object)
            
            # Calculate diversity as average unique ratio
            total_ratio = sum(
                len(unique) / len(self.ideas)
                for unique in components.values()
            )
            self.diversity_score = total_ratio / len(components)
            return self.diversity_score
        except ZeroDivisionError:
            return 0.0
        except Exception as e:
            print(f"Warning: Error calculating diversity: {e}")
            return self.diversity_score  # Return last known score
    
    def sort_by_fitness(self) -> None:
        """Sort ideas by their fitness scores in descending order."""
        self.ideas.sort(key=lambda x: x.fitness_score, reverse=True)
    
    @property
    def top_quartile(self) -> List[Idea]:
        """Get the top quartile of ideas."""
        return self.ideas[:self.quartile_size]
    
    def get_diverse_parents(self) -> Tuple[Idea, Idea]:
        """Select diverse parents for crossover."""
        if len(self.top_quartile) < 2:
            # Fallback if not enough parents
            return random.sample(self.ideas, 2) if len(self.ideas) >= 2 else (self.ideas[0], self.ideas[0])
            
        # Enhanced parent selection with fitness weighting
        parent1 = self._weighted_selection(self.top_quartile)
        
        # Select second parent that's different in at least 2 components
        candidates = [
            idea for idea in self.top_quartile
            if idea != parent1 and self._component_difference(idea, parent1) >= 2
        ]
        
        if candidates:
            parent2 = self._weighted_selection(candidates)
        else:
            # Fallback to any different parent from top quartile
            alternatives = [idea for idea in self.top_quartile if idea != parent1]
            parent2 = self._weighted_selection(alternatives) if alternatives else parent1
        
        return parent1, parent2
    
    def _component_difference(self, idea1: Idea, idea2: Idea) -> int:
        """Calculate how many components are different between two ideas."""
        return sum(
            1 for attr in ['technical', 'industry', 'action', 'object']
            if getattr(idea1, attr) != getattr(idea2, attr)
        )
    
    def _weighted_selection(self, candidates: List[Idea]) -> Idea:
        """Select an idea weighted by fitness score."""
        if not candidates:
            raise ValueError("No candidates for selection")
            
        weights = [idea.fitness_score + 1.0 for idea in candidates]  # Add 1.0 to avoid zero weights
        total = sum(weights)
        weights = [w/total for w in weights]
        
        return random.choices(candidates, weights=weights, k=1)[0]
    
    def ensure_minimum_population(self) -> None:
        """Ensure population meets minimum size requirements."""
        while len(self.ideas) < self.min_population:
            self.ideas.append(Idea.generate_random())

class GeneticAlgorithm:
    """Class implementing the genetic algorithm for idea generation."""
    
    def __init__(self, 
                 seed_ideas: Optional[List[Dict[str, str]]] = None,
                 population_size: Optional[int] = None,
                 min_population: Optional[int] = None,
                 target_fitness: Optional[float] = None,
                 max_generations: Optional[int] = None,
                 max_generations_without_improvement: Optional[int] = None,
                 min_diversity_threshold: Optional[float] = None,
                 fitness_improvement_threshold: Optional[float] = None):
        """Initialize the genetic algorithm."""
        print("\nInitializing Idea Generator...")
        self._running = True
        self._shutting_down = False
        self.csv_filename = self._generate_csv_filename()
        self.seed_ideas = seed_ideas or []
        
        # Initialize token and cost tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self._reset_generation_tokens()
        
        # Override settings with provided parameters
        self.population_size = population_size or POPULATION_SIZE
        self.min_population = min_population or DEFAULT_MIN_POPULATION
        
        # Initialize population and evaluator
        self.population = self._initialize_population()
        self.evaluator = create_evaluator()
        
        # Detect and set cost model
        self.cost_model = self._detect_cost_model()
        
        self.avg_eval_time = None
        self.generation_stats = []
        
        # Initialize adaptive rates
        self.adaptive_rates = {
            'mutation': MUTATION_RATE,
            'crossover': DEFAULT_CROSSOVER_RATE,
            'random': DEFAULT_RANDOM_RATE,
            'max_mutation': DEFAULT_MAX_MUTATION,
            'min_crossover': DEFAULT_MIN_CROSSOVER
        }
        
        # Initialize convergence parameters with overrides
        self.convergence_params = {
            'max_generations': max_generations or DEFAULT_MAX_GENERATIONS,
            'max_generations_without_improvement': (
                max_generations_without_improvement or 
                DEFAULT_MAX_GENERATIONS_WITHOUT_IMPROVEMENT
            ),
            'min_diversity_threshold': (
                min_diversity_threshold or 
                DEFAULT_MIN_DIVERSITY_THRESHOLD
            ),
            'target_fitness': target_fitness or DEFAULT_TARGET_FITNESS,
            'fitness_improvement_threshold': (
                fitness_improvement_threshold or 
                DEFAULT_FITNESS_IMPROVEMENT_THRESHOLD
            )
        }
        
        # Print configuration
        self._print_configuration()
    
    def _print_configuration(self) -> None:
        """Print the current configuration settings."""
        print("\nConfiguration:")
        print(f"- Population Size: {self.population_size}")
        print(f"- Minimum Population: {self.min_population}")
        print(f"- Target Fitness: {self.convergence_params['target_fitness']}")
        print(f"- Maximum Generations: {self.convergence_params['max_generations']}")
        print(f"- Max Generations Without Improvement: {self.convergence_params['max_generations_without_improvement']}")
        print(f"- Minimum Diversity Threshold: {self.convergence_params['min_diversity_threshold']:.2%}")
        print(f"- Fitness Improvement Threshold: {self.convergence_params['fitness_improvement_threshold']}")
        
        if self.seed_ideas:
            print(f"\nSeed Ideas: {len(self.seed_ideas)}")
            for i, seed in enumerate(self.seed_ideas, 1):
                components = [f"{k}={v}" for k, v in seed.items() if v is not None]
                print(f"{i}. {', '.join(components)}")
    
    def _initialize_population(self) -> Population:
        """Create initial population with seed ideas and random ideas."""
        print("\nInitializing population...")
        
        ideas = []
        
        # Add seed ideas first
        if self.seed_ideas:
            print(f"\nIncorporating {len(self.seed_ideas)} seed ideas:")
            for seed in self.seed_ideas:
                idea = Idea.from_components(**seed)
                print(f"- {idea}")
                ideas.append(idea)
                
                # Add variations of seed ideas
                for _ in range(2):  # Create 2 variations of each seed
                    variant = idea.mutate()
                    ideas.append(variant)
        
        # Fill remaining slots with random ideas
        remaining = self.population_size - len(ideas)
        if remaining > 0:
            print(f"\nGenerating {remaining} random ideas to complete population...")
            with tqdm(total=remaining, desc="Generating ideas", unit="idea") as pbar:
                for _ in range(remaining):
                    ideas.append(Idea.generate_random())
                    pbar.update(1)
        
        print(f"\nCreated initial population of {len(ideas)} ideas "
              f"({len(self.seed_ideas)} seeded, {remaining} random).")
        return Population(ideas=ideas, min_population=self.min_population)
    
    def _generate_csv_filename(self) -> str:
        """Generate a timestamped CSV filename."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{CSV_BASE_NAME}_{timestamp}.csv"
    
    def _setup_csv_file(self) -> None:
        """Set up the CSV file for logging results."""
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        print(f"\nCreated log file: {self.csv_filename}")
    
    def _log_evaluation(self, idea: Idea, eval_time: float, token_info: Optional[Dict] = None) -> None:
        """Log an idea evaluation to the CSV file."""
        timestamp = datetime.datetime.now().isoformat()
        
        # Extract evaluation metrics matching CSV_HEADERS structure
        metrics = {
            'viability': 0,
            'value_potential': 0,
            'simplicity': 0,
            'novelty': 0,
            'scalability': 0,
            'reasoning': 'No evaluation',
            'diversity': self.population.diversity_score,
            'population_size': len(self.population.ideas),
            'mutation_rate': self.adaptive_rates['mutation'],
            'crossover_rate': self.adaptive_rates['crossover']
        }
        
        if idea.evaluation:
            # Update only defined metrics
            for key in ['viability', 'value_potential', 'simplicity', 'novelty', 'scalability', 'reasoning']:
                if key in idea.evaluation:
                    metrics[key] = idea.evaluation[key]
        
        # Extract token information
        tokens = token_info or {}
        prompt_tokens = tokens.get('prompt_tokens', 0)
        completion_tokens = tokens.get('completion_tokens', 0)
        total_tokens = tokens.get('total_tokens', prompt_tokens + completion_tokens)
        
        # Calculate cost using current model
        cost = self._calculate_cost(prompt_tokens, completion_tokens)
        
        # Ensure row matches CSV_HEADERS exactly
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,                    # Timestamp
                self.population.generation,   # Generation
                str(idea),                    # Idea Description
                idea.fitness_score,           # Fitness Score
                round(eval_time, 2),          # Evaluation Time
                metrics['viability'],         # Viability
                metrics['value_potential'],   # Value Potential
                metrics['simplicity'],        # Simplicity
                metrics['novelty'],           # Novelty
                metrics['scalability'],       # Scalability
                metrics['reasoning'],         # Reasoning
                metrics['diversity'],         # Diversity
                metrics['population_size'],   # Population Size
                metrics['mutation_rate'],     # Mutation Rate
                metrics['crossover_rate'],    # Crossover Rate
                prompt_tokens,                # Prompt Tokens
                completion_tokens,            # Completion Tokens
                total_tokens,                 # Total Tokens
                round(cost, 6)                # Cost
            ])
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signal."""
        if self._shutting_down:
            print("\n\nForce quitting... Please wait for file operations to complete.")
            sys.exit(1)
        else:
            print("\n\nReceived interrupt signal. Starting graceful shutdown...")
            print("(Press Ctrl+C again to force quit)")
            self._running = False
            self._shutting_down = True
    
    def _cleanup(self) -> None:
        """Perform cleanup operations."""
        with tqdm(total=3, desc="Cleaning up", unit="task") as pbar:
            # Save any pending evaluations
            pbar.set_description("Saving pending evaluations")
            self._save_pending_evaluations()
            pbar.update(1)

            # Sort population for final results
            pbar.set_description("Sorting population")
            self.population.sort_by_fitness()
            pbar.update(1)

            # Ensure CSV file is properly closed
            pbar.set_description("Finalizing log file")
            self._finalize_log_file()
            pbar.update(1)

    def _save_pending_evaluations(self) -> None:
        """Save any pending evaluations to the CSV file."""
        pending = [idea for idea in self.population.ideas 
                  if idea.evaluation is not None and idea.fitness_score > 0]
        if pending:
            current_time = time.time()
            for idea in pending:
                eval_time = self.avg_eval_time if self.avg_eval_time else 0.0
                self._log_evaluation(idea, eval_time)

    def _finalize_log_file(self) -> None:
        """Ensure the CSV file is properly finalized."""
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().isoformat(),
                self.population.generation,
                "SESSION_END",
                self.generation_stats[-1]['avg_fitness'] if self.generation_stats else 0,
                0,  # eval time
                0, 0, 0, 0, 0,  # metrics placeholders
                f"Session ended at generation {self.population.generation}. "
                f"Total tokens: {self.total_prompt_tokens + self.total_completion_tokens:,}, "
                f"Total cost: ${self.total_cost:.4f}",
                self.population.diversity_score,
                len(self.population.ideas),
                self.adaptive_rates['mutation'],
                self.adaptive_rates['crossover'],
                self.total_prompt_tokens,
                self.total_completion_tokens,
                self.total_prompt_tokens + self.total_completion_tokens,
                self.total_cost
            ])
    
    def _estimate_completion_time(self, total_ideas: int) -> str:
        """Estimate completion time based on average evaluation time."""
        if not self.avg_eval_time:
            return "Calculating..."
        
        total_seconds = total_ideas * self.avg_eval_time
        
        if total_seconds < 60:
            return f"{total_seconds:.1f} seconds"
        elif total_seconds < 3600:
            minutes = total_seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = total_seconds / 3600
            return f"{hours:.1f} hours"
    
    def _detect_cost_model(self) -> str:
        """Detect and validate the cost model based on evaluator model name."""
        if not hasattr(self.evaluator, 'model_name'):
            print("\nWarning: Could not detect model name, using default cost model (lm-studio)")
            return DEFAULT_COST_MODEL
            
        model_name = self.evaluator.model_name.lower()
        
        # Map model names to cost models
        model_mapping = {
            'gpt-4': ['gpt-4', 'gpt-4-turbo', 'gpt-4-32k', 'gpt-4-0314'],
            'gpt-3.5-turbo': ['gpt-3.5', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k']
        }
        
        # First try exact matches
        if model_name in COST_PER_1K_TOKENS:
            print(f"\nDetected model: {model_name}")
            return model_name
            
        # Then try pattern matching
        for cost_model, patterns in model_mapping.items():
            if any(pattern in model_name for pattern in patterns):
                if cost_model in COST_PER_1K_TOKENS:
                    print(f"\nDetected model: {model_name}")
                    print(f"Using cost rates for: {cost_model}")
                    return cost_model
                    
        print(f"\nWarning: Unknown model '{model_name}', using default cost model (lm-studio)")
        return DEFAULT_COST_MODEL

    def _reset_generation_tokens(self) -> None:
        """Reset generation token counters."""
        self.generation_tokens = {'prompt': 0, 'completion': 0}

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for given token counts."""
        if prompt_tokens < 0 or completion_tokens < 0:
            print(f"\nWarning: Invalid token counts detected: prompt={prompt_tokens}, completion={completion_tokens}")
            return 0.0
            
        cost_rates = COST_PER_1K_TOKENS.get(self.cost_model, {'prompt': 0.0, 'completion': 0.0})
        return (prompt_tokens * cost_rates['prompt'] + 
                completion_tokens * cost_rates['completion']) / 1000

    def _update_token_counts(self, token_info: Optional[Dict]) -> Tuple[int, int, float]:
        """Update token counts and calculate cost.
        
        Args:
            token_info: Dictionary containing token usage information
            
        Returns:
            Tuple of (prompt_tokens, completion_tokens, cost)
        """
        if not token_info:
            print("\nWarning: No token information provided")
            return 0, 0, 0.0
            
        try:
            # Extract and validate token counts
            prompt_tokens = max(0, int(token_info.get('prompt_tokens', 0)))
            completion_tokens = max(0, int(token_info.get('completion_tokens', 0)))
            total_tokens = token_info.get('total_tokens')
            
            # Validate total if provided
            if total_tokens is not None:
                total_tokens = int(total_tokens)
                calculated_total = prompt_tokens + completion_tokens
                if total_tokens != calculated_total:
                    print(f"\nWarning: Token count mismatch - "
                          f"Reported total: {total_tokens}, "
                          f"Calculated total: {calculated_total} "
                          f"(prompt: {prompt_tokens}, completion: {completion_tokens})")
            
            # Update model if provided
            if 'model' in token_info:
                model_name = str(token_info['model']).lower()
                for known_model in COST_PER_1K_TOKENS.keys():
                    if known_model in model_name:
                        self.cost_model = known_model
                        break
            
            # Update counters
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.generation_tokens['prompt'] += prompt_tokens
            self.generation_tokens['completion'] += completion_tokens
            
            # Calculate cost
            cost = self._calculate_cost(prompt_tokens, completion_tokens)
            self.total_cost += cost
            
            return prompt_tokens, completion_tokens, cost
            
        except (TypeError, ValueError) as e:
            print(f"\nWarning: Invalid token count format - {str(e)}")
            print("Expected format: {'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int}")
            return 0, 0, 0.0

    def _format_cost(self, cost: float) -> str:
        """Format cost for display, using K suffix for large numbers."""
        return (f"${cost/1000:.1f}K" if cost > 1000 else f"${cost:.2f}")

    def evaluate_population(self) -> None:
        """Evaluate all ideas in the population that haven't been evaluated yet."""
        print(f"\n{'='*80}")
        print(f"Generation {self.population.generation} - Evaluating Population")
        print(f"{'='*80}")
        
        unevaluated = [idea for idea in self.population.ideas if idea.evaluation is None]
        total = len(unevaluated)
        
        if total == 0:
            return
            
        est_time = self._estimate_completion_time(total)
        print(f"\nEstimated completion time: {est_time}")
        
        eval_times = []
        high_fitness_ideas = []
        
        try:
            with tqdm(total=total, desc="Evaluating ideas", unit="idea") as pbar:
                for idea in unevaluated:
                    if not self._running:
                        remaining = total - pbar.n
                        print(f"\nStopping evaluation ({remaining} ideas remaining)")
                        break
                    
                    pbar.set_description(f"Evaluating: {str(idea)[:40]}...")
                    
                    try:
                        # Handle both 2-value and 3-value return formats
                        eval_result = self.evaluator.evaluate(idea)
                        
                        # Default token info based on model type
                        if len(eval_result) == 2:
                            evaluation, eval_time = eval_result
                            # For local models, use zero tokens
                            if self.cost_model == DEFAULT_COST_MODEL:
                                token_info = {
                                    'prompt_tokens': 0,
                                    'completion_tokens': 0,
                                    'total_tokens': 0,
                                    'model': 'lm-studio'
                                }
                            # For paid models, use conservative estimates
                            else:
                                token_info = {
                                    'prompt_tokens': 500,  # Conservative estimate
                                    'completion_tokens': 100,  # Conservative estimate
                                    'total_tokens': 600,
                                    'model': self.cost_model
                                }
                                print(f"\nNote: Using estimated token counts for {self.cost_model}")
                        else:
                            evaluation, eval_time, token_info = eval_result
                            
                        eval_times.append(eval_time)
                        
                        # Update token counts and get cost
                        prompt_tokens, completion_tokens, cost = self._update_token_counts(token_info)
                        
                        idea.evaluation = evaluation
                        idea.calculate_fitness()
                        self._log_evaluation(idea, eval_time, token_info)
                        
                        # Update progress bar postfix with current metrics
                        self.avg_eval_time = sum(eval_times) / len(eval_times)
                        remaining = total - pbar.n
                        new_est = self._estimate_completion_time(remaining)
                        
                        # Calculate and format costs for display
                        gen_cost = self._calculate_cost(
                            self.generation_tokens['prompt'],
                            self.generation_tokens['completion']
                        )
                        
                        pbar.set_postfix({
                            'Score': f"{idea.fitness_score:.0f}",
                            'Avg Time': f"{self.avg_eval_time:.1f}s",
                            'Gen Cost': self._format_cost(gen_cost),
                            'Total': self._format_cost(self.total_cost),
                            'Remaining': new_est
                        })
                        
                        if idea.fitness_score >= 400:
                            high_fitness_ideas.append(idea)
                            
                    except Exception as e:
                        print(f"\nError evaluating idea: {str(idea)}")
                        print(f"Error details: {str(e)}")
                        # Set empty evaluation to avoid re-processing
                        idea.evaluation = {}
                        idea.calculate_fitness()  # Will set to 0
                    
                    pbar.update(1)
                    
            # Print generation token usage
            if sum(self.generation_tokens.values()) > 0:
                print(f"\nGeneration {self.population.generation} Token Usage:")
                print(f"- Prompt Tokens: {self.generation_tokens['prompt']:,}")
                print(f"- Completion Tokens: {self.generation_tokens['completion']:,}")
                print(f"- Total Tokens: {sum(self.generation_tokens.values()):,}")
                print(f"- Generation Cost: {self._format_cost(gen_cost)}")
                print(f"- Total Cost: {self._format_cost(self.total_cost)}")
                
        except KeyboardInterrupt:
            if not self._shutting_down:
                self._signal_handler(None, None)
        
        if high_fitness_ideas:
            print("\nHigh Fitness Ideas from this Generation:")
            for idea in high_fitness_ideas:
                print(f"- {idea} (Score: {idea.fitness_score:.0f})")
    
    def _adjust_rates(self) -> None:
        """Adjust genetic algorithm rates based on population diversity and performance."""
        diversity = self.population.calculate_diversity()
        
        # Get performance trend if available
        improving = False
        if len(self.generation_stats) >= 2:
            last_avg = self.generation_stats[-1].get('avg_fitness', 0)
            prev_avg = self.generation_stats[-2].get('avg_fitness', 0)
            improving = last_avg > prev_avg
        
        # Adjust rates based on diversity and performance
        if diversity < 0.3:  # Low diversity
            if improving:
                # Small adjustments if we're improving
                self.adaptive_rates['mutation'] = min(
                    self.adaptive_rates['max_mutation'],
                    self.adaptive_rates['mutation'] * 1.1
                )
                self.adaptive_rates['random'] = min(0.4, self.adaptive_rates['random'] * 1.1)
            else:
                # Larger adjustments if we're stagnating
                self.adaptive_rates['mutation'] = min(
                    self.adaptive_rates['max_mutation'],
                    self.adaptive_rates['mutation'] * 1.2
                )
                self.adaptive_rates['random'] = min(0.4, self.adaptive_rates['random'] * 1.2)
            
            self.adaptive_rates['crossover'] = max(
                self.adaptive_rates['min_crossover'],
                self.adaptive_rates['crossover'] * 0.9
            )
        
        elif diversity > 0.7:  # High diversity
            if improving:
                # Favor crossover if we're improving
                self.adaptive_rates['mutation'] = max(
                    self.adaptive_rates['min_mutation'],
                    self.adaptive_rates['mutation'] * 0.9
                )
                self.adaptive_rates['crossover'] = min(
                    self.adaptive_rates['max_mutation'],
                    self.adaptive_rates['crossover'] * 1.1
                )
            else:
                # Balance rates if we're not improving
                self.adaptive_rates['mutation'] = self.adaptive_rates['mutation']  # Keep current
                self.adaptive_rates['crossover'] = min(
                    self.adaptive_rates['max_mutation'],
                    self.adaptive_rates['crossover'] * 1.05
                )
            
            self.adaptive_rates['random'] = max(0.1, self.adaptive_rates['random'] * 0.9)
    
    def _update_generation_stats(self) -> None:
        """Update statistics for the current generation."""
        if not self.population.ideas:
            return
            
        fitness_scores = [idea.fitness_score for idea in self.population.ideas]
        gen_cost = self._calculate_cost(
            self.generation_tokens['prompt'],
            self.generation_tokens['completion']
        )
        
        stats = {
            'generation': self.population.generation,
            'max_fitness': max(fitness_scores),
            'min_fitness': min(fitness_scores),
            'avg_fitness': sum(fitness_scores) / len(fitness_scores),
            'diversity': self.population.diversity_score,
            'population_size': len(self.population.ideas),
            'mutation_rate': self.adaptive_rates['mutation'],
            'crossover_rate': self.adaptive_rates['crossover'],
            'random_rate': self.adaptive_rates['random'],
            'prompt_tokens': self.generation_tokens['prompt'],
            'completion_tokens': self.generation_tokens['completion'],
            'total_tokens': sum(self.generation_tokens.values()),
            'generation_cost': gen_cost
        }
        self.generation_stats.append(stats)
        
        # Reset generation token counters
        self._reset_generation_tokens()
        
        # Log generation stats
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().isoformat(),
                stats['generation'],
                'GENERATION_STATS',
                stats['avg_fitness'],
                0,  # eval time placeholder
                stats['max_fitness'],  # viability column for max fitness
                stats['avg_fitness'],  # value_potential column for avg fitness
                0, 0, 0,  # simplicity, novelty, scalability placeholders
                f"Max: {stats['max_fitness']}, Avg: {stats['avg_fitness']}, Diversity: {stats['diversity']:.2%}",
                stats['diversity'],
                stats['population_size'],
                stats['mutation_rate'],
                stats['crossover_rate'],
                stats['prompt_tokens'],
                stats['completion_tokens'],
                stats['total_tokens'],
                stats['generation_cost']
            ])

    def _create_next_generation(self) -> None:
        """Create the next generation of ideas through evolution."""
        # Sort current population by fitness
        self.population.sort_by_fitness()
        
        # Calculate current diversity
        diversity = self.population.calculate_diversity()
        
        # Adjust rates based on diversity
        if diversity < self.convergence_params['min_diversity_threshold']:
            # Increase mutation and random rates to promote diversity
            self.adaptive_rates['mutation'] = min(
                self.adaptive_rates['mutation'] * 1.2,
                self.adaptive_rates['max_mutation']
            )
            self.adaptive_rates['random'] = min(
                self.adaptive_rates['random'] * 1.2,
                0.4  # Max random rate
            )
            self.adaptive_rates['crossover'] = max(
                self.adaptive_rates['crossover'] * 0.9,
                self.adaptive_rates['min_crossover']
            )
        else:
            # Restore default rates
            self.adaptive_rates['mutation'] = MUTATION_RATE
            self.adaptive_rates['crossover'] = 0.7
            self.adaptive_rates['random'] = 0.3
        
        # Keep top performers (25%)
        next_generation = self.population.ideas[:self.population.quartile_size]
        
        # Fill remaining slots
        remaining_slots = self.population_size - len(next_generation)
        
        # Calculate number of ideas to generate for each method
        crossover_count = int(remaining_slots * self.adaptive_rates['crossover'])
        mutation_count = int(remaining_slots * self.adaptive_rates['mutation'])
        random_count = remaining_slots - (crossover_count + mutation_count)
        
        # Add crossover offspring
        for _ in range(crossover_count):
            parent1, parent2 = self.population.get_diverse_parents()
            child = Idea.crossover(parent1, parent2)
            next_generation.append(child)
        
        # Add mutations
        for _ in range(mutation_count):
            parent = random.choice(self.population.top_quartile)
            mutated = parent.mutate()
            next_generation.append(mutated)
        
        # Add random ideas
        for _ in range(random_count):
            next_generation.append(Idea.generate_random())
        
        # Update population
        self.population.ideas = next_generation
        self.population.generation += 1
        
        # Ensure minimum population size
        self.population.ensure_minimum_population()

    def _display_progress(self) -> None:
        """Display progress of the current generation."""
        if not self.generation_stats:
            return
            
        stats = self.generation_stats[-1]
        print(f"\n{'='*80}")
        print(f"Generation {stats['generation']} - Progress")
        print(f"{'='*80}")
        print(f"\nPopulation Size: {stats['population_size']}")
        print(f"Diversity: {stats['diversity']:.2%}")
        print(f"\nFitness Scores:")
        print(f"- Maximum: {stats['max_fitness']:.2f}")
        print(f"- Average: {stats['avg_fitness']:.2f}")
        print(f"- Minimum: {stats['min_fitness']:.2f}")
        
        if len(self.generation_stats) >= 2:
            prev_stats = self.generation_stats[-2]
            avg_change = stats['avg_fitness'] - prev_stats['avg_fitness']
            print(f"\nProgress:")
            print(f"- Average Fitness Change: {avg_change:+.2f}")
            print(f"- Generations without improvement: {self._generations_without_improvement()}")
        
        print("\nCurrent Rates:")
        print(f"- Mutation: {stats['mutation_rate']:.2%}")
        print(f"- Crossover: {stats['crossover_rate']:.2%}")
        print(f"- Random: {stats['random_rate']:.2%}")
    
    def _generations_without_improvement(self) -> int:
        """Calculate number of generations without improvement in max fitness."""
        if len(self.generation_stats) < 2:
            return 0
            
        current_max = self.generation_stats[-1]['max_fitness']
        count = 0
        
        for stats in reversed(self.generation_stats[:-1]):
            if stats['max_fitness'] >= current_max:
                count += 1
            else:
                break
                
        return count
    
    def show_final_results(self) -> None:
        """Display final results of the evolution process."""
        print("\n\n" + "="*80)
        print("Evolution Process Stopped")
        print("="*80)
        print(f"\nFinal Results from Generation {self.population.generation}:")
        print("\nTop 10 Ideas:")
        self.population.sort_by_fitness()
        for i, idea in enumerate(self.population.ideas[:10], 1):
            print(f"\n{i}. {idea}")
            print(f"   Score: {idea.fitness_score}")
            print(f"   Reasoning: {idea.evaluation.get('reasoning', 'No reasoning provided') if idea.evaluation else 'No evaluation'}")
        
        print("\nToken Usage Statistics:")
        print(f"- Total Prompt Tokens: {self.total_prompt_tokens:,}")
        print(f"- Total Completion Tokens: {self.total_completion_tokens:,}")
        print(f"- Total Tokens: {self.total_prompt_tokens + self.total_completion_tokens:,}")
        print(f"- Total Cost: ${self.total_cost:.4f}")
        
        print(f"\nAll results have been logged to: {self.csv_filename}")
        print("="*80)
    
    def _check_convergence(self) -> Tuple[bool, str]:
        """Check if the algorithm has converged."""
        if len(self.generation_stats) < 2:
            return False, ""
            
        current_stats = self.generation_stats[-1]
        
        # Check if we've reached maximum generations
        if current_stats['generation'] >= self.convergence_params['max_generations']:
            return True, f"Maximum generations ({self.convergence_params['max_generations']}) reached"
            
        # Check if we've reached target fitness
        if current_stats['max_fitness'] >= self.convergence_params['target_fitness']:
            return True, "Target fitness score reached"
        
        # Check for stagnation
        generations_without_improvement = self._generations_without_improvement()
        if generations_without_improvement >= self.convergence_params['max_generations_without_improvement']:
            return True, f"No improvement for {generations_without_improvement} generations"
        
        # Check for low diversity
        if (current_stats['diversity'] < self.convergence_params['min_diversity_threshold'] and 
            generations_without_improvement >= 2):
            return True, "Population diversity too low with no recent improvements"
        
        # Check for minimal fitness changes
        if len(self.generation_stats) >= 3:
            recent_improvements = [
                abs(self.generation_stats[i]['max_fitness'] - self.generation_stats[i-1]['max_fitness'])
                for i in range(len(self.generation_stats)-1, len(self.generation_stats)-3, -1)
            ]
            if all(imp < self.convergence_params['fitness_improvement_threshold'] for imp in recent_improvements):
                return True, "Fitness improvements below threshold"
        
        return False, ""
    
    def run(self) -> None:
        """Run the genetic algorithm continuously until interrupted or converged."""
        print("\n" + "="*80)
        print("Welcome to the Idea Generator Genetic Algorithm!")
        print("="*80)
        
        if not self.evaluator.test_connectivity():
            print("\nAPI connectivity test failed. Check if server is running.")
            return
            
        print("\nAPI connectivity test successful.")
        print("\nStarting evolution process...")
        print("Press Ctrl+C to stop and view final results.")
        
        # Set up signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        self._running = True
        
        try:
            # Initial evaluation of population
            self.evaluate_population()
            
            while self._running:
                # Create next generation
                self._create_next_generation()
                
                # Evaluate new population
                self.evaluate_population()
                
                # Update statistics and display progress
                self._update_generation_stats()
                self._display_progress()
                
                # Adjust rates for next generation
                self._adjust_rates()
                
                # Check for convergence
                converged, reason = self._check_convergence()
                if converged:
                    print(f"\nAlgorithm converged: {reason}")
                    self._running = False
                    
        except KeyboardInterrupt:
            if not self._shutting_down:
                self._signal_handler(None, None)
        finally:
            if self.population.generation > 0:
                print("\nStarting cleanup process...")
                self._cleanup()
                self.show_final_results()
                
                # Show convergence statistics
                if len(self.generation_stats) >= 2:
                    print("\nConvergence Statistics:")
                    print(f"- Total Generations: {len(self.generation_stats)}")
                    print(f"- Final Diversity: {self.generation_stats[-1]['diversity']:.2%}")
                    print(f"- Generations without improvement: {self._generations_without_improvement()}")
                    print(f"- Best Fitness Score: {self.generation_stats[-1]['max_fitness']:.2f}")
                    print(f"- Average Fitness (last gen): {self.generation_stats[-1]['avg_fitness']:.2f}") 
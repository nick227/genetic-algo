"""Main entry point for the idea generator."""

import argparse
import sys
from typing import List, Dict, Optional
from .__version__ import __version__

from .core.genetic_algorithm import GeneticAlgorithm
from .config.settings import (
    POPULATION_SIZE,
    DEFAULT_MIN_POPULATION,
    DEFAULT_TARGET_FITNESS,
    DEFAULT_MAX_GENERATIONS_WITHOUT_IMPROVEMENT,
    DEFAULT_MIN_DIVERSITY_THRESHOLD,
    DEFAULT_FITNESS_IMPROVEMENT_THRESHOLD,
    DEFAULT_MAX_GENERATIONS
)

def parse_seed_idea(idea_str: str) -> Optional[Dict[str, str]]:
    """Parse a seed idea string into components.
    
    Formats accepted:
    1. Full idea: "A machine learning healthcare system to analyze patient data"
    2. Components: "technical=machine learning,industry=healthcare,action=analyze,object=patient data"
    3. Partial components: "technical=AI,action=analyze" (other components will be random)
    """
    if '=' in idea_str:
        # Parse component format
        components = {}
        for pair in idea_str.split(','):
            if '=' in pair:
                key, value = pair.split('=', 1)
                if key.strip() in ['technical', 'industry', 'action', 'object']:
                    components[key.strip()] = value.strip()
        return components if len(components) > 0 else None
    else:
        # Parse full idea format
        from .models.idea import Idea
        idea = Idea.from_string(idea_str)
        if idea:
            return {
                'technical': idea.technical,
                'industry': idea.industry,
                'action': idea.action,
                'object': idea.object
            }
    return None

def main():
    """Main entry point for the idea generator."""
    parser = argparse.ArgumentParser(
        description='Generate and evolve programming project ideas using genetic algorithms.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings:
  python -m src.idea_generator
  
  # Seed with a complete idea:
  python -m src.idea_generator --seed-idea "A machine learning healthcare system to analyze patient data"
  
  # Seed with specific components (some random):
  python -m src.idea_generator --seed-idea "technical=AI,industry=healthcare"
  
  # Multiple seed ideas:
  python -m src.idea_generator --seed-idea "A blockchain finance system to process transactions" --seed-idea "technical=AI,industry=education"
  
  # Load seeds from file:
  python -m src.idea_generator --seed-file my_ideas.txt
  
  # Customize algorithm parameters:
  python -m src.idea_generator --min-population 30 --target-fitness 500 --max-stagnant-generations 8
"""
    )
    
    # Version
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    # Seed ideas
    parser.add_argument(
        '--seed-idea',
        action='append',
        help='Seed idea to include in initial population. Can be specified multiple times. '
             'Format 1: "A [technical] [industry] system to [action] [object]" '
             'Format 2: "technical=X,industry=Y,action=Z,object=W" '
             'Format 3: "technical=AI,action=analyze" (other components random)'
    )
    
    parser.add_argument(
        '--seed-file',
        type=str,
        help='File containing seed ideas, one per line in either format'
    )
    
    # Algorithm parameters
    parser.add_argument(
        '--population-size',
        type=int,
        default=POPULATION_SIZE,
        help=f'Size of the population (default: {POPULATION_SIZE})'
    )
    
    parser.add_argument(
        '--min-population',
        type=int,
        default=DEFAULT_MIN_POPULATION,
        help=f'Minimum population size (default: {DEFAULT_MIN_POPULATION})'
    )
    
    parser.add_argument(
        '--target-fitness',
        type=float,
        default=DEFAULT_TARGET_FITNESS,
        help=f'Stop when an idea reaches this fitness score (default: {DEFAULT_TARGET_FITNESS})'
    )
    
    parser.add_argument(
        '--max-generations',
        type=int,
        default=DEFAULT_MAX_GENERATIONS,
        help=f'Maximum number of generations to run (default: {DEFAULT_MAX_GENERATIONS})'
    )
    
    parser.add_argument(
        '--max-stagnant-generations',
        type=int,
        default=DEFAULT_MAX_GENERATIONS_WITHOUT_IMPROVEMENT,
        help=f'Maximum generations without improvement before stopping (default: {DEFAULT_MAX_GENERATIONS_WITHOUT_IMPROVEMENT})'
    )
    
    parser.add_argument(
        '--min-diversity',
        type=float,
        default=DEFAULT_MIN_DIVERSITY_THRESHOLD,
        help=f'Minimum diversity threshold (default: {DEFAULT_MIN_DIVERSITY_THRESHOLD})'
    )
    
    parser.add_argument(
        '--improvement-threshold',
        type=float,
        default=DEFAULT_FITNESS_IMPROVEMENT_THRESHOLD,
        help=f'Minimum fitness improvement to consider progress (default: {DEFAULT_FITNESS_IMPROVEMENT_THRESHOLD})'
    )
    
    args = parser.parse_args()
    
    # Collect seed ideas
    seed_ideas = []
    
    # Process direct seed ideas
    if args.seed_idea:
        print("\nProcessing seed ideas:")
        for idea_str in args.seed_idea:
            if components := parse_seed_idea(idea_str):
                print(f"✓ Successfully parsed: {idea_str}")
                seed_ideas.append(components)
            else:
                print(f"✗ Could not parse seed idea: {idea_str}")
    
    # Process seed file if provided
    if args.seed_file:
        try:
            print(f"\nReading seed ideas from {args.seed_file}:")
            with open(args.seed_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if components := parse_seed_idea(line):
                            print(f"✓ Line {line_num}: Successfully parsed")
                            seed_ideas.append(components)
                        else:
                            print(f"✗ Line {line_num}: Could not parse: {line}")
        except Exception as e:
            print(f"✗ Error reading seed file: {e}")
    
    try:
        # Configure algorithm parameters
        config = {
            'population_size': args.population_size,
            'min_population': args.min_population,
            'target_fitness': args.target_fitness,
            'max_generations': args.max_generations,
            'max_generations_without_improvement': args.max_stagnant_generations,
            'min_diversity_threshold': args.min_diversity,
            'fitness_improvement_threshold': args.improvement_threshold
        }
        
        # Run the genetic algorithm
        algorithm = GeneticAlgorithm(
            seed_ideas=seed_ideas if seed_ideas else None,
            **config
        )
        algorithm.run()
        
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 
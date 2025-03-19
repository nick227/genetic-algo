"""Command-line interface for the idea generator."""

import click
import json
from pathlib import Path
from typing import List, Dict, Optional
from .models.idea import Idea

def load_category_file(path: str) -> Dict[str, Dict[str, List[str]]]:
    """Load category definitions from a JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise click.BadParameter(f"Could not load category file: {e}")

def parse_word_list(ctx, param, value: Optional[str]) -> List[str]:
    """Parse comma-separated word list."""
    if not value:
        return []
    return [word.strip() for word in value.split(',') if word.strip()]

@click.command()
@click.version_option()
# Existing options
@click.option('--seed-idea', multiple=True, help='Add a seed idea (can be used multiple times)')
@click.option('--seed-file', type=click.Path(exists=True), help='Load seed ideas from a file')
@click.option('--population-size', type=int, default=30, help='Set population size')
@click.option('--min-population', type=int, default=20, help='Set minimum population size')
@click.option('--target-fitness', type=float, default=450, help='Stop when an idea reaches this fitness')
@click.option('--max-stagnant-generations', type=int, default=5, help='Max generations without improvement')
@click.option('--min-diversity', type=float, default=0.2, help='Minimum diversity threshold')
@click.option('--improvement-threshold', type=float, default=1.0, help='Minimum fitness improvement')

# New word pool options
@click.option('--category', help='Use a specific category for word pools (e.g., wordpress, ai)')
@click.option('--category-file', type=click.Path(exists=True), help='Load category definitions from JSON file')
@click.option('--add-technical', callback=parse_word_list, help='Add technical words (comma-separated)')
@click.option('--add-industry', callback=parse_word_list, help='Add industry words (comma-separated)')
@click.option('--add-action', callback=parse_word_list, help='Add action words (comma-separated)')
@click.option('--add-object', callback=parse_word_list, help='Add object words (comma-separated)')
@click.option('--show-pools', is_flag=True, help='Show available word pools and categories')
@click.option('--show-stats', is_flag=True, help='Show word usage statistics')

def main(seed_idea: List[str], seed_file: Optional[str], 
         population_size: int, min_population: int,
         target_fitness: float, max_stagnant_generations: int,
         min_diversity: float, improvement_threshold: float,
         category: Optional[str], category_file: Optional[str],
         add_technical: List[str], add_industry: List[str],
         add_action: List[str], add_object: List[str],
         show_pools: bool, show_stats: bool) -> None:
    """Generate and evolve programming project ideas using genetic algorithms."""
    
    # Handle word pool management first
    if category_file:
        categories = load_category_file(category_file)
        for cat_name, words in categories.items():
            Idea.add_category(cat_name, words)
            click.echo(f"Added category '{cat_name}' with {sum(len(w) for w in words.values())} words")
    
    # Add user-defined words
    if any([add_technical, add_industry, add_action, add_object]):
        words = {
            'technical': add_technical,
            'industry': add_industry,
            'action': add_action,
            'object': add_object
        }
        Idea.add_category('user_defined', words)
        for component, word_list in words.items():
            if word_list:
                click.echo(f"Added {len(word_list)} {component} words: {', '.join(word_list)}")
    
    # Show pool information if requested
    if show_pools:
        stats = Idea.get_statistics()
        click.echo("\nWord Pool Statistics:")
        click.echo("-" * 40)
        for component, count in stats['total_words'].items():
            click.echo(f"{component.title()}: {count} words total")
            if stats['user_defined_words'][component]:
                click.echo(f"  ({stats['user_defined_words'][component]} user-defined)")
        
        if stats['categories']:
            click.echo("\nCategories:")
            for cat, comps in stats['categories'].items():
                click.echo(f"\n{cat.title()}:")
                for comp, count in comps.items():
                    click.echo(f"  {comp.title()}: {count} words")
    
    if show_stats:
        stats = Idea.get_statistics()
        click.echo("\nMost Used Words:")
        click.echo("-" * 40)
        for component, words in stats['most_used'].items():
            if words:
                click.echo(f"\n{component.title()}:")
                for word, count in words:
                    click.echo(f"  {word}: used {count} times")
    
    # If only showing statistics, exit
    if show_pools or show_stats:
        return
    
    # Process seeds
    seeds = list(seed_idea)
    if seed_file:
        with open(seed_file) as f:
            seeds.extend(line.strip() for line in f if line.strip() and not line.startswith('#'))
    
    # Initialize genetic algorithm with category context
    from .core.genetic_algorithm import GeneticAlgorithm
    from .core.evaluator import create_evaluator
    
    evaluator = create_evaluator()
    algorithm = GeneticAlgorithm(
        evaluator=evaluator,
        population_size=population_size,
        min_population=min_population,
        target_fitness=target_fitness,
        max_stagnant_generations=max_stagnant_generations,
        min_diversity=min_diversity,
        improvement_threshold=improvement_threshold,
        category=category
    )
    
    # Process seeds with category context
    if seeds:
        click.echo("\nProcessing seed ideas:")
        for seed in seeds:
            if '=' in seed:
                # Component format
                components = dict(item.split('=') for item in seed.split(','))
                idea = Idea.from_components(**components, category=category)
            else:
                # Full idea format
                idea = Idea.from_string(seed)
                if idea and category:
                    idea.category = category
            
            if idea:
                click.echo(f"✓ Successfully parsed: {idea}")
                algorithm.add_seed(idea)
            else:
                click.echo(f"✗ Failed to parse: {seed}")
    
    # Run the algorithm
    click.echo("\nInitializing Idea Generator...")
    algorithm.run()

if __name__ == '__main__':
    main() 
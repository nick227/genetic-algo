# Idea Generator

A genetic algorithm-based tool that generates and evolves programming project ideas using LM Studio's API or OpenAI's API.

## Features

- Generates programming project ideas by combining technical, industry, action, and object components
- Supports seeding the initial population with your own ideas:
  - Full idea format: "A [technical] [industry] system to [action] [object]"
  - Component format: "technical=X,industry=Y,action=Z,object=W"
  - Partial components: Specify some components, others randomly generated
  - Multiple seed ideas via command line or file input
- Uses genetic algorithm principles to evolve and improve ideas over generations:
  - Adaptive mutation and crossover rates based on population diversity
  - Intelligent parent selection weighted by fitness scores
  - Semantic validation to prevent nonsensical combinations
  - Early stopping when convergence is detected
- Evaluates ideas using either LM Studio or OpenAI's API for:
  - Viability (25%)
  - Value potential (25%)
  - Simplicity (20%)
  - Novelty (15%)
  - Scalability (15%)
- Advanced population management:
  - Maintains minimum viable population size
  - Tracks and optimizes population diversity
  - Prevents stagnation through adaptive rates
- Comprehensive progress monitoring:
  - Real-time evaluation progress with time estimates
  - Generation statistics and improvement tracking
  - Convergence detection and reporting
- Flexible model selection from either LM Studio or OpenAI
- Detailed logging with timestamped files

## Requirements

- Python 3.8+
- Either:
  - LM Studio running locally (recommended for free usage)
  - OpenAI API key (for using OpenAI's models)
- NLTK data (downloaded automatically on first run)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd idea-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Have either:
   - LM Studio running locally on port 1234
   - Or an OpenAI API key ready

## Usage

Run the idea generator with various options:

```bash
# Basic usage with default settings
python -m src.idea_generator

# Seed with a complete idea
python -m src.idea_generator --seed-idea "A machine learning healthcare system to analyze patient data"

# Seed with specific components (some random)
python -m src.idea_generator --seed-idea "technical=AI,industry=healthcare"

# Multiple seed ideas
python -m src.idea_generator \
  --seed-idea "A blockchain finance system to process transactions" \
  --seed-idea "technical=AI,industry=education"

# Load seeds from file
python -m src.idea_generator --seed-file wordpress_ai_seeds.txt

# Customize algorithm parameters
python -m src.idea_generator \
  --seed-idea "technical=wordpress,industry=plugin,action=specialize,object=AI tools" \
  --population-size 1 \
  --min-population 1 \
  --target-fitness 450 \
  --max-stagnant-generations 1

# Focused seed file
python -m src.idea_generator \
  --seed-file wordpress_ai_focus.txt \
  --population-size 3 \
  --min-population 3 \
  --min-diversity 0.1 \
  --max-stagnant-generations 2
```

### Command Line Options

- Seeding Options:
  - `--seed-idea TEXT`: Add a seed idea (can be used multiple times)
  - `--seed-file PATH`: Load seed ideas from a file

- Algorithm Parameters:
  - `--population-size INT`: Set population size (default: 30)
  - `--min-population INT`: Set minimum population size (default: 20)
  - `--target-fitness FLOAT`: Stop when an idea reaches this fitness (default: 450)
  - `--max-stagnant-generations INT`: Max generations without improvement (default: 5)
  - `--min-diversity FLOAT`: Minimum diversity threshold (default: 0.2)
  - `--improvement-threshold FLOAT`: Minimum fitness improvement (default: 1.0)

- Other Options:
  - `--version`: Show program version
  - `--help`: Show help message and examples

### Seed File Format

Create a text file with one idea per line using either format:

```text
# Full idea format:
A machine learning healthcare system to analyze patient data
A blockchain finance system to process transactions

# Component format:
technical=AI,industry=education,action=generate,object=study materials
technical=blockchain,industry=finance

# Comments start with #
# Empty lines are ignored
# Mix and match formats as needed
```

The program will:
1. Check for LM Studio availability and prompt for model selection
2. If LM Studio is not available, ask for OpenAI API key and model selection
3. Process any provided seed ideas:
   - Parse and validate each idea
   - Create variations of seed ideas through mutation
4. Generate additional random ideas to complete the population
5. Evaluate each idea using the selected API
6. Evolve the population through:
   - Selection (keeping top performers)
   - Weighted crossover (combining successful ideas)
   - Adaptive mutation (introducing variations)
   - Random addition (maintaining diversity)
7. Monitor convergence and population health:
   - Track diversity and fitness improvements
   - Adjust genetic algorithm rates dynamically
   - Stop when convergence criteria are met
8. Log results to timestamped files (e.g., `idea_log_20240320_143022.csv`)

Press Ctrl+C to stop the program and view the final top ideas. A second Ctrl+C will force quit (with warning).

## Configuration

Edit `src/idea_generator/config/settings.py` to modify:
- API settings (base URL, default model)
- Algorithm parameters:
  - Population size and minimum thresholds
  - Initial mutation and crossover rates
  - Convergence parameters
  - Fitness weights
- Word pools and seed words
- Evaluation criteria

## Output

The program generates timestamped CSV files containing:
- Timestamp
- Generation number
- Idea description
- Fitness score
- Evaluation time
- AI reasoning
- Generation statistics:
  - Population diversity
  - Average/Max/Min fitness
  - Current genetic algorithm rates

Each run creates a new file with format: `idea_log_YYYYMMDD_HHMMSS.csv`

## Progress Monitoring

The program provides real-time feedback on:
- Evaluation progress with time estimates
- Population diversity and fitness trends
- Convergence status and stopping conditions
- Generation statistics and improvements
- Adaptive rate adjustments

## Architecture

The codebase is organized into modules:
- `config/`: Configuration settings
- `core/`: Core algorithm implementation
  - `genetic_algorithm.py`: Main evolution logic
  - `evaluator.py`: Idea evaluation interface
- `models/`: Data models
  - `idea.py`: Idea representation and operations
  - `evaluator/`: Model-specific evaluators
- `utils/`: Utility functions

## License

MIT License

# wordpress_ai_seeds.txt
A wordpress plugin system to develop AI tools
A wordpress extension system to integrate AI features
technical=wordpress,industry=plugin,action=automate,object=AI workflows
technical=wordpress,industry=saas,action=manage,object=AI services
technical=wordpress,industry=plugin,action=specialize,object=AI tools
technical=wordpress,industry=plugin,action=integrate,object=AI features
technical=wordpress,industry=plugin,action=develop,object=AI tools

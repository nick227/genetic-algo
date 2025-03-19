"""Configuration settings for the idea generator."""

# API Settings
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Aliases for backward compatibility
API_BASE_URL = LM_STUDIO_BASE_URL  # Default to LM Studio
API_KEY = None  # Not needed for LM Studio, required for OpenAI

# Model Parameters
MAX_RETRIES = 3
MAX_TOKENS = 500
TEMPERATURE = 0.7
DEFAULT_MODEL = "gpt-3.5-turbo"  # Default model for OpenAI

# Word Pools for Idea Generation
TECHNICAL_WORDS = [
    "blockchain", "AI", "machine learning", "cloud", "IoT", "quantum", 
    "neural network", "distributed", "serverless", "microservice",
    "edge computing", "augmented reality", "virtual reality", "robotics",
    "biometric", "cryptographic", "autonomous", "predictive", "wordpress", "php", "cms", "plugin"
]

INDUSTRY_WORDS = [
    "healthcare", "finance", "education", "retail", "manufacturing",
    "transportation", "agriculture", "energy", "entertainment",
    "security", "logistics", "real estate", "telecommunications",
    "plugin", "theme", "extension", "addon"
]

ACTION_WORDS = [
    "analyze", "optimize", "automate", "monitor", "predict",
    "transform", "manage", "secure", "integrate", "visualize",
    "track", "enhance", "streamline", "validate",
    "specialize"
]

OBJECT_WORDS = [
    "data", "processes", "resources", "operations", "transactions",
    "systems", "infrastructure", "networks", "assets", "workflows",
    "performance", "security", "compliance", "efficiency",
    "themes", "plugins", "features", "functionality"
]

# Evaluation Parameters
EVALUATION_WEIGHTS = {
    'viability': 0.25,
    'value_potential': 0.25,
    'simplicity': 0.20,
    'novelty': 0.15,
    'scalability': 0.15
}

EVALUATION_RANGES = {
    'viability': (0, 300),
    'value_potential': (0, 100),
    'simplicity': (0, 50),
    'novelty': (0, 25),
    'scalability': (0, 25)
}

# Genetic Algorithm Parameters
POPULATION_SIZE = 30
DEFAULT_MIN_POPULATION = 20  # Renamed from MIN_POPULATION for clarity
MUTATION_RATE = 0.3
MUTATION_WEIGHTS = [0.5, 0.3, 0.2]  # Weights for 1, 2, or 3 mutations

# Default rates for genetic operations
DEFAULT_CROSSOVER_RATE = 0.7
DEFAULT_RANDOM_RATE = 0.3
DEFAULT_MAX_MUTATION = 0.8
DEFAULT_MIN_CROSSOVER = 0.4

# Convergence Parameters
DEFAULT_MAX_GENERATIONS = 50
DEFAULT_MAX_GENERATIONS_WITHOUT_IMPROVEMENT = 5
DEFAULT_MIN_DIVERSITY_THRESHOLD = 0.2
DEFAULT_TARGET_FITNESS = 450
DEFAULT_FITNESS_IMPROVEMENT_THRESHOLD = 1.0

INITIAL_RATES = {
    'mutation': MUTATION_RATE,
    'crossover': DEFAULT_CROSSOVER_RATE,
    'random': DEFAULT_RANDOM_RATE,
    'min_mutation': 0.1,
    'min_crossover': DEFAULT_MIN_CROSSOVER,
    'max_mutation': DEFAULT_MAX_MUTATION,
    'max_crossover': 0.8
}

# Semantic Validation
INCOMPATIBLE_PAIRS = {
    ('blockchain', 'agriculture'): ['predict', 'analyze'],
    ('quantum', 'retail'): ['monitor', 'track'],
    ('biometric', 'entertainment'): ['transform', 'optimize']
}

# CSV Settings
CSV_BASE_NAME = "idea_log"
CSV_HEADERS = [
    'timestamp',
    'generation',
    'idea',
    'fitness_score',
    'evaluation_time',
    'viability',
    'value_potential',
    'simplicity',
    'novelty',
    'scalability',
    'reasoning',
    'diversity_score',
    'population_size',
    'mutation_rate',
    'crossover_rate',
    'prompt_tokens',
    'completion_tokens',
    'total_tokens',
    'estimated_cost'
]

# Token Cost Settings (per 1K tokens)
COST_PER_1K_TOKENS = {
    'gpt-4': {'prompt': 0.03, 'completion': 0.06},
    'gpt-3.5-turbo': {'prompt': 0.001, 'completion': 0.002},
    'lm-studio': {'prompt': 0.0, 'completion': 0.0}  # Free for local models
}

# Default to free local model
DEFAULT_COST_MODEL = 'lm-studio'

# System Prompt for Evaluation
SYSTEM_PROMPT = """You are an AI evaluating ideas for programmatic implementation. For each idea (e.g., "A Parser for Processing Analytics Data"), return a JSON object with these exact keys and integer ranges:

{
    "viability": (0-300) Technical feasibility and implementation possibility,
    "value_potential": (0-100) Potential business or user value,
    "simplicity": (0-50) Ease of implementation and maintenance,
    "novelty": (0-25) Uniqueness and innovation level,
    "scalability": (0-25) Potential for growth and adaptation,
    "reasoning": A brief explanation of the scores
}

Focus on practical implementation while considering:
1. Technical complexity and required resources
2. Market need and potential impact
3. Development time and maintenance effort
4. Innovation level in the current tech landscape
5. Potential for future expansion

Provide objective scores based on real-world software development constraints."""
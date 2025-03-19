"""Module for generating and managing word pools."""

import random
import nltk
from nltk.corpus import words
from typing import List, Set

from ..config.settings import (
    TECH_SEEDS,
    INDUSTRY_SEEDS,
    VERB_SEEDS,
    NOUN_SEEDS,
    POOL_SIZE,
    SEED_WEIGHT
)

# Download NLTK words if necessary
nltk.download('words', quiet=True)
english_words = set(words.words())

def expand_pool(seed_list: List[str], target_size: int = POOL_SIZE) -> List[str]:
    """
    Expand a seed list into a larger pool of related words.
    
    Args:
        seed_list: Initial list of seed words
        target_size: Desired size of the expanded pool
        
    Returns:
        List of words including seeds and related words
    """
    pool: Set[str] = set(seed_list)
    
    if seed_list == TECH_SEEDS:
        candidates = [w for w in english_words if len(w) > 3 and 
                     (w.endswith(("er", "ing", "ion", "or", "et")) or w in TECH_SEEDS) and 
                     not w.endswith(("ness", "ity"))]
    elif seed_list == INDUSTRY_SEEDS:
        candidates = [w for w in english_words if len(w) > 4 and 
                     (w.endswith(("ness", "ity", "ics", "ing")) or w in INDUSTRY_SEEDS) and 
                     not w.endswith(("er", "or"))]
    elif seed_list == VERB_SEEDS:
        candidates = [w for w in english_words if len(w) > 3 and 
                     (w.endswith(("e", "t", "n", "d", "ing")) or w in VERB_SEEDS) and 
                     not w.endswith(("ness", "ity"))]
    elif seed_list == NOUN_SEEDS:
        candidates = [w for w in english_words if len(w) > 3 and 
                     not w.endswith(("ing", "ness", "ity")) and 
                     w not in VERB_SEEDS]
    else:
        candidates = list(english_words)
    
    while len(pool) < target_size and candidates:
        word = random.choice(candidates)
        pool.add(word)
        candidates.remove(word)
    
    return list(pool)

def weighted_choice(pool: List[str], seeds: List[str], seed_weight: float = SEED_WEIGHT) -> str:
    """
    Choose a word from the pool with weighted probability favoring seed words.
    
    Args:
        pool: Complete pool of words to choose from
        seeds: List of seed words that should have higher probability
        seed_weight: Probability of choosing from seeds vs pool
        
    Returns:
        Selected word
    """
    if random.random() < seed_weight:
        return random.choice(seeds)
    return random.choice(pool)

# Generate the expanded pools
technologies = expand_pool(TECH_SEEDS)
industries = expand_pool(INDUSTRY_SEEDS)
verbs = expand_pool(VERB_SEEDS)
nouns = expand_pool(NOUN_SEEDS) 
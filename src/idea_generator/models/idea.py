"""Module for idea generation and manipulation."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, TypeVar, ClassVar, Union, Literal
import random
import numpy as np
from itertools import product
from numpy.typing import NDArray
import re

from ..config.settings import (
    TECHNICAL_WORDS,
    INDUSTRY_WORDS,
    ACTION_WORDS,
    OBJECT_WORDS,
    MUTATION_RATE,
    EVALUATION_WEIGHTS,
    EVALUATION_RANGES,
    INCOMPATIBLE_PAIRS,
    MUTATION_WEIGHTS
)
from .word_pool_manager import WordPoolManager

T = TypeVar('T')

# Type aliases for clarity
EvaluationMetrics = Literal['viability', 'value_potential', 'simplicity', 'novelty', 'scalability']
ComponentType = Literal['technical', 'industry', 'action', 'object']

@dataclass
class Idea:
    """Class representing a programming project idea.
    
    Attributes:
        technical: Technical component of the idea
        industry: Industry/domain component
        action: Action/verb component
        object: Object/target component
        evaluation: Evaluation metrics with ranges:
            - viability (0-300): Technical feasibility
            - value_potential (0-100): Business/user value
            - simplicity (0-50): Implementation complexity
            - novelty (0-25): Uniqueness
            - scalability (0-25): Growth potential
        fitness_score: Normalized fitness score (0-100)
        category: Optional category for context-aware word matching
    """
    technical: str
    industry: str
    action: str
    object: str
    evaluation: Optional[Dict[EvaluationMetrics, float]] = None
    fitness_score: float = 0.0
    category: Optional[str] = None
    _recent_choices: Dict[ComponentType, List[str]] = field(default_factory=lambda: {
        component: [] for component in ('technical', 'industry', 'action', 'object')
    })
    
    # Class constants
    _COMPONENTS: ClassVar[Tuple[ComponentType, ...]] = ('technical', 'industry', 'action', 'object')
    _MAX_RECENT_CHOICES: ClassVar[int] = 5
    _word_pool_manager: ClassVar[WordPoolManager] = WordPoolManager({
        'technical': TECHNICAL_WORDS,
        'industry': INDUSTRY_WORDS,
        'action': ACTION_WORDS,
        'object': OBJECT_WORDS
    })
    
    def __post_init__(self):
        """Validate instance after initialization."""
        for component in self._COMPONENTS:
            value = getattr(self, component)
            if not isinstance(value, str):
                raise ValueError(f"{component} must be a string, got {type(value)}")
            if not value:
                raise ValueError(f"{component} cannot be empty")
    
    @classmethod
    def add_category(cls, category: str, words: Dict[str, List[str]]) -> None:
        """Add a new category with associated words."""
        if not category or not isinstance(category, str):
            raise ValueError("Category must be a non-empty string")
        if not words or not all(isinstance(v, list) for v in words.values()):
            raise ValueError("Words must be a non-empty dictionary of lists")
            
        for component, word_list in words.items():
            if component not in cls._COMPONENTS:
                raise ValueError(f"Invalid component: {component}")
            cls._word_pool_manager.add_words(category, component, word_list)
    
    @classmethod
    def get_pool(cls, component: str, category: Optional[str] = None) -> NDArray[np.str_]:
        """Get word pool for a component."""
        if component not in cls._COMPONENTS:
            raise ValueError(f"Invalid component: {component}")
        return cls._word_pool_manager.get_pool(component, category)
    
    def _update_recent_choices(self, component: str, word: str) -> None:
        """Update recent choices for a component."""
        if component not in self._COMPONENTS:
            raise ValueError(f"Invalid component: {component}")
        
        recent = self._recent_choices[component]
        recent.insert(0, word)
        self._recent_choices[component] = recent[:self._MAX_RECENT_CHOICES]

    @classmethod
    def generate_random(cls, category: Optional[str] = None) -> 'Idea':
        """Generate a random idea with weighted word selection."""
        components = {}
        
        for component in cls._COMPONENTS:
            pool = cls.get_pool(component, category)
            if len(pool) == 0:
                raise ValueError(f"Word pool for {component} is empty")
                
            try:
                word = cls._get_weighted_choice(pool, [])
                word = cls._word_pool_manager.find_best_match(word, component, category)
                if not word:
                    raise ValueError(f"{component} word selection failed")
                components[component] = word
            except Exception as e:
                raise ValueError(f"Failed to generate {component}: {str(e)}")
        
        instance = cls(**components, category=category)
        
        # Update recent choices after successful creation
        for component, word in components.items():
            instance._update_recent_choices(component, word)
        
        return instance

    @staticmethod
    def crossover(parent1: 'Idea', parent2: 'Idea') -> 'Idea':
        """Create a new idea by combining aspects of two parent ideas."""
        if not all(isinstance(p, Idea) for p in (parent1, parent2)):
            raise ValueError("Both parents must be Idea instances")
            
        p1_weight = (0.5 if parent1.fitness_score + parent2.fitness_score == 0 
                    else parent1.fitness_score / (parent1.fitness_score + parent2.fitness_score))
        
        mask = np.random.random(len(Idea._COMPONENTS)) < p1_weight
        components = {
            attr: getattr(parent1 if m else parent2, attr)
            for attr, m in zip(Idea._COMPONENTS, mask)
        }
        
        if not Idea._is_valid_combination(components):
            components = random.choice(
                Idea._generate_alternative_combinations(parent1, parent2)
            )
        
        # Use category from fitter parent
        category = (parent1.category if parent1.fitness_score >= parent2.fitness_score 
                   else parent2.category)
        
        child = Idea(**components, category=category)
        # Efficiently merge recent choices with deduplication
        for comp in Idea._COMPONENTS:
            merged = dict.fromkeys(
                parent1._recent_choices[comp] + parent2._recent_choices[comp]
            )
            child._recent_choices[comp] = list(merged)[:child._MAX_RECENT_CHOICES]
        
        return child

    def mutate(self) -> 'Idea':
        """Create a new idea by mutating this one."""
        components = self._get_components_dict()
        mutated = Idea(**components, category=self.category)
        # Efficiently copy recent choices
        mutated._recent_choices = {
            comp: choices[:self._MAX_RECENT_CHOICES] 
            for comp, choices in self._recent_choices.items()
        }
        
        mutation_count = np.random.choice(3, p=MUTATION_WEIGHTS) + 1
        to_mutate = np.random.choice(
            self._COMPONENTS,
            size=mutation_count,
            replace=False
        )
        
        mutations = np.random.random(len(to_mutate)) < MUTATION_RATE
        for component, should_mutate in zip(to_mutate, mutations):
            if should_mutate:
                word_pool = self.get_pool(component, self.category)
                recent = mutated._recent_choices[component]
                new_word = self._word_pool_manager.find_best_match(
                    self._get_weighted_choice(word_pool, recent),
                    component,
                    self.category
                )
                setattr(mutated, component, new_word)
                # Update recent choices more efficiently
                recent = mutated._recent_choices[component]
                if new_word not in recent:
                    recent.insert(0, new_word)
                    if len(recent) > self._MAX_RECENT_CHOICES:
                        recent.pop()
        
        return mutated

    @classmethod
    def from_string(cls, idea_str: str, category: Optional[str] = None) -> Optional['Idea']:
        """Create an idea from a string description.
        
        Args:
            idea_str: String description of the idea
            category: Optional category for word matching context
            
        Returns:
            Idea instance if parsing successful, None otherwise
        """
        if not idea_str or not isinstance(idea_str, str):
            return None
            
        idea_str = idea_str.lower().strip()
        if idea_str.startswith('a '):
            idea_str = idea_str[2:]
        idea_str = idea_str.replace(' system to ', ' ')
        
        components = idea_str.split()
        if len(components) < 4:
            return None
            
        technical = components[0]
        industry = components[1]
        action_start = 2
        object_start = -1
        
        # Use word pool manager for consistent matching
        tech_match = cls._word_pool_manager.find_best_match(technical, 'technical', category)
        industry_match = cls._word_pool_manager.find_best_match(industry, 'industry', category)
        
        action_words = components[action_start:object_start]
        object_words = components[object_start:]
        
        action_match = cls._word_pool_manager.find_best_match(' '.join(action_words), 'action', category)
        object_match = cls._word_pool_manager.find_best_match(' '.join(object_words), 'object', category)
        
        if all([tech_match, industry_match, action_match, object_match]):
            return cls(
                technical=tech_match,
                industry=industry_match,
                action=action_match,
                object=object_match,
                category=category
            )
        return None

    @classmethod
    def from_components(cls, 
                       technical: Optional[str] = None,
                       industry: Optional[str] = None,
                       action: Optional[str] = None,
                       object: Optional[str] = None,
                       category: Optional[str] = None) -> 'Idea':
        """Create an idea from specified components."""
        components = {
            'technical': technical,
            'industry': industry,
            'action': action,
            'object': object
        }
        
        for component, value in components.items():
            if not value:
                components[component] = cls._get_weighted_choice(
                    cls.get_pool(component, category),
                    []  # No recent choices for seeded components
                )
            else:
                components[component] = cls._word_pool_manager.find_best_match(
                    value,
                    component,
                    category
                )
        
        return cls(**components, category=category)

    @classmethod
    def get_statistics(cls) -> Dict:
        """Get statistics about word pools and usage."""
        return cls._word_pool_manager.get_statistics()

    @staticmethod
    def _get_weighted_choice(words: NDArray[np.str_], recent_choices: List[str]) -> str:
        """Get a weighted random choice, avoiding recent selections."""
        if len(words) == 0:
            raise ValueError("Cannot make a weighted choice from an empty word pool")
            
        weights = np.ones(len(words), dtype=np.float32)
        
        if recent_choices:
            # Vectorized operations for recency penalties
            mask = np.isin(words, recent_choices)
            if mask.any():
                positions = np.where(mask)[0]
                weights[positions] *= np.power(0.5, np.arange(len(positions)) + 1)
        
        # Ensure no zero weights
        if np.sum(weights) == 0:
            weights = np.ones(len(words), dtype=np.float32)
            
        # Normalize and select
        weights /= np.sum(weights)
        selected = str(np.random.choice(words, p=weights))
        
        # Double check the result is not empty
        if not selected:
            selected = str(np.random.choice(words))
            
        return selected

    @staticmethod
    def _is_valid_combination(components: Dict[str, str]) -> bool:
        """Check if the combination of components makes logical sense."""
        tech, ind, act = map(components.get, ('technical', 'industry', 'action'))
        return not any(
            tech == t and ind == i and act in invalid
            for (t, i), invalid in INCOMPATIBLE_PAIRS.items()
        )
    
    @staticmethod
    def _generate_alternative_combinations(parent1: 'Idea', parent2: 'Idea') -> List[Dict[str, str]]:
        """Generate alternative valid combinations from parent components."""
        # Vectorized options creation
        options = {
            comp: np.array([getattr(parent1, comp), getattr(parent2, comp)])
            for comp in Idea._COMPONENTS
        }
        
        # Efficient combination generation
        combinations = product(*(options[comp] for comp in Idea._COMPONENTS))
        valid_combinations = [
            dict(zip(Idea._COMPONENTS, combo))
            for combo in combinations
            if Idea._is_valid_combination(dict(zip(Idea._COMPONENTS, combo)))
        ]
        
        return valid_combinations or [{
            'technical': parent1.technical,
            'industry': parent2.industry,
            'action': parent1.action,
            'object': parent2.object
        }]

    def calculate_fitness(self) -> None:
        """Calculate fitness score based on evaluation metrics.
        
        The fitness score is calculated using the following metrics and ranges:
        - Viability (0-300): Technical feasibility
        - Value Potential (0-100): Business/user value
        - Simplicity (0-50): Implementation complexity
        - Novelty (0-25): Uniqueness of the idea
        - Scalability (0-25): Growth potential
        
        Each metric is normalized to a 0-1 range before applying weights.
        The final score is scaled to 0-100.
        """
        if not self.evaluation:
            self.fitness_score = 0.0
            return
            
        try:
            metrics = np.zeros(len(EVALUATION_WEIGHTS))
            for i, (metric, weight) in enumerate(EVALUATION_WEIGHTS.items()):
                value = float(self.evaluation.get(metric, 0))
                min_val, max_val = EVALUATION_RANGES[metric]
                
                if not min_val <= value <= max_val:
                    print(f"Warning: {metric} score {value} outside range [{min_val}, {max_val}]")
                    value = np.clip(value, min_val, max_val)
                
                # Normalize to 0-1 range before applying weight
                normalized_value = (value - min_val) / (max_val - min_val)
                metrics[i] = normalized_value * weight
            
            # Scale to 0-100 range
            self.fitness_score = 100 * np.sum(metrics)
        except Exception as e:
            print(f"Error calculating fitness: {e}")
            self.fitness_score = 0.0
    
    def __str__(self) -> str:
        """Return string representation of the idea."""
        return f"A {self.technical} {self.industry} system to {self.action} {self.object}"

    def _get_components_dict(self) -> Dict[ComponentType, str]:
        """Get a dictionary of all components."""
        return {
            component: getattr(self, component)
            for component in self._COMPONENTS
        }

    @staticmethod
    def _find_closest_match(word: str, choices: NDArray[np.str_]) -> str:
        """Find the closest matching word in the choices array.
        
        Uses basic string similarity to find best match.
        
        Args:
            word: Word to match
            choices: Array of possible choices
            
        Returns:
            Best matching word from choices
        """
        word = word.lower()
        # First try exact match
        if word in choices:
            return word
            
        # Then try substring match
        matches = [c for c in choices if word in c or c in word]
        if matches:
            return matches[0]
            
        # Finally try character similarity
        best_match = None
        best_score = 0
        
        for choice in choices:
            score = sum(c in choice for c in word) / max(len(word), len(choice))
            if score > best_score:
                best_score = score
                best_match = choice
                
        return best_match if best_score > 0.5 else str(np.random.choice(choices)) 
"""Module for managing dynamic word pools and user categories."""

from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import json
import os
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

class WordPoolManager:
    """Manages dynamic word pools with user-defined categories and automatic expansion."""
    
    def __init__(self, base_pools: Dict[str, List[str]], user_pools_dir: str = "user_pools"):
        """Initialize word pool manager.
        
        Args:
            base_pools: Dictionary of base word pools (technical, industry, etc.)
            user_pools_dir: Directory to store user-defined pools
        """
        # Convert all words to lowercase for case-insensitive matching
        self.base_pools = {k: {w.lower() for w in v} for k, v in base_pools.items()}
        self.user_pools = {k: set() for k in base_pools.keys()}
        self.frequency = {k: defaultdict(int) for k in base_pools.keys()}
        self.user_pools_dir = user_pools_dir
        self.categories = defaultdict(set)  # e.g., {'wordpress': {'technical': {'wordpress', 'php'}}}
        
        # Create user pools directory if it doesn't exist
        Path(user_pools_dir).mkdir(parents=True, exist_ok=True)
        self._load_user_pools()
    
    def add_words(self, category: str, component: str, words: List[str]) -> None:
        """Add words to a specific category and component.
        
        Args:
            category: Category name (e.g., 'wordpress', 'ai')
            component: Component type (technical, industry, etc.)
            words: List of words to add
        """
        # Convert words to lowercase
        words = [w.lower() for w in words]
        
        # Add to user pools
        self.user_pools[component].update(words)
        
        # Add to category
        if category not in self.categories:
            self.categories[category] = defaultdict(set)
        self.categories[category][component].update(words)
        
        # Save to file
        self._save_user_pools()
        
    def get_pool(self, component: str, category: Optional[str] = None) -> NDArray[np.str_]:
        """Get combined word pool for a component.
        
        Args:
            component: Component type (technical, industry, etc.)
            category: Optional category to filter by
            
        Returns:
            Array of words for the component
        """
        if category and category in self.categories:
            # Combine base pool with category-specific words
            words = self.base_pools[component].union(self.categories[category][component])
        else:
            # Combine base pool with all user-defined words
            words = self.base_pools[component].union(self.user_pools[component])
        
        return np.array(list(words), dtype=str)
    
    def find_best_match(self, word: str, component: str, category: Optional[str] = None) -> str:
        """Find best matching word in pool with context awareness.
        
        Args:
            word: Word to match
            component: Component type
            category: Optional category for context
            
        Returns:
            Best matching word from pool
        """
        if not word:
            # If input is empty, return a random word from the pool
            pool = self.get_pool(component, category)
            return str(np.random.choice(pool))
            
        word = word.lower()
        pool = self.get_pool(component, category)
        pool_lower = np.array([w.lower() for w in pool])
        
        # Update frequency
        self.frequency[component][word] += 1
        
        # Exact match (case-insensitive)
        if word in pool_lower:
            idx = np.where(pool_lower == word)[0][0]
            return str(pool[idx])
            
        # Check compound words
        parts = word.split()
        if any(part in pool_lower for part in parts):
            # Add compound word to user pool if parts match
            self.add_words('user_defined', component, [word])
            return word
            
        # Find similar matches
        matches = []
        for choice, choice_lower in zip(pool, pool_lower):
            # Check for substrings
            if word in choice_lower or choice_lower in word:
                matches.append((choice, 0.8))
                continue
                
            # Character similarity
            score = sum(c in choice_lower for c in word) / max(len(word), len(choice))
            if score > 0.5:
                matches.append((choice, score))
        
        if matches:
            # Sort by similarity score
            best_match = max(matches, key=lambda x: x[1])[0]
            return best_match
            
        # If no good match, add to user pool and return original
        if category:
            self.add_words(category, component, [word])
        else:
            self.add_words('user_defined', component, [word])
        return word
    
    def _load_user_pools(self) -> None:
        """Load user-defined pools from files."""
        try:
            pool_file = Path(self.user_pools_dir) / "user_pools.json"
            if pool_file.exists():
                with open(pool_file) as f:
                    data = json.load(f)
                    # Convert all words to lowercase
                    self.user_pools = {k: {w.lower() for w in v} for k, v in data['pools'].items()}
                    self.categories = defaultdict(lambda: defaultdict(set))
                    for cat, comps in data['categories'].items():
                        for comp, words in comps.items():
                            self.categories[cat][comp] = {w.lower() for w in words}
        except Exception as e:
            print(f"Warning: Could not load user pools: {e}")
    
    def _save_user_pools(self) -> None:
        """Save user-defined pools to files."""
        try:
            pool_file = Path(self.user_pools_dir) / "user_pools.json"
            data = {
                'pools': {k: list(v) for k, v in self.user_pools.items()},
                'categories': {
                    cat: {comp: list(words) 
                          for comp, words in comps.items()}
                    for cat, comps in self.categories.items()
                }
            }
            with open(pool_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save user pools: {e}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about word usage."""
        return {
            'total_words': {
                component: len(self.base_pools[component].union(self.user_pools[component]))
                for component in self.base_pools.keys()
            },
            'user_defined_words': {
                component: len(self.user_pools[component])
                for component in self.user_pools.keys()
            },
            'categories': {
                category: {
                    component: len(words)
                    for component, words in comps.items()
                }
                for category, comps in self.categories.items()
            },
            'most_used': {
                component: sorted(
                    self.frequency[component].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                for component in self.frequency.keys()
            }
        } 
"""Base evaluator module."""

import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

from ...config.settings import MAX_RETRIES, MAX_TOKENS, TEMPERATURE, SYSTEM_PROMPT
from ...models.idea import Idea

class BaseEvaluator(ABC):
    """Abstract base class for idea evaluators."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.client = None
        self.model_name = None
    
    @abstractmethod
    def setup_client(self) -> None:
        """Set up the API client."""
        pass
    
    @abstractmethod
    def get_completion(self, messages: list) -> str:
        """Get completion from the model."""
        pass
    
    def evaluate(self, idea: Idea) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Evaluate an idea using the configured API.
        
        Args:
            idea: The idea to evaluate
            
        Returns:
            Tuple of (evaluation result dict, evaluation time in seconds)
        """
        start_time = time.time()
        
        for attempt in range(MAX_RETRIES):
            try:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Evaluate this idea: {str(idea)}"}
                ]
                
                content = self.get_completion(messages)
                if not content:
                    raise ValueError("Empty response from model")
                
                # Strip ```json markers if present
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                evaluation = json.loads(content)
                end_time = time.time()
                return evaluation, end_time - start_time
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing response: {e}")
                if attempt < MAX_RETRIES - 1:
                    print("Retrying...")
                    time.sleep(2 ** attempt)
                else:
                    end_time = time.time()
                    return {
                        "viability": 0,
                        "reasoning": f"Failed to evaluate '{str(idea)}' due to invalid response.",
                        "value_potential": 0,
                        "simplicity": 0,
                        "novelty": 0,
                        "scalability": 0
                    }, end_time - start_time
                    
            except Exception as e:
                print(f"API Error: {e}")
                if attempt < MAX_RETRIES - 1:
                    print("Retrying...")
                    time.sleep(2 ** attempt)
                else:
                    end_time = time.time()
                    return None, end_time - start_time
    
    def test_connectivity(self) -> bool:
        """Test API connectivity with a simple evaluation."""
        test_idea = Idea(technical="Parser", action="Processing", industry="Analytics", object="Data")
        evaluation, _ = self.evaluate(test_idea)
        return evaluation is not None 
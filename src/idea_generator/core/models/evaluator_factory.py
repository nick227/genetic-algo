"""Factory module for creating evaluators."""

from typing import Optional
from .base_evaluator import BaseEvaluator
from .lm_studio_evaluator import LMStudioEvaluator
from .openai_evaluator import OpenAIEvaluator

class EvaluatorFactory:
    """Factory class for creating evaluators."""
    
    @staticmethod
    def create_evaluator() -> BaseEvaluator:
        """
        Create and configure an appropriate evaluator.
        First tries LM Studio, falls back to OpenAI if needed.
        """
        # Try LM Studio first
        lm_studio = LMStudioEvaluator()
        if lm_studio.setup_client():
            print("\nUsing LM Studio evaluator")
            return lm_studio
            
        # Fall back to OpenAI
        print("\nFalling back to OpenAI evaluator")
        openai = OpenAIEvaluator()
        openai.setup_client()
        return openai 
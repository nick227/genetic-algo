"""Module for evaluating ideas using AI models.

This module provides the interface for evaluating generated ideas using either
LM Studio or OpenAI models. It uses the factory pattern to create appropriate
evaluators based on available configurations.

The evaluation process:
1. An idea is submitted to the evaluator
2. The evaluator formats the idea into a prompt
3. The prompt is sent to either LM Studio or OpenAI
4. The response is parsed and validated
5. The evaluation metrics are extracted and normalized
6. The fitness score is calculated based on weighted metrics

Evaluation metrics:
- Viability (0-300): Technical feasibility
- Value Potential (0-100): Business/user value
- Simplicity (0-50): Implementation complexity
- Novelty (0-25): Uniqueness of the idea
- Scalability (0-25): Growth potential

Returns:
    Tuple containing:
    - evaluation (Dict): The evaluation metrics
    - eval_time (float): Time taken for evaluation
    - token_info (Dict): Token usage information with keys:
        - prompt_tokens: Number of tokens in the prompt
        - completion_tokens: Number of tokens in the response
        - total_tokens: Total tokens used
        - model: Model used for evaluation

See settings.SYSTEM_PROMPT for the exact evaluation criteria.
"""

from .models.evaluator_factory import EvaluatorFactory

def create_evaluator():
    """Create and configure an appropriate evaluator.
    
    The factory will:
    1. Check for LM Studio availability
    2. If available, list and validate available models
    3. If not available, check for OpenAI configuration
    4. Create and return the appropriate evaluator
    
    Returns:
        BaseEvaluator: An instance of either LMStudioEvaluator or OpenAIEvaluator
        configured with appropriate settings.
    
    Raises:
        RuntimeError: If no valid evaluator can be created
    """
    return EvaluatorFactory.create_evaluator() 
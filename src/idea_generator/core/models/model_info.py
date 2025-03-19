"""Module for handling model information."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelInfo:
    """Class representing model information."""
    id: str
    model_name: str
    organization: Optional[str] = None
    repository: Optional[str] = None

def parse_model_path(model_id: str) -> ModelInfo:
    """Parse a model ID into its components.
    
    Example:
        "mradermacher/DeepSeek-R1-Distill-Llama-70B-Uncensored-i1-GGUF/DeepSeek-R1-Distill-Llama-70B-Uncensored.i1-IQ3_M.gguf"
        -> ModelInfo(
            id="mradermacher/DeepSeek-R1-Distill-Llama-70B-Uncensored-i1-GGUF",
            model_name="DeepSeek-R1-Distill-Llama-70B-Uncensored.i1-IQ3_M",
            organization="mradermacher",
            repository="DeepSeek-R1-Distill-Llama-70B-Uncensored-i1-GGUF"
        )
    """
    parts = model_id.split('/')
    
    if len(parts) >= 3:
        # Full path format
        organization = parts[0]
        repository = parts[1]
        model_name = parts[-1].replace('.gguf', '')
        model_id = f"{organization}/{repository}"
    else:
        # Simple format
        organization = None
        repository = None
        model_name = model_id.replace('.gguf', '')
        model_id = model_name
    
    return ModelInfo(
        id=model_id,
        model_name=model_name,
        organization=organization,
        repository=repository
    ) 
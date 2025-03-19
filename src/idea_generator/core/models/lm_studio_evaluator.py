"""LM Studio evaluator module."""

from typing import List
from openai import OpenAI
import time
import requests
from requests.exceptions import Timeout

from ...config.settings import API_BASE_URL, MAX_TOKENS, TEMPERATURE, MAX_RETRIES
from .base_evaluator import BaseEvaluator
from .model_info import ModelInfo, parse_model_path

class LMStudioEvaluator(BaseEvaluator):
    """Class for evaluating ideas using LM Studio API."""
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models with parsed information."""
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key="lm-studio")
            models = client.models.list()
            return [parse_model_path(model.id) for model in models.data]
        except Exception as e:
            print(f"Error: Failed to connect to LM Studio - {str(e)}")
            return []
    
    def setup_client(self) -> None:
        """Set up the LM Studio client."""
        # First check if LM Studio is running
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key="lm-studio")
            client.models.list()  # Test connection
        except Exception:
            print("\nError: Cannot connect to LM Studio. Please ensure it's running on http://localhost:1234")
            return None
            
        available_models = self.get_available_models()
        
        if not available_models:
            print("\nNo models available in LM Studio. Please load a model first.")
            return None
            
        print("\nAvailable models in LM Studio:")
        for i, model in enumerate(available_models, 1):
            print(f"\n{i}. Model: {model.model_name}")
            if model.organization:
                print(f"   Organization: {model.organization}")
            if model.repository:
                print(f"   Repository: {model.repository}")
        
        print("\n0. Use OpenAI instead")
        
        while True:
            try:
                model_number = int(input(f"\nSelect a model (0-{len(available_models)}): "))
                if model_number == 0:
                    return None
                if 1 <= model_number <= len(available_models):
                    selected_model = available_models[model_number - 1]
                    self.model_name = selected_model.id  # Use the organization/repository format
                    print(f"\nSelected model: {selected_model.model_name}")
                    if selected_model.organization:
                        print(f"Organization: {selected_model.organization}")
                    if selected_model.repository:
                        print(f"Repository: {selected_model.repository}")
                    
                    # Test the model with a simple request
                    if self._test_model():
                        return True
                    else:
                        print("\nError: Selected model is not responding correctly. Please try another model.")
                else:
                    print(f"Please enter a number between 0 and {len(available_models)}")
            except ValueError:
                print("Please enter a valid number")
    
    def _test_model(self) -> bool:
        """Test if the selected model is working correctly."""
        try:
            print(f"\nTesting model with ID: {self.model_name}")
            client = OpenAI(base_url=API_BASE_URL, api_key="lm-studio")
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test"}],
                temperature=TEMPERATURE,
                max_tokens=10,
                timeout=30  # 30 second timeout for test
            )
            
            if completion.choices[0].message.content:
                print("Model test successful!")
                return True
            else:
                print("Model test failed: Empty response")
                return False
                
        except Timeout:
            print("Model test timed out after 30 seconds")
            return False
        except Exception as e:
            print(f"Model test failed with error: {str(e)}")
            return False
    
    def get_completion(self, messages: list) -> str:
        """Get completion from LM Studio model."""
        for attempt in range(MAX_RETRIES):
            try:
                # Add a small delay between retries
                if attempt > 0:
                    time.sleep(2 ** attempt)
                
                print(f"\nSending request to model: {self.model_name}")
                client = OpenAI(base_url=API_BASE_URL, api_key="lm-studio")
                
                # Use a shorter timeout for each attempt
                timeout = 60  # 60 seconds timeout
                if attempt > 0:
                    timeout = 30  # 30 seconds for retries
                
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    timeout=timeout
                )
                
                content = completion.choices[0].message.content.strip()
                if not content:
                    print("Warning: Empty response from model")
                    continue
                return content
                    
            except Timeout:
                print(f"Error: Request timed out after {timeout} seconds (attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    continue
            except Exception as e:
                print(f"Error: Request failed - {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    continue
        
        # If we get here, all retries failed
        print("Error: All attempts to get completion failed")
        return "" 
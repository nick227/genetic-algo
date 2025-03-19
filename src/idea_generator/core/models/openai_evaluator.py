"""OpenAI evaluator module."""

import os
from typing import List
from openai import OpenAI

from ...config.settings import MAX_TOKENS, TEMPERATURE
from .base_evaluator import BaseEvaluator

class OpenAIEvaluator(BaseEvaluator):
    """Class for evaluating ideas using OpenAI API."""
    
    def setup_client(self) -> None:
        """Set up the OpenAI client."""
        # Try to get API key from environment first
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            print("\nOpenAI API key not found in environment.")
            api_key = input("Enter your OpenAI API key: ").strip()
        
        try:
            temp_client = OpenAI(api_key=api_key)
            models = temp_client.models.list()
            
            print("\nAvailable OpenAI models:")
            model_list = [model.id for model in models.data]
            for i, model_id in enumerate(model_list, 1):
                print(f"{i}. {model_id}")
            
            while True:
                try:
                    model_number = int(input(f"\nSelect a model (1-{len(model_list)}): "))
                    if 1 <= model_number <= len(model_list):
                        self.model_name = model_list[model_number - 1]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(model_list)}")
                except ValueError:
                    print("Please enter a valid number")
            
            self.client = temp_client
            print(f"\nUsing OpenAI model: {self.model_name}")
            return self.client
            
        except Exception as e:
            print(f"Error setting up OpenAI client: {e}")
            raise SystemExit(1)
    
    def get_completion(self, messages: list) -> str:
        """Get completion from OpenAI model."""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        return completion.choices[0].message.content.strip() 
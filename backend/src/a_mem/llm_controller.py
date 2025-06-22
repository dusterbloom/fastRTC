from typing import Dict, Optional, Literal, Any
import os
import json
import logging
from abc import ABC, abstractmethod
import ollama

logger = logging.getLogger(__name__)

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Get completion from LLM"""
        pass

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
    
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content

class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        try:
            # Direct Ollama API call
            messages = [
                {"role": "system", "content": "You must respond with a valid JSON object."},
                {"role": "user", "content": prompt}
            ]
            
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "format": "json" if response_format else None
                }
            )
            
            result = response['message']['content']
            
            # Validate JSON if response_format is specified
            if response_format and result:
                try:
                    # Clean markdown code blocks if present
                    cleaned_result = result.strip()
                    if cleaned_result.startswith('```json'):
                        # Remove ```json at start and ``` at end
                        cleaned_result = cleaned_result[7:]  # Remove ```json
                        if cleaned_result.endswith('```'):
                            cleaned_result = cleaned_result[:-3]  # Remove ```
                        cleaned_result = cleaned_result.strip()
                    elif cleaned_result.startswith('```'):
                        # Remove generic ``` blocks
                        lines = cleaned_result.split('\n')
                        if len(lines) > 2 and lines[0].startswith('```') and lines[-1].strip() == '```':
                            cleaned_result = '\n'.join(lines[1:-1])
                    
                    # Test if it's valid JSON
                    json.loads(cleaned_result)
                    return cleaned_result
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ LLM returned invalid JSON: {result[:100]}...")
                    # Return fallback empty response
                    empty_response = self._generate_empty_response(response_format)
                    return json.dumps(empty_response)
            
            # Return result if no JSON validation needed or result is valid
            return result if result else '{"error": "Empty response"}'
            
        except Exception as e:
            logger.error(f"❌ Ollama LLM call failed: {e}")
            # Fallback to empty JSON response
            if response_format:
                empty_response = self._generate_empty_response(response_format)
                return json.dumps(empty_response)
            return '{"error": "LLM call failed"}'

class LLMController:
    """LLM-based controller for memory metadata generation"""
    def __init__(self, 
                 backend: Literal["openai", "ollama"] = "openai",
                 model: str = "gpt-4", 
                 api_key: Optional[str] = None):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        else:
            raise ValueError("Backend must be one of: 'openai', 'ollama'")
            
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)
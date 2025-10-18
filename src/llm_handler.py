from typing import Optional, Dict, Any
import ollama
from config.settings import LLM_MODEL, TEMPERATURE, MAX_TOKENS


class LLMHandler:
    """Handles interactions with Ollama LLM."""
    
    def __init__(self, 
                 model_name: str = LLM_MODEL,
                 temperature: float = TEMPERATURE,
                 max_tokens: int = MAX_TOKENS):
        """
        Initialize LLM handler.
        
        Args:
            model_name: Name of the Ollama model
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Verify model is available
        self._verify_model()
        
    def _verify_model(self):
        """Check if the model is available in Ollama."""
        try:
            # List available models
            models = ollama.list()
            available_models = [model['name'] for model in models.get('models', [])]
            
            # Check if our model is in the list
            model_available = any(self.model_name in model for model in available_models)
            
            if model_available:
                print(f"✓ Model '{self.model_name}' is available")
            else:
                print(f"⚠ Warning: Model '{self.model_name}' not found")
                print(f"Available models: {available_models}")
                print(f"\nTo download the model, run: ollama pull {self.model_name}")
                
        except Exception as e:
            print(f"⚠ Warning: Could not verify model availability: {e}")
            print("Make sure Ollama is running")
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: Input prompt
            stream: Whether to stream the response
            
        Returns:
            Generated text response
        """
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                },
                stream=stream
            )
            
            if stream:
                # Handle streaming response
                full_response = ""
                for chunk in response:
                    if 'response' in chunk:
                        full_response += chunk['response']
                return full_response
            else:
                return response['response']
                
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def chat(self, messages: list, stream: bool = False) -> str:
        """
        Generate response using chat format.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
            
        Returns:
            Generated text response
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                },
                stream=stream
            )
            
            if stream:
                # Handle streaming response
                full_response = ""
                for chunk in response:
                    if 'message' in chunk and 'content' in chunk['message']:
                        full_response += chunk['message']['content']
                return full_response
            else:
                return response['message']['content']
                
        except Exception as e:
            raise Exception(f"Error in chat: {str(e)}")
    
    def generate_answer(self, 
                       question: str, 
                       context: str,
                       system_message: Optional[str] = None) -> str:
        """
        Generate answer given question and context.
        
        Args:
            question: User's question
            context: Retrieved context
            system_message: Optional system message
            
        Returns:
            Generated answer
        """
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({
                'role': 'system',
                'content': system_message
            })
        
        # Add user message with context and question
        user_content = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        messages.append({
            'role': 'user',
            'content': user_content
        })
        
        return self.chat(messages)
    
    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            ollama.list()
            return True
        except:
            return False


# Test function
if __name__ == "__main__":
    print("Testing LLM Handler...\n")
    
    # Initialize handler
    llm = LLMHandler()
    
    # Check availability
    if not llm.is_available():
        print("❌ Ollama is not running. Please start Ollama and try again.")
        exit(1)
    
    print("✓ Ollama is running\n")
    
    # Test 1: Simple generation
    print("=== Test 1: Simple Generation ===")
    prompt = "What is Python? Give a brief answer in 2 sentences."
    response = llm.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")
    
    # Test 2: Chat format
    print("=== Test 2: Chat Format ===")
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'What is machine learning in one sentence?'}
    ]
    response = llm.chat(messages)
    print(f"Response: {response}\n")
    
    # Test 3: Question answering with context
    print("=== Test 3: Question Answering with Context ===")
    context = """
    Python is a high-level, interpreted programming language known for its simplicity and readability.
    It was created by Guido van Rossum and first released in 1991.
    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
    """
    
    question = "Who created Python and when?"
    
    response = llm.generate_answer(
        question=question,
        context=context,
        system_message="You are a helpful AI assistant. Answer based only on the provided context."
    )
    
    print(f"Context: {context.strip()}")
    print(f"\nQuestion: {question}")
    print(f"Answer: {response}")
import requests
import json
import time

from ollama import ListResponse, list

class OllamaClient:
    def __init__(self, base_url="http://127.0.0.1:11434"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
    
    def check_server(self):
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self):
        """List available models"""
        try:
            response = requests.get(f"{self.api_url}/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")
            return []
    
    def send_message(self, model, message, stream=False):
        """Send a message to the specified model"""
        url = f"{self.api_url}/generate"
        payload = {
            "model": model,
            "prompt": message,
            "stream": stream
        }
        
        try:
            if stream:
                response = requests.post(url, json=payload, stream=True, timeout=30)
                if response.status_code == 200:
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                chunk = data.get('response', '')
                                full_response += chunk
                                print(chunk, end='', flush=True)
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                    print()  # New line after streaming
                    return full_response
                else:
                    print(f"Error: {response.status_code} - {response.text}")
                    return None
            else:
                response = requests.post(url, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    return data.get('response', '')
                else:
                    print(f"Error: {response.status_code} - {response.text}")
                    return None
                    
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None

def main():
    print("🦙 Ollama Test Script")
    print("=" * 30)
    
    # Initialize client
    client = OllamaClient()
    
    # Check if server is running
    print("Checking Ollama server...")
    if not client.check_server():
        print("❌ Ollama server is not running or not accessible at http://localhost:11434")
        print("Make sure to start Ollama with: ollama serve")
        return
    
    print("✅ Ollama server is running!")
    
    # List available models
    print("\nListing available models...")
    models = client.list_models()
    if not models:
        print("❌ No models found. Please pull a model first with: ollama pull llama2")
        return
    
    print("✅ Available models:")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model}")
    
    # Select model (use first one for testing)
    selected_model = models[0]
    print(f"\n🎯 Using model: {selected_model}")
    
    # Test messages
    test_messages = [
        "Hello! Can you introduce yourself briefly?",
        "What's 2+2?",
        "Write a haiku about programming"
    ]
    
    print(f"\n🧪 Running tests with {len(test_messages)} messages...")
    print("=" * 50)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n[Test {i}] Prompt: {message}")
        print("-" * 40)
        
        start_time = time.time()
        response = client.send_message(selected_model, message, stream=True)
        end_time = time.time()
        
        if response:
            print(f"\n⏱️  Response time: {end_time - start_time:.2f} seconds")
        else:
            print("❌ Failed to get response")
        
        print("=" * 50)
    
    print("\n✅ Test completed!")
    print("\nTip: You can modify the test_messages list to try different prompts")

if __name__ == "__main__":
    main()
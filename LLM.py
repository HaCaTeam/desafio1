import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
load_dotenv()


MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
KEY = os.getenv("KEY")

class LLM:

    def __init__(self, model=MODEL, key=KEY):
        self.model = model
        self.key = key

    
    def query_cloud(self, query: str) -> str:
        """
        Query the model on the cloud

        Args:
            query (str): The query to ask the model

        Returns:
            str: The response from the model
        """
        client = InferenceClient(api_key=self.key)
        messages = [
	        { "role": "user", "content": query }
        ]
        output = client.chat.completions.create(
            model="meta-llama/Llama-3.1-70B-Instruct", 
            messages=messages, 
            stream=True, 
            temperature=0.5,
            max_tokens=1024,
            top_p=0.7
        )
        answer = ''.join(
            chunk.choices[0].delta.content for chunk in output
        )
        return answer

    

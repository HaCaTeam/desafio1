import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
load_dotenv()


# MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
KEY = os.getenv("HF_KEY")

class LLM:
    """
    A class to interact with the LLM model
    """

    def __init__(
        self, 
        model=MODEL, 
        key=KEY
    ):
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
            model=self.model,
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
    

    def transcript_to_query(self, transcripts: list) -> str:
        """
        Convert a list of transcripts to a query

        Args:
            transcripts (list): A list of transcripts

        Returns:
            str: The query to ask the model
        """
        text = " ".join(transcripts)
        query = (
            "Preciso de um titulo chamativo (max: 70 chars), resumo em formato expositivo e 3 keywords para a seguinte transcrição de um clipe de um programa de TV:\n\n" +  
            f"{text}\n\n" +
            "A resposta deve ser em português de portugal e ter o seguinte formato: \n Titulo: ... \n Resumo: ... \n Keywords: ..."
        )
        return query
    

    def process_transcripts(self, transcripts: list) -> tuple[str, str, list[str]]:
        """
        Process a list of transcripts

        Args:
            transcripts (list): A list of transcripts

        Returns:
            tuple(str, str, list(str)): Title, summary, and keywords for the transcripts
        """
        query = self.transcript_to_query(transcripts)
        response = self.query_cloud(query)
        response = response.split("\n")
        response = [line.strip() for line in response if line]
        title = response[0][8:]
        summary = response[1][8:]
        keywords = response[2][10:].split(",")
        return title, summary, keywords

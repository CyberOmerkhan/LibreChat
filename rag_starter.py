from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv() 
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
documents = SimpleDirectoryReader("data", recursive=True).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

def multiply(a: float, b: float) -> float:
    return a * b

def search_documents(query: str) -> str:
    response = query_engine.query(query)  # Use .query() not .aquery()
    return str(response)

agent = FunctionAgent(
    tools = [multiply, search_documents],
    llm = OpenAI(model='gpt-5.2'), 
    system_prompt="You are a helpful assistant."
)

async def main():
    prompt = input()
    response = await agent.run(prompt)        
    print(response)

if __name__ == '__main__':
    asyncio.run(main())
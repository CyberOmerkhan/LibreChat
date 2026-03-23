from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv() 
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
documents = SimpleDirectoryReader("data_confluence").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

def multiply(a: float, b: float) -> float:
    return a * b

async def search_documents(query: str) -> str:
    response = await query_engine.aquery(query)
    return str(response)

agent = FunctionAgent(
    tools = [multiply, search_documents],
    llm = OpenAI(model='gpt-5.1'),
    system_prompt="You are a helpful assistant that can perform calculations and search through documents to answer questions"
)

async def main():
    response = await agent.run(
        "How do I use a printer in Cornell?"
    )
    print(response)

if __name__ == '__main__':
    asyncio.run(main())
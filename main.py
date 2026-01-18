import asyncio
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


async def main():
    print(f"Connecting to Neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    graph = Neo4jGraph(
        url="neo4j://127.0.0.1:7687",
        username="neo4j",
        password=password,
        refresh_schema=False,
    )
    graph.query("MATCH (n) DETACH DELETE n")
    print(f"Connected to Neo4j\n")

    print(f"Defining LLM and Graph Transformer")
    # llm = ChatGoogleGenerativeAI(temperature=0, model='gemini-2.5-flash')
    # llm = ChatOpenAI(temperature=0, model='gpt-4o')
    # llm = ChatOllama(temperature=0, model="nemotron-3-nano:30b")
    llm = ChatOllama(temperature=0, model="gpt-oss:20b")

    graph_transformer = LLMGraphTransformer(llm=llm)
    print(f"Defined LLM and Graph Transformer: \n{llm}\n")

    print(f"Loading document")
    loader = TextLoader("marie_curie_summary.txt")
    documents = loader.load()
    print(f"Loaded document: \n{documents}\n")

    print(f"Creating graph")
    graph_documents = await graph_transformer.aconvert_to_graph_documents(documents)
    print(f"Created graph: \n{graph_documents}\n")

    print(f"Adding graph")
    graph.add_graph_documents(graph_documents)
    print(f"Added graph \n")


if __name__ == "__main__":
    asyncio.run(main())

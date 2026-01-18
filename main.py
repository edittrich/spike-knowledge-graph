import asyncio
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader, TextLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


async def main():
    print(f"Connecting to Neo4j and Deleting Graph")
    password = os.getenv("NEO4J_PASSWORD")
    graph = Neo4jGraph(
        url="neo4j://127.0.0.1:7687",
        username="neo4j",
        password=password,
        refresh_schema=False,
    )
    graph.query("MATCH (n)\nDETACH DELETE n")
    print(f"Connected to Neo4j and Deleted Graph\n")

    jsonloader = JSONLoader(
        file_path="marie_curie_summary.json",
        jq_schema=".",
        text_content=False,
        json_lines=True
    )
    json_documents = jsonloader.load()

    print(f"Defining LLM and Graph Transformer")
    # llm = ChatGoogleGenerativeAI(temperature=0, model='gemini-2.5-flash')
    # llm = ChatOpenAI(temperature=0, model='gpt-4o')
    llm = ChatOllama(temperature=0, model="nemotron-3-nano:30b")
    # llm = ChatOllama(temperature=0, model="gpt-oss:20b")

    graph_transformer = LLMGraphTransformer(llm=llm)
    print(f"Defined LLM and Graph Transformer: \n{llm}\n")

    print(f"Loading Document")
    textloader = TextLoader("marie_curie_summary.txt")
    documents = textloader.load()
    print(f"Loaded Document: \n{documents}\n")

    print(f"Creating Graph")
    graph_documents = await graph_transformer.aconvert_to_graph_documents(documents)
    print(f"Created Graph: \n{graph_documents}\n")

    print(f"Adding Graph")
    graph.add_graph_documents(graph_documents)
    print(f"Added Graph \n")


if __name__ == "__main__":
    asyncio.run(main())

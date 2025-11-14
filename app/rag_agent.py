from langchain.tools import tool
from app.schemas import ChatInput, ChatOutput
from app.gen_embeddings import GenEmbeddings 
from langchain.agents import create_agent
from langchain.messages import SystemMessage
from langchain_ollama import ChatOllama


@tool
def retrieve_doc(message):
    """Retrieve the most relevant documents from the vector database."""
    gen_embeddings = GenEmbeddings("")
    print("Retrieving documents...")
    db = gen_embeddings.vector_store()

    retrieve_docs = db.similarity_search(message, k=2)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieve_docs
    )
    return serialized, retrieve_docs

llm = ChatOllama(
    model="qwen3:1.7b",
    temperature=0,
)

tools = [retrieve_doc]
prompt = (
    "You have access to a tool that retrieves context from a pdf file called retrieve_doc"
    "Use the tool to help answer user queries always"
)


rag_agent = create_agent(llm, tools, system_prompt=prompt)







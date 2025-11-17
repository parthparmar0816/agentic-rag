from langchain.tools import tool
from app.schemas import ChatInput, ChatOutput
from app.gen_embeddings import GenEmbeddings 
from langchain.agents import create_agent
from langchain.messages import SystemMessage
from langchain_ollama import ChatOllama


@tool
def retrieve_doc(message):
    """Retrieve the most relevant documents from the vector database for the given content always"""
    gen_embeddings = GenEmbeddings("")
    print("Retrieving documents...")
    db = gen_embeddings.vector_store()

    retrieve_docs = db.similarity_search(message, k=2)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieve_docs
    )
    return serialized, retrieve_docs

@tool
def financial_summary(message: str):
    """
    Always Retrieve relevant financial context and create a financial summary.
    """
    # Step 1: Retrieve documents
    print("Financial summary...")
    gen_embeddings = GenEmbeddings("")
    db = gen_embeddings.vector_store()
    retrieved_docs = db.similarity_search(message, k=4)

    combined_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Step 2: Use LLM to summarize
    llm = ChatOllama(
        model="qwen3:1.7b",
        temperature=0,
    )

    summary = llm.invoke(
        [
            SystemMessage(content="Summarize the following into a concise, accurate financial summary."),
            {"role": "user", "content": combined_content}
        ]
    )

    return summary.content

def make_agent():
    llm = ChatOllama(
        model="qwen3:1.7b",
        temperature=0,
    )
    tools = [retrieve_doc, financial_summary]
    prompt = (
        "You have access to a tool that retrieves context from a pdf file called retrieve_doc or financial_summary.\n"
        "Use the tool to help answer user queries always"
    )

    return create_agent(llm, tools, system_prompt=prompt)







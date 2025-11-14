from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class GenEmbeddings:
    def __init__(self, input_file=None):
        self.input_file = input_file
        
    def load_data(self):
        loader = PyPDFLoader(self.input_file)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(docs)
        print("Data loaded and split into chunks")
        return docs
    def vector_store(self,embeddings=OllamaEmbeddings(model="nomic-embed-text")):
        db = Chroma(collection_name="test", embedding_function=embeddings, persist_directory="db")
        return db
    
    def get_embeddings(self):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return embeddings
    
    def store_embeddings(self):
        docs = self.load_data()
        embeddings = self.get_embeddings()
        db = self.vector_store(embeddings)
        db.add_documents(docs)
        print("Embeddings stored in ChromaDB")
        return None
        
       



# vectorstore.py — Handles PDF loading, chunking, and ChromaDB vector store creation.
# Loads all PDFs from the documents/ folder, splits them into chunks,
# and creates a retriever for similarity search.

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

# Embedding model used to convert text chunks into vectors
embeddings = OllamaEmbeddings(model="nomic-embed-text")

def load_documents():
    """Loads all PDFs from the documents/ folder and returns a list of pages."""
    path = path = os.path.join(os.path.dirname(__file__), "documents")
    if not os.path.exists(path):
        os.makedirs(path)
        print("Folder 'documents' created, please insert PDFs inside!")
        return []
    
    all_pages = []
    
    for file_path in os.listdir(path):
        full_path = os.path.join(path, file_path)
        if file_path.endswith(".pdf"):
            pdf_loader = PyPDFLoader(full_path)

            try:
                pages = pdf_loader.load()
                # We add the file name to each page's metadata 
                # so we can track which PDF a chunk came from
                for page in pages:
                    page.metadata["source_file"] = file_path

                all_pages.extend(pages)
                print(f"PDF has been loaded and is formed by {len(pages)} pages")
            except Exception as e:
                print(f"Error loading PDF: {e}")
                continue # If a PDF is corrupted, skip it and continue with the others
            
    print(f"The amount of PDF pages is {len(all_pages)}")
    
    return all_pages

def create_vectorstore():
    """Chunks the documents and creates a ChromaDB vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 175
    )
    
    docs = load_documents()

    if not docs:
        print("No documents found!")
        return None

    pages_split = text_splitter.split_documents(docs)
    
    persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
    collection_name = "study_materials"
    
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        
    try:
        vectorstore = Chroma.from_documents(
            documents=pages_split,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print(f"Created ChromaDB vector store!")
    
    except Exception as e:
        print(f"Error setting up ChromaDB: {str(e)}")
        raise
    return vectorstore

def get_retriever():
    """Creates and returns a retriever for similarity search over the vector store."""
    vectorstore = create_vectorstore()
    
    if vectorstore is None:
        print("Unable to create retriever!")
        return None
    
    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 8}
    )
    return retriever

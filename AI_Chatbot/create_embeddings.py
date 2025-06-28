
import os
import shutil # Needed for removing old FAISS directories

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Note: The following imports are for your chatbot's generation part,
# not strictly for creating embeddings, but kept as per your original file structure.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# --- Configuration Paths ---

# Folder where your PDF documents are located
PDF_FOLDER = "data/"

# Path where the FAISS vector database will be saved
FAISS_DB_PATH = "vectorstore/db_faiss"


# --- Functions for Document Processing ---

def load_documents_from_folder(folder_path):
    """Loads all PDF files from the specified folder."""
    print(f"\n--- Loading Documents ---")
    print(f"Checking for PDFs in: '{folder_path}'")

    # Set up the loader to find PDFs recursively in the folder

    pdf_loader = DirectoryLoader(
        folder_path,
        glob='**/*.pdf', # Look for all .pdf files in subfolders too
        loader_cls=PyPDFLoader # Use PyPDFLoader to read each PDF
    )

    all_loaded_docs = pdf_loader.load()

    if not all_loaded_docs:
        print("Heads up: No PDF documents found or loaded. "
              "Make sure your 'data/' folder exists and contains text-searchable PDFs.")
    else:
        print(f"Successfully loaded {len(all_loaded_docs)} pages from PDFs.")
        # Quick check for text content in the first page
        if all_loaded_docs[0].page_content:
            preview = all_loaded_docs[0].page_content[:500]
            print(f"Preview of first 500 characters from first loaded page:\n'{preview}'\n")
        else:
            print("Important: First loaded page seems empty. "
                  "This often means your PDFs are image-based (screenshots converted to PDF) "
                  "without proper text. You need to OCR them first.")

    return all_loaded_docs


def split_docs_into_chunks(loaded_documents):

    """Splits loaded documents into smaller text chunks."""

    print(f"\n--- Creating Text Chunks ---")
    if not loaded_documents:
        print("No documents to split. Skipping chunk creation.")
        return []

    # Configure how documents are split into smaller pieces
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # Size of each text chunk in characters
        chunk_overlap=230, # Overlap between chunks to maintain context
        length_function=len, # Method to calculate chunk length
    )

    document_chunks = text_splitter.split_documents(loaded_documents)
    print(f"Generated {len(document_chunks)} usable text chunks.")

    #checking if no chunks are created, we can't build the DB
    if not document_chunks:
        raise ValueError("No text chunks could be created from your documents. "
                         "This typically means your PDFs have no extractable text "
                         "or your chunk size is too large for the content. Exiting.")

    return document_chunks


def initialize_embedding_model():
    """Initializes the Sentence-Transformers embedding model."""
    print(f"\n--- Setting up Embedding Model ---")
    print("Loading 'sentence-transformers/all-MiniLM-L6-v2' model...")
    # This model converts text into numerical vectors (embeddings)
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model loaded and ready.")
    return embed_model


def create_and_save_faiss_db(chunks, embedding_model_instance, save_path):
    """Builds and saves the FAISS vector database."""
    print(f"\n--- Building and Saving FAISS Database ---")

    # Clean up old database to ensure a fresh start
    if os.path.exists(save_path):
        print(f"Removing old database at '{save_path}' for a clean build...")
        try:
            shutil.rmtree(save_path)
            print("Old database successfully deleted.")
        except OSError as e:
            print(f"Warning: Could not remove old database directory '{save_path}': {e}")
            print("Please delete it manually if you face issues, then retry.")
            exit("Failed to clear old vector store. Stopping script.")

    # Ensure the directory exists before saving
    os.makedirs(save_path, exist_ok=True)
    print(f"Ensuring save directory exists: '{save_path}'")

    print("Generating embeddings and building FAISS index... This may take a moment.")
    try:
        # Create the FAISS index from your chunks using the embedding model
        faiss_database = FAISS.from_documents(chunks, embedding_model_instance)
        faiss_database.save_local(save_path)
        print(f"FAISS database successfully created and saved locally at: '{save_path}'")
    except Exception as e:
        print(f"Error encountered during FAISS database creation: {e}")
        print("Check your text chunks and embedding model. Stopping script.")
        exit("Failed to build FAISS database. Exiting.")


# --- Main Execution Flow ---

if __name__ == "__main__":
    print("--- Starting Document Processing for Chatbot ---")

    # Step 1: Load documents
    loaded_documents = load_documents_from_folder(folder_path=PDF_FOLDER)

    if not loaded_documents:
        print("No documents were loaded. Please ensure PDFs are in the 'data/' folder and are text-searchable. Exiting.")
        exit() # Stop if no documents to process

    # Step 2: Split documents into chunks
    try:
        processed_chunks = split_docs_into_chunks(loaded_documents=loaded_documents)
    except ValueError as e:
        print(f"Error: {e}")
        exit() # Stop if no chunks could be created

    # Step 3: Get the embedding model
    active_embedding_model = initialize_embedding_model()

    # Step 4: Build and save the FAISS vector database
    create_and_save_faiss_db(
        chunks=processed_chunks,
        embedding_model_instance=active_embedding_model,
        save_path=FAISS_DB_PATH
    )

    print("\n--- Document Processing Complete ---")
    print("Your knowledge base is now prepared. You can proceed with running your chatbot!")
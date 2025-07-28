from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings


# Extract text from PDF files

def load_pdf_files(data):
    loader=DirectoryLoader(data,
                           glob="*.pdf",
                           loader_cls=PyPDFLoader)
    
    documents=loader.load()
    
    return documents


def filter_to_minimal_docs(docs: List[Document])-> List[Document]:
    
    ''' 
    '''
    minimal_docs:List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    
    return minimal_docs

def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
   
    )
    
    texts_chunk=text_splitter.split_documents(minimal_docs)
    return texts_chunk



def download_hugging_face_embeddings():
    ''' 
    Download and return the HuggingFace embeddings model.
    '''
    
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embeddings=HuggingFaceBgeEmbeddings(
        model_name=model_name
        
    )
    
    return embeddings



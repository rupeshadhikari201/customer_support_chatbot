import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedRAGPipeline:
    def __init__(self, 
                 docs_path: str = 'doc/', 
                 chunk_size: int = 500, 
                 chunk_overlap: int = 100,
                 embedding_model: str = 'sentence-transformers/all-mpnet-base-v2',
                 use_reranker: bool = True,
                 use_compression: bool = True):
        """
        Initialize the optimized RAG pipeline
        
        Args:
            docs_path (str): Path to the documents directory
            chunk_size (int): Size of document chunks
            chunk_overlap (int): Overlap between chunks
            embedding_model (str): HuggingFace embedding model name
            use_reranker (bool): Whether to use reranking
            use_compression (bool): Whether to use contextual compression
        """
        self.docs_path = docs_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.use_reranker = use_reranker
        self.use_compression = use_compression
        
        # Initialize components
        self.documents = self._load_documents()
        self.embeddings = self._initialize_embeddings()
        self.vector_store = self._create_vector_store()
        self.retriever = self._build_retriever()
        
    def _load_documents(self) -> List[Document]:
        """Load and split documents from the specified path"""
        logger.info(f"Loading documents from {self.docs_path}")
        try:
            # Load documents from directory
            loader = DirectoryLoader(
                self.docs_path,
                glob='*.txt',
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            
            # Split documents with optimized chunking strategy
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len
            )
            
            documents = loader.load_and_split(text_splitter)
            logger.info(f"Loaded {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return []
            
    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        logger.info(f"Initializing embeddings with model: {self.embedding_model}")
        try:
            return HuggingFaceEmbeddings(model_name=self.embedding_model)
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            # Fallback to a simpler model if the specified one fails
            return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            
    def _create_vector_store(self):
        """Create a vector store from documents and embeddings"""
        logger.info("Creating vector store")
        try:
            return FAISS.from_documents(self.documents, self.embeddings)
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return None
            
    def _build_retriever(self, k: int = 4):
        """Build an optimized retriever with multiple strategies"""
        logger.info("Building optimized retriever")
        
        # Create base vector retriever
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        
        # Create BM25 retriever for keyword-based retrieval
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = k
        
        # Combine retrievers with weights
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        
        # Add contextual compression if enabled
        if self.use_compression:
            try:
                # Use LLM to extract relevant parts of retrieved documents
                llm = ChatOpenAI(
                    openai_api_key = os.getenv('OPENROUTER_API_KEY'),
        openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0, 
                    model="gpt-3.5-turbo")
                compressor = LLMChainExtractor.from_llm(llm)
                
                return ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=ensemble_retriever
                )
            except Exception as e:
                logger.warning(f"Compression retriever initialization failed: {str(e)}")
                return ensemble_retriever
        else:
            return ensemble_retriever
            
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query (str): The search query
            k (int): Number of documents to retrieve
            
        Returns:
            List[Document]: List of retrieved documents
        """
        logger.info(f"Retrieving documents for query: {query}")
        try:
            return self.retriever.get_relevant_documents(query)
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            # Fallback to simple vector retrieval if ensemble fails
            try:
                return self.vector_store.similarity_search(query, k=k)
            except:
                return []
                
    def get_retrieval_content(self, query: str, k: int = 4) -> List[str]:
        """
        Get the content of retrieved documents for a query
        
        Args:
            query (str): The search query
            k (int): Number of documents to retrieve
            
        Returns:
            List[str]: List of document contents
        """
        docs = self.retrieve(query, k)
        return [doc.page_content for doc in docs]

# Initialize the RAG pipeline
rag_pipeline = OptimizedRAGPipeline() 
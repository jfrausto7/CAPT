from typing import List, Optional, Dict, Tuple
import logging
from langchain_together import TogetherEmbeddings
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import math
import faiss
import gc

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from ArticleFetcher import ArticleFetcher

# Get the directory where DocumentIngestor.py is located
SCRIPT_DIR = Path(__file__).parent.absolute()

class DocumentIngestor:
    def __init__(
        self,
        data_dir: str = "data/raw",
        vector_dir: str = "app/vectorstore/store",
        chunk_size: int = 2000,
        chunk_overlap: int = 100,
        batch_size: int = 32,
        num_workers: int = 4,
        api_keys: Dict[str, str] = None
    ):
        """Initialize the document ingestor with optimized batch processing."""
        self.data_dir = Path(data_dir)
        self.vector_dir = Path(vector_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.logger = self._setup_logger()
        self.embeddings = TogetherEmbeddings()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        
        self.article_fetcher = ArticleFetcher(
            cache_dir=str(self.data_dir / "cache"),
            api_keys=api_keys
        )
        
        # Ensure directories exist
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "cache").mkdir(parents=True, exist_ok=True)

    def _process_batch(self, batch: List[Document]) -> Tuple[List[np.ndarray], List[str], List[Dict]]:
        """Process a batch of documents in parallel."""
        try:
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            
            # Generate embeddings for the batch
            embeddings = self.embeddings.embed_documents(texts)
            
            return embeddings, texts, metadatas
        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}")
            return [], [], []

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('DocumentIngestor')
        logger.setLevel(logging.INFO)

        log_dir = SCRIPT_DIR / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        if not logger.handlers:
            c_handler = logging.StreamHandler()
            f_handler = logging.FileHandler(log_dir / 'document_ingestion.log')

            log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            c_handler.setFormatter(log_format)
            f_handler.setFormatter(log_format)

            logger.addHandler(c_handler)
            logger.addHandler(f_handler)

        return logger

    def load_pdfs(self, directory: Path) -> List:
        """Load PDF documents from specified directory."""
        self.logger.info(f"Loading PDFs from {directory}")

        loader = DirectoryLoader(
            str(directory),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )

        documents = loader.load()
        self.logger.info(f"Loaded {len(documents)} PDF documents")
        return documents

    def process_documents(self, documents: List) -> List:
        """Process and split documents into chunks."""
        self.logger.info("Processing documents...")
        chunks = self.text_splitter.split_documents(documents)
        self.logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def process_csv_files(self, directory: Path) -> List[Dict]:
        """Process CSV files and extract article information."""
        articles = []

        for csv_file in directory.glob("**/*.csv"):
            self.logger.info(f"Processing CSV file: {csv_file}")

            try:
                # Try reading with utf-8 encoding first
                df = pd.read_csv(csv_file, encoding='utf-8')
            except UnicodeDecodeError:
                # If utf-8 fails, try with iso-8859-1
                self.logger.warning(f"utf-8 decode error for {csv_file}, trying iso-8859-1")
                df = pd.read_csv(csv_file, encoding='iso-8859-1')

        # Check for both DOI and URL columns
            
            # Check for both DOI and URL columns
            doi_columns = ['DOI', 'doi']
            url_columns = ['ArticleURL', 'URL', 'FullTextURL']
            
            available_doi_columns = [col for col in doi_columns if col in df.columns]
            available_url_columns = [col for col in url_columns if col in df.columns]

            if not (available_doi_columns or available_url_columns):
                self.logger.warning(f"No DOI or URL columns found in {csv_file}")
                continue

            for _, row in df.iterrows():
                # Try DOI first
                doi = None
                for col in available_doi_columns:
                    if pd.notna(row.get(col)):
                        doi = str(row[col]).strip()
                        break
                
                # If no DOI, try URL
                url = None
                if not doi:
                    for col in available_url_columns:
                        if pd.notna(row.get(col)):
                            url = str(row[col]).strip()
                            break
                
                if not (doi or url):
                    self.logger.warning("No valid DOI or URL found for row")
                    continue

                article = {
                    'identifier': doi if doi else url,
                    'type': 'doi' if doi else 'url',
                    'metadata': {
                        'title': row.get('Title', ''),
                        'authors': row.get('Authors', ''),
                        'year': row.get('Year', ''),
                        'source': row.get('Source', ''),
                        'abstract': row.get('Abstract', '')
                    }
                }
                articles.append(article)

        return articles

    def fetch_article_content(self, article: Dict) -> Optional[str]:
        """Fetch full text content for an article."""
        try:
            if article['type'] == 'doi':
                content = self.article_fetcher.fetch_from_doi(article['identifier'])
            else:  # url type
                content = self.article_fetcher.fetch_from_url(article['identifier'])
            
            if content:
                # If content already includes metadata, return as is
                if content.strip().startswith('Title:') or content.strip().startswith('Abstract:'):
                    return content
                    
                # Otherwise, format with metadata
                return f"""
                Title: {article['metadata']['title']}
                Authors: {article['metadata']['authors']}
                Year: {article['metadata']['year']}
                Source: {article['metadata']['source']}
                
                Abstract:
                {article['metadata']['abstract']}
                
                Full Text:
                {content}
                """
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching content for {article['identifier']}: {str(e)}")
            return None

    def create_documents_from_articles(self, articles: List[Dict]) -> List[Document]:
        """Create document objects from fetched articles."""
        documents = []
        for article in tqdm(articles, desc="Processing articles"):
            try:
                content = self.fetch_article_content(article)
                if content:
                    chunks = self.text_splitter.split_text(content)
                    for chunk in chunks:
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                **article['metadata'],
                                'identifier': article['identifier'],
                                'chunk_size': self.chunk_size,
                                'chunk_overlap': self.chunk_overlap
                            }
                        ))
                    # Rate limiting
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error creating documents from articles: {str(e)}")
                # Continue processing other articles in case of errors
                continue
        return documents

    def create_vectorstore(
        self,
        documents: List[Document],
        collection_name: str
    ) -> FAISS:
        """Create or update vector store with documents using optimized batch processing."""
        persist_directory = self.vector_dir / collection_name
        self.logger.info(f"Creating FAISS vector store in {persist_directory}")

        # Check if there's an existing index
        index_path = persist_directory / "index.faiss"
        store_path = persist_directory / "store.pkl"
        
        if index_path.exists() and store_path.exists():
            self.logger.info("Loading existing FAISS index")
            vectorstore = FAISS.load_local(
                str(persist_directory),
                self.embeddings,
                index_name="index"
            )
            
            # Process documents in batches with periodic saves
            for i in tqdm(range(0, len(documents), self.batch_size), desc="Processing document batches"):
                batch = documents[i:i + self.batch_size]
                vectorstore.add_documents(batch)
                
                # Save periodically and clear memory
                if (i // self.batch_size) % 10 == 0:
                    vectorstore.save_local(str(persist_directory))
                    gc.collect()  # Release memory
                
        else:
            self.logger.info("Creating new FAISS index")
            
            # Calculate embedding dimension
            sample_embedding = self.embeddings.embed_documents([documents[0].page_content])[0]
            dim = len(sample_embedding)
            
            # Initialize FAISS index
            index = faiss.IndexFlatL2(dim)
            
            # Process documents in batches using ThreadPoolExecutor
            total_batches = math.ceil(len(documents) / self.batch_size)
            processed_vectors = []
            processed_texts = []
            processed_metadatas = []
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                
                # Submit batches for processing
                for i in range(0, len(documents), self.batch_size):
                    batch = documents[i:i + self.batch_size]
                    futures.append(executor.submit(self._process_batch, batch))
                
                # Process results as they complete
                for future in tqdm(futures, total=total_batches, desc="Processing batches"):
                    embeddings, texts, metadatas = future.result()
                    if embeddings:  # Only process if batch was successful
                        processed_vectors.extend(embeddings)
                        processed_texts.extend(texts)
                        processed_metadatas.extend(metadatas)
                    
                    # Periodically add to index and clear memory
                    if len(processed_vectors) >= self.batch_size * 10:
                        vectors_array = np.array(processed_vectors).astype('float32')
                        index.add(vectors_array)
                        processed_vectors = []
                        gc.collect()  # Release memory
            
            # Add any remaining vectors
            if processed_vectors:
                vectors_array = np.array(processed_vectors).astype('float32')
                index.add(vectors_array)
            
            # Create FAISS vectorstore
            vectorstore = FAISS(
                self.embeddings.embed_query,
                index,
                processed_texts,
                processed_metadatas,
                normalize_L2=True
            )
        
        # Ensure directory exists and save
        persist_directory.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(persist_directory))
        
        self.logger.info("FAISS vector store created and persisted")
        return vectorstore

    def ingest_documents(
        self,
        collection_name: str,
        source_dir: Optional[str] = None,
    ) -> FAISS:
        """Complete ingestion pipeline with batch processing optimization."""
        try:
            documents = []
            target_dir = Path(source_dir) if source_dir else self.data_dir

            self.logger.info(f"Starting document ingestion from: {target_dir}")

            # Process PDFs
            pdf_files = list(target_dir.glob("**/*.pdf"))
            if pdf_files:
                self.logger.info(f"Processing {len(pdf_files)} PDF files")
                pdf_documents = self.load_pdfs(target_dir)
                documents.extend(self.process_documents(pdf_documents))
                gc.collect()  # Release memory after PDF processing

            # Process CSVs
            csv_files = list(target_dir.glob("**/*.csv"))
            if csv_files:
                self.logger.info(f"Processing {len(csv_files)} CSV files")
                articles = self.process_csv_files(target_dir)
                
                if articles:
                    self.logger.info(f"Creating documents from {len(articles)} articles")
                    # Process articles in batches to manage memory
                    for i in range(0, len(articles), self.batch_size):
                        batch = articles[i:i + self.batch_size]
                        article_documents = self.create_documents_from_articles(batch)
                        documents.extend(article_documents)
                        gc.collect()  # Release memory after each batch

            if not documents:
                raise ValueError(f"No documents found in {target_dir}")

            self.logger.info(f"Total documents to process: {len(documents)}")

            # Create vector store with batch processing
            vectorstore = self.create_vectorstore(documents, collection_name)

            return vectorstore

        except Exception as e:
            self.logger.error(f"Error during document ingestion: {str(e)}")
            raise
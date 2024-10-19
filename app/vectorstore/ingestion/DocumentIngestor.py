from typing import List, Optional, Dict
import logging
from langchain_together import TogetherEmbeddings
import pandas as pd
from pathlib import Path
import time
import torch
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma

from ArticleFetcher import ArticleFetcher

# Get the directory where DocumentIngestor.py is located
SCRIPT_DIR = Path(__file__).parent.absolute()
print(torch.cuda.is_available())

class DocumentIngestor:
    def __init__(
        self,
        data_dir: str = "data/raw",
        vector_dir: str = "app/vectorstore/store",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        api_keys: Dict[str, str] = None
    ):
        """Initialize the document ingestor."""
        self.data_dir = Path(data_dir)
        self.vector_dir = Path(vector_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
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
        # Ensure the vector directory exists
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        # Create cache directory for ArticleFetcher
        (self.data_dir / "cache").mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('DocumentIngestor')
        logger.setLevel(logging.INFO)

        # Ensure logs directory exists
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
    ) -> Chroma:
        """Create or update vector store with documents."""
        persist_directory = self.vector_dir / collection_name
        self.logger.info(f"Creating vector store in {persist_directory}")

        # Initialize the vector store without documents first
        vectorstore = Chroma(embedding_function=self.embeddings, persist_directory=str(persist_directory), collection_name=collection_name)
        
        # Add documents one by one with a progress bar
        for document in tqdm(documents, desc="Adding documents to vector store"):
            vectorstore.add_documents([document])

        vectorstore.persist()
        self.logger.info("Vector store created and persisted")
        return vectorstore

    def ingest_documents(
        self,
        collection_name: str,
        source_dir: Optional[str] = None
    ) -> Chroma:
        """Complete ingestion pipeline."""
        try:
            documents = []
            target_dir = Path(source_dir) if source_dir else self.data_dir

            self.logger.info(f"Checking directory: {target_dir}")

            # Process PDFs
            pdf_files = list(target_dir.glob("**/*.pdf"))
            self.logger.info(f"Found {len(pdf_files)} PDF files")

            if pdf_files:
                pdf_documents = self.load_pdfs(target_dir)
                documents.extend(self.process_documents(pdf_documents))

            # Process CSVs
            csv_files = list(target_dir.glob("**/*.csv"))
            self.logger.info(f"Found {len(csv_files)} CSV files")

            articles = self.process_csv_files(target_dir)
            self.logger.info(f"Processed {len(articles)} articles from CSV files")

            if articles:
                article_documents = self.create_documents_from_articles(articles)
                documents.extend(article_documents)

            if not documents:
                raise ValueError(f"No documents found in {target_dir}")

            self.logger.info(f"Total documents to process: {len(documents)}")

            # Create vector store
            vectorstore = self.create_vectorstore(documents, collection_name)

            return vectorstore

        except Exception as e:
            self.logger.error(f"Error during document ingestion: {str(e)}")
            raise

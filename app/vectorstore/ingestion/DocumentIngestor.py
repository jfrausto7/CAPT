from typing import List, Optional, Dict, Tuple
import logging
from langchain_together import TogetherEmbeddings
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import math
import faiss
import gc
import chardet

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

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

        self.common_encodings = [
            'utf-8', 
            'utf-8-sig',  # UTF-8 with BOM
            'iso-8859-1',
            'cp1252',     # Windows-1252
            'latin1'
        ]
        
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
        
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect the file encoding using chardet."""
        # Read a sample of the file to detect encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            confidence = result['confidence']
            encoding = result['encoding']
            
            self.logger.info(f"Detected encoding for {file_path.name}: {encoding} (confidence: {confidence:.2f})")
            return encoding if confidence > 0.7 else None
        
    def _read_csv_with_encoding(self, file_path: Path) -> pd.DataFrame:
        """Try reading CSV file with different encodings."""
        errors = []
        
        # First try detected encoding
        detected_encoding = self._detect_encoding(file_path)
        if detected_encoding:
            try:
                return pd.read_csv(file_path, encoding=detected_encoding)
            except Exception as e:
                errors.append(f"Detected encoding ({detected_encoding}) failed: {str(e)}")

        # Try common encodings
        for encoding in self.common_encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except Exception as e:
                errors.append(f"{encoding} failed: {str(e)}")
                continue

        # If all attempts fail, try with the most permissive encoding and error handling
        try:
            return pd.read_csv(file_path, encoding='iso-8859-1', on_bad_lines='skip')
        except Exception as e:
            errors.append(f"Final attempt with iso-8859-1 failed: {str(e)}")
            
        # If everything fails, raise an exception with the error history
        raise ValueError(f"Failed to read {file_path.name} with all attempted encodings:\n" + "\n".join(errors))



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
            c_handler.setLevel(logging.WARNING)
            f_handler.setLevel(logging.INFO)

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
                df = self._read_csv_with_encoding(csv_file)
                self.logger.info(f"Successfully read {csv_file.name}")
                
                # Count rows before filtering
                initial_rows = len(df)
                
                # Remove rows with all NaN values
                df = df.dropna(how='all')
                
                # Remove rows where all string columns are empty strings
                string_cols = df.select_dtypes(include=['object']).columns
                df = df[~(df[string_cols] == '').all(axis=1)]
                
                # Log if any rows were removed
                final_rows = len(df)
                if final_rows < initial_rows:
                    self.logger.info(f"Removed {initial_rows - final_rows} empty or invalid rows from {csv_file.name}")

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
                        continue

                    # Clean metadata values
                    metadata = {
                        'title': str(row.get('Title', '')).strip(),
                        'authors': str(row.get('Authors', '')).strip(),
                        'year': str(row.get('Year', '')).strip(),
                        'source': str(row.get('Source', '')).strip(),
                        'abstract': str(row.get('Abstract', '')).strip()
                    }
                    
                    # Remove any empty metadata fields
                    metadata = {k: v for k, v in metadata.items() if v and v.lower() != 'nan'}

                    article = {
                        'identifier': doi if doi else url,
                        'type': 'doi' if doi else 'url',
                        'metadata': metadata
                    }
                    articles.append(article)

            except Exception as e:
                self.logger.error(f"Error processing {csv_file}: {str(e)}")
                continue

        self.logger.info(f"Successfully processed {len(articles)} articles from CSV files")
        return articles

    def fetch_article_content(self, article: Dict) -> Optional[str]:
        """Fetch full text content for an article with improved error handling."""
        try:
            if article['type'] == 'doi':
                content = self.article_fetcher.fetch_from_doi(article['identifier'])
            else:  # url type
                content = self.article_fetcher.fetch_from_url(article['identifier'])
            
            if content:
                # If content already includes metadata, verify it has required fields
                if content.strip().startswith('Title:'):
                    # Verify the content has basic structure we expect
                    required_fields = ['Title:', 'Abstract:', 'Full Text:']
                    has_required_fields = all(field in content for field in required_fields)
                    if has_required_fields:
                        return content
                
                # If we get here, either content doesn't have metadata or is missing required fields
                # Build content with metadata from article dict, safely accessing fields
                metadata = article.get('metadata', {})
                formatted_content = []
                
                # Add each metadata field, checking for None or empty values
                if metadata.get('title'):
                    formatted_content.append(f"Title: {metadata['title']}")
                else:
                    formatted_content.append("Title: [No title available]")
                    
                if metadata.get('authors'):
                    formatted_content.append(f"Authors: {metadata['authors']}")
                
                if metadata.get('year'):
                    formatted_content.append(f"Year: {metadata['year']}")
                    
                if metadata.get('source'):
                    formatted_content.append(f"Source: {metadata['source']}")
                
                # Add abstract and content sections
                formatted_content.append("\nAbstract:")
                if metadata.get('abstract'):
                    formatted_content.append(metadata['abstract'])
                else:
                    formatted_content.append("[No abstract available]")
                
                formatted_content.append("\nFull Text:")
                formatted_content.append(content)
                
                return "\n".join(formatted_content)
                
            # Skip article if no content retrieved
            return None
                
        except Exception as e:
            self.logger.error(f"Error fetching content for {article['identifier']}: {str(e)}", exc_info=True)
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
                    # time.sleep(1)
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
        store_path = persist_directory / "index.pkl"
        
        if index_path.exists() and store_path.exists():
            self.logger.info("Loading existing FAISS index")
            vectorstore = FAISS.load_local(
                str(persist_directory),
                self.embeddings,
                index_name="index",
                allow_dangerous_deserialization=True
            )
            
            # Calculate starting point
            start_batch = 34300
            start_index = start_batch * self.batch_size

            # Process documents in batches using ThreadPoolExecutor
            total_batches = math.ceil(len(documents) / self.batch_size)
            processed_vectors = []
            processed_texts = {}
            index_to_docstore_id = {}
            unique_id_counter = len(vectorstore.docstore._dict)  # Initialize with existing docstore size
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                
                # Submit batches for processing
                for i in range(start_index, len(documents), self.batch_size):
                    batch = documents[i:i + self.batch_size]
                    futures.append(executor.submit(self._process_batch, batch))
                
                # Process results as they complete
                for future in tqdm(futures, 
                                total=total_batches, 
                                initial=start_batch,
                                desc=f"Processing document batches (resuming from batch {start_batch})"):
                    embeddings, texts, metadatas = future.result()
                    if embeddings:  # Only process if batch was successful
                        processed_vectors.extend(embeddings)
                        for text, metadata in zip(texts, metadatas):
                            doc_id = unique_id_counter
                            processed_texts[doc_id] = Document(page_content=text, metadata=metadata)
                            index_to_docstore_id[unique_id_counter] = doc_id
                            unique_id_counter += 1
                    
                    # Periodically add to index, clear memory, and save
                    if len(processed_vectors) >= self.batch_size * 1000:
                        vectors_array = np.array(processed_vectors).astype('float32')
                        vectorstore.index.add(vectors_array)
                        vectorstore.docstore.add(processed_texts)
                        vectorstore.index_to_docstore_id.update(index_to_docstore_id)
                        processed_vectors = []
                        processed_texts = {}
                        index_to_docstore_id = {}
                        gc.collect()  # Release memory
                        vectorstore.save_local(str(persist_directory))
            
            # Add any remaining vectors
            if processed_vectors:
                vectors_array = np.array(processed_vectors).astype('float32')
                vectorstore.index.add(vectors_array)
                vectorstore.docstore.add(processed_texts)
                vectorstore.index_to_docstore_id.update(index_to_docstore_id)
                vectorstore.save_local(str(persist_directory))
                
        else:
            self.logger.info("Creating new FAISS index")
            
            # Calculate embedding dimension
            sample_embedding = self.embeddings.embed_documents([documents[0].page_content])[0]
            dim = len(sample_embedding)
            
            # Initialize FAISS index
            index = faiss.IndexFlatL2(dim)

            # Initialize FAISS vectorstore
            docstore = InMemoryDocstore()
            index_to_docstore_id = {}
            vectorstore = FAISS(
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
                embedding_function=self.embeddings,
                normalize_L2=True
            )
            
            # Process documents in batches using ThreadPoolExecutor
            total_batches = math.ceil(len(documents) / self.batch_size)
            processed_vectors = []
            processed_texts = {}
            unique_id_counter = 0
            
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
                        for text, metadata in zip(texts, metadatas):
                            doc_id = unique_id_counter
                            processed_texts[doc_id] = Document(page_content=text, metadata=metadata)
                            index_to_docstore_id[unique_id_counter] = doc_id
                            unique_id_counter += 1
                    
                    # Periodically add to index, clear memory, and save
                    if len(processed_vectors) >= self.batch_size * 1000:
                        vectors_array = np.array(processed_vectors).astype('float32')
                        vectorstore.index.add(vectors_array)
                        vectorstore.docstore.add(processed_texts)
                        vectorstore.index_to_docstore_id.update(index_to_docstore_id)
                        processed_vectors = []
                        processed_texts = {}
                        index_to_docstore_id = {}
                        gc.collect()  # Release memory
                        vectorstore.save_local(str(persist_directory))
            
            # Add any remaining vectors
            if processed_vectors:
                vectors_array = np.array(processed_vectors).astype('float32')
                vectorstore.index.add(vectors_array)
                vectorstore.docstore.add(processed_texts)
                vectorstore.index_to_docstore_id.update(index_to_docstore_id)
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
            all_documents = []
            target_dir = Path(source_dir) if source_dir else self.data_dir

            self.logger.info(f"Starting document ingestion from: {target_dir}")

            # Process PDFs
            pdf_files = list(target_dir.glob("**/*.pdf"))
            if pdf_files:
                self.logger.info(f"Processing {len(pdf_files)} PDF files")
                pdf_documents = self.load_pdfs(target_dir)
                all_documents.extend(self.process_documents(pdf_documents))
                gc.collect()  # Release memory after PDF processing

            # Process CSVs
            csv_files = list(target_dir.glob("**/*.csv"))
            if csv_files:
                self.logger.info(f"Processing {len(csv_files)} CSV files")
                articles = self.process_csv_files(target_dir)
                
                if articles:
                    total_articles = len(articles)
                    total_batches = (total_articles + self.batch_size - 1) // self.batch_size
                    self.logger.info(f"Processing {total_articles} articles in {total_batches} batches of size {self.batch_size}")
                    
                    documents_from_articles = []
                    for batch_num in range(total_batches):
                        start_idx = batch_num * self.batch_size
                        end_idx = min((batch_num + 1) * self.batch_size, total_articles)
                        batch = articles[start_idx:end_idx]
                        
                        self.logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch)} articles)")
                        batch_documents = self.create_documents_from_articles(batch)
                        if batch_documents:  # Check if we got valid documents back
                            documents_from_articles.extend(batch_documents)
                            print(f"Completed batch {batch_num + 1}/{total_batches}. "
                                        f"Total documents so far: {len(documents_from_articles)}")
                            self.logger.info(f"Completed batch {batch_num + 1}/{total_batches}. "
                                        f"Total documents so far: {len(documents_from_articles)}")
                        gc.collect()  # Release memory after each batch
                    
                    all_documents.extend(documents_from_articles)

            if not all_documents:
                raise ValueError(f"No documents found in {target_dir}")

            self.logger.info(f"Total documents to process: {len(all_documents)}")

            # Create vector store with batch processing
            vectorstore = self.create_vectorstore(all_documents, collection_name)

            return vectorstore

        except Exception as e:
            self.logger.error(f"Error during document ingestion: {str(e)}")
            raise
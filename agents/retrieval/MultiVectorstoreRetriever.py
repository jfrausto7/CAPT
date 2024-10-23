from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
from langchain.schema import BaseRetriever, Document
from langchain_community.vectorstores import FAISS
from langchain_together import TogetherEmbeddings
import heapq
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pydantic import Field, PrivateAttr

@dataclass
class ScoredDocument:
    """Document with its similarity score."""
    document: Document
    score: float
    collection: str

    def __lt__(self, other):
        return self.score > other.score  # Reverse for max-heap

class MultiVectorstoreRetriever(BaseRetriever):
    """Retriever that searches across multiple FAISS vectorstores."""
    
    vector_dir: Union[str, Path] = Field(description="Directory containing vectorstores")
    embeddings: TogetherEmbeddings = Field(default_factory=TogetherEmbeddings)
    k: int = Field(default=4, description="Number of documents to retrieve per collection")
    score_threshold: float = Field(default=0.0, description="Minimum similarity score threshold")
    num_workers: int = Field(default=4, description="Number of parallel workers for search")
    show_progress: bool = Field(default=True, description="Whether to show progress bars")
    
    # Private attributes
    _vectorstores: Dict[str, FAISS] = PrivateAttr(default_factory=dict)
    _logger: logging.Logger = PrivateAttr()

    def __init__(
        self,
        vector_dir: Union[str, Path],
        collections: Optional[List[str]] = None,
        embeddings: Optional[TogetherEmbeddings] = None,
        k: int = 4,
        score_threshold: float = 0.0,
        num_workers: int = 4,
        show_progress: bool = True,
        **kwargs
    ):
        """Initialize the multi-vectorstore retriever."""
        super().__init__(
            vector_dir=vector_dir,
            embeddings=embeddings or TogetherEmbeddings(),
            k=k,
            score_threshold=score_threshold,
            num_workers=num_workers,
            show_progress=show_progress,
            **kwargs
        )
        self._logger = self._setup_logger()
        self._vectorstores = {}
        self._load_vectorstores(collections)

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('MultiVectorstoreRetriever')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _load_vectorstores(self, collections: Optional[List[str]] = None) -> None:
        """Load FAISS vectorstores from disk with progress bar."""
        vector_dir = Path(self.vector_dir)
        if not vector_dir.exists():
            raise ValueError(f"Vector store directory not found: {vector_dir}")
            
        # Get all collection directories if none specified
        if collections is None:
            collections = [d.name for d in vector_dir.iterdir() if d.is_dir()]
        
        # Create progress bar for loading
        collections_iter = tqdm(
            collections,
            desc="Loading vectorstores",
            disable=not self.show_progress
        )
            
        for collection in collections_iter:
            collection_path = vector_dir / collection
            if collection_path.exists():
                try:
                    collections_iter.set_postfix({"current": collection})
                    self._vectorstores[collection] = FAISS.load_local(
                        str(collection_path),
                        self.embeddings,
                        index_name="index",
                        allow_dangerous_deserialization=True
                    )
                    self._logger.info(f"Loaded vectorstore: {collection}")
                except Exception as e:
                    self._logger.error(f"Error loading vectorstore {collection}: {str(e)}")
            else:
                self._logger.warning(f"Collection directory not found: {collection_path}")

    def _search_collection(
        self,
        collection: str,
        vectorstore: FAISS,
        query: str,
        k: int
    ) -> List[ScoredDocument]:
        """Search a single collection."""
        try:
            docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
            return [
                ScoredDocument(doc, score, collection)
                for doc, score in docs_and_scores
            ]
        except Exception as e:
            self._logger.error(f"Error searching collection {collection}: {str(e)}")
            return []

    def get_relevant_documents(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        **kwargs
    ) -> List[Document]:
        """Get relevant documents from all or specified collections."""
        scored_docs = self.get_scored_documents(query, collections)
        return [sd.document for sd in scored_docs]

    def get_scored_documents(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        **kwargs
    ) -> List[ScoredDocument]:
        """Get relevant documents with their scores from all or specified collections."""
        collections = collections or list(self._vectorstores.keys())
        all_scored_docs = []
        
        # Use ThreadPoolExecutor for parallel search with progress bar
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_collection = {
                executor.submit(
                    self._search_collection,
                    collection,
                    self._vectorstores[collection],
                    query,
                    self.k
                ): collection
                for collection in collections
                if collection in self._vectorstores
            }
            
            # Create progress bar for search operations
            search_progress = tqdm(
                as_completed(future_to_collection),
                total=len(future_to_collection),
                desc="Searching collections",
                disable=not self.show_progress
            )
            
            # Gather results
            for future in search_progress:
                collection = future_to_collection[future]
                try:
                    search_progress.set_postfix({"collection": collection})
                    scored_docs = future.result()
                    all_scored_docs.extend(scored_docs)
                except Exception as e:
                    self._logger.error(f"Error searching collection {collection}: {str(e)}")

        # Filter by score threshold and get top k overall
        filtered_docs = [
            doc for doc in all_scored_docs
            if doc.score >= self.score_threshold
        ]
        
        # Use heap for efficient top-k selection
        return heapq.nlargest(self.k, filtered_docs)


"""Initialize the multi-vectorstore retriever.
        
        Args:
            vector_dir: Directory containing vectorstores
            collections: List of collection names to load (loads all if None)
            embeddings: TogetherEmbeddings instance
            k: Number of documents to retrieve per collection
            score_threshold: Minimum similarity score threshold
            num_workers: Number of parallel workers for search

        Usage example:

        # Example usage with your model:
        def create_retrieval_chain(vector_dir: str, model) -> None:
            # Initialize the retriever
            retriever = MultiVectorstoreRetriever(
                vector_dir=vector_dir,
                k=4,  # Number of documents to retrieve per query
                score_threshold=0.7  # Minimum similarity score
            )
            
            # Create a retrieval chain
            from langchain.chains import RetrievalQA
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=model,
                chain_type="stuff",  # or "map_reduce" for longer contexts
                retriever=retriever,
                return_source_documents=True  # Include source docs in response
            )
            
            return qa_chain

        # Initialize your model
        model = Together(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.7,
            max_tokens=1024
        )

        # Create the retrieval chain
        qa_chain = create_retrieval_chain(
            vector_dir="app/vectorstore/store",
            model=model
        )

        # Query across all collections
        response = qa_chain("What are the safety guidelines for ketamine therapy?")
        answer = response['result']
        sources = response['source_documents']

        # Query specific collections
        response = qa_chain(
            "What are the safety guidelines for ketamine therapy?",
            collections=['guidelines_ketamine', 'guidelines_general']
        )
        """
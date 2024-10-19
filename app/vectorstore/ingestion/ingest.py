import logging
from pathlib import Path
from typing import Dict
import yaml

from DocumentIngestor import DocumentIngestor

# Get the directory where ingest.py is located
SCRIPT_DIR = Path(__file__).parent.absolute()

def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('Ingestion')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(SCRIPT_DIR / 'logs/ingestion.log')
        
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)
        
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    return logger

def load_config(config_path: str = SCRIPT_DIR / "config.yaml") -> Dict:
    """Load ingestion configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_collection(
    ingestor: DocumentIngestor,
    collection_name: str,
    source_dir: str,
    logger: logging.Logger
) -> None:
    """Process a single collection."""
    try:
        logger.info(f"Processing collection: {collection_name}")
        vectorstore = ingestor.ingest_documents(
            collection_name=collection_name,
            source_dir=source_dir
        )
        logger.info(f"Successfully processed {collection_name}")
    except Exception as e:
        logger.error(f"Error processing {collection_name}: {str(e)}")

def main():
    # Set up logging
    logger = setup_logging()
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize ingestor
        ingestor = DocumentIngestor(
            data_dir=config.get('data_dir', 'data/raw'),
            vector_dir=config.get('vector_dir', SCRIPT_DIR / 'app/vectorstore/store'),
            chunk_size=config.get('chunk_size', 1000),
            chunk_overlap=config.get('chunk_overlap', 200)
        )
        
        # Process collections
        collections = {
            'articles': 'data/raw/articles',
            'guidelines_ketamine': 'data/raw/guidelines/Ketamine',
            'guidelines_mdma': 'data/raw/guidelines/MDMA',
            'guidelines_psilocybin': 'data/raw/guidelines/Psilocybin',
            'guidelines_LSD': 'data/raw/guidelines/LSD',
            'guidelines_general': 'data/raw/guidelines/General'
        }
        
        for collection_name, source_dir in collections.items():
            process_collection(ingestor, collection_name, source_dir, logger)
            
    except Exception as e:
        logger.error(f"Error in main ingestion process: {str(e)}")

if __name__ == "__main__":
    main()
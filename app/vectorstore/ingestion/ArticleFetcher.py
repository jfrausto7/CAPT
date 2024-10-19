import hashlib
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Optional, Dict
import doi2bib.crossref as crossref
import logging
from urllib.parse import urlparse

class ArticleFetcher:
    """Handles fetching full text content from various sources."""

    def __init__(self, cache_dir: str = "data/cache", api_keys: Dict[str, str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        })
        
        # Initialize API keys
        self.api_keys = api_keys or {}
        if not self.api_keys.get('elsevier'):
            self.api_keys['elsevier'] = "836373b117a9a87cbbca087555a7c42f"  # Default key
            
        self.logger = logging.getLogger('ArticleFetcher')
        
        # Define publisher DOI prefixes and domains
        self.publisher_info = {
            'elsevier': {
                'doi_prefixes': ['10.1016'],
                'domains': ['sciencedirect.com', 'elsevier.com']
            },
            'springer': {
                'doi_prefixes': ['10.1007'],
                'domains': ['springer.com', 'springerlink.com']
            },
            'wiley': {
                'doi_prefixes': ['10.1002', '10.1111'],
                'domains': ['wiley.com', 'onlinelibrary.wiley.com']
            },
            'sage': {
                'doi_prefixes': ['10.1177'],
                'domains': ['sagepub.com']
            },
            'taylor_francis': {
                'doi_prefixes': ['10.1080'],
                'domains': ['tandfonline.com']
            },
            'nature': {
                'doi_prefixes': ['10.1038'],
                'domains': ['nature.com']
            },
            'oxford': {
                'doi_prefixes': ['10.1093'],
                'domains': ['oxford.com', 'oup.com']
            },
            'ieee': {
                'doi_prefixes': ['10.1109'],
                'domains': ['ieee.org', 'ieee.com']
            }
        }

    def _get_publisher(self, identifier: str, is_url: bool = False) -> Optional[str]:
        """Identify the publisher based on DOI prefix or domain."""
        if is_url:
            domain = urlparse(identifier).netloc.lower()
            for publisher, info in self.publisher_info.items():
                if any(pub_domain in domain for pub_domain in info['domains']):
                    return publisher
        else:
            for publisher, info in self.publisher_info.items():
                if any(identifier.startswith(prefix) for prefix in info['doi_prefixes']):
                    return publisher
        return None

    def _get_cache_path(self, identifier: str) -> Path:
        """Generate cache file path from identifier."""
        # Remove invalid characters and hash the identifier
        safe_identifier = hashlib.md5(identifier.encode()).hexdigest()
        return self.cache_dir / f"{safe_identifier}.txt"

    def _is_cached(self, identifier: str) -> bool:
        """Check if article is already cached."""
        return self._get_cache_path(identifier).exists()

    def _cache_content(self, identifier: str, content: str) -> None:
        """Cache article content."""
        try:
            with open(self._get_cache_path(identifier), 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            self.logger.error(f"Error caching content for {identifier}: {str(e)}")

    def _get_cached_content(self, identifier: str) -> Optional[str]:
        """Retrieve cached content."""
        try:
            with open(self._get_cache_path(identifier), 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            # self.logger.error(f"Error retrieving cached content for {identifier}: {str(e)}")
            return None

    def _modify_url_for_access(self, url: str) -> str:
        """Modify URLs to try to access available versions."""
        url = re.sub(r'/doi/pdf/', '/doi/abs/', url)
        url = re.sub(r'/doi/full-xml/', '/doi/full/', url)
        
        if 'sciencedirect.com' in url:
            url = url.replace('/science/article/pii/', '/science/article/abs/pii/')
        
        return url

    def fetch_from_elsevier(self, doi: str) -> Optional[str]:
        """Enhanced Elsevier API fetcher with better error handling."""
        if 'elsevier' not in self.api_keys:
            # self.logger.warning("No Elsevier API key provided")
            return None
            
        doi = doi.strip('{}')
        url = f"https://api.elsevier.com/content/article/doi/{doi}"
        
        headers = {
            'Accept': 'application/json',
            'X-ELS-APIKey': self.api_keys['elsevier'],
            'X-ELS-ResourceVersion': 'FULL'
        }
        
        try:
            response = self.session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if "full-text-retrieval-response" in data:
                full_text = data["full-text-retrieval-response"]
                content_parts = []
                
                if "coredata" in full_text:
                    if "dc:title" in full_text["coredata"]:
                        content_parts.append(f"Title: {full_text['coredata']['dc:title']}")
                    if "dc:description" in full_text["coredata"]:
                        content_parts.append(f"Abstract: {full_text['coredata']['dc:description']}")
                
                if "originalText" in full_text:
                    content_parts.append(f"Full Text: {full_text['originalText']}")
                elif not content_parts:
                    # self.logger.warning(f"No accessible content found for DOI: {doi}")
                    return None
                    
                return "\n\n".join(content_parts)
                
        except Exception as e:
            # self.logger.error(f"Error accessing Elsevier API: {str(e)}")
            return None

    def fetch_from_tandfonline(self, url: str) -> Optional[str]:
        """NOTE: We cannot handle articles from T&F unfortunately."""
        return None

    def fetch_from_sage(self, url: str) -> Optional[str]:
        """NOTE: We cannot handle articles from Sage unfortunately."""
        return None

    def fetch_from_crossref(self, url: str) -> Optional[str]:
        """Fetch article content using Crossref URL."""
        try:
            url = self._modify_url_for_access(url)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            # self.logger.error(f"Error fetching from Crossref URL {url}: {str(e)}")
            return None

    def fetch_from_doi(self, doi: str) -> Optional[str]:
        """Enhanced DOI fetcher with publisher-specific handling."""
        if self._is_cached(doi):
            return self._get_cached_content(doi)
            
        try:
            publisher = self._get_publisher(doi)
            
            # Try publisher-specific APIs first
            if publisher == 'elsevier':
                content = self.fetch_from_elsevier(doi)
                if content:
                    self._cache_content(doi, content)
                    return content
                    
            # Get metadata from Crossref
            metadata = crossref.get_json(doi)[1]
            if "message" in metadata and "link" in metadata["message"]:
                for link in metadata["message"]["link"]:
                    url = link["URL"]
                    content = None
                    
                    # Apply publisher-specific handling
                    if 'tandfonline.com' in url:
                        content = self.fetch_from_tandfonline(url)
                    elif 'sagepub.com' in url:
                        content = self.fetch_from_sage(url)
                    else:
                        content = self.fetch_from_crossref(url)
                        
                    if content:
                        self._cache_content(doi, content)
                        return content
                        
        except Exception as e:
            # self.logger.error(f"Error fetching content for DOI {doi}: {str(e)}")
            return None
    def fetch_from_url(self, url: str) -> Optional[str]:
        """Fetch article content from direct URL."""
        if self._is_cached(url):
            return self._get_cached_content(url)

        publisher = self._get_publisher(url, is_url=True)
        content = None

        try:
            if publisher == 'elsevier':
                # Extract DOI or PII from ScienceDirect URL
                if 'pii' in url.lower():
                    pii = re.search(r'[Pp][Ii][Ii]/([^/]+)', url).group(1)
                    content = self.fetch_from_elsevier(f"pii/{pii}")
                elif 'doi' in url.lower():
                    doi = re.search(r'doi/([^/]+/[^/]+)', url).group(1)
                    content = self.fetch_from_elsevier(doi)
            elif publisher == 'taylor_francis':
                content = self.fetch_from_tandfonline(url)
            elif publisher == 'sage':
                content = self.fetch_from_sage(url)
            
            # Generic fetch for other publishers or if specific fetch failed
            if not content:
                modified_url = self._modify_url_for_access(url)
                response = self.session.get(modified_url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try to extract title
                title = soup.find('h1') or soup.find('title')
                title_text = title.get_text(strip=True) if title else ""
                
                # Try to extract abstract
                abstract = (
                    soup.find('section', {'id': 'abstract'}) or
                    soup.find('div', {'class': 'abstract'}) or
                    soup.find('div', {'id': 'abstract'}) or
                    soup.find('abstract')
                )
                abstract_text = abstract.get_text(strip=True) if abstract else ""
                
                # Try to extract main content
                main_content = (
                    soup.find('main') or
                    soup.find('article') or
                    soup.find('div', {'id': 'main-content'})
                )
                content_text = main_content.get_text(strip=True) if main_content else ""
                
                if any([title_text, abstract_text, content_text]):
                    content = f"""
                    Title: {title_text}
                    
                    Abstract:
                    {abstract_text}
                    
                    Content:
                    {content_text}
                    """
                    
            if content:
                self._cache_content(url, content)
                return content
                
        except Exception as e:
            # self.logger.error(f"Error fetching from URL {url}: {str(e)}")
            return None
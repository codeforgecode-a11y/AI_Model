#!/usr/bin/env python3
"""
Web Search Integration for AugmentCode Knowledge System

Provides integration between the intelligent knowledge system and
available web search capabilities.
"""

import logging
import json
import requests
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


class WebSearchProvider:
    """Provides web search functionality for the knowledge acquisition system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the web search provider."""
        self.config = config or {}
        self.search_engines = {
            'duckduckgo': self._search_duckduckgo,
            'google_custom': self._search_google_custom,
            'bing': self._search_bing
        }
        self.default_engine = self.config.get('default_engine', 'duckduckgo')
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform web search using the configured search engine.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results with title, url, and snippet
        """
        try:
            engine = self.config.get('search_engine', self.default_engine)
            search_function = self.search_engines.get(engine, self._search_duckduckgo)
            
            results = search_function(query, num_results)
            
            # Standardize result format
            standardized_results = []
            for result in results:
                standardized_result = {
                    'title': result.get('title', 'No Title'),
                    'url': result.get('url', ''),
                    'snippet': result.get('snippet', result.get('description', 'No description available')),
                    'source': self._extract_domain(result.get('url', '')),
                    'engine': engine
                }
                standardized_results.append(standardized_result)
            
            logger.info(f"Web search completed: {len(standardized_results)} results for '{query}'")
            return standardized_results
            
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            return []
    
    def _search_duckduckgo(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo Instant Answer API."""
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Extract abstract if available
            if data.get('Abstract'):
                results.append({
                    'title': data.get('AbstractText', query),
                    'url': data.get('AbstractURL', ''),
                    'snippet': data.get('Abstract', ''),
                    'source': 'duckduckgo'
                })
            
            # Extract related topics
            for topic in data.get('RelatedTopics', [])[:num_results-1]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('Text', '')[:100] + '...',
                        'url': topic.get('FirstURL', ''),
                        'snippet': topic.get('Text', ''),
                        'source': 'duckduckgo'
                    })
            
            return results[:num_results]
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    def _search_google_custom(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API (requires API key)."""
        try:
            api_key = self.config.get('google_api_key')
            search_engine_id = self.config.get('google_search_engine_id')
            
            if not api_key or not search_engine_id:
                logger.warning("Google Custom Search API credentials not configured")
                return []
            
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': api_key,
                'cx': search_engine_id,
                'q': query,
                'num': min(num_results, 10)
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'google'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Google Custom Search failed: {e}")
            return []
    
    def _search_bing(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using Bing Search API (requires API key)."""
        try:
            api_key = self.config.get('bing_api_key')
            
            if not api_key:
                logger.warning("Bing Search API key not configured")
                return []
            
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {'Ocp-Apim-Subscription-Key': api_key}
            params = {
                'q': query,
                'count': min(num_results, 50),
                'responseFilter': 'Webpages'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('webPages', {}).get('value', []):
                results.append({
                    'title': item.get('name', ''),
                    'url': item.get('url', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'bing'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Bing search failed: {e}")
            return []
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace('www.', '')
        except:
            return 'unknown'


def create_web_search_function(config: Dict[str, Any] = None):
    """
    Create a web search function that can be used by the knowledge system.
    
    Args:
        config: Configuration for web search provider
        
    Returns:
        Function that performs web search
    """
    provider = WebSearchProvider(config)
    
    def web_search_function(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Web search function compatible with the knowledge system.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        return provider.search(query, num_results)
    
    return web_search_function


# Example configuration for different search engines
SEARCH_CONFIG_EXAMPLES = {
    'duckduckgo_only': {
        'search_engine': 'duckduckgo'
    },
    'google_custom': {
        'search_engine': 'google_custom',
        'google_api_key': 'your_google_api_key_here',
        'google_search_engine_id': 'your_search_engine_id_here'
    },
    'bing': {
        'search_engine': 'bing',
        'bing_api_key': 'your_bing_api_key_here'
    }
}


if __name__ == "__main__":
    # Test the web search functionality
    config = SEARCH_CONFIG_EXAMPLES['duckduckgo_only']
    search_func = create_web_search_function(config)
    
    test_query = "Python programming tutorial"
    results = search_func(test_query, 3)
    
    print(f"Search results for '{test_query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Snippet: {result['snippet'][:100]}...")
        print()

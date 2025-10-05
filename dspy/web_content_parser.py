#!/usr/bin/env python3
"""
Web Content Parser Script
Extracts text content from web pages given URLs
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebContentParser:
    def __init__(self, delay=1, timeout=10):
        """
        Initialize the web content parser
        
        Args:
            delay (int): Delay between requests in seconds
            timeout (int): Request timeout in seconds
        """
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def clean_text(self, text):
        """
        Clean extracted text by removing extra whitespace and special characters
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/]', '', text)
        
        return text
    
    def extract_content(self, url):
        """
        Extract text content from a web page
        
        Args:
            url (str): URL to parse
            
        Returns:
            dict: Dictionary containing url and content_string
        """
        result = {
            'url': url,
            'content_string': '',
            'status': 'success',
            'error': None
        }
        
        try:
            logger.info(f"Fetching content from: {url}")
            
            # Make the request
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text content
            # Try to get main content areas first
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main', re.I))
            
            if main_content:
                content_text = main_content.get_text()
            else:
                # Fall back to body content
                content_text = soup.get_text()
            
            # Clean the extracted text
            result['content_string'] = self.clean_text(content_text)
            
            logger.info(f"Successfully extracted {len(result['content_string'])} characters from {url}")
            
        except requests.exceptions.RequestException as e:
            result['status'] = 'error'
            result['error'] = f"Request error: {str(e)}"
            logger.error(f"Request error for {url}: {e}")
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = f"Parsing error: {str(e)}"
            logger.error(f"Parsing error for {url}: {e}")
        
        # Add delay to be respectful to the server
        time.sleep(self.delay)
        
        return result
    
    def parse_multiple_urls(self, urls):
        """
        Parse multiple URLs and return results
        
        Args:
            urls (list): List of URLs to parse
            
        Returns:
            list: List of dictionaries with results
        """
        results = []
        
        for i, url in enumerate(urls, 1):
            logger.info(f"Processing URL {i}/{len(urls)}: {url}")
            result = self.extract_content(url)
            results.append(result)
        
        return results

def main():
    """
    Main function to demonstrate usage
    """
    # Initialize parser
    parser = WebContentParser(delay=1, timeout=15)
    
    # Test URL
    test_url = "https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.contractSpecs.options.html"
    
    print("=" * 80)
    print("WEB CONTENT PARSER")
    print("=" * 80)
    
    # Parse single URL
    print(f"\nParsing URL: {test_url}")
    result = parser.extract_content(test_url)
    
    print(f"\nResults:")
    print(f"URL: {result['url']}")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        content_preview = result['content_string'][:500] + "..." if len(result['content_string']) > 500 else result['content_string']
        print(f"Content Length: {len(result['content_string'])} characters")
        print(f"Content Preview: {content_preview}")
        
        # Save to file
        output_file = 'extracted_content.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"URL: {result['url']}\n")
            f.write(f"Content Length: {len(result['content_string'])} characters\n")
            f.write("=" * 80 + "\n")
            f.write(result['content_string'])
        
        print(f"\nContent saved to: {output_file}")
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame([{
            'url': result['url'],
            'content_string': result['content_string']
        }])
        
        csv_file = 'extracted_content.csv'
        df.to_csv(csv_file, index=False)
        print(f"Data saved to CSV: {csv_file}")
        
    else:
        print(f"Error: {result['error']}")

def parse_urls_from_list(urls_list):
    """
    Parse multiple URLs from a list
    
    Args:
        urls_list (list): List of URLs to parse
        
    Returns:
        pandas.DataFrame: DataFrame with url and content_string columns
    """
    parser = WebContentParser(delay=1, timeout=15)
    results = parser.parse_multiple_urls(urls_list)
    
    # Convert to DataFrame
    df_data = []
    for result in results:
        df_data.append({
            'url': result['url'],
            'content_string': result['content_string'] if result['status'] == 'success' else '',
            'status': result['status'],
            'error': result['error']
        })
    
    df = pd.DataFrame(df_data)
    return df

def parse_urls_from_csv(csv_file, url_column='url'):
    """
    Parse URLs from a CSV file
    
    Args:
        csv_file (str): Path to CSV file containing URLs
        url_column (str): Name of column containing URLs
        
    Returns:
        pandas.DataFrame: DataFrame with url and content_string columns
    """
    # Read URLs from CSV
    urls_df = pd.read_csv(csv_file)
    urls_list = urls_df[url_column].dropna().tolist()
    
    print(f"Found {len(urls_list)} URLs to parse")
    
    # Parse URLs
    results_df = parse_urls_from_list(urls_list)
    
    # Save results
    output_file = 'parsed_content_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    return results_df

if __name__ == "__main__":
    main()

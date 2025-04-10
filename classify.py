#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
News Article Scraper and Classifier
-----------------------------------
This module scrapes news articles from various websites and 
classifies them into categories using NLP techniques.
"""

import requests
import time
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download NLTK resources if not already present
def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    for resource in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)


# Constants
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'
}


class WebScraper(ABC):
    """Abstract base class for website scrapers."""
    
    @abstractmethod
    def scrape(self, url: str) -> List[str]:
        """Scrape content from the given URL."""
        pass
    
    def _get_soup(self, url: str) -> BeautifulSoup:
        """Get BeautifulSoup object from URL."""
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
            return BeautifulSoup(response.content, 'html.parser')
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return BeautifulSoup("", 'html.parser')  # Return empty soup on error


class NYTimesScraper(WebScraper):
    """Scraper for New York Times articles."""
    
    def scrape(self, url: str) -> List[str]:
        """Scrape content from NYTimes article."""
        soup = self._get_soup(url)
        article_content = []
        
        for div in soup.find_all('div', class_="StoryBodyCompanionColumn"):
            for paragraph in div.find_all('p'):
                content = paragraph.get_text().strip()
                if content:  # Only add non-empty paragraphs
                    article_content.append(content)
        
        return article_content


class GuardianScraper(WebScraper):
    """Scraper for The Guardian articles."""
    
    def scrape(self, url: str) -> List[str]:
        """Scrape content from Guardian article."""
        soup = self._get_soup(url)
        article_content = []
        
        for paragraph in soup.find_all('p', class_="dcr-n6w1lc"):
            content = paragraph.get_text().strip()
            if content:  # Only add non-empty paragraphs
                article_content.append(content)
        
        return article_content


class ScraperFactory:
    """Factory for creating appropriate scrapers based on URL."""
    
    @staticmethod
    def get_scraper(url: str) -> WebScraper:
        """Return appropriate scraper based on URL."""
        if "nytimes.com" in url:
            return NYTimesScraper()
        elif "theguardian.com" in url:
            return GuardianScraper()
        else:
            raise ValueError(f"No scraper available for URL: {url}")


class TextProcessor:
    """Class for processing text data."""
    
    def __init__(self):
        download_nltk_resources()
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuations = r".,\"-\\/#!?$%\^&\*;:{}=\-_'~()"
    
    def clean_text(self, text: str, min_word_length: int = 4) -> str:
        """Clean and preprocess text."""
        # Convert to lowercase
        text = text.lower() if isinstance(text, str) else ""
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens
        filtered_tokens = [
            token for token in tokens 
            if len(token) > min_word_length 
            and token not in self.stopwords 
            and token not in self.punctuations
        ]
        
        # Lemmatize tokens
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        
        # Join tokens back into text
        return " ".join(lemmatized_tokens)
    
    def process_article(self, article_parts: List[str]) -> str:
        """Process a list of article paragraphs into cleaned text."""
        full_text = " ".join(article_parts)
        return self.clean_text(full_text)


class NewsClassifier:
    """Class for classifying news articles."""
    
    def __init__(self, vectorizer_type: str = 'tfidf'):
        """Initialize classifier with vectorizer type (count or tfidf)."""
        if vectorizer_type.lower() == 'count':
            self.vectorizer = CountVectorizer()
        elif vectorizer_type.lower() == 'tfidf':
            self.vectorizer = TfidfVectorizer()  # TF-IDF often performs better than Count
        else:
            raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")
            
        self.model = MultinomialNB()
        self.is_trained = False
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """Train the classifier and return performance metrics."""
        # Prepare data
        X = self.vectorizer.fit_transform(df['Text'])
        y = df['Category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred) * 100,
            'precision': precision_score(y_test, y_pred, average='weighted') * 100,
            'recall': recall_score(y_test, y_pred, average='weighted') * 100,
            'f1': f1_score(y_test, y_pred, average='weighted') * 100
        }
        
        return metrics
    
    def predict(self, text: str) -> str:
        """Predict category for text."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
            
        # Vectorize text
        text_vectorized = self.vectorizer.transform([text])
        
        # Make prediction
        return self.model.predict(text_vectorized)[0]


class NewsAnalyzer:
    """Main application class that combines scraping, processing, and classification."""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.classifier = NewsClassifier(vectorizer_type='tfidf')
    
    def analyze_url(self, url: str) -> Dict[str, Any]:
        """Analyze news article from URL."""
        # Get appropriate scraper
        try:
            scraper = ScraperFactory.get_scraper(url)
        except ValueError as e:
            return {'error': str(e)}
        
        # Scrape content
        article_parts = scraper.scrape(url)
        if not article_parts:
            return {'error': 'Failed to scrape article content'}
        
        # Process text
        processed_text = self.text_processor.process_article(article_parts)
        
        # Classify (if model is trained)
        result = {
            'raw_text': ' '.join(article_parts),
            'processed_text': processed_text,
            'paragraph_count': len(article_parts)
        }
        
        if self.classifier.is_trained:
            result['category'] = self.classifier.predict(processed_text)
        
        return result
    
    def train_classifier(self, dataset_path: str) -> Dict[str, float]:
        """Train classifier with dataset."""
        # Load dataset
        try:
            df = pd.read_csv(dataset_path)
            # Remove unnecessary columns
            if 'ArticleId' in df.columns:
                df.drop("ArticleId", axis=1, inplace=True)
            
            # Process text in dataset
            print("Processing dataset text...")
            df['Text'] = df['Text'].apply(self.text_processor.clean_text)
            
            # Train classifier
            print("Training classifier...")
            metrics = self.classifier.train(df)
            
            return metrics
            
        except Exception as e:
            return {'error': f"Failed to train classifier: {str(e)}"}


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='News Article Scraper and Classifier')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train classifier command
    train_parser = subparsers.add_parser('train', help='Train the classifier')
    train_parser.add_argument('dataset', help='Path to training dataset CSV file')
    
    # Analyze URL command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a news article')
    analyze_parser.add_argument('url', help='URL of the news article to analyze')
    analyze_parser.add_argument('--model', help='Path to trained model file (if available)')
    
    args = parser.parse_args()
    
    analyzer = NewsAnalyzer()
    
    if args.command == 'train':
        print(f"Training classifier with dataset: {args.dataset}")
        metrics = analyzer.train_classifier(args.dataset)
        
        if 'error' in metrics:
            print(f"Error: {metrics['error']}")
        else:
            print("\nTraining results:")
            for metric, value in metrics.items():
                print(f"{metric.capitalize()}: {value:.2f}%")
            
    elif args.command == 'analyze':
        print(f"Analyzing article at: {args.url}")
        result = analyzer.analyze_url(args.url)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print("\nArticle Analysis:")
            print(f"Category: {result.get('category', 'Not classified')}")
            print(f"Paragraph count: {result['paragraph_count']}")
            print("\nProcessed text sample (first 200 chars):")
            print(result['processed_text'][:200] + "...")
    else:
        parser.print_help()


if __name__ == '__main__':
    main() 
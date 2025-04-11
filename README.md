# News Article Classifier

A Python tool that scrapes news articles from websites and classifies them into categories using Natural Language Processing techniques.

## Features

- Web scraping from multiple news sources (currently supports NY Times and The Guardian)
- Text preprocessing with NLTK (tokenization, stopword removal, lemmatization)
- News article classification using Naive Bayes
- Command-line interface for training and analyzing articles

## Requirements

- Python 3.6+
- BeautifulSoup4
- NLTK
- Pandas
- Scikit-learn
- Requests

## Installation

```bash
# Clone the repository
git clone https://github.com/blackaly/classify
cd classify

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Train the classifier:
```bash
python classify.py train /path/to/training_data.csv
```

Analyze a news article:
```bash
python classify.py analyze https://www.nytimes.com/path/to/article.html
```

### As a Module

```python
from classify import NewsAnalyzer

analyzer = NewsAnalyzer()

# Train classifier
metrics = analyzer.train_classifier("/path/to/training_data.csv")
print(f"Accuracy: {metrics['accuracy']:.2f}%")

# Analyze article
result = analyzer.analyze_url("https://www.nytimes.com/path/to/article.html")
print(f"Category: {result['category']}")
```

## Improvements in the Refactored Code

1. **Modular Architecture**
   - Separated scraping, text processing, and classification into distinct classes
   - Used object-oriented programming principles (inheritance, abstraction)
   - Implemented the Factory pattern for scraper creation

2. **Error Handling**
   - Added robust error handling throughout the code
   - Graceful degradation when network requests fail

3. **Performance Improvements**
   - Added TF-IDF vectorization option (often performs better than CountVectorizer)
   - Optimized text processing pipeline
   - Added type hints for better IDE support and code reliability

4. **New Features**
   - Command-line interface for easy usage
   - Extensible architecture for adding new scrapers
   - Improved results formatting

5. **Code Readability**
   - Comprehensive docstrings
   - Consistent code style
   - Logical organization of functions and classes

## License

MIT License.

# spaCy NLP API

A FastAPI-based REST API wrapper for spaCy, providing endpoints for common Natural Language Processing (NLP) tasks.

## Features

- Named Entity Recognition (NER)
- Part of Speech (POS) Tagging
- Text Similarity Analysis
- Basic Text Analysis (tokenization, lemmatization, sentence segmentation)
- Dependency Parsing

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Install spaCy's English language model:

```bash
python -m spacy download en_core_web_sm
```

3. Run the API:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive API documentation.

### Endpoints

#### POST /ner

Extract named entities from text.

#### POST /pos

Get part-of-speech tags for each token in the text.

#### POST /similarity

Calculate similarity between two texts.

#### POST /basic-analysis

Perform basic text analysis including tokenization and lemmatization.

#### POST /dependency-parse

Generate dependency parse information for the input text.

## Example Usage

```python
import requests

# NER Analysis
response = requests.post(
    "http://localhost:8000/ner",
    json={"text": "Apple is looking at buying U.K. startup for $1 billion"}
)
print(response.json())

# POS Tagging
response = requests.post(
    "http://localhost:8000/pos",
    json={"text": "The quick brown fox jumps over the lazy dog"}
)
print(response.json())

# Text Similarity
response = requests.post(
    "http://localhost:8000/similarity",
    json={"texts": ["The quick brown fox", "The fast brown fox"]}
)
print(response.json())
```

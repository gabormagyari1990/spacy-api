from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import spacy
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="spaCy NLP API",
    description="API for common Natural Language Processing tasks using spaCy",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "Spacy model 'en_core_web_sm' not found. Please install it by running: python -m spacy download en_core_web_sm"
    )

# Request/Response Models
class TextRequest(BaseModel):
    text: str

class TextsRequest(BaseModel):
    texts: List[str]

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int

class Token(BaseModel):
    text: str
    pos: str
    tag: str
    dep: str
    lemma: str

class NERResponse(BaseModel):
    entities: List[Entity]

class POSResponse(BaseModel):
    tokens: List[Token]

class SimilarityResponse(BaseModel):
    similarity: float

@app.post("/ner", response_model=NERResponse, tags=["NLP"])
async def named_entity_recognition(request: TextRequest):
    """
    Extract named entities from the input text.
    Returns entities with their labels and positions.
    """
    doc = nlp(request.text)
    entities = [
        Entity(
            text=ent.text,
            label=ent.label_,
            start=ent.start_char,
            end=ent.end_char
        )
        for ent in doc.ents
    ]
    return NERResponse(entities=entities)

@app.post("/pos", response_model=POSResponse, tags=["NLP"])
async def parts_of_speech(request: TextRequest):
    """
    Analyze parts of speech in the input text.
    Returns tokens with their POS tags, dependencies, and lemmas.
    """
    doc = nlp(request.text)
    tokens = [
        Token(
            text=token.text,
            pos=token.pos_,
            tag=token.tag_,
            dep=token.dep_,
            lemma=token.lemma_
        )
        for token in doc
    ]
    return POSResponse(tokens=tokens)

@app.post("/similarity", response_model=SimilarityResponse, tags=["NLP"])
async def text_similarity(request: TextsRequest):
    """
    Calculate similarity between two texts.
    Requires exactly two texts in the input list.
    """
    if len(request.texts) != 2:
        raise HTTPException(
            status_code=400,
            detail="Exactly two texts must be provided for similarity comparison"
        )
    
    doc1 = nlp(request.texts[0])
    doc2 = nlp(request.texts[1])
    similarity = doc1.similarity(doc2)
    
    return SimilarityResponse(similarity=float(similarity))

@app.post("/basic-analysis", tags=["NLP"])
async def basic_analysis(request: TextRequest):
    """
    Perform basic text analysis including:
    - Tokenization
    - Lemmatization
    - Sentence segmentation
    """
    doc = nlp(request.text)
    
    analysis = {
        "tokens": [token.text for token in doc],
        "lemmas": [token.lemma_ for token in doc],
        "sentences": [str(sent) for sent in doc.sents],
        "noun_phrases": [chunk.text for chunk in doc.noun_chunks]
    }
    
    return analysis

@app.post("/dependency-parse", tags=["NLP"])
async def dependency_parse(request: TextRequest):
    """
    Generate dependency parse information for the input text.
    Returns token relationships and syntactic dependencies.
    """
    doc = nlp(request.text)
    
    deps = [
        {
            "token": token.text,
            "dep": token.dep_,
            "head": token.head.text,
            "children": [child.text for child in token.children]
        }
        for token in doc
    ]
    
    return {"dependencies": deps}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

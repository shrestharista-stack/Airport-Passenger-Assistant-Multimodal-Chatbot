# text preprocessing- preparing text by normalizing text, semantic embeddings

import json
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#defining list of stop words
STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been",
    "i", "you", "he", "she", "it", "we", "they", "my", "your",
    "do", "does", "did", "can", "could", "will", "would", "should",
    "to", "of", "in", "on", "at", "for", "and", "or", "there",
}

# gate like b12 or B12 (letter + 1 to 3 digits)
GATE_PATTERN = re.compile(r"\b([a-z]\d{1,3})\b")
TERMINAL_PATTERN = re.compile(r"terminal\s?(\d)")

#defining function to convert all words into strings and lower case, remove punctuations, and removes spaces at the start and end of word as well as multiple spaces in between. This cleans the texts.
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

#defining function to convert texts into individual word tokes.
def tokenize(text):
    cleaned = clean_text(text)
    tokens = cleaned.split()
    return tokens

#defining function to remove stopwords as listed in STOPWORDS
def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]

#defining function to identify and extract entities from passengers queries
def extract_entities(text):
    cleaned = clean_text(text)
    entities = {}
    gate_match = GATE_PATTERN.search(cleaned)
    if gate_match:
        entities["gate_number"] = gate_match.group(1).upper()
    terminal_match = TERMINAL_PATTERN.search(cleaned)
    if terminal_match:
        entities["terminal_number"] = terminal_match.group(1)
    return entities

#defining function to split data into train and validation data
def split_data(queries, test_size=0.2, random_state=42):
    intents = [q["intent"] for q in queries]
    try:
        train_data, val_data = train_test_split(
            queries,
            test_size=test_size,
            random_state=random_state,
            stratify=intents,
        )
    except ValueError:
        print("Stratify did not work since there are too little examples for some intents. Hence, normal split is performed.")
        train_data, val_data = train_test_split(
            queries,
            test_size=test_size,
            random_state=random_state,
        )
    return train_data, val_data

#defining function to convert text into vectors to perform embeddings
def tfidf_embeddings(texts):
    cleaned = [clean_text(t) for t in texts]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(cleaned)
    return vectorizer, matrix

#defining function to conduct semantic embedding using allMiniLM to capture word similarity
def minilm_embeddings(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    cleaned = [clean_text(t) for t in texts]
    vectors = model.encode(cleaned, convert_to_numpy=True)
    return vectors

#defining function to load passenger queries json file
def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

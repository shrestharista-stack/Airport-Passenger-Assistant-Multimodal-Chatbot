# MiniLM + FAISS text search over the airport knowledge base
# typed text and Whisper transcripts both use clean_text()

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from text_pipeline import clean_text, extract_entities


class TextSearcher:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        kb_path = os.path.join(data_dir, "kb", "airport_kb.json")
        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)

        self.docs = []
        for rec in self.kb:
            text = rec["name"] + " " + rec["category"] + " " + rec["description"] + " " + rec["direction_text"]
            self.docs.append(text)

        print("Loading MiniLM all-MiniLM-L6-v2")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        cleaned = [clean_text(t) for t in self.docs]
        vectors = self.model.encode(cleaned, convert_to_numpy=True)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        self.vectors = vectors.astype("float32")
        d = self.vectors.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.vectors)
        print("Text FAISS size:", self.index.ntotal)
        print("KB records:", len(self.kb))

    def encode_query(self, query):
        q = clean_text(query)
        vec = self.model.encode([q], convert_to_numpy=True)
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        return vec.astype("float32")

    def gate_record(self, entities):
        if "gate_number" not in entities:
            return None
        gate = entities["gate_number"]
        for rec in self.kb:
            blob = (rec["id"] + " " + rec["name"]).upper()
            if gate.upper() in blob:
                return rec
        return None

    def search_text(self, query, k=3, category=None):
        entities = extract_entities(query)
        gate_rec = self.gate_record(entities)
        vec = self.encode_query(query)

        if gate_rec is not None:
            i = self.kb.index(gate_rec)
            score = float(np.dot(vec[0], self.vectors[i]))
            top = [{"name": gate_rec["name"], "score": score, "category": gate_rec["category"]}]
            return gate_rec, score, entities, top

        if category is None:
            scores, idx = self.index.search(vec, k)
            i = int(idx[0][0])
            top = []
            for rank in range(len(idx[0])):
                j = int(idx[0][rank])
                top.append({
                    "name": self.kb[j]["name"],
                    "category": self.kb[j]["category"],
                    "score": float(scores[0][rank]),
                })
            return self.kb[i], float(scores[0][0]), entities, top

        best_i = None
        best_score = -1.0
        top = []
        for i, rec in enumerate(self.kb):
            if rec["category"] != category:
                continue
            score = float(np.dot(vec[0], self.vectors[i]))
            top.append({"name": rec["name"], "category": rec["category"], "score": score})
            if score > best_score:
                best_score = score
                best_i = i
        top = sorted(top, key=lambda x: x["score"], reverse=True)[:k]
        if best_i is None:
            print("No KB rows in category", category, "- searching all records")
            return self.search_text(query, k=k, category=None)
        return self.kb[best_i], best_score, entities, top

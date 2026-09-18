
import os
import csv
import numpy as np
import torch
from PIL import Image
import faiss
import clip

#defining class to handle airport image analysis
class ImageSearcher:
    # Set up the processor and choose the hardware (CPU or GPU)
    def __init__(self, data_dir, device='cuda'):
        #save the hardware choice
        self.device = device
        #store the data directory
        self.data_dir = data_dir
        #creating path to image directory
        self.image_dir = os.path.join(data_dir, "images", "downloaded")
        #creating path to manifest.csv
        self.manifest_path = os.path.join(data_dir, "images", "manifest.csv")

        print("Loading CLIP ViT-B/32 on", device)
        #load the CLIP model and preprocessing function
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        # Set the model to 'evaluation mode' (not training mode)
        self.model.eval()
        self.build_index()

    #defining function to encode images
    def encode_image_file(self, path):
        image = Image.open(path).convert("RGB")
        #applying CLIP preprocessing and prepares image for CLIP model
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        # Tell the model not to learn right now (saves memory)
        with torch.no_grad():
            #converting images into numerical vectors
            features = self.model.encode_image(tensor)
        features = features.float()
        #normalize the vectors
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def build_index(self):
        rows = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        vectors = []
        self.filenames = []
        self.categories = []

        for row in rows:
            path = os.path.join(self.image_dir, row["filename"])
            if not os.path.exists(path):
                print("Missing image (skip):", row["filename"])
                continue
            features = self.encode_image_file(path)
            vectors.append(features.cpu().numpy()[0])
            self.filenames.append(row["filename"])
            self.categories.append(row["category"])

        matrix = np.stack(vectors).astype("float32")
        d = matrix.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(matrix)
        print("Image FAISS size:", self.index.ntotal)

    def search_image(self, path, k=3):
        features = self.encode_image_file(path)
        q = features.cpu().numpy().astype("float32")
        scores, idx = self.index.search(q, k)
        i = int(idx[0][0])
        top = []
        for rank in range(len(idx[0])):
            j = int(idx[0][rank])
            top.append({
                "filename": self.filenames[j],
                "category": self.categories[j],
                "score": float(scores[0][rank]),
            })
        return {
            "filename": self.filenames[i],
            "category": self.categories[i],
            "score": float(scores[0][0]),
            "embedding": features,
            "top_k": top,
        }

    def pick_kb_record(self, kb, category, image_features):
        # several KB rows can share a category (two terminals)
        rows = [r for r in kb if r["category"] == category]
        if len(rows) == 0:
            print("No KB record for category:", category)
            return None
        if len(rows) == 1:
            return rows[0]

        texts = []
        for rec in rows:
            texts.append("a photo of " + rec["name"] + ". " + rec["description"])
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        sims = (text_features @ image_features.T).squeeze(1)
        i = int(torch.argmax(sims).item())
        return rows[i]

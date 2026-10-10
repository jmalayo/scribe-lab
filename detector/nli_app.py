import os

import torch
from fastapi import FastAPI
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ["MODEL_ID"]
REVISION = os.environ["NLI_REVISION"]
BATCH_SIZE = int(os.environ.get("NLI_BATCH_SIZE", "16"))

path = snapshot_download(MODEL_ID, revision=REVISION)
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path).eval()
labels = [
    model.config.id2label[i] 
        for i in range(model.config.num_labels)
]

app = FastAPI()

class Pair(BaseModel):
    premise: str
    hypothesis: str

class NliRequest(BaseModel):
    pairs: list[Pair]

@app.get("/health")
def health():
    return {
        "model": MODEL_ID,
        "revision": REVISION,
        "labels": labels
    }

@app.post("/predict")
def predict(req: NliRequest):

    probs = []

    for i in range(0, len(req.pairs), BATCH_SIZE):
        batch = req.pairs[i:i + BATCH_SIZE]
        enc = tokenizer(
            [
                p.premise 
                    for p in batch
            ],
            [
                p.hypothesis 
                    for p in batch
            ],
            truncation="only_first",
            max_length=512,
            padding=True,
            return_tensors="pt"
        )

        with torch.inference_mode():
            probs.extend(model(**enc).logits.softmax(-1).tolist())

    return {
        "probs": [
            dict(zip(labels, p, strict=True)) 
                for p in probs
        ]
    }

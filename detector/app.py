import os

import torch
from fastapi import FastAPI
from huggingface_hub import snapshot_download
from lettucedetect.models.inference import HallucinationDetector
from pydantic import BaseModel

MODEL_ID = os.environ["MODEL_ID"]
REVISION = os.environ["LETTUCE_REVISION"]
LANG_CODE = os.environ.get("LANG_CODE", "es")

detector = HallucinationDetector(
    method="transformer",
    model_path=snapshot_download(MODEL_ID, revision=REVISION),
    lang=LANG_CODE,
    device=torch.device("cpu"),
    trust_remote_code=True,
)

app = FastAPI()

class DetectRequest(BaseModel):
    context: list[str]
    question: str
    answer: str

@app.get("/health")
def health():
    return {
        "model": MODEL_ID, 
        "revision": REVISION, 
        "lang": LANG_CODE
    }

@app.post("/predict")
def predict(req: DetectRequest):

    if not req.answer.strip():
        return {
            "spans": []
        }

    return {
        "spans": detector.predict(
            context=req.context, 
            question=req.question, 
            answer=req.answer, 
            output_format="spans"
        )
    }

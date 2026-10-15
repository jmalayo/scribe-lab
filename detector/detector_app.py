import os
from typing import Literal

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
    output_format: Literal["spans", "tokens"] = "spans"

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
            req.output_format: []
        }

    result = detector.predict(
        context=req.context,
        question=req.question,
        answer=req.answer,
        output_format=req.output_format
    )

    if req.output_format == "spans":
        return {
            "spans": result
        }

    offsets = detector.detector.tokenizer(
        req.answer, 
        add_special_tokens=False, 
        return_offsets_mapping=True
    )["offset_mapping"]

    return {
        "tokens": [
            {
                "start": start,
                "end": end, 
                "prob": tok["prob"]
            } for tok, (start, end) 
                in zip(result, offsets, strict=False)
        ]
    }

import re

import requests
from pydantic import BaseModel

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡`\"'(*\-•\d])")
# Separa en bloques tras terminar una oración

_BLOCK_RE = re.compile(r"\n\s*\n|\n(?=\s*(?:[-*•]|\d+[.)])\s)") 
# Separa en bloques cuando
# 1. Existe una linea en blanco entre ellos
# 2. Existe elementos listas (inclusive sin linea en blanco)

class NliScores(BaseModel):
    entailment: float
    contradiction: float


class UnsupportedSentence(BaseModel):
    sentence: str
    start: int
    end: int
    score: float
    reasons: list[str]
    numbers: list[str]
    nli: NliScores | None = None


class GroundingResult(BaseModel):
    score: float
    unsupported_sentences: list[UnsupportedSentence]


def split_sentences(answer: str) -> list[str]:

    sentences = []

    for block in _BLOCK_RE.split(answer or ""):
        for sentence in _SENTENCE_END_RE.split(block.strip()):
            sentence = " ".join(sentence.split())
            if sentence:
                sentences.append(sentence)

    return sentences

_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

def sentence_ranges(answer: str, sentences: list[str]) -> list[tuple[int, int]]:

    ranges, pos = [], 0

    for sentence in sentences:
        pattern = r"\s+".join(re.escape(word) for word in sentence.split())
        match = re.compile(pattern).search(answer, pos)

        if match is None:
            raise ValueError(f"oración no encontrada en la respuesta: {sentence!r}")

        ranges.append((match.start(), match.end()))
        pos = match.end()

    return ranges


def sentence_scores(answer: str, tokens: list[dict]) -> list[dict]:

    sentences = split_sentences(answer)
    scores = []

    for sentence, (start, end) in zip(sentences, sentence_ranges(answer, sentences), strict=True):
        probs = [
            t["prob"] 
                for t in tokens 
                    if t["start"] < end and 
                    t["end"] > start and 
                    t["start"] != t["end"]
        ]

        scores.append({
            "sentence": sentence, 
            "start": start, 
            "end": end, 
            "score": max(probs, default=0.0)
        })

    return scores

def _numbers(text: str) -> set[str]:
    return {
        n.replace(",", ".") 
            for n in _NUMBER_RE.findall(text)
    }

def unsupported_numbers(sentence: str, context: str, question: str = "") -> list[str]:

    allowed = _numbers(context) | _numbers(question)
    found = _NUMBER_RE.findall(_LIST_MARKER_RE.sub("", sentence))

    return [
        n for n in found 
        if n.replace(",", ".") not in allowed
    ]

def should_abstain(chunks: list[dict], min_score: float, score_key: str = "rerank_score") -> bool:

    return max(
        (c[score_key] 
            for c in chunks), 
        default=float("-inf")
    ) < min_score

def detect_tokens(
    detector_url: str, 
    chunks: list[str], 
    question: str, 
    answer: str,
    timeout: float = 120
) -> list[dict]:

    resp = requests.post(
        f"{detector_url}/predict", 
        timeout=timeout, 
        json={
            "context": chunks,
            "question": question, 
            "answer": answer,
            "output_format": "tokens"
        }
    )

    resp.raise_for_status()

    return resp.json()["tokens"]

def nli_predictor(model_id: str, revision: str, device: str = "cpu", batch_size: int = 16):

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    
    labels = [
        model.config.id2label[i] 
            for i in range(model.config.num_labels)
    ]

    def predict(pairs: list[tuple[str, str]]) -> list[dict]:
        out = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]

            enc = tokenizer(
                [p for p, _ in batch], 
                [h for _, h in batch], 
                truncation="only_first",
                max_length=512, 
                padding=True, 
                return_tensors="pt"
            ).to(device)

            with torch.inference_mode():
                probs = model(**enc).logits.softmax(-1).cpu().tolist()

            out.extend(dict(zip(labels, p, strict=True)) for p in probs)

        return out

    return predict

def nli_http_predictor(nli_url: str, timeout: float = 300):

    import requests

    def predict(pairs: list[tuple[str, str]]) -> list[dict]:

        resp = requests.post(f"{nli_url}/predict", 
            timeout=timeout, 
            json={
                "pairs": [
                    {
                        "premise": p, 
                        "hypothesis": h
                    } for p, h in pairs
                ]
            }
        )

        resp.raise_for_status()

        return resp.json()["probs"]

    return predict

def nli_scores(sentences: list[str], chunks: list[str], predict) -> list[dict]:

    if not sentences or not chunks:
        return [
            {
                "entailment": 0.0, 
                "contradiction": 0.0
            } for _ in sentences
        ]

    probs = predict(
        [
            (c, s) for s in sentences 
                for c in chunks
        ]
    )
    n = len(chunks)

    return [
        {
            "entailment": max(p["entailment"] for p in probs[i * n:(i + 1) * n]),
            "contradiction": max(p["contradiction"] for p in probs[i * n:(i + 1) * n]),
        }
        for i in range(len(sentences))
    ]

def ground_answer(
    answer: str, 
    chunks: list[str], 
    question: str, 
    tokens: list[dict],
    threshold: float, nli: list[dict] | None = None,
    entailment_threshold: float = 0.5, 
    contradiction_threshold: float = 0.5
) -> dict:

    context = "\n\n".join(chunks)
    scored = sentence_scores(answer, tokens)

    nli = nli or [None] * len(scored)

    unsupported = []

    for s, n in zip(scored, nli, strict=True):
        numbers = unsupported_numbers(s["sentence"], context, question)
        reasons = (["lettucedetect"] if s["score"] > threshold else []) + (["numeric"] if numbers else [])

        if n is not None:
            reasons += ["nli_contradiction"] if n["contradiction"] > contradiction_threshold else []
            reasons += ["nli_not_entailed"] if n["entailment"] <= entailment_threshold else []

        if reasons:
            unsupported.append({
                **s, 
                "reasons": reasons, 
                "numbers": numbers, 
                "nli": n
            })

    return {
        "score": 1 - len(unsupported) / len(scored) if scored else 1.0,
        "unsupported_sentences": unsupported,
    }

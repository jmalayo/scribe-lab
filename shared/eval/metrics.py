from __future__ import annotations

import re
import statistics

from rapidfuzz import fuzz


def _normalize(text: str) -> str:
    text = re.sub(r"(\*\*|__)", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()



def _find_numbers(text: str) -> list[str]:
    
    return [
        re.sub(r"\s+", "", n)
            for n in re.findall(r"\d+\.?\d*\s*%?", text)
    ]

def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)

    return all(n in it for n in needle)

def _span_matches(span: str, chunk_text: str) -> bool:

    fuzzy_threshold = 0.8
    coverage_threshold = 0.9

    alignment = fuzz.partial_ratio_alignment(span, chunk_text)

    if alignment.score / 100 < fuzzy_threshold:
        return False

    coverage = min(
        alignment.src_end - alignment.src_start,
        alignment.dest_end - alignment.dest_start
    ) / len(span)

    if coverage < coverage_threshold:
        return False

    window = chunk_text[alignment.dest_start:alignment.dest_end]

    return _is_subsequence(
        _find_numbers(span),
        _find_numbers(window)
    )

def is_chunk_correct(chunk: dict, question: dict) -> bool:

    valid_docs = [
        d.strip() for d in question["source_doc"].split(";")
    ]

    if chunk.get("doc_id") not in valid_docs:
        return False

    texts = [chunk.get("text", "")]
    
    for t in chunk.get("tables", []):
        texts.append(t.get("text_content", ""))

    chunk_text = _normalize(" ".join(texts))

    return any(
        _span_matches(_normalize(span), chunk_text)
            for span in question["gold_spans"]
    )

def calculate_hit_at_k(results: dict, questions: list, k: int) -> list[float]:

    return [
        1.0 if any(is_chunk_correct(c, q) for c in results[q["id"]][:k]) else 0.0
        for q in questions
    ]

def recall_at_k(results: dict[str, list[dict]], questions: list[dict], k: int) -> float:
    
    hits = 0

    for q in questions:
        retrieved = results.get(q["id"], [])[:k]

        if any(is_chunk_correct(c, q) for c in retrieved):
            hits += 1

    return hits / len(questions) if questions else 0.0

def mrr(results: dict[str, list[dict]], questions: list[dict], k: int | None = None) -> float:
    
    reciprocal_ranks = []

    for q in questions:

        retrieved = results.get(q["id"], [])

        if k is not None:
            retrieved = retrieved[:k]

        rank = next(
            (
                i + 1 for i, c in enumerate(retrieved) 
                    if is_chunk_correct(c, q)
            ),
            None
        )

        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

def groundedness_rate(verdicts: list[bool]) -> float:

    return sum(verdicts) / len(verdicts) if verdicts else 0.0

def hallucination_rate(verdicts: list[bool]) -> float:

    return 1.0 - groundedness_rate(verdicts) if verdicts else 0.0

def percentile(values: list[float], p: float) -> float:

    # interpolación lineal entre los dos vecinos (mismo resultado que np.percentile por defecto);
    # con n par, la p50 es la mediana
    if not values:
        return 0.0

    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(ordered) - 1)

    if f == c:  # freno en el percentil 100
        return ordered[f]

    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)

def latency_summary(latencies_ms: list[float]) -> dict:

    return {
        "p50_ms": round(percentile(latencies_ms, 50), 1),
        "p95_ms": round(percentile(latencies_ms, 95), 1),
        "mean_ms": round(statistics.mean(latencies_ms), 1) if latencies_ms else 0.0,
    }

def bootstrap_ci(values: list[float], n_resamples: int = 1000, ci: float = 0.95, seed: int = 42):

    import random

    if not values:
        return (0.0, 0.0)

    rng = random.Random(seed)
    means = []
    n = len(values)

    for _ in range(n_resamples):
        sample = [
            values[rng.randrange(n)] for _ in range(n)
        ]

        means.append(sum(sample) / n)

    means.sort()

    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = int((1 + ci) / 2 * n_resamples) - 1

    return (round(means[lo_idx], 4), round(means[hi_idx], 4))

def auroc(scores: list[float], positive: list[bool]) -> float:

    pos = [s for s, is_pos in zip(scores, positive, strict=True) if is_pos]
    neg = [s for s, is_pos in zip(scores, positive, strict=True) if not is_pos]

    if not pos or not neg:
        raise ValueError("auroc necesita al menos un positivo y un negativo")

    wins = 0.0

    for p in pos:
        for q in neg:

            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5

    return round(wins / (len(pos) * len(neg)), 3)

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:

    from math import sqrt

    if n == 0:
        return (0.0, 0.0)

    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))

def mcnemar_exact(only_a: int, only_b: int) -> float:

    from math import comb

    n = only_a + only_b

    if n == 0:
        return 1.0

    tail = sum(
        comb(n, k)
            for k in range(min(only_a, only_b) + 1)
    ) / 2 ** n

    return round(min(1.0, 2 * tail), 3)

def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:

    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / n

    expected = sum(
        (labels_a.count(k) / n) * (labels_b.count(k) / n)
            for k in set(labels_a) | set(labels_b)
    )

    if expected == 1:
        return 1.0

    return round((observed - expected) / (1 - expected), 3)

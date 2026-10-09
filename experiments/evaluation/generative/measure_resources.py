import json
import os
import re
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from shared.eval.metrics import percentile

OUT_DIR = Path(__file__).resolve().parent
ANSWERS = REPO_ROOT / "experiments/evaluation/benchmark-models/results/exp_7/llama31_base/base_answers.csv"
SENTENCES = OUT_DIR / "llama_3_1_8_B/labels_sentences.csv"

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
NLI_REVISION = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
LETTUCE_MODEL = "KRLabsOrg/lettucedect-210m-eurobert-es-v1"
QWEN_MODEL = "qwen3:8b"
LLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"
OLLAMA = "http://127.0.0.1:11434"
DETECTOR = "http://127.0.0.1:8100"
DETECTOR_CONTAINER = "rag-system-lab-detector-1"

RELEVANCE_PROMPT = """Pregunta: {question}

Respuesta: {answer}

¿La respuesta aborda la pregunta, sin importar si es correcta? Devuelve JSON con las claves "relevante" (bool) y "motivo" (str)."""

RELEVANCE_SCHEMA = {"type": "object", "required": ["relevante", "motivo"],
                    "properties": {"relevante": {"type": "boolean"}, "motivo": {"type": "string"}}}


def items() -> list[dict]:
    df = pd.read_csv(ANSWERS, sep=";").fillna({"answer": ""})
    sentences = pd.read_csv(SENTENCES, sep=";")
    return [{
        "id": r.id, "question": r.question, "answer": r.answer, "context": r.context,
        "chunks": re.split(r"\n\n(?=\[music-tagger/)", r.context),
        "sentences": sentences[sentences["id"] == r.id].sort_values("oracion_idx")["oracion"].tolist(),
    } for r in df.itertuples()]


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0])


class GpuPeak:

    def __enter__(self):
        self.peak, self._stop = 0, threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, gpu_used_mib())
            time.sleep(0.2)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def row(component, model, device, latencies, load_s, **extra) -> dict:
    lat = sorted(latencies)
    return {
        "component": component, "model": model, "device": device, "n_items": len(lat),
        "load_s": round(load_s, 2), "p50_s": round(percentile(lat, 50), 2),
        "p95_s": round(percentile(lat, 95), 2),
        "max_s": round(lat[-1], 2), "total_s": round(sum(lat), 1), **extra,
    }


def run_nli(device: str) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rss0 = peak_rss_gb()
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(NLI_MODEL, revision=NLI_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, revision=NLI_REVISION).eval().to(device)
    load_s = time.perf_counter() - t0

    latencies = []
    for it in items():
        pairs = [(c, s) for s in it["sentences"] for c in it["chunks"]]
        t0 = time.perf_counter()
        if pairs:
            enc = tok([p for p, _ in pairs], [h for _, h in pairs], truncation=True, max_length=512,
                      padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                model(**enc).logits.softmax(-1).cpu()
        latencies.append(time.perf_counter() - t0)

    vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    return row("nli_mdeberta", NLI_MODEL, device, latencies, load_s,
               peak_ram_gb=round(peak_rss_gb(), 2), ram_delta_gb=round(peak_rss_gb() - rss0, 2),
               vram_gb=round(vram, 2))


def container_kb(field: str) -> int:
    status = subprocess.run(["docker", "exec", DETECTOR_CONTAINER, "cat", "/proc/1/status"],
                            capture_output=True, text=True, check=True).stdout
    return int(re.search(rf"{field}:\s+(\d+)", status).group(1))


def run_lettuce() -> dict:
    subprocess.run(["docker", "restart", DETECTOR_CONTAINER], capture_output=True, check=True)
    t0 = time.perf_counter()
    while True:
        try:
            requests.get(f"{DETECTOR}/health", timeout=2).raise_for_status()
            break
        except requests.RequestException:
            time.sleep(0.2)
    load_s = time.perf_counter() - t0
    rss0 = container_kb("VmRSS")

    latencies = []
    for it in items():
        t0 = time.perf_counter()
        requests.post(f"{DETECTOR}/predict", timeout=120, json={
            "context": it["chunks"], "question": it["question"], "answer": it["answer"]}).raise_for_status()
        latencies.append(time.perf_counter() - t0)

    peak = container_kb("VmHWM")
    return row("lettucedetect", LETTUCE_MODEL, "cpu", latencies, load_s,
               peak_ram_gb=round(peak / 1024 / 1024, 2), ram_delta_gb=round((peak - rss0) / 1024 / 1024, 2), vram_gb=0.0)


def ollama(model: str, prompt: str, options: dict, **kwargs) -> dict:
    resp = requests.post(f"{OLLAMA}/api/generate", timeout=600, json={
        "model": model, "prompt": prompt, "stream": False, "options": options, **kwargs})
    resp.raise_for_status()
    return resp.json()


def run_ollama(component: str, model: str, build_prompt, options: dict, **kwargs) -> dict:
    for m in requests.get(f"{OLLAMA}/api/ps", timeout=10).json().get("models", []):
        ollama(m["name"], "", {}, keep_alive=0)
    time.sleep(3)
    load_s = ollama(model, "hola", {**options, "num_predict": 1}).get("load_duration", 0) / 1e9

    latencies = []
    with GpuPeak() as gpu:
        for it in items():
            t0 = time.perf_counter()
            ollama(model, build_prompt(it), options, **kwargs)
            latencies.append(time.perf_counter() - t0)
        vram = next((m.get("size_vram", 0) / 1e9 for m in requests.get(f"{OLLAMA}/api/ps", timeout=10).json()["models"]
                     if m["name"] == model), 0.0)

    return row(component, model, "gpu", latencies, load_s,
               vram_gb=round(vram, 2), vram_peak_nvidia_smi_gb=round(gpu.peak / 1024, 2))


def run_qwen(think: bool) -> dict:
    kwargs = {"think": True} if think else {"think": False, "format": RELEVANCE_SCHEMA}
    return run_ollama("qwen3_relevance_think" if think else "qwen3_relevance_json", QWEN_MODEL,
                      lambda it: RELEVANCE_PROMPT.format(question=it["question"], answer=it["answer"] or "(vacía)"),
                      {"temperature": 0.0, "num_ctx": 4096, "num_predict": 3000}, **kwargs)


def run_llama() -> dict:
    from experiments.evaluation.prompts import ANSWER_PROMPT

    return run_ollama("llama31_generation", LLAMA_MODEL,
                      lambda it: ANSWER_PROMPT.format(context=it["context"], question=it["question"]),
                      {"temperature": 0.0, "num_ctx": 4096, "num_thread": 8, "num_predict": 500, "repeat_penalty": 1.0})


RUNNERS = {
    "nli_cpu": lambda: run_nli("cpu"),
    "nli_cuda": lambda: run_nli("cuda"),
    "lettuce": run_lettuce,
    "qwen3_think": lambda: run_qwen(True),
    "qwen3_json": lambda: run_qwen(False),
    "llama31": run_llama,
}

RUN = list(RUNNERS)

COLUMNS = ["component", "model", "device", "n_items", "load_s", "p50_s", "p95_s", "max_s", "total_s",
           "peak_ram_gb", "ram_delta_gb", "vram_gb", "vram_peak_nvidia_smi_gb"]


def main():
    if len(sys.argv) > 1:
        print("RESULT " + json.dumps(RUNNERS[sys.argv[1]]()))
        return

    rows = []
    for name in RUN:
        out = subprocess.run([sys.executable, __file__, name], capture_output=True, text=True, check=False)
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            raise RuntimeError(f"{name} falló:\n{out.stderr[-3000:]}")
        rows.append(json.loads(line[len("RESULT "):]))
        print(rows[-1], flush=True)

    path = OUT_DIR / "resources_summary.csv"
    table = pd.read_csv(path, sep=";").to_dict("records") if path.exists() else []
    for r in rows:
        i = next((i for i, t in enumerate(table) if (t["component"], t["device"]) == (r["component"], r["device"])), None)
        if i is None:
            table.append(r)
        else:
            table[i] = r
    pd.DataFrame(table).reindex(columns=COLUMNS).to_csv(path, index=False, sep=";")


if __name__ == "__main__":
    main()

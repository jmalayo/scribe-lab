from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from shared.settings import settings


def load_questions(path: str | None = None) -> list[dict]:

    root = Path(path or settings.questions_path)

    questions = [
        json.loads(line)
            for file in sorted(root.glob("*.jsonl"))
                for line in file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
    ]

    return sorted(questions, key=lambda q: q["id"])

def load_answers(path: str | Path, chunks_path: str | Path | None = None) -> pd.DataFrame:

    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", keep_default_na=False)

    if chunks_path is None:
        return df

    contexts = {id_: "\n\n".join(chunks) for id_, chunks in load_chunks(chunks_path).items()}

    return df.assign(context=df["id"].map(contexts))


def load_chunks(path: str | Path) -> dict[str, list[str]]:

    df = pd.read_csv(path, sep=";", encoding="utf-8", keep_default_na=False)

    return df.groupby("id", sort=False)["context"].agg(list).to_dict()
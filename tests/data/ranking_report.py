import pandas as pd

from shared.eval.data import load_questions
from shared.eval.metrics import is_chunk_correct
from shared.ingest import build_qdrant_client, qualified_collection_name
from shared.retrieval import dense_search

COLLECTION = qualified_collection_name("exp_chunking_dynamic_tables")
K = 100
TARGET_K = 5

def find_rank(chunks: list[dict], question: dict) -> int | None:

    for position, chunk in enumerate(chunks, start=1):

        if is_chunk_correct(chunk, question):
            return position

    return None

def main():

    client = build_qdrant_client()
    questions = load_questions()

    rows = []

    for question in questions:

        chunks = dense_search(client, COLLECTION, question["question"], K)
        rank = find_rank(chunks, question)

        rows.append({
            "id": question["id"],
            "rank": rank if rank is not None else "fuera de rango",
        })

    df = pd.DataFrame(rows)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 200)

    print(f"Colección: {COLLECTION} | top-{K}\n")
    print(df[["id", "rank"]].to_string(index=False))

    print()

    out_of_range = df["rank"].isna().sum()
    in_scope = df[
       df["rank"].apply(lambda x: isinstance(x, int) and x <= TARGET_K)
    ]

    print(f"fuera de rango (> {K}): {out_of_range}/{len(df)}")
    print(f"dentro del rango objetivo (< {TARGET_K}): {len(in_scope)}")

if __name__ == "__main__":
    main()

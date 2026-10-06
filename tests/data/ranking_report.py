import pandas as pd

from shared.eval.data import load_questions
from shared.eval.metrics import is_chunk_correct
from shared.ingest import build_qdrant_client, qualified_collection_name
from shared.retrieval import dense_search

COLLECTION = qualified_collection_name("exp_chunking_dynamic_tables")
K = 100
RECALL_KS = [5, 10, 15, 20, 100]


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
            "rank": rank,
            "posicion": rank if rank is not None else "fuera de rango",
            "pregunta": question["question"][:80],
        })

    df = pd.DataFrame(rows)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 200)

    print(f"Colección: {COLLECTION} | top-{K}\n")
    print(df[["id", "posicion", "pregunta"]].to_string(index=False))

    print()

    for k in RECALL_KS:

        hits = df["rank"].notna() & (df["rank"] <= k)
        print(f"recall@{k}: {hits.mean():.3f} ({hits.sum()}/{len(df)})")

    out_of_range = df["rank"].isna().sum()
    print(f"fuera de rango (> {K}): {out_of_range}/{len(df)}")


if __name__ == "__main__":
    main()

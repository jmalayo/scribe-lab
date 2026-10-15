import logging
import sys

from shared.ingest import (
    build_qdrant_client,
    chunk_documents,
    get_tables_from_docs,
    load_corpus,
    sync_index,
)
from shared.pipeline import PipelineConfig
from shared.settings import settings

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("app.reindex")


def main() -> None:

    config = PipelineConfig.from_yaml(settings.pipeline_config_path)
    client = build_qdrant_client()

    docs = load_corpus(config.corpus_dir)

    if not docs:
        raise SystemExit(f"{config.corpus_dir}: no hay documentos .md")

    tables = [
        t for doc in docs 
                for t in get_tables_from_docs(doc)[0]
    ]
    
    chunks = chunk_documents(docs, tables, config.chunk_size, config.chunk_overlap)

    result = sync_index(
        client,
        chunks,
        alias=config.collection,
        config={
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "embedding_revision": settings.embedding_revision,
        },
    )

    logger.info("alias=%s docs=%s chunks=%s result=%s", config.collection, len(docs), len(chunks), result)


if __name__ == "__main__":
    main()

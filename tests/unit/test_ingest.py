import logging
import uuid

import pytest

import shared.ingest as ingest
from shared.ingest import (
    SourceDoc,
    chunk_documents,
    corpus_hash,
    get_current_config,
    load_corpus,
    qualified_collection_name,
    validate_chunk,
)

DOC_ID = "music-tagger/benchmark.md"
TABLE_RF = "| Modelo | Acc |\n|---|---|\n| RF | 48.9 % |"
TABLE_KERNEL = "| Kernel | Spearman |\n|---|---|\n| 9 | 0.888 |"
LONG_WORD = "x" * 60
PARAGRAPH = (
    "El pipeline de producción corre sin HPSS y extrae las noventa y seis canciones en unos tres minutos. "
    "Random Forest alcanza un balanced accuracy de cuarenta y ocho coma nueve por ciento con validación cruzada. "
)

# ~1 500 caracteres con dos tablas: con los chunk_size reales del barrido se generan entre 4 y 14 cortes
LONG_TEXT = PARAGRAPH * 3 + f"\n\n{TABLE_RF}\n\n" + PARAGRAPH * 2 + f"\n\n{TABLE_KERNEL}\n\n" + PARAGRAPH * 2
ORIGINAL_GET_EMBEDDER = ingest.get_embedder

class FakeTokenizer:
    # un token por carácter: permite verificar el truncado sin descargar el tokenizer real
    model_max_length = 10_000

    def encode(self, text, add_special_tokens=False, truncation=False):
        return list(text[: self.model_max_length] if truncation else text)

    def decode(self, tokens):
        return "".join(tokens)

class FakeEmbedder:
    tokenizer = FakeTokenizer()
    max_seq_length = 128

class FakePoint:
    def __init__(self, payload: dict):
        self.payload = payload

class FakeQdrant:
    # solo lo que usa get_current_config: si la colección existe y el primer punto
    def __init__(self, points: list | None):
        self._points = points

    def collection_exists(self, name):
        return self._points is not None

    def scroll(self, **kwargs):
        return self._points, None

@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    # mocking - intercepción del modulo ingest
    monkeypatch.setattr(ingest, "get_embedder", lambda: FakeEmbedder())

def doc(text: str, doc_id: str = DOC_ID) -> SourceDoc:
    return SourceDoc(
        doc_id=doc_id,
        library=doc_id.split("/")[0],
        text=text
    )

def table(text_content: str, doc_id: str = DOC_ID) -> dict:
    return {
        "text_content": text_content,
        "doc_id": doc_id,
        "type": "Table"
    }

def attached(chunks) -> list[str]:
    return [
        t["text_content"]
            for c in chunks for t in c.tables
    ]

def sweep_configs() -> list:
    # todas las configs que corrió el barrido de chunking (fases 1 a 3), con su overlap = int(size * frac)
    configs = []
    for size in [128, 256, 277, 284, 512]:
        for frac in [0.0, 0.10, 0.25]:
            overlap = int(size * frac)
            marks = []

            if (size, overlap) == (512, 128):
                marks = pytest.mark.xfail(
                    strict=True,
                    reason="el marcador cae en los últimos 128 caracteres del corte y el overlap lo repite",
                )

            configs.append(pytest.param(size, overlap, marks=marks, id=f"cs{size}_ov{int(frac * 100)}"))

    return configs

# SEP-12: El marcador aislado duplicaba tablas por overlap; el inline solo lo evita si no cae en la zona de overlap.
@pytest.mark.parametrize("chunk_size, overlap", sweep_configs())
def test_table_leaves_the_text_and_is_attached_once(chunk_size, overlap):

    chunks = chunk_documents([doc(LONG_TEXT)], [table(TABLE_RF), table(TABLE_KERNEL)], chunk_size, overlap)

    assert attached(chunks) == [TABLE_RF, TABLE_KERNEL]
    assert all("[[TABLE" not in c.text and "| RF |" not in c.text for c in chunks)
    assert all(c.text for c in chunks)
    assert all(len(c.text) <= chunk_size for c in chunks)

# fase 3 de chunking: un split con solo el marcador generaba un chunk "" y TEI rechazaba el batch
# entero (413, "inputs cannot be empty"); el fix acarrea la tabla (pending_tables) al próximo chunk
def test_marker_only_split_is_carried_to_the_next_chunk():

    chunks = chunk_documents([doc(f"{TABLE_RF}\n\n{LONG_WORD} fin del documento")], [table(TABLE_RF)], 40, 0)

    assert all(c.text for c in chunks)
    assert chunks[0].tables == [table(TABLE_RF)]
    assert attached(chunks) == [TABLE_RF]

# mismo fix, del otro lado: si el último split es solo el marcador, la tabla va al último chunk con texto
def test_table_at_the_end_goes_to_the_last_chunk():

    chunks = chunk_documents([doc(f"Texto inicial {LONG_WORD}\n\n{TABLE_RF}")], [table(TABLE_RF)], 40, 0)

    assert chunks[-1].tables == [table(TABLE_RF)]
    assert attached(chunks) == [TABLE_RF]

# music-tagger-benchmark.md tiene 9 tablas: cada [[TABLE:i]] tiene que volver con su propio contenido
def test_each_marker_maps_to_its_own_table():

    text = f"Intro corta.\n\n{TABLE_RF}\n\nTexto del medio.\n\n{TABLE_KERNEL}\n\nCierre."

    chunks = chunk_documents([doc(text)], [table(TABLE_RF), table(TABLE_KERNEL)], 284, 71)

    assert attached(chunks) == [TABLE_RF, TABLE_KERNEL]

# un doc que es solo tabla no tiene texto que embeber: no genera chunks y lo deja registrado
def test_doc_with_only_tables_yields_no_chunks_and_warns(caplog):

    with caplog.at_level(logging.WARNING, logger="shared.ingest"):
        chunks = chunk_documents([doc(TABLE_RF)], [table(TABLE_RF)], 284, 71)

    assert chunks == []
    assert "solo por tablas" in caplog.text

# las tablas se extraen de todo el corpus a la vez: la de genesis.md no puede terminar en benchmark.md
def test_tables_of_other_docs_are_ignored():

    chunks = chunk_documents([doc("Texto " * 20)], [table(TABLE_RF, doc_id="music-tagger/genesis.md")], 284, 71)

    assert attached(chunks) == []

# cada experimento re-indexa su colección: ids estables (uuid5 de doc#índice) hacen upsert sobre los
# mismos puntos en vez de duplicarlos
def test_chunk_ids_are_deterministic():

    first = chunk_documents([doc("Texto " * 30)], [], 60, 10)
    second = chunk_documents([doc("Texto " * 30)], [], 60, 10)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].chunk_id == str(uuid.uuid5(uuid.NAMESPACE_URL, f"{DOC_ID}#0"))

# AGO-23 / SEP-02: el embedder trunca en silencio a su límite (128 tokens); el 512/25% de la fase 1
# perdía ~19 % de cada chunk y por eso quedó inválido. validate_chunk deja el texto como lo ve el embedder
def test_validate_chunk_truncates_to_the_tokenizer_limit():

    tokenizer = FakeTokenizer()
    tokenizer.model_max_length = 20

    assert validate_chunk("abcdefghijklmnopqrstuvwxyz", tokenizer, 128) == "abcdefghijklmnopqrst"

# a16e47b: corpus_hash decide si una colección se puede reusar; el 2026-08-18 cambió el corpus (drift
# que desactualizó el gold span de q012) y el hash tiene que detectarlo aunque chunk_size no cambie
def test_corpus_hash_ignores_order_and_detects_changes():

    a, b = doc("texto a", "lib/a.md"), doc("texto b", "lib/b.md")

    assert corpus_hash([a, b]) == corpus_hash([b, a])
    assert len(corpus_hash([a, b])) == 64
    assert corpus_hash([a, b]) != corpus_hash([a, doc("texto b editado", "lib/b.md")])
    assert corpus_hash([a, b]) != corpus_hash([a, doc("texto b", "lib/c.md")])

# el doc_id tiene que coincidir con source_doc de questions.jsonl ("music-tagger/...md") o ningún chunk
# cuenta como correcto; el front matter no es contenido y no debe indexarse
def test_load_corpus_strips_front_matter_and_builds_doc_ids(tmp_path):

    lib = tmp_path / "music-tagger"
    lib.mkdir()
    (lib / "genesis.md").write_text("---\ntitle: Genesis\n---\n\n# Genesis\n\nTexto.\n", encoding="utf-8")
    (lib / "benchmark.md").write_text("# Benchmark\n", encoding="utf-8")

    docs = load_corpus(str(tmp_path))

    assert [d.doc_id for d in docs] == ["music-tagger/benchmark.md", "music-tagger/genesis.md"]
    assert docs[1].library == "music-tagger"
    assert docs[1].text == "# Genesis\n\nTexto."

# a16e47b: las colecciones llevan el modelo en el nombre para no mezclar vectores de modelos distintos
# bajo el mismo nombre (p. ej. exp_chunking_dynamic_tables__paraphrase-multilingual-MiniLM-L12-v2)
def test_qualified_collection_name_includes_the_model(monkeypatch):

    monkeypatch.setattr(ingest.settings, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

    assert qualified_collection_name("exp_chunking") == "exp_chunking__sentence-transformers--all-MiniLM-L6-v2"

# a16e47b: hubo una segunda definición de get_current_config que leía "overlap_frac" en vez de
# "chunk_overlap" y la comparación para reusar la colección nunca coincidía; además "tables" (fase 3)
# es payload del chunk, no config
def test_get_current_config_returns_only_the_indexing_config():

    payload = {
        "chunk_id": "id", "text": "t", "doc_id": DOC_ID, "library": "music-tagger", "chunk_index": 0,
        "tables": [table(TABLE_RF)], "chunk_size": 284, "chunk_overlap": 71, "corpus_hash": "abc",
    }

    assert get_current_config(FakeQdrant([FakePoint(payload)]), "c") == {
        "chunk_size": 284, "chunk_overlap": 71, "corpus_hash": "abc",
    }
    assert get_current_config(FakeQdrant(None), "c") is None
    assert get_current_config(FakeQdrant([]), "c") is None

# 2026-09-01: el modo server cargaba el modelo dos veces (SentenceTransformer local + TEI); ahora usa
# RemoteEmbedder (tokenizer + /info de TEI) y lru_cache lo crea una sola vez por proceso
def test_get_embedder_builds_one_remote_embedder(monkeypatch):

    created = []

    class FakeRemote:
        def __init__(self, url, model_name):
            created.append((url, model_name))

    monkeypatch.setattr(ingest.settings, "text_embedder_mode", "server")
    monkeypatch.setattr(ingest, "RemoteEmbedder", FakeRemote)
    ORIGINAL_GET_EMBEDDER.cache_clear()

    try:
        assert ORIGINAL_GET_EMBEDDER() is ORIGINAL_GET_EMBEDDER()
    finally:
        ORIGINAL_GET_EMBEDDER.cache_clear()

    assert created == [(ingest.settings.text_embedder_url, f"sentence-transformers/{ingest.settings.embedding_model}")]

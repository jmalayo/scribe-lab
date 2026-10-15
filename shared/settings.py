from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    qdrant_mode: str = "server"
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333

    text_embedder_mode: str = "server"
    text_embedder_url: str = "http://127.0.0.1:1010"

    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"

    detector_url: str = "http://127.0.0.1:1011"
    nli_url: str = "http://127.0.0.1:1012"

    llm_host: str = "http://127.0.0.1:1014"
    llm_backend: str = "ollama"
    llm_model_base: str = "llama3.2:3b"
    llm_model_answer: str = "llama3.1:8b-instruct-q4_K_M"
    llm_model_judge: str = "qwen3:8b"

    pipeline_config_path: str = "app/config/pipeline.yaml"

    mlflow_experiment_prefix: str = "rag-system-eval"
    mlflow_tracking_uri: str = "http://127.0.0.1:5000/"

    corpus_dir: str = "shared/corpus"
    questions_path: str = "shared/eval/questions.jsonl"

settings = Settings()

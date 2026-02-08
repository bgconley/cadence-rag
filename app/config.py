from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    expected_pg_version: str = "18.1"
    expected_pg_search_version: str = "0.21.5"
    expected_pgvector_version: str = "0.8.1"
    skip_version_check: bool = False
    embeddings_base_url: str = ""
    embeddings_model_id: str = "Qwen/Qwen3-Embedding-4B"
    embeddings_dim: int = 1024
    embeddings_timeout_s: float = 180.0
    embeddings_batch_size: int = 32
    embeddings_exact_scan_threshold: int = 2000
    embeddings_hnsw_ef_search: int = 80
    redis_url: str = "redis://localhost:6379/0"
    ingest_queue_name: str = "ingest"
    ingest_root_dir: str = "./ingest"
    ingest_poll_seconds: int = 5
    ingest_auto_manifest: bool = True
    ingest_single_file_min_age_s: int = 5
    ingest_job_max_attempts: int = 3
    ingest_job_retry_backoff_s: int = 10
    analysis_pdf_ocr_enabled: bool = False
    analysis_pdf_ocr_command: str = "ocrmypdf"
    analysis_pdf_ocr_languages: str = "eng"
    analysis_pdf_ocr_min_chars: int = 400
    analysis_pdf_ocr_min_alpha_ratio: float = 0.55
    analysis_pdf_ocr_max_pages: int = 150
    analysis_pdf_ocr_timeout_s: int = 600
    analysis_pdf_ocr_force: bool = False
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
    )


settings = Settings()

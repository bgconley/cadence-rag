from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    expected_pg_version: str = "18.1"
    expected_pg_search_version: str = "0.21.5"
    expected_pgvector_version: str = "0.8.1"
    skip_version_check: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
    )


settings = Settings()

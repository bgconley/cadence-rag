"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-02-03

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search;")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS corpora (
          corpus_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          name        TEXT NOT NULL,
          created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE (name)
        );
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS calls (
          call_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          corpus_id     UUID REFERENCES corpora(corpus_id) ON DELETE SET NULL,
          started_at    TIMESTAMPTZ NOT NULL,
          ended_at      TIMESTAMPTZ,
          title         TEXT,
          source_uri    TEXT,
          source_hash   TEXT,
          participants  JSONB,
          tags          TEXT[],
          metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
          created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS calls_started_at_idx ON calls (started_at DESC);")
    op.execute("CREATE INDEX IF NOT EXISTS calls_tags_gin_idx ON calls USING GIN (tags);")
    op.execute("CREATE INDEX IF NOT EXISTS calls_meta_gin_idx ON calls USING GIN (metadata);")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS utterances (
          utterance_id  BIGSERIAL PRIMARY KEY,
          call_id       UUID NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
          speaker       TEXT,
          speaker_id    TEXT,
          start_ts_ms   BIGINT NOT NULL,
          end_ts_ms     BIGINT NOT NULL,
          confidence    REAL,
          text          TEXT NOT NULL,
          metadata      JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS utterances_call_ts_idx "
        "ON utterances (call_id, start_ts_ms);"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
          chunk_id        BIGSERIAL PRIMARY KEY,
          call_id         UUID NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
          corpus_id       UUID REFERENCES corpora(corpus_id) ON DELETE SET NULL,
          call_started_at TIMESTAMPTZ NOT NULL,
          speaker         TEXT,
          start_ts_ms     BIGINT NOT NULL,
          end_ts_ms       BIGINT NOT NULL,
          token_count     INT NOT NULL,
          text            TEXT NOT NULL,
          embedding       vector(1024),
          tech_tokens     TEXT[] NOT NULL DEFAULT '{}',
          metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS chunks_call_ts_idx ON chunks (call_id, start_ts_ms);")
    op.execute("CREATE INDEX IF NOT EXISTS chunks_started_at_idx ON chunks (call_started_at DESC);")
    op.execute("CREATE INDEX IF NOT EXISTS chunks_meta_gin_idx ON chunks USING GIN (metadata);")
    op.execute("CREATE INDEX IF NOT EXISTS chunks_tech_tokens_gin ON chunks USING GIN (tech_tokens);")
    op.execute("CREATE INDEX IF NOT EXISTS chunks_text_trgm ON chunks USING GIN (text gin_trgm_ops);")
    op.execute(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
        "ON chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64);"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_utterances (
          chunk_id     BIGINT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
          utterance_id BIGINT NOT NULL REFERENCES utterances(utterance_id) ON DELETE CASCADE,
          ordinal      INT NOT NULL,
          PRIMARY KEY (chunk_id, utterance_id)
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS chunk_utterances_chunk_ordinal_idx "
        "ON chunk_utterances (chunk_id, ordinal);"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_artifacts (
          artifact_id     BIGSERIAL PRIMARY KEY,
          call_id         UUID NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
          corpus_id       UUID REFERENCES corpora(corpus_id) ON DELETE SET NULL,
          call_started_at TIMESTAMPTZ NOT NULL,
          kind            TEXT NOT NULL,
          content         TEXT NOT NULL,
          token_count     INT NOT NULL,
          embedding       vector(1024),
          tech_tokens     TEXT[] NOT NULL DEFAULT '{}',
          metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
          created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS artifacts_call_kind_idx "
        "ON analysis_artifacts (call_id, kind);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS artifacts_started_at_idx "
        "ON analysis_artifacts (call_started_at DESC);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS artifacts_meta_gin "
        "ON analysis_artifacts USING GIN (metadata);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS artifacts_tech_tokens_gin "
        "ON analysis_artifacts USING GIN (tech_tokens);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS artifacts_embedding_hnsw "
        "ON analysis_artifacts USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64);"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS entities (
          entity_id   BIGSERIAL PRIMARY KEY,
          label       TEXT NOT NULL,
          canonical   TEXT NOT NULL,
          normalized  TEXT NOT NULL,
          metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
          UNIQUE (label, normalized)
        );
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_entities (
          chunk_id   BIGINT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
          entity_id  BIGINT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
          start_char INT NOT NULL,
          end_char   INT NOT NULL,
          confidence REAL,
          PRIMARY KEY (chunk_id, entity_id, start_char, end_char)
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS chunk_entities_entity_idx "
        "ON chunk_entities (entity_id);"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS artifact_entities (
          artifact_id BIGINT NOT NULL REFERENCES analysis_artifacts(artifact_id) ON DELETE CASCADE,
          entity_id   BIGINT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
          start_char  INT NOT NULL,
          end_char    INT NOT NULL,
          confidence  REAL,
          PRIMARY KEY (artifact_id, entity_id, start_char, end_char)
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS artifact_entities_entity_idx "
        "ON artifact_entities (entity_id);"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_runs (
          ingestion_run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          call_id          UUID NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
          pipeline_version TEXT NOT NULL,
          chunking_config  JSONB NOT NULL,
          embedding_config JSONB NOT NULL,
          ner_config       JSONB NOT NULL,
          created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ingestion_runs_call_idx "
        "ON ingestion_runs (call_id);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ingestion_runs;")
    op.execute("DROP TABLE IF EXISTS artifact_entities;")
    op.execute("DROP TABLE IF EXISTS chunk_entities;")
    op.execute("DROP TABLE IF EXISTS entities;")
    op.execute("DROP TABLE IF EXISTS analysis_artifacts;")
    op.execute("DROP TABLE IF EXISTS chunk_utterances;")
    op.execute("DROP TABLE IF EXISTS chunks;")
    op.execute("DROP TABLE IF EXISTS utterances;")
    op.execute("DROP TABLE IF EXISTS calls;")
    op.execute("DROP TABLE IF EXISTS corpora;")

import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

# Load .env if present
load_dotenv()

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

db_url = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:5432/rag"
)
config.set_main_option("sqlalchemy.url", db_url.replace("%", "%%"))
version_table_schema = os.getenv("ALEMBIC_VERSION_TABLE_SCHEMA")

target_metadata = None


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    kwargs = dict(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    if version_table_schema:
        kwargs["version_table_schema"] = version_table_schema
    context.configure(**kwargs)

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        kwargs = dict(connection=connection, target_metadata=target_metadata)
        if version_table_schema:
            kwargs["version_table_schema"] = version_table_schema
        context.configure(**kwargs)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

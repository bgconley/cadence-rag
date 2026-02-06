FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
# Packaging expects the `app` package to exist at build time.
COPY app ./app
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

COPY alembic.ini ./alembic.ini
COPY alembic ./alembic

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

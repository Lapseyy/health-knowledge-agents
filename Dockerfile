# Cloud Run image: the API, the web UI, the data files, and a Chroma index built at image
# build time, so a deploy from a fresh clone needs no prebuilt artifacts.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY data ./data
COPY static ./static
COPY main.py NOTICE ./

# Download the local embedding model (~70 MB, into models/) and embed the textbook
# (2,679 chunks, into chroma_index/). Needs no API keys. The index is read-only at run
# time; conversation memory is not stored here.
RUN python -m src.rag.ingest

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
CMD ["python", "main.py"]

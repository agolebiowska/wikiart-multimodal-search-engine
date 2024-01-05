FROM python:3.9-slim

COPY requirements.txt .

RUN python3 -m pip install -r \
    requirements.txt --quiet --no-cache-dir \
    && rm -f requirements.txt

COPY ./src /generate_embeddings/src

ENTRYPOINT python3 /generate_embeddings/src/main.py
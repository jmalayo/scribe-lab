FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home --uid 10001 app

WORKDIR /srv

COPY app/requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

COPY shared shared
COPY app app
COPY corpus_demo corpus_demo

RUN mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache

USER app

EXPOSE 1013

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1013"]

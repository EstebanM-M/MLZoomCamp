FROM python:3.14.7-slim

RUN pip install uv

WORKDIR /app
COPY ["pyproject.toml", "uv.lock", "./"]

RUN uv sync --frozen --no-dev

COPY ["predict.py","model_C=1.0.bin", "./"]

EXPOSE 8080

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8080", "predict:app"]
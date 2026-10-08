FROM python:3.12-slim-bookworm

RUN pip install uv

WORKDIR /app
COPY ["pyproject.toml", "uv.lock", "./"]

RUN uv sync --frozen --no-dev

COPY ["predict_service.py", "fraud_model.bin", "./"]

EXPOSE 9696

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict_service:app"]
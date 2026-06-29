# Credit Risk + Profit Decision API
# Serves the trained PD + LGD models via FastAPI / uvicorn.
#
# Build:  docker build -t credit-risk-api .
# Run:    docker run -p 8000:8000 -v $(pwd)/models:/app/models credit-risk-api
#         (mount a models/ dir containing pd_model.pkl + lgd_model.pkl produced
#          by `make train`)
FROM python:3.11-slim

# libgomp1 is required by LightGBM / XGBoost / CatBoost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

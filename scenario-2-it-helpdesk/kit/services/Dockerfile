# Simulated helpdesk API (python:3.12-slim). Build from the kit folder:
#   docker compose up --build        (see ../docker-compose.yml)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Same pins as ../requirements.txt (only what the API needs)
RUN pip install --no-cache-dir fastapi==0.141.1 uvicorn==0.52.1 jsonschema==4.26.0

COPY services/ /app/services/
COPY dataset/manifest.json /app/dataset/manifest.json
COPY dataset/schemas/ /app/dataset/schemas/
COPY dataset/tickets/ /app/dataset/tickets/

EXPOSE 8102
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8102/health', timeout=2)"

CMD ["uvicorn", "services.helpdesk_api:app", "--host", "0.0.0.0", "--port", "8102"]

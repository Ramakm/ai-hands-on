FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SEQNET_HOST=0.0.0.0
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY seqnet ./seqnet
COPY web ./web
# Bake the trained weights into the image so the container starts instantly.
RUN python -m seqnet.train

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["python", "-m", "uvicorn", "seqnet.api:app", "--host", "0.0.0.0", "--port", "8000"]

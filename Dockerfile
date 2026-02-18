FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY vectorbt_bot.py README.md .env.example ./

RUN useradd -m -u 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python", "vectorbt_bot.py", "live", "--bars", "1500"]

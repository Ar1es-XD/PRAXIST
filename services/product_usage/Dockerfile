FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN addgroup --system collector && adduser --system --ingroup collector collector

COPY . .
RUN python -m pip install --no-cache-dir '.[product-usage-server]'

USER collector
EXPOSE 8000

CMD ["praxist-collector"]

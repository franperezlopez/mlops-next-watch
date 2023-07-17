# - stage 0 -
FROM python:3.10-slim AS builder
RUN apt-get update && apt-get install -y build-essential libpq-dev
COPY docker/mlflow/requirements.txt .
RUN pip install --user -r ./requirements.txt

# - stage 1 -
FROM python:3.10-slim
EXPOSE 5000
RUN apt-get update && apt-get install -y libpq-dev
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH


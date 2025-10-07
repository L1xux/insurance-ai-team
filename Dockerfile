# Build stage
FROM python:3.11 AS build-stage

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Production stage
FROM python:3.11
WORKDIR /app

COPY --from=build-stage /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY . .

EXPOSE 8080

CMD ["python3", "server.py", "--port", "8080"]
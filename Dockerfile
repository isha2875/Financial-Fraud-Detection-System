FROM python:3.8-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3-dev \
       python-dev-is-python3 \
       build-essential \
       apt-utils \
       curl \
       git \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip setuptools

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY . .

CMD gunicorn -w 3 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8501

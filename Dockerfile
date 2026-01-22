FROM python:3.10-slim

WORKDIR /app

# system deps (optional but helpful for many ML/data libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# install python deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the project
COPY . .

# default behavior (optional)
CMD ["python", "src/predict_risk.py", "--help"]
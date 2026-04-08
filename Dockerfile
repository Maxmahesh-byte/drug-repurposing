FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create all directories early (before any COPY)
RUN mkdir -p /app/data/raw \
             /app/data/processed \
             /app/data/new \
             /app/mlruns \
             /app/artifacts

# Copy only the code (data folder is now ignored by .dockerignore)
COPY . .

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"]
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Create required directories safely (absolute paths + force)
RUN mkdir -p /app/data/processed /app/mlruns /app/artifacts

EXPOSE 8000 8501

# Start both FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"]
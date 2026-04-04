# Base image
FROM python:3.10

# Working directory
WORKDIR /app

# Copy all project files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose FastAPI port
EXPOSE 10000

# Run FastAPI server
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000"]
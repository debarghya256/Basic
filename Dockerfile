FROM python:3.12-slim

WORKDIR /code

# Install dependencies first (cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code, then train the model inside the image
COPY train.py .
COPY app ./app
RUN python train.py

EXPOSE 8000
# Many hosts (Render, Railway, Cloud Run) provide $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

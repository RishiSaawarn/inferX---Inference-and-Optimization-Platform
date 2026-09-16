FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install -e .

COPY . .

# Download models during image build to bake them into the image
RUN python scripts/download_models.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

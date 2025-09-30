# Dockerfile
FROM python:3.10

# Set working directory
WORKDIR /app
# Copy files
COPY . .
# Install dependencies
RUN pip install fastapi uvicorn scikit-learn==1.7.1 pydantic category_encoders pandas numpy
# Run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
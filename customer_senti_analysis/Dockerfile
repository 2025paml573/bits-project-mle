# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements/dependency file or install directly
# (Best practice: create a requirements.txt, but we can install inline here)
# RUN pip install --no-cache-dir fastapi uvicorn nltk pydantic sentence-transformers

# Copy the local models directory into the container's expected path
COPY ./models/* /app/models/

# Copy your FastAPI application code into the container
COPY ./api.py /app/

# install requirements
COPY ./requirements.txt /app/

RUN pip install --upgrade pip

RUN pip install -r /app/requirements.txt

# Download required NLTK data inside the container image so it doesn't download at runtime
RUN python -c "import nltk; nltk.download('rte'); nltk.download('stopwords')"

# Update MODEL_DIR in your script if necessary to match container path, 
# or ensure it points to "/app/models/" (your current script already uses an absolute path, 
# so make sure to change MODEL_DIR = "/app/models/" in your Python script for Docker).

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["python", "/app/api.py"]
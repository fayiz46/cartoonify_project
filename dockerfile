# Base Image
FROM python:3.9

# Set working directory inside container
WORKDIR /app

# Copy requirements file if you have it (optional)
# COPY requirements.txt .

# Install dependencies
RUN pip install fastapi uvicorn jinja2 python-multipart opencv-python-headless numpy

# Copy project files into container
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

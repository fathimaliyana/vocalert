# Use official Python image with slim Debian base
FROM python:3.10-slim

# Install system dependencies for PortAudio and other libs
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app code and model
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app1.py", "--server.port=8501", "--server.address=0.0.0.0"]

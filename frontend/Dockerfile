# Use a slim but workable base image
FROM python:3.11

# Install system dependencies required for Plotly + Streamlit
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
ENTRYPOINT ["streamlit", "run"]

CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
FROM python:3.13-slim

# Install libgomp (required by LightGBM for OpenMP support)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create data/model/log directories
RUN mkdir -p data models logs

# Run the bot
CMD ["python", "-m", "src.telegram_bot"]

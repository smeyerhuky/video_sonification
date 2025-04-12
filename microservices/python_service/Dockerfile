# Use Python slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make the server script executable
RUN chmod +x server.py

# Expose the service port
EXPOSE 5000

# Start the service
CMD ["python", "server.py"]
# Project Sonique: Quick Start Guide

This guide will help you get the Project Sonique proof-of-concept up and running, demonstrating the Multi-Band Motion Analysis Framework with GPU acceleration.

## Prerequisites

- Docker with Docker Compose
- NVIDIA GPU with appropriate drivers
- NVIDIA Container Toolkit installed and configured

### Checking NVIDIA Container Toolkit Installation

To verify that your system is properly configured for GPU passthrough to containers:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 nvidia-smi
```

You should see output showing your GPU information. If you get an error, you may need to install or configure the NVIDIA Container Toolkit.

## Running the System

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd video_sonification
   ```

2. Start the services with Docker Compose:
   ```bash
   docker-compose up
   ```

3. Access the frontend in your browser:
   ```
   http://localhost:5173
   ```

4. Navigate to the "Multi-Band Analysis" tab to upload and analyze videos.

## System Components

### 1. Multi-Band Analysis Service

- **Port**: 5002
- **Endpoint**: `http://localhost:5002/analyze`
- **Features**:
  - GPU-accelerated video processing
  - Multi-band frequency decomposition
  - Wobble bass generation

### 2. Frontend Application

- **Port**: 5173
- **Features**:
  - Video upload interface
  - Visualization of frequency band analysis
  - Audio playback of generated wobble bass

### 3. NATS Message Broker

- **Port**: 4222 (Client), 8222 (HTTP Monitoring)
- **Features**:
  - Service communication infrastructure
  - Event distribution
  - JetStream for persistent messaging

## Testing the System

1. **Prepare a Test Video**:
   - Short videos (10-30 seconds) work best for initial testing
   - Videos with clear motion patterns will produce more interesting results
   - Examples: swaying trees, dancing, sports footage

2. **Upload and Analyze**:
   - Go to the Multi-Band Analysis tab
   - Upload your video
   - Wait for processing to complete (this may take some time depending on video length)
   - Explore the analysis results across different frequency bands
   - Listen to the generated wobble bass audio

3. **Monitoring**:
   - Check Docker logs for processing details:
     ```bash
     docker logs video_sonification-multi-band-analysis
     ```
   - Monitor GPU usage:
     ```bash
     nvidia-smi
     ```

## Troubleshooting

### Dependency Issues

If you encounter errors related to Python dependencies (like the Werkzeug/Flask compatibility issue), you can rebuild the affected service:

```bash
# Stop the services
docker-compose down

# Rebuild the specific service
docker-compose build multi-band-analysis

# Start the services again
docker-compose up
```

The `requirements.txt` file for the multi-band analysis service has pinned versions of Flask (2.0.1) and Werkzeug (2.0.3) to ensure compatibility.

### GPU Issues

If the multi-band analysis service fails to use the GPU:

1. Verify NVIDIA drivers are installed:
   ```bash
   nvidia-smi
   ```

2. Check NVIDIA Container Toolkit installation:
   ```bash
   dpkg -l | grep nvidia-container-toolkit
   ```

3. Ensure Docker is configured to use NVIDIA runtime:
   ```bash
   cat /etc/docker/daemon.json
   ```
   Should contain:
   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

### Service Connection Issues

If the frontend cannot connect to the multi-band analysis service:

1. Check if the service is running:
   ```bash
   docker ps | grep multi-band-analysis
   ```

2. Verify the service is accessible:
   ```bash
   curl http://localhost:5002/health
   ```

3. Check CORS configuration if needed.

## Next Steps

This proof-of-concept demonstrates the core functionality of Project Sonique's Multi-Band Motion Analysis Framework. Future development will include:

1. Enhanced pattern classification
2. More sophisticated audio generation
3. Integration with LLaVA for semantic understanding
4. Full implementation of the shared context system
5. Advanced visualization of frequency bands

Refer to the implementation guide and README files for more detailed information about the system architecture and components.
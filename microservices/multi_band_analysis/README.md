# Multi-Band Motion Analysis Service

This microservice implements the core Multi-Band Motion Analysis Framework for Project Sonique, providing frequency-domain decomposition of video motion patterns and basic audio generation capabilities.

## Overview

The Multi-Band Motion Analysis service analyzes video content across multiple frequency bands:

- **Ultra-Low Frequency (ULF)**: 0.01-0.1 Hz - Scene transitions, camera movements
- **Low Frequency (LF)**: 0.1-1 Hz - Swaying, gentle motion
- **Mid Frequency (MF)**: 1-5 Hz - Regular rhythmic movements
- **High Frequency (HF)**: 5-15 Hz - Rapid movements
- **Ultra-High Frequency (UHF)**: 15+ Hz - Micro-movements, vibrations

For each frequency band, the service analyzes motion patterns, spatial distribution, and temporal evolution, providing a comprehensive analysis that can be mapped to audio parameters.

## Features

- **GPU-Accelerated Processing**: Uses NVIDIA CUDA for efficient video analysis
- **Multi-Band Frequency Decomposition**: Analyzes motion across five frequency bands
- **Spatial Distribution Analysis**: Identifies regions with significant motion
- **Temporal Evolution Analysis**: Tracks how motion patterns change over time
- **Basic Wobble Bass Generation**: Creates audio based on motion analysis
- **RESTful API**: Simple HTTP endpoints for video upload and analysis

## API Endpoints

### Health Check

```
GET /health
```

Returns the service status and CUDA availability.

### Video Analysis

```
POST /analyze
```

Analyzes a video file and returns multi-band motion analysis results along with generated audio.

**Request:**
- Content-Type: `multipart/form-data`
- Body: 
  - `video`: Video file to analyze

**Response:**
```json
{
  "video_id": "unique-identifier",
  "analysis_results": {
    "ulf": { /* Ultra-Low Frequency band results */ },
    "lf": { /* Low Frequency band results */ },
    "mf": { /* Mid Frequency band results */ },
    "hf": { /* High Frequency band results */ },
    "uhf": { /* Ultra-High Frequency band results */ }
  },
  "audio_info": {
    "lfo_rate": 2.5,
    "cutoff_min": 300,
    "cutoff_max": 3000,
    "duration": 10,
    "sample_rate": 44100,
    "file_path": "/path/to/audio.wav"
  }
}
```

## Technical Implementation

### Frequency Band Decomposition

The service uses Butterworth bandpass filters to decompose motion signals into different frequency bands. For each pixel in the video, the temporal motion signal is filtered to extract motion components in each frequency range.

### Optical Flow Computation

Motion is detected using the Farneback optical flow algorithm from OpenCV, which provides dense motion vectors for each pixel in the video.

### Audio Generation

A simple wobble bass generator creates audio based on the analysis results:
- MF band periodicity maps to LFO rate
- LF band magnitude maps to filter cutoff range
- Basic sawtooth carrier with time-varying filter

## Docker Configuration

The service is containerized with NVIDIA GPU support:

```dockerfile
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
```

To run the container with GPU support, the Docker host must have:
1. NVIDIA GPU with appropriate drivers
2. NVIDIA Container Toolkit installed

## Development

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- Docker with NVIDIA Container Toolkit

### Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the server:
   ```
   python server.py
   ```

### Docker Development

1. Build the container:
   ```
   docker build -t multi-band-analysis .
   ```

2. Run with GPU support:
   ```
   docker run --gpus all -p 5002:5000 multi-band-analysis
   ```

## Integration with Project Sonique

This service is part of the Project Sonique architecture, which includes:

1. Frontend microservices for user interaction
2. Backend microservices for specialized processing
3. NATS message broker for service communication

In the current implementation, the service communicates via HTTP, but future versions will integrate with the NATS message broker for event-driven communication.

## Known Issues and Fixes

### Filter Frequency Normalization

When processing videos with low frame rates, the frequency band normalization can result in values outside the valid range (0 < Wn < 1) for the Butterworth filter, or cases where the lower frequency is not less than the higher frequency. We've implemented a robust fix that:

1. Sets low_norm in the range [0.01, 0.8] to ensure it's always greater than 0 but not too high
2. Sets high_norm in the range [low_norm+0.1, 0.95] to ensure it's always greater than low_norm
3. Logs the actual normalized frequencies used for each band for debugging
4. Handles any frame rate by properly normalizing against the Nyquist frequency

This ensures the filter initialization works correctly regardless of the video's frame rate.

### Flask/Werkzeug Compatibility

We've pinned specific versions of Flask (2.0.1) and Werkzeug (2.0.3) to ensure compatibility and prevent import errors with `url_quote`.

## Future Enhancements

- Full NATS integration for event-driven architecture
- More sophisticated audio generation
- Advanced pattern classification
- Integration with LLaVA for semantic understanding
- Improved visualization of frequency bands
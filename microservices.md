# Video Sonification Microservices Architecture

This document outlines the core microservices required for the Video Sonification project, with a focus on the initial v0.0.1 implementation.

## Overview

The microservices architecture divides the video processing and audio generation pipeline into specialized, independently deployable services. Each service has a specific role in the overall system and communicates with others through well-defined Thrift interfaces.

```
┌─────────────┐         ┌───────────────┐         ┌─────────────────┐
│             │  Thrift │               │  Thrift │                 │
│   Frontend  │ ◄─────► │ Video Analysis │ ◄─────► │ Audio Generation │
│             │         │               │         │                 │
└─────────────┘         └───────────────┘         └─────────────────┘
                              │                           ▲
                              │                           │
                              │         ┌─────────────────┘
                              │         │
                              ▼         │
                        ┌─────────────────┐
                        │                 │
                        │  Feature Store  │
                        │                 │
                        └─────────────────┘
```

## Core Microservices for v0.0.1

### 1. Video Analysis Service

**Purpose:** Process video frames and extract visual features for sonification.

**Responsibilities:**
- Frame extraction from video input
- Motion detection using optical flow
- Color analysis and dominant color extraction
- Edge detection and pattern analysis
- Feature data preparation for audio mapping

**Implementation:**
- **Language:** Python
- **Key Libraries:** OpenCV, NumPy, scikit-image
- **Container:** Dockerfile in `microservices/python_service/`
- **API:** Defined in `thrift/video_analysis_service.thrift` (to be created)

**Data Flow:**
- Receives video frames from the frontend
- Processes frames to extract features
- Stores feature data in the Feature Store
- Returns summarized analysis results to the frontend

### 2. Audio Generation Service

**Purpose:** Generate audio based on visual features extracted from video.

**Responsibilities:**
- Generate wobble bass sounds based on motion data
- Create rhythm patterns from edge detection data
- Produce atmospheric elements from color data
- Mix and master audio outputs
- Stream audio to the frontend

**Implementation:**
- **Language:** Go
- **Key Libraries:** PortAudio, go-dsp
- **Container:** Dockerfile in `microservices/go_service/`
- **API:** Defined in `thrift/audio_generation_service.thrift` (to be created)

**Data Flow:**
- Reads visual feature data from the Feature Store
- Maps visual features to audio parameters
- Generates audio samples based on mappings
- Streams audio data to the frontend

### 3. Feature Store Service

**Purpose:** Store and manage extracted visual features and mapping parameters.

**Responsibilities:**
- Store time-series data of extracted visual features
- Maintain mapping configurations between visual features and audio parameters
- Provide query capabilities for other services
- Cache frequently accessed data

**Implementation:**
- **Language:** Go
- **Storage:** In-memory for v0.0.1, expandable to Redis/TimescaleDB
- **Container:** New Dockerfile in `microservices/feature_store_service/` (to be created)
- **API:** Defined in `thrift/feature_store_service.thrift` (to be created)

**Data Flow:**
- Receives feature data from the Video Analysis Service
- Provides feature data to the Audio Generation Service
- Maintains temporal synchronization between video and audio

## Service Communication

All service communication is handled through Thrift interfaces, which provide:
- Type safety across language boundaries
- Efficient binary serialization
- Clear API contracts

### Key Thrift Interfaces (to be implemented)

1. **Video Analysis Service Interface:**
```thrift
service VideoAnalysisService {
    AnalysisResult analyzeFrame(1: binary frameData, 2: i32 frameNumber, 3: string sessionId)
    AnalysisResult analyzeVideo(1: string videoUrl, 2: string sessionId)
}
```

2. **Audio Generation Service Interface:**
```thrift
service AudioGenerationService {
    AudioStreamInfo generateAudio(1: string sessionId, 2: AudioParameters params)
    void updateParameters(1: string sessionId, 2: AudioParameters params)
    void stopGeneration(1: string sessionId)
}
```

3. **Feature Store Service Interface:**
```thrift
service FeatureStoreService {
    void storeFeatures(1: string sessionId, 2: i32 frameNumber, 3: FeatureData features)
    FeatureData getFeatures(1: string sessionId, 2: i32 frameNumber)
    FeatureSeries getFeatureSeries(1: string sessionId, 2: i32 startFrame, 3: i32 endFrame)
}
```

## Data Models

Core data structures used across services:

```thrift
struct MotionFeatures {
    1: double averageMagnitude
    2: list<double> directionHistogram
    3: list<Region> highMotionRegions
}

struct ColorFeatures {
    1: list<Color> dominantColors
    2: double colorfulness
    3: double brightness
}

struct EdgeFeatures {
    1: double edgeDensity
    2: double horizontalEdgeIntensity
    3: double verticalEdgeIntensity
    4: list<EdgePattern> patterns
}

struct FeatureData {
    1: MotionFeatures motion
    2: ColorFeatures color
    3: EdgeFeatures edges
    4: i64 timestamp
}

struct AudioParameters {
    1: double bassIntensity
    2: double modulationRate
    3: double filterResonance
    4: i32 bpm
    5: double rhythmComplexity
}
```

## Deployment

For v0.0.1, all services will be deployed using Docker Compose:

```yaml
# Excerpt from docker-compose.yml
services:
  frontend:
    # Frontend configuration...
    
  video-analysis-service:
    build: ./microservices/python_service
    ports:
      - "9090:9090"
    volumes:
      - ./shared_data:/data
      
  audio-generation-service:
    build: ./microservices/go_service
    ports:
      - "9091:9091"
    volumes:
      - ./shared_data:/data
      
  feature-store-service:
    build: ./microservices/feature_store_service
    ports:
      - "9092:9092"
```

## Future Extensions (Post v0.0.1)

As the project evolves, the microservices architecture can be extended with:

1. **Scene Detection Service:** Specialized service for identifying scene changes and structural elements in videos.

2. **Machine Learning Service:** Advanced feature detection using ML models for better audio mapping.

3. **Audio Mixing Service:** Dedicated service for professional-quality audio mixing and mastering.

4. **User Profile Service:** Store and manage user preferences and custom mapping configurations.

5. **Export Service:** Handle exporting generated audio in various formats and quality levels.

## Implementation Strategy

For v0.0.1, focus on:

1. Implementing the core Thrift interfaces
2. Building minimal viable implementations of each service
3. Establishing the basic pipeline flow from video input to audio output
4. Creating simple, functional mapping from visual features to audio parameters

This foundation will allow for incremental enhancement of individual services as the project evolves.
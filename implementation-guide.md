# Project Sonique: Implementation Guide

This technical guide provides detailed implementation instructions for the core components of Project Sonique, focusing on our priorities:

1. Multi-Band Motion Analysis Framework
2. Microservice Infrastructure with Shared Context/Tagging System

## 1. Microservice Infrastructure Implementation

### 1.1 Project Structure

Create a new directory structure that supports both frontend and backend microservices:

```
video_sonification/
├── frontend/
│   ├── services/
│   │   ├── analysis-dashboard/
│   │   ├── pattern-explorer/
│   │   ├── parameter-mapper/
│   │   └── music-generator/
│   ├── lib/
│   │   ├── shared-ui/
│   │   ├── contexts/
│   │   └── utils/
│   └── shell/
│       ├── src/
│       ├── public/
│       └── ...
├── microservices/
│   ├── multi-band-analysis/
│   ├── jitter-analysis/
│   ├── llava-integration/
│   └── audio-generation/
├── shared/
│   ├── types/
│   ├── schemas/
│   └── utils/
├── infra/
│   ├── docker/
│   ├── k8s/
│   └── nats/
└── docs/
```

### 1.2 Docker Configuration with NVIDIA Support

Create a base Docker configuration for GPU-accelerated services:

```yaml
# infra/docker/docker-compose.yml
version: '3.8'

services:
  nats:
    image: nats:latest
    ports:
      - "4222:4222"
      - "8222:8222"
    volumes:
      - ./nats/nats-server.conf:/etc/nats/nats-server.conf
    command: "-c /etc/nats/nats-server.conf"

  frontend-shell:
    build:
      context: ../../frontend/shell
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    volumes:
      - ../../frontend/shell:/app
      - /app/node_modules
    environment:
      - NATS_URL=nats://nats:4222
      - NODE_ENV=development
      - CHOKIDAR_USEPOLLING=true

  multi-band-analysis:
    build:
      context: ../../microservices/multi-band-analysis
      dockerfile: Dockerfile
    volumes:
      - ../../microservices/multi-band-analysis:/app
    environment:
      - NATS_URL=nats://nats:4222
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Other services follow similar pattern
```

For the GPU-enabled Dockerfile:

```dockerfile
# microservices/multi-band-analysis/Dockerfile
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the service
CMD ["python3", "app.py"]
```

Example requirements.txt:

```
# microservices/multi-band-analysis/requirements.txt
numpy>=1.21.0
scipy>=1.7.0
opencv-python>=4.5.3
nats-py>=2.2.0
pydantic>=1.9.0
torch>=2.0.0
librosa>=0.9.1
matplotlib>=3.5.0
```

### 1.3 NATS Message Broker Setup

Configure NATS for the message broker:

```conf
# infra/nats/nats-server.conf
# Basic NATS Server Configuration
port: 4222
http_port: 8222

# Debug options
debug: false
trace: false

# Performance Options
max_payload: 8MB
write_deadline: "2s"

# Security Options (for production, uncomment and configure)
# tls {
#   cert_file: "/etc/nats/certs/server-cert.pem"
#   key_file: "/etc/nats/certs/server-key.pem"
#   ca_file: "/etc/nats/certs/ca.pem"
#   verify: true
# }
```

Create a NATS client wrapper for consistent service connection:

```typescript
// shared/utils/nats-client.ts
import { connect, NatsConnection, Subscription, JSONCodec } from 'nats';

const jsonCodec = JSONCodec();

export class NatsClient {
  private connection: NatsConnection | null = null;
  private subscriptions: Map<string, Subscription> = new Map();
  
  async connect(url: string = 'nats://localhost:4222'): Promise<void> {
    this.connection = await connect({ servers: url });
    console.log(`Connected to NATS at ${url}`);
  }
  
  async disconnect(): Promise<void> {
    if (this.connection) {
      for (const sub of this.subscriptions.values()) {
        sub.unsubscribe();
      }
      await this.connection.drain();
      this.connection = null;
      console.log('Disconnected from NATS');
    }
  }
  
  async publish(subject: string, data: any): Promise<void> {
    if (!this.connection) throw new Error('Not connected to NATS');
    this.connection.publish(subject, jsonCodec.encode(data));
  }
  
  subscribe(subject: string, callback: (data: any, subject: string) => void): void {
    if (!this.connection) throw new Error('Not connected to NATS');
    
    const subscription = this.connection.subscribe(subject);
    this.subscriptions.set(subject, subscription);
    
    (async () => {
      for await (const message of subscription) {
        const data = jsonCodec.decode(message.data);
        callback(data, message.subject);
      }
    })().catch(err => console.error(`Subscription error: ${err}`));
  }
  
  // Request-reply pattern
  async request(subject: string, data: any, timeout: number = 5000): Promise<any> {
    if (!this.connection) throw new Error('Not connected to NATS');
    
    const response = await this.connection.request(
      subject, 
      jsonCodec.encode(data),
      { timeout }
    );
    
    return jsonCodec.decode(response.data);
  }
}

// Singleton instance
export const natsClient = new NatsClient();
```

### 1.4 Shared Context Implementation

Create a shared context system using NATS and a React Context API wrapper:

```typescript
// frontend/lib/contexts/SharedContext.tsx
import React, { createContext, useContext, useEffect, useState } from 'react';
import { natsClient } from '../../../shared/utils/nats-client';

// Define context state interface based on your needs
interface ContextState {
  analysisResults: any | null;
  selectedPatterns: any[];
  parameterMappings: any[];
  // Add other shared state as needed
}

const initialState: ContextState = {
  analysisResults: null,
  selectedPatterns: [],
  parameterMappings: [],
};

interface ContextValue extends ContextState {
  updateAnalysisResults: (results: any) => void;
  addSelectedPattern: (pattern: any) => void;
  removeSelectedPattern: (patternId: string) => void;
  updateParameterMapping: (mapping: any) => void;
  // Add other actions as needed
}

const SharedContext = createContext<ContextValue | undefined>(undefined);

export function SharedContextProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<ContextState>(initialState);
  
  useEffect(() => {
    // Connect to NATS when component mounts
    natsClient.connect()
      .catch(err => console.error('Failed to connect to NATS:', err));
    
    // Subscribe to analysis results
    natsClient.subscribe('analysis.results', (data) => {
      setState(prev => ({ ...prev, analysisResults: data }));
    });
    
    // Subscribe to pattern selections
    natsClient.subscribe('patterns.selected', (data) => {
      setState(prev => ({ ...prev, selectedPatterns: data }));
    });
    
    // Subscribe to parameter mappings
    natsClient.subscribe('parameters.mappings', (data) => {
      setState(prev => ({ ...prev, parameterMappings: data }));
    });
    
    return () => {
      // Disconnect from NATS when component unmounts
      natsClient.disconnect()
        .catch(err => console.error('Error disconnecting from NATS:', err));
    };
  }, []);
  
  // Context actions
  const updateAnalysisResults = (results: any) => {
    setState(prev => ({ ...prev, analysisResults: results }));
    natsClient.publish('analysis.results', results)
      .catch(err => console.error('Failed to publish analysis results:', err));
  };
  
  const addSelectedPattern = (pattern: any) => {
    const updatedPatterns = [...state.selectedPatterns, pattern];
    setState(prev => ({ ...prev, selectedPatterns: updatedPatterns }));
    natsClient.publish('patterns.selected', updatedPatterns)
      .catch(err => console.error('Failed to publish selected patterns:', err));
  };
  
  const removeSelectedPattern = (patternId: string) => {
    const updatedPatterns = state.selectedPatterns.filter(p => p.id !== patternId);
    setState(prev => ({ ...prev, selectedPatterns: updatedPatterns }));
    natsClient.publish('patterns.selected', updatedPatterns)
      .catch(err => console.error('Failed to publish selected patterns:', err));
  };
  
  const updateParameterMapping = (mapping: any) => {
    const existingIndex = state.parameterMappings.findIndex(m => m.id === mapping.id);
    let updatedMappings;
    
    if (existingIndex >= 0) {
      updatedMappings = [...state.parameterMappings];
      updatedMappings[existingIndex] = mapping;
    } else {
      updatedMappings = [...state.parameterMappings, mapping];
    }
    
    setState(prev => ({ ...prev, parameterMappings: updatedMappings }));
    natsClient.publish('parameters.mappings', updatedMappings)
      .catch(err => console.error('Failed to publish parameter mappings:', err));
  };
  
  const contextValue: ContextValue = {
    ...state,
    updateAnalysisResults,
    addSelectedPattern,
    removeSelectedPattern,
    updateParameterMapping,
  };
  
  return (
    <SharedContext.Provider value={contextValue}>
      {children}
    </SharedContext.Provider>
  );
}

// Custom hook for using this context
export function useSharedContext() {
  const context = useContext(SharedContext);
  if (context === undefined) {
    throw new Error('useSharedContext must be used within a SharedContextProvider');
  }
  return context;
}
```

### 1.5 Frontend Module Federation Setup

Configure the Module Federation architecture for micro-frontends:

```javascript
// frontend/shell/webpack.config.js
const { ModuleFederationPlugin } = require('webpack').container;

module.exports = {
  // ... other webpack configuration
  plugins: [
    new ModuleFederationPlugin({
      name: 'shell',
      filename: 'remoteEntry.js',
      remotes: {
        analysisDashboard: 'analysisDashboard@http://localhost:3001/remoteEntry.js',
        patternExplorer: 'patternExplorer@http://localhost:3002/remoteEntry.js',
        parameterMapper: 'parameterMapper@http://localhost:3003/remoteEntry.js',
        musicGenerator: 'musicGenerator@http://localhost:3004/remoteEntry.js',
      },
      shared: {
        react: { singleton: true },
        'react-dom': { singleton: true },
        // other shared dependencies
      },
    }),
  ],
};

// frontend/services/analysis-dashboard/webpack.config.js
const { ModuleFederationPlugin } = require('webpack').container;

module.exports = {
  // ... other webpack configuration
  plugins: [
    new ModuleFederationPlugin({
      name: 'analysisDashboard',
      filename: 'remoteEntry.js',
      exposes: {
        './AnalysisDashboard': './src/AnalysisDashboard',
        './FrequencyBandVisualizer': './src/components/FrequencyBandVisualizer',
      },
      shared: {
        react: { singleton: true },
        'react-dom': { singleton: true },
        // other shared dependencies
      },
    }),
  ],
};
```

## 2. Multi-Band Motion Analysis Framework Implementation

### 2.1 Frequency Band Decomposition

Implement the core frequency band decomposition in Python:

```python
# microservices/multi-band-analysis/frequency_bands.py
import numpy as np
import cv2
from scipy import signal

class FrequencyBandDecomposition:
    """Process video motion into multiple frequency bands."""
    
    def __init__(self):
        # Define frequency bands in Hz
        self.bands = {
            'ulf': (0.01, 0.1),  # Ultra-Low Frequency
            'lf': (0.1, 1.0),    # Low Frequency
            'mf': (1.0, 5.0),    # Mid Frequency
            'hf': (5.0, 15.0),   # High Frequency
            'uhf': (15.0, 50.0)  # Ultra-High Frequency
        }
        
        # Initialize filters for each band
        self.filters = {}
        self.initialize_filters()
        
    def initialize_filters(self, fps=30.0, filter_order=4):
        """Initialize bandpass filters for each frequency band."""
        nyquist = fps / 2.0
        
        for band_name, (low_freq, high_freq) in self.bands.items():
            # Normalize frequencies to Nyquist rate
            low_norm = low_freq / nyquist
            high_norm = high_freq / nyquist
            
            # Create Butterworth bandpass filter
            b, a = signal.butter(filter_order, [low_norm, high_norm], btype='band')
            self.filters[band_name] = (b, a)
    
    def analyze_motion_sequence(self, flow_sequence, fps=30.0):
        """
        Analyze a sequence of optical flow frames and decompose into frequency bands.
        
        Args:
            flow_sequence: List of optical flow frames (x and y components)
            fps: Frame rate of the video
            
        Returns:
            Dictionary of band-specific motion signals and analysis results
        """
        # Prepare storage for temporal signals
        sequence_length = len(flow_sequence)
        height, width = flow_sequence[0][0].shape
        
        # Reshape flow into temporal signals (one per pixel)
        x_flow_temporal = np.zeros((height, width, sequence_length))
        y_flow_temporal = np.zeros((height, width, sequence_length))
        
        # Fill in the temporal data
        for t, (flow_x, flow_y) in enumerate(flow_sequence):
            x_flow_temporal[:, :, t] = flow_x
            y_flow_temporal[:, :, t] = flow_y
        
        # Analyze each frequency band
        band_results = {}
        
        for band_name, (b, a) in self.filters.items():
            # Apply bandpass filter to each pixel's temporal signal
            x_filtered = signal.filtfilt(b, a, x_flow_temporal, axis=2)
            y_filtered = signal.filtfilt(b, a, y_flow_temporal, axis=2)
            
            # Calculate motion magnitude in this band
            magnitude = np.sqrt(x_filtered**2 + y_filtered**2)
            
            # Store filtered signals and derived metrics
            band_results[band_name] = {
                'x_component': x_filtered,
                'y_component': y_filtered,
                'magnitude': magnitude,
                'mean_magnitude': np.mean(magnitude),
                'peak_magnitude': np.max(magnitude),
                'variance': np.var(magnitude),
                'spatial_distribution': self._analyze_spatial_distribution(magnitude),
                'temporal_evolution': self._analyze_temporal_evolution(magnitude)
            }
        
        return band_results
    
    def _analyze_spatial_distribution(self, magnitude):
        """Analyze spatial distribution of motion in a frequency band."""
        # Compute average across time dimension
        avg_magnitude = np.mean(magnitude, axis=2)
        
        # Find regions with significant motion
        threshold = np.mean(avg_magnitude) + np.std(avg_magnitude)
        significant_regions = avg_magnitude > threshold
        
        # Label connected regions
        num_labels, labels = cv2.connectedComponents(significant_regions.astype(np.uint8))
        
        # Gather statistics for each significant region
        regions = []
        for label in range(1, num_labels):  # Skip background (0)
            region_mask = labels == label
            if np.sum(region_mask) > 10:  # Minimum region size
                region_y, region_x = np.where(region_mask)
                
                regions.append({
                    'x': int(np.min(region_x)),
                    'y': int(np.min(region_y)),
                    'width': int(np.max(region_x) - np.min(region_x)),
                    'height': int(np.max(region_y) - np.min(region_y)),
                    'strength': float(np.mean(avg_magnitude[region_mask])),
                })
        
        # Calculate spatial uniformity
        uniformity = 1.0 - np.std(avg_magnitude) / np.mean(avg_magnitude) if np.mean(avg_magnitude) > 0 else 0
        
        return {
            'regions': regions,
            'uniformity': float(uniformity)
        }
    
    def _analyze_temporal_evolution(self, magnitude):
        """Analyze how motion in a frequency band evolves over time."""
        # Calculate motion energy over time (summed across spatial dimensions)
        energy_over_time = np.sum(np.sum(magnitude, axis=0), axis=0)
        
        # Determine trend direction
        if len(energy_over_time) > 1:
            first_half = np.mean(energy_over_time[:len(energy_over_time)//2])
            second_half = np.mean(energy_over_time[len(energy_over_time)//2:])
            
            if second_half > first_half * 1.1:
                trend = "increasing"
            elif first_half > second_half * 1.1:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        # Calculate periodicity
        if len(energy_over_time) > 1:
            # Use autocorrelation to detect periodicity
            autocorr = np.correlate(energy_over_time, energy_over_time, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr /= autocorr[0]  # Normalize
            
            # Detect peaks in autocorrelation
            peaks, _ = signal.find_peaks(autocorr, height=0.2)
            
            if len(peaks) > 0:
                periodicity = 1.0 - np.std(np.diff(peaks)) / np.mean(np.diff(peaks)) if len(peaks) > 1 else 0.5
            else:
                periodicity = 0.0
        else:
            periodicity = 0.0
        
        # Detect change points
        change_points = []
        if len(energy_over_time) > 10:
            # Simple change point detection
            window_size = max(3, len(energy_over_time) // 10)
            for i in range(window_size, len(energy_over_time) - window_size):
                before = np.mean(energy_over_time[i-window_size:i])
                after = np.mean(energy_over_time[i:i+window_size])
                
                if abs(after - before) > 0.5 * np.std(energy_over_time):
                    change_points.append(float(i))
        
        return {
            'trend': trend,
            'periodicity': float(periodicity),
            'change_points': change_points
        }
```

### 2.2 Motion Type Classification

Implement motion classification to categorize patterns in each frequency band:

```python
# microservices/multi-band-analysis/motion_classifier.py
import numpy as np
from scipy import signal, stats

class MotionTypeClassifier:
    """Classify motion patterns into different types based on their characteristics."""
    
    def __init__(self):
        # Classification thresholds - these should be tuned based on your data
        self.thresholds = {
            'direction_coherence': 0.7,    # Threshold for directional motion
            'periodicity': 0.7,            # Threshold for oscillatory motion
            'entropy_max': 4.0,            # Maximum entropy for chaotic motion
            'spatial_correlation': 0.5     # Threshold for ambient motion
        }
    
    def classify_band_motion(self, band_data):
        """
        Classify motion patterns in a frequency band.
        
        Args:
            band_data: Dictionary containing band-specific motion signals
            
        Returns:
            Classification results with confidence scores
        """
        # Extract motion components from band data
        x_component = band_data['x_component']
        y_component = band_data['y_component']
        magnitude = band_data['magnitude']
        
        # Calculate metrics for different motion types
        directional_metrics = self._analyze_directional_motion(x_component, y_component)
        oscillatory_metrics = self._analyze_oscillatory_motion(magnitude)
        chaotic_metrics = self._analyze_chaotic_motion(magnitude)
        ambient_metrics = self._analyze_ambient_motion(magnitude)
        
        # Calculate classification confidence based on metrics
        dir_confidence = min(1.0, directional_metrics['direction_coherence'] / self.thresholds['direction_coherence'])
        osc_confidence = min(1.0, oscillatory_metrics['frequency_coherence'] / self.thresholds['periodicity'])
        chaos_confidence = min(1.0, chaotic_metrics['entropy'] / self.thresholds['entropy_max'])
        ambient_confidence = min(1.0, ambient_metrics['spatial_correlation'] / self.thresholds['spatial_correlation'])
        
        # Normalize confidences to sum to 1.0
        total_confidence = dir_confidence + osc_confidence + chaos_confidence + ambient_confidence
        if total_confidence > 0:
            dir_confidence /= total_confidence
            osc_confidence /= total_confidence
            chaos_confidence /= total_confidence
            ambient_confidence /= total_confidence
        
        # Assign motion type percentages
        motion_types = {
            'directional': {
                'percentage': float(dir_confidence),
                'confidence': float(min(1.0, directional_metrics['direction_coherence'] * 1.5)),
                'properties': directional_metrics
            },
            'oscillatory': {
                'percentage': float(osc_confidence),
                'confidence': float(min(1.0, oscillatory_metrics['frequency_coherence'] * 1.5)),
                'properties': oscillatory_metrics
            },
            'chaotic': {
                'percentage': float(chaos_confidence),
                'confidence': float(min(1.0, (1.0 - chaotic_metrics['predictability_score']) * 1.5)),
                'properties': chaotic_metrics
            },
            'ambient': {
                'percentage': float(ambient_confidence),
                'confidence': float(min(1.0, ambient_metrics['homogeneity'] * 1.5)),
                'properties': ambient_metrics
            }
        }
        
        return motion_types
    
    def _analyze_directional_motion(self, x_component, y_component):
        """Analyze directional motion characteristics."""
        # Average over time
        avg_x = np.mean(x_component, axis=2)
        avg_y = np.mean(y_component, axis=2)
        
        # Calculate average direction and magnitude
        magnitude = np.sqrt(avg_x**2 + avg_y**2)
        direction = np.arctan2(avg_y, avg_x) * 180 / np.pi
        
        # Non-zero magnitude points
        valid_points = magnitude > 0.01
        
        if np.sum(valid_points) > 0:
            # Calculate direction statistics
            valid_directions = direction[valid_points]
            mean_direction = stats.circmean(valid_directions * np.pi/180) * 180/np.pi
            direction_variance = stats.circstd(valid_directions * np.pi/180) * 180/np.pi
            
            # Direction coherence (higher is more coherent)
            direction_coherence = 1.0 - min(1.0, direction_variance / 90.0)
            
            # Mean velocity
            mean_velocity = np.mean(magnitude[valid_points])
        else:
            mean_direction = 0.0
            direction_variance = 180.0
            direction_coherence = 0.0
            mean_velocity = 0.0
        
        return {
            'primary_direction': float(mean_direction),
            'direction_variance': float(direction_variance),
            'direction_coherence': float(direction_coherence),
            'mean_velocity': float(mean_velocity)
        }
    
    def _analyze_oscillatory_motion(self, magnitude):
        """Analyze oscillatory motion characteristics."""
        # Average spatially to get temporal signal
        spatial_avg = np.mean(np.mean(magnitude, axis=0), axis=0)
        
        # Spectral analysis
        if len(spatial_avg) > 4:
            # Calculate FFT
            spectrum = np.abs(np.fft.rfft(spatial_avg))
            freqs = np.fft.rfftfreq(len(spatial_avg))
            
            # Find dominant frequency
            dominant_idx = np.argmax(spectrum[1:]) + 1  # Skip DC component
            primary_frequency = float(freqs[dominant_idx])
            
            # Calculate bandwidth (range around dominant frequency with significant power)
            threshold = 0.5 * spectrum[dominant_idx]
            bandwidth_indices = np.where(spectrum > threshold)[0]
            if len(bandwidth_indices) > 0:
                bandwidth = freqs[max(bandwidth_indices)] - freqs[min(bandwidth_indices)]
            else:
                bandwidth = 0.0
            
            # Calculate frequency coherence (how concentrated the energy is)
            frequency_coherence = spectrum[dominant_idx] / np.sum(spectrum) if np.sum(spectrum) > 0 else 0.0
            
            # Calculate phase coherence across spatial points
            phase_coherence = self._calculate_phase_coherence(magnitude)
            
            # Calculate amplitude statistics
            amplitude_mean = np.mean(spatial_avg)
            amplitude_variance = np.var(spatial_avg) / (amplitude_mean**2) if amplitude_mean > 0 else 0
        else:
            primary_frequency = 0.0
            bandwidth = 0.0
            frequency_coherence = 0.0
            phase_coherence = 0.0
            amplitude_mean = np.mean(spatial_avg) if len(spatial_avg) > 0 else 0.0
            amplitude_variance = 0.0
        
        return {
            'primary_frequency': float(primary_frequency),
            'frequency_bandwidth': float(bandwidth),
            'frequency_coherence': float(frequency_coherence),
            'amplitude_mean': float(amplitude_mean),
            'amplitude_variance': float(amplitude_variance),
            'phase_coherence': float(phase_coherence)
        }
    
    def _calculate_phase_coherence(self, magnitude):
        """Calculate phase coherence across spatial points."""
        # Sample a subset of points for computational efficiency
        h, w, t = magnitude.shape
        n_samples = min(100, h * w)
        
        if t < 4:  # Not enough time points for meaningful phase analysis
            return 0.0
        
        # Randomly sample points
        points_h = np.random.randint(0, h, size=n_samples)
        points_w = np.random.randint(0, w, size=n_samples)
        
        # Extract time series for sampled points
        time_series = magnitude[points_h, points_w, :]
        
        # Calculate phase using Hilbert transform
        analytic_signal = signal.hilbert(time_series)
        phase = np.angle(analytic_signal)
        
        # Calculate phase coherence
        # (average length of the resultant vector when phases are represented as unit vectors)
        n_times = phase.shape[1]
        sum_cos = np.sum(np.cos(phase), axis=0)
        sum_sin = np.sum(np.sin(phase), axis=0)
        resultant_length = np.sqrt(sum_cos**2 + sum_sin**2) / n_samples
        
        # Average across time
        return float(np.mean(resultant_length))
    
    def _analyze_chaotic_motion(self, magnitude):
        """Analyze chaotic motion characteristics."""
        # Flatten spatial dimensions
        h, w, t = magnitude.shape
        flattened = magnitude.reshape(h*w, t)
        
        # Sample a subset of points
        n_samples = min(100, h * w)
        sample_indices = np.random.choice(h*w, size=n_samples, replace=False)
        sampled = flattened[sample_indices]
        
        if t < 4:  # Not enough time points for meaningful analysis
            return {
                'entropy': 0.0,
                'lyapunov_exponent': 0.0,
                'predictability_score': 1.0
            }
        
        # Calculate sample entropy for each time series
        entropy_values = []
        for i in range(n_samples):
            if np.std(sampled[i]) > 0:
                # Normalize time series
                normalized = (sampled[i] - np.mean(sampled[i])) / np.std(sampled[i])
                # Approximate entropy using histogram
                hist, _ = np.histogram(normalized, bins=10)
                hist = hist / np.sum(hist)
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                entropy_values.append(entropy)
        
        # Average entropy across samples
        entropy = np.mean(entropy_values) if entropy_values else 0.0
        
        # Approximate Lyapunov exponent (measure of chaos)
        # For simplicity, we use a proxy measure based on prediction error
        lyapunov_exponent = 0.0
        predictability_score = 0.0
        
        if t > 10:
            prediction_error = 0.0
            count = 0
            
            # For each sample, try to predict next value from previous values
            for i in range(n_samples):
                if np.std(sampled[i]) > 0:
                    for j in range(5, t-1):
                        # Use previous 5 points to predict next point (simple average)
                        predicted = np.mean(sampled[i, j-5:j])
                        actual = sampled[i, j]
                        error = abs(predicted - actual) / np.std(sampled[i])
                        prediction_error += error
                        count += 1
            
            if count > 0:
                prediction_error /= count
                # Convert to Lyapunov exponent proxy
                lyapunov_exponent = np.log(max(1.01, prediction_error))
                # Predictability score (inverse of chaos)
                predictability_score = 1.0 / (1.0 + prediction_error)
        
        return {
            'entropy': float(entropy),
            'lyapunov_exponent': float(lyapunov_exponent),
            'predictability_score': float(predictability_score)
        }
    
    def _analyze_ambient_motion(self, magnitude):
        """Analyze ambient motion characteristics."""
        h, w, t = magnitude.shape
        
        # Spatial correlation
        spatial_correlation = 0.0
        if h > 1 and w > 1:
            # Calculate correlation between adjacent pixels
            correlations = []
            for time_idx in range(t):
                frame = magnitude[:, :, time_idx]
                # Horizontal correlations
                for i in range(h):
                    corr = np.corrcoef(frame[i, :-1], frame[i, 1:])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                
                # Vertical correlations
                for j in range(w):
                    corr = np.corrcoef(frame[:-1, j], frame[1:, j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                spatial_correlation = np.mean(correlations)
        
        # Temporal consistency
        temporal_consistency = 0.0
        if t > 1:
            # Calculate temporal correlation between consecutive frames
            correlations = []
            for i in range(min(10, h)):  # Sample some rows
                for j in range(min(10, w)):  # Sample some columns
                    pixel_series = magnitude[i, j, :]
                    if np.std(pixel_series) > 0:
                        autocorr = np.correlate(pixel_series, pixel_series, mode='full')
                        autocorr = autocorr[len(autocorr)//2:]
                        autocorr /= autocorr[0]
                        if len(autocorr) > 1:
                            correlations.append(autocorr[1])
            
            if correlations:
                temporal_consistency = np.mean(correlations)
        
        # Homogeneity
        homogeneity = 0.0
        if t > 0:
            # Calculate spatial homogeneity across frames
            variances = []
            for time_idx in range(t):
                frame = magnitude[:, :, time_idx]
                if np.mean(frame) > 0:
                    normalized_variance = np.var(frame) / (np.mean(frame)**2)
                    variances.append(normalized_variance)
            
            if variances:
                # Lower variance -> higher homogeneity
                homogeneity = 1.0 / (1.0 + np.mean(variances))
        
        return {
            'spatial_correlation': float(spatial_correlation),
            'temporal_consistency': float(temporal_consistency),
            'homogeneity': float(homogeneity)
        }
```

### 2.3 Main Service Application

Create a NATS-connected service application:

```python
# microservices/multi-band-analysis/app.py
import asyncio
import json
import cv2
import numpy as np
import nats
from nats.errors import TimeoutError

from frequency_bands import FrequencyBandDecomposition
from motion_classifier import MotionTypeClassifier
from optical_flow import OpticalFlowProcessor

class MultiFrequencyMotionAnalyzer:
    """Main service for multi-band motion analysis."""
    
    def __init__(self):
        self.band_decomposer = FrequencyBandDecomposition()
        self.motion_classifier = MotionTypeClassifier()
        self.flow_processor = OpticalFlowProcessor()
        self.nc = None  # NATS connection
        
    async def connect_nats(self, url="nats://nats:4222"):
        """Connect to NATS server."""
        self.nc = await nats.connect(url)
        print(f"Connected to NATS at {url}")
        
        # Subscribe to video analysis requests
        await self.nc.subscribe("video.analyze", cb=self.handle_video_analysis)
        await self.nc.subscribe("flow.analyze", cb=self.handle_flow_analysis)
        
        print("Subscribed to analysis requests")
    
    async def handle_video_analysis(self, msg):
        """Handle incoming video analysis requests."""
        try:
            # Parse the request
            request = json.loads(msg.data.decode())
            video_path = request.get("video_path")
            frame_range = request.get("frame_range", [0, -1])
            
            print(f"Processing video: {video_path}, frames {frame_range}")
            
            # Process the video
            flow_sequence, fps = self.flow_processor.process_video(video_path, frame_range)
            
            # Process the flow sequence
            result = await self.process_flow_sequence(flow_sequence, fps)
            
            # Send the response
            if msg.reply:
                response = {
                    "status": "success",
                    "result": result
                }
                await self.nc.publish(msg.reply, json.dumps(response).encode())
                
            # Publish the results as an event
            await self.nc.publish("analysis.results", json.dumps({
                "video_path": video_path,
                "result": result
            }).encode())
            
        except Exception as e:
            print(f"Error processing video analysis request: {e}")
            if msg.reply:
                error_response = {
                    "status": "error",
                    "message": str(e)
                }
                await self.nc.publish(msg.reply, json.dumps(error_response).encode())
    
    async def handle_flow_analysis(self, msg):
        """Handle incoming optical flow analysis requests."""
        try:
            # Parse the request
            request = json.loads(msg.data.decode())
            flow_data = request.get("flow_sequence")
            fps = request.get("fps", 30.0)
            
            # Convert flow data to numpy arrays
            flow_sequence = []
            for frame in flow_data:
                flow_x = np.array(frame["x"], dtype=np.float32)
                flow_y = np.array(frame["y"], dtype=np.float32)
                flow_sequence.append((flow_x, flow_y))
            
            # Process the flow sequence
            result = await self.process_flow_sequence(flow_sequence, fps)
            
            # Send the response
            if msg.reply:
                response = {
                    "status": "success",
                    "result": result
                }
                await self.nc.publish(msg.reply, json.dumps(response).encode())
            
        except Exception as e:
            print(f"Error processing flow analysis request: {e}")
            if msg.reply:
                error_response = {
                    "status": "error",
                    "message": str(e)
                }
                await self.nc.publish(msg.reply, json.dumps(error_response).encode())
    
    async def process_flow_sequence(self, flow_sequence, fps=30.0):
        """Process a sequence of optical flow frames."""
        # Analyze motion in different frequency bands
        band_results = self.band_decomposer.analyze_motion_sequence(flow_sequence, fps)
        
        # Classify motion types in each band
        analysis_results = {}
        for band_name, band_data in band_results.items():
            motion_types = self.motion_classifier.classify_band_motion(band_data)
            
            # Create band summary with global statistics and motion types
            band_summary = {
                "frequency_range": list(self.band_decomposer.bands[band_name]),
                "global_statistics": {
                    "mean_magnitude": float(band_data["mean_magnitude"]),
                    "peak_magnitude": float(band_data["peak_magnitude"]),
                    "variance": float(band_data["variance"])
                },
                "motion_types": motion_types,
                "spatial_distribution": band_data["spatial_distribution"],
                "temporal_evolution": band_data["temporal_evolution"]
            }
            
            analysis_results[band_name] = band_summary
        
        return analysis_results

async def main():
    """Main function to start the service."""
    analyzer = MultiFrequencyMotionAnalyzer()
    await analyzer.connect_nats()
    
    # Keep the service running
    while True:
        try:
            await asyncio.sleep(3600)  # Sleep for an hour
        except KeyboardInterrupt:
            break
    
    # Clean up
    await analyzer.nc.drain()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2.4 Frontend Visualization Component

Create a WebGL-powered visualization component for the frequency bands:

```jsx
// frontend/services/analysis-dashboard/src/components/FrequencyBandVisualizer.jsx
import React, { useEffect, useRef } from 'react';
import { useSharedContext } from '../../../../lib/contexts/SharedContext';
import * as THREE from 'three';

const FrequencyBandVisualizer = ({ bandName }) => {
  const canvasRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const meshRef = useRef(null);
  
  const { analysisResults } = useSharedContext();
  
  useEffect(() => {
    if (!canvasRef.current) return;
    
    // Set up Three.js scene
    const width = canvasRef.current.clientWidth;
    const height = canvasRef.current.clientHeight;
    
    // Create scene if it doesn't exist
    if (!sceneRef.current) {
      sceneRef.current = new THREE.Scene();
      
      // Create camera
      cameraRef.current = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
      cameraRef.current.position.z = 2;
      
      // Create renderer
      rendererRef.current = new THREE.WebGLRenderer({ 
        canvas: canvasRef.current,
        alpha: true,
        antialias: true
      });
      rendererRef.current.setSize(width, height);
      
      // Create initial mesh with placeholder geometry
      const geometry = new THREE.PlaneGeometry(2, 2, 50, 50);
      const material = new THREE.MeshBasicMaterial({
        color: getBandColor(bandName),
        wireframe: true,
        transparent: true,
        opacity: 0.7
      });
      
      meshRef.current = new THREE.Mesh(geometry, material);
      sceneRef.current.add(meshRef.current);
    }
    
    // Set up animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      
      if (meshRef.current) {
        meshRef.current.rotation.x += 0.005;
        meshRef.current.rotation.y += 0.005;
      }
      
      rendererRef.current.render(sceneRef.current, cameraRef.current);
    };
    
    animate();
    
    // Handle window resize
    const handleResize = () => {
      if (!canvasRef.current || !cameraRef.current || !rendererRef.current) return;
      
      const width = canvasRef.current.clientWidth;
      const height = canvasRef.current.clientHeight;
      
      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      
      rendererRef.current.setSize(width, height);
    };
    
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      
      // Clean up Three.js resources
      if (meshRef.current) {
        sceneRef.current.remove(meshRef.current);
        meshRef.current.geometry.dispose();
        meshRef.current.material.dispose();
      }
      
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, [bandName]);
  
  // Update visualization when analysis results change
  useEffect(() => {
    if (!analysisResults || !analysisResults[bandName] || !meshRef.current) return;
    
    const bandData = analysisResults[bandName];
    
    // Update the geometry based on motion data
    updateVisualization(bandData);
    
  }, [analysisResults, bandName]);
  
  const updateVisualization = (bandData) => {
    if (!meshRef.current) return;
    
    const geometry = meshRef.current.geometry;
    const positionAttribute = geometry.getAttribute('position');
    const positions = positionAttribute.array;
    
    // Example: Update vertex positions based on motion types
    const motionTypes = bandData.motion_types;
    const amplitude = bandData.global_statistics.mean_magnitude * 0.5;
    
    // Apply different patterns based on dominant motion type
    const dominantType = getDominantMotionType(motionTypes);
    
    for (let i = 0; i < positions.length; i += 3) {
      const vertexIndex = i / 3;
      const x = positions[i];
      const y = positions[i + 1];
      
      let z = 0;
      
      // Pattern based on motion type
      switch (dominantType) {
        case 'directional':
          // Linear gradient based on primary direction
          const angle = motionTypes.directional.properties.primary_direction * Math.PI / 180;
          z = amplitude * (Math.cos(angle) * x + Math.sin(angle) * y);
          break;
          
        case 'oscillatory':
          // Ripple pattern
          const frequency = motionTypes.oscillatory.properties.primary_frequency * 10;
          const distance = Math.sqrt(x * x + y * y);
          z = amplitude * Math.sin(distance * frequency);
          break;
          
        case 'chaotic':
          // Random terrain
          z = amplitude * (Math.sin(x * 10) * Math.cos(y * 10) * Math.sin(x * y * 5));
          break;
          
        case 'ambient':
          // Gentle noise
          z = amplitude * (Math.sin(x * 5) * Math.sin(y * 5));
          break;
          
        default:
          z = 0;
      }
      
      positions[i + 2] = z;
    }
    
    positionAttribute.needsUpdate = true;
    geometry.computeVertexNormals();
  };
  
  const getDominantMotionType = (motionTypes) => {
    let maxPercentage = 0;
    let dominantType = 'none';
    
    for (const [type, data] of Object.entries(motionTypes)) {
      if (data.percentage > maxPercentage) {
        maxPercentage = data.percentage;
        dominantType = type;
      }
    }
    
    return dominantType;
  };
  
  const getBandColor = (bandName) => {
    // Color scheme for different frequency bands
    const colors = {
      ulf: 0x3333ff, // Deep blue for ultra-low
      lf: 0x33cc33,  // Green for low
      mf: 0xffcc00,  // Yellow for mid
      hf: 0xff6600,  // Orange for high
      uhf: 0xff3333,  // Red for ultra-high
    };
    
    return colors[bandName] || 0xcccccc;
  };
  
  return (
    <div className="frequency-band-visualizer">
      <h3>{getBandLabel(bandName)}</h3>
      <canvas 
        ref={canvasRef} 
        className="visualizer-canvas" 
        style={{ width: '100%', height: '200px' }}
      />
    </div>
  );
};

// Helper function to get user-friendly band labels
const getBandLabel = (bandName) => {
  const labels = {
    ulf: 'Ultra-Low Frequency (0.01-0.1 Hz)',
    lf: 'Low Frequency (0.1-1 Hz)',
    mf: 'Mid Frequency (1-5 Hz)',
    hf: 'High Frequency (5-15 Hz)',
    uhf: 'Ultra-High Frequency (15+ Hz)',
  };
  
  return labels[bandName] || bandName;
};

export default FrequencyBandVisualizer;
```

## 3. EDM Production Integration

### 3.1 Audio Parameter Mapping Matrix

Create a comprehensive mapping matrix for audio parameters:

```typescript
// frontend/services/parameter-mapper/src/models/ParameterMatrix.ts
export interface TransferFunction {
  type: 'linear' | 'exponential' | 'logarithmic' | 'sigmoid' | 'custom';
  inputRange: [number, number];
  outputRange: [number, number];
  // Additional parameters based on function type
  slope?: number;       // for linear
  exponent?: number;    // for exponential/logarithmic
  midpoint?: number;    // for sigmoid
  custom?: string;      // for custom (function expression)
}

export interface ParameterModulation {
  source: string;       // Source parameter path
  amount: number;       // Modulation amount (0-1)
  type: 'additive' | 'multiplicative' | 'frequency' | 'amplitude';
}

export interface ParameterMapping {
  id: string;
  source: {
    type: 'oscillatory' | 'directional' | 'chaotic' | 'ambient';
    band: 'ulf' | 'lf' | 'mf' | 'hf' | 'uhf';
    property: string;   // e.g. 'primary_frequency', 'amplitude_mean'
  };
  destination: string;  // e.g. 'wobble_bass.lfo_rate'
  transferFunction: TransferFunction;
  modulation?: ParameterModulation;
  enabled: boolean;
}

export interface ParameterMatrixTemplate {
  id: string;
  name: string;
  description: string;
  author: string;
  created: string;
  mappings: ParameterMapping[];
  metadata: {
    targetGenre?: string;
    bpm?: number;
    tags?: string[];
  };
}

// Available audio synthesis parameters
export const AUDIO_PARAMETERS = {
  wobble_bass: {
    lfo_rate: {
      name: 'LFO Rate',
      min: 0.1,
      max: 20,
      default: 1,
      unit: 'Hz',
      description: 'Rate of LFO modulation for wobble effect'
    },
    filter_cutoff: {
      name: 'Filter Cutoff',
      min: 80,
      max: 10000,
      default: 1000,
      unit: 'Hz',
      description: 'Cutoff frequency of the filter'
    },
    filter_resonance: {
      name: 'Filter Resonance',
      min: 0,
      max: 20,
      default: 5,
      unit: 'Q',
      description: 'Resonance of the filter for emphasizing cutoff frequency'
    },
    filter_type: {
      name: 'Filter Type',
      options: ['lowpass', 'highpass', 'bandpass', 'notch'],
      default: 'lowpass',
      description: 'Type of filter used in the wobble bass'
    },
    waveform: {
      name: 'Waveform',
      options: ['sine', 'triangle', 'sawtooth', 'square', 'custom'],
      default: 'sawtooth',
      description: 'Basic waveform shape for the oscillator'
    },
    distortion_amount: {
      name: 'Distortion',
      min: 0,
      max: 1,
      default: 0.2,
      description: 'Amount of distortion applied to the sound'
    }
  },
  rhythm: {
    kick_pattern: {
      name: 'Kick Pattern',
      type: 'rhythm',
      description: 'Pattern of kick drum hits'
    },
    hat_density: {
      name: 'Hi-hat Density',
      min: 0,
      max: 1,
      default: 0.5,
      description: 'Density of hi-hat patterns'
    },
    percussion_complexity: {
      name: 'Percussion Complexity',
      min: 0,
      max: 1,
      default: 0.3,
      description: 'Complexity of percussion patterns'
    },
    swing_amount: {
      name: 'Swing',
      min: 0,
      max: 0.5,
      default: 0,
      description: 'Amount of swing/groove applied to the rhythm'
    }
  },
  harmony: {
    chord_complexity: {
      name: 'Chord Complexity',
      min: 1,
      max: 5,
      default: 2,
      description: 'Complexity of generated chords (triad, 7th, 9th, etc.)'
    },
    progression_type: {
      name: 'Progression Type',
      options: ['simple', 'moderate', 'complex', 'chromatic'],
      default: 'moderate',
      description: 'Type of chord progression to generate'
    },
    key: {
      name: 'Key',
      options: ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
      default: 'C',
      description: 'Musical key for harmonic content'
    },
    scale: {
      name: 'Scale',
      options: ['major', 'minor', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'locrian'],
      default: 'minor',
      description: 'Musical scale for melodic/harmonic content'
    }
  },
  atmosphere: {
    pad_intensity: {
      name: 'Pad Intensity',
      min: 0,
      max: 1,
      default: 0.5,
      description: 'Intensity of atmospheric pad sounds'
    },
    reverb_size: {
      name: 'Reverb Size',
      min: 0,
      max: 1,
      default: 0.5,
      description: 'Size of reverb space'
    },
    texture_density: {
      name: 'Texture Density',
      min: 0,
      max: 1,
      default: 0.3,
      description: 'Density of atmospheric textures'
    },
    modulation_depth: {
      name: 'Modulation Depth',
      min: 0,
      max: 1,
      default: 0.2,
      description: 'Depth of modulation effects on atmospheric sounds'
    }
  },
  global: {
    bpm: {
      name: 'Tempo',
      min: 60,
      max: 200,
      default: 140,
      unit: 'BPM',
      description: 'Tempo of the generated music'
    },
    mix_balance: {
      name: 'Mix Balance',
      min: 0,
      max: 1,
      default: 0.5,
      description: 'Balance between different elements in the mix'
    },
    energy: {
      name: 'Energy Level',
      min: 0,
      max: 1,
      default: 0.7,
      description: 'Overall energy level of the output'
    },
    dynamics_range: {
      name: 'Dynamics Range',
      min: 0,
      max: 1,
      default: 0.6,
      description: 'Range between quiet and loud sections'
    }
  }
};

// Default parameter matrix templates
export const DEFAULT_TEMPLATES: ParameterMatrixTemplate[] = [
  {
    id: 'template-dubstep-basic',
    name: 'Basic Dubstep',
    description: 'Standard dubstep parameter mappings focused on wobble bass and aggressive rhythms',
    author: 'System',
    created: new Date().toISOString(),
    mappings: [
      {
        id: 'map-mf-lfo',
        source: {
          type: 'oscillatory',
          band: 'mf',
          property: 'primary_frequency'
        },
        destination: 'wobble_bass.lfo_rate',
        transferFunction: {
          type: 'exponential',
          inputRange: [0.5, 5.0],
          outputRange: [0.25, 8.0],
          exponent: 1.5
        },
        enabled: true
      },
      {
        id: 'map-lf-filter',
        source: {
          type: 'oscillatory',
          band: 'lf',
          property: 'amplitude_mean'
        },
        destination: 'wobble_bass.filter_cutoff',
        transferFunction: {
          type: 'exponential',
          inputRange: [0.0, 1.0],
          outputRange: [100, 5000],
          exponent: 2.0
        },
        enabled: true
      },
      {
        id: 'map-hf-hats',
        source: {
          type: 'directional',
          band: 'hf',
          property: 'mean_velocity'
        },
        destination: 'rhythm.hat_density',
        transferFunction: {
          type: 'linear',
          inputRange: [0.0, 0.5],
          outputRange: [0.2, 0.9],
          slope: 1.4
        },
        enabled: true
      },
      {
        id: 'map-uhf-texture',
        source: {
          type: 'ambient',
          band: 'uhf',
          property: 'spatial_correlation'
        },
        destination: 'atmosphere.texture_density',
        transferFunction: {
          type: 'linear',
          inputRange: [0.0, 1.0],
          outputRange: [0.1, 0.8],
          slope: 0.7
        },
        enabled: true
      },
      {
        id: 'map-ulf-energy',
        source: {
          type: 'directional',
          band: 'ulf',
          property: 'mean_velocity'
        },
        destination: 'global.energy',
        transferFunction: {
          type: 'sigmoid',
          inputRange: [0.0, 0.5],
          outputRange: [0.3, 0.9],
          midpoint: 0.25
        },
        enabled: true
      }
    ],
    metadata: {
      targetGenre: 'Dubstep',
      bpm: 140,
      tags: ['wobble', 'bass-heavy', 'aggressive']
    }
  }
];
```

## 4. Next Steps

To start implementing the Project Sonique pivot:

1. **Set up the microservice infrastructure**:
   - Create the directory structure
   - Set up Docker configurations with NVIDIA support
   - Configure NATS message broker
   - Implement the shared context system

2. **Implement the Multi-Band Motion Analysis Framework**:
   - Build the frequency band decomposition
   - Create the motion type classification
   - Develop the Python service
   - Create the WebGL visualization components

3. **Set up frontend micro-frontend architecture**:
   - Configure module federation
   - Implement the shared context providers
   - Create the basic UI components

4. **Develop initial mappings and templates**:
   - Implement the parameter mapping matrix
   - Create default templates for common EDM styles
   - Set up basic audio generation

By focusing on these key areas, we can quickly establish the foundation for the Project Sonique vision while providing a clear path for future development.
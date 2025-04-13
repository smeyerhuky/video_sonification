#!/usr/bin/env python3
import os
import json
import time
import uuid
import logging
import numpy as np
import cv2
from tqdm import tqdm
from scipy import signal
import torch
import librosa
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Check for CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA is not available. Using CPU only.")

# Create upload directory if it doesn't exist
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'video_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        
    def initialize_filters(self, fps=30.0, filter_order=4):
        """Initialize bandpass filters for each frequency band."""
        nyquist = fps / 2.0
        
        for band_name, (low_freq, high_freq) in self.bands.items():
            # Normalize frequencies to Nyquist rate and ensure valid range
            # Must have 0 < low_norm < high_norm < 1
            
            # Set low_norm between 0.01 and 0.8
            low_norm = max(0.01, min(0.8, low_freq / nyquist))
            
            # Set high_norm between low_norm+0.1 and 0.95
            # This ensures high_norm is always > low_norm
            high_norm = max(low_norm + 0.1, min(0.95, high_freq / nyquist))
            
            # Log the normalized frequencies
            logger.info(f"Band {band_name}: Low cutoff = {low_norm}, High cutoff = {high_norm}")
            
            # Create Butterworth bandpass filter
            b, a = signal.butter(filter_order, [low_norm, high_norm], btype='band')
            self.filters[band_name] = (b, a)
    
    def compute_optical_flow(self, video_path):
        """Compute optical flow for a video file."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize filters with video's FPS
        self.initialize_filters(fps=fps)
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Could not read video file")
        
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Store flow vectors
        flow_sequence = []
        processed_frames = 0
        
        # Initialize progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        while True:
            # Read next frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Store flow components
            flow_sequence.append((flow[..., 0], flow[..., 1]))
            
            # Update previous frame
            prev_gray = gray
            processed_frames += 1
        # Close progress bar
        if pbar:
            pbar.close()
            pbar.update(1)
        
        # Close progress bar
        pbar.close()
        cap.release()
        logger.info(f"Computed optical flow for {processed_frames} frames at {fps} FPS")
        return flow_sequence, fps
    
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
        if sequence_length == 0:
            raise ValueError("Empty flow sequence")
            
        height, width = flow_sequence[0][0].shape
        
        # Reshape flow into temporal signals (one per pixel)
        x_flow_temporal = np.zeros((height, width, sequence_length))
        y_flow_temporal = np.zeros((height, width, sequence_length))
        
        # Fill in the temporal data with progress tracking
        for t, (flow_x, flow_y) in enumerate(tqdm(flow_sequence, desc="Processing temporal data")):
            x_flow_temporal[:, :, t] = flow_x
            y_flow_temporal[:, :, t] = flow_y
        
        # Analyze each frequency band
        band_results = {}
        
        for band_name, (b, a) in self.filters.items():
            logger.info(f"Analyzing {band_name} band")
            
            # Apply bandpass filter to each pixel's temporal signal
            x_filtered = signal.filtfilt(b, a, x_flow_temporal, axis=2)
            y_filtered = signal.filtfilt(b, a, y_flow_temporal, axis=2)
            
            # Calculate motion magnitude in this band
            magnitude = np.sqrt(x_filtered**2 + y_filtered**2)
            
            # Store filtered signals and derived metrics
            band_results[band_name] = {
                'mean_magnitude': float(np.mean(magnitude)),
                'peak_magnitude': float(np.max(magnitude)),
                'variance': float(np.var(magnitude)),
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

class SimpleAudioGenerator:
    """Generate audio based on motion analysis results."""
    
    def __init__(self, sr=44100, duration=10):
        self.sr = sr
        self.duration = duration
    
    def generate_wobble_bass(self, analysis_results, output_path):
        """Generate a wobble bass sound based on motion analysis."""
        # Extract parameters from analysis results
        mf_band = analysis_results.get('mf', {})
        lf_band = analysis_results.get('lf', {})
        
        # Default parameters
        lfo_rate = 1.0  # Hz
        cutoff_min = 200  # Hz
        cutoff_max = 2000  # Hz
        
        # Map MF band to LFO rate if available
        if mf_band:
            # Use periodicity as primary driver for LFO rate
            periodicity = mf_band.get('temporal_evolution', {}).get('periodicity', 0.5)
            lfo_rate = 0.5 + 7.5 * periodicity  # Map to range 0.5-8 Hz
        
        # Map LF band to filter cutoff range if available
        if lf_band:
            mean_magnitude = lf_band.get('mean_magnitude', 0.1)
            cutoff_min = 100 + 400 * mean_magnitude  # Map to range 100-500 Hz
            cutoff_max = 1000 + 4000 * mean_magnitude  # Map to range 1000-5000 Hz
        
        # Generate time array
        t = np.linspace(0, self.duration, int(self.duration * self.sr), endpoint=False)
        
        # Generate carrier signal (sawtooth wave at 55 Hz - A1)
        carrier_freq = 55
        carrier = 0.5 * signal.sawtooth(2 * np.pi * carrier_freq * t)
        
        # Generate LFO for filter cutoff
        lfo = 0.5 * (1 + np.sin(2 * np.pi * lfo_rate * t))
        
        # Map LFO to filter cutoff
        cutoff = cutoff_min + (cutoff_max - cutoff_min) * lfo
        
        # Apply time-varying filter
        filtered_signal = np.zeros_like(carrier)
        # Process in small chunks to simulate time-varying filter
        chunk_size = int(0.01 * self.sr)  # 10ms chunks
        num_chunks = (len(t) + chunk_size - 1) // chunk_size
        
        logger.info(f"Processing {num_chunks} audio chunks")
        for i in tqdm(range(0, len(t), chunk_size), desc="Generating audio", total=num_chunks):
            chunk_end = min(i + chunk_size, len(t))
            chunk = carrier[i:chunk_end]
            
            
            # Get average cutoff for this chunk
            avg_cutoff = np.mean(cutoff[i:chunk_end])
            
            # Design filter for this cutoff
            nyquist = self.sr / 2
            normalized_cutoff = avg_cutoff / nyquist
            b, a = signal.butter(4, normalized_cutoff, btype='low')
            
            # Apply filter to chunk
            filtered_chunk = signal.lfilter(b, a, chunk)
            filtered_signal[i:chunk_end] = filtered_chunk
        
        # Apply resonance (slight boost around cutoff)
        resonance = np.zeros_like(filtered_signal)
        for i in tqdm(range(0, len(t), chunk_size), desc="Adding resonance", total=num_chunks):
            chunk_end = min(i + chunk_size, len(t))
            chunk = filtered_signal[i:chunk_end]
            
            # Get average cutoff for this chunk
            avg_cutoff = np.mean(cutoff[i:chunk_end])
            
            # Design bandpass filter centered at cutoff
            nyquist = self.sr / 2
            normalized_cutoff = avg_cutoff / nyquist
            
            # Ensure reasonable values for bandpass filter
            normalized_cutoff = max(0.2, min(0.8, normalized_cutoff))
            bandwidth = max(0.1, 0.1 * normalized_cutoff)
            
            # Calculate low and high cutoffs ensuring sufficient separation
            low_cut = max(0.05, normalized_cutoff - bandwidth/2)
            high_cut = min(0.95, normalized_cutoff + bandwidth/2)
            
            # Ensure high_cut is always greater than low_cut
            if high_cut <= low_cut + 0.1:
                high_cut = min(0.95, low_cut + 0.1)
            
            # Add debug logging
            logger.info(f"Audio filter: Low cutoff = {low_cut}, High cutoff = {high_cut}")
            
            b, a = signal.butter(2, [low_cut, high_cut], btype='band')
            
            # Apply filter to chunk
            resonant_chunk = signal.lfilter(b, a, chunk)
            resonance[i:chunk_end] = resonant_chunk
        
        # Mix filtered signal with resonance
        mixed_signal = 0.7 * filtered_signal + 0.3 * resonance
        
        # Normalize
        mixed_signal = mixed_signal / np.max(np.abs(mixed_signal))
        
        # Save as WAV file
        import scipy.io.wavfile as wav
        wav.write(output_path, self.sr, (mixed_signal * 32767).astype(np.int16))
        
        return {
            'lfo_rate': lfo_rate,
            'cutoff_min': cutoff_min,
            'cutoff_max': cutoff_max,
            'duration': self.duration,
            'sample_rate': self.sr,
            'file_path': output_path
        }

# Initialize analysis components
frequency_decomposer = FrequencyBandDecomposition()
audio_generator = SimpleAudioGenerator()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'cuda_available': CUDA_AVAILABLE,
        'timestamp': time.time()
    })

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Analyze video and generate audio based on motion patterns."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'Empty video filename'}), 400
    
    # Save uploaded file
    video_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_FOLDER, f"{video_id}_{video_file.filename}")
    video_file.save(video_path)
    
    try:
        # Compute optical flow
        logger.info(f"Computing optical flow for {video_path}")
        flow_sequence, fps = frequency_decomposer.compute_optical_flow(video_path)
        
        # Analyze motion
        logger.info("Analyzing motion patterns")
        analysis_results = frequency_decomposer.analyze_motion_sequence(flow_sequence, fps)
        
        # Generate audio
        logger.info("Generating audio")
        audio_path = os.path.join(UPLOAD_FOLDER, f"{video_id}_wobble.wav")
        audio_info = audio_generator.generate_wobble_bass(analysis_results, audio_path)
        
        # Prepare response
        response = {
            'video_id': video_id,
            'analysis_results': analysis_results,
            'audio_info': audio_info
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up video file, but keep the audio file
        try:
            os.remove(video_path)
        except:
            pass
# Audio endpoint
@app.route('/audio/<video_id>', methods=['GET'])
def get_audio(video_id):
    """Serve the generated audio file for a specific video ID."""
    # Construct the expected audio file path
    audio_path = os.path.join(UPLOAD_FOLDER, f"{video_id}_wobble.wav")
    
    # Check if file exists
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return jsonify({'error': 'Audio file not found'}), 404
    
    # Return the audio file
    return send_file(audio_path, mimetype='audio/wav')

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('SERVICE_PORT', 5000))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)
    app.run(host='0.0.0.0', port=port, debug=False)
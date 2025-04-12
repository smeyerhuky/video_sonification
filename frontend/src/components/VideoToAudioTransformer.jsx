import React, { useState, useEffect, useRef } from 'react';

const VideoToAudioTransformer = () => {
    // State for component configuration
    const [inputSource, setInputSource] = useState('webcam'); // 'webcam' or 'upload'
    const [isCapturing, setIsCapturing] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [audioEnabled, setAudioEnabled] = useState(false);

    // Motion analysis state
    const [motionLevel, setMotionLevel] = useState(0);
    const [motionThreshold, setMotionThreshold] = useState(15);

    // Audio parameters
    const [bassFreq, setBassFreq] = useState(60);
    const [lfoRate, setLfoRate] = useState(4);
    const [filterQ, setFilterQ] = useState(10);

    // Refs for DOM elements and processing
    const videoRef = useRef(null);
    const uploadRef = useRef(null);
    const canvasRef = useRef(null);
    const outputCanvasRef = useRef(null);
    const prevFrameRef = useRef(null);
    const animationRef = useRef(null);

    // Audio context and nodes
    const audioContextRef = useRef(null);
    const oscillatorRef = useRef(null);
    const filterRef = useRef(null);
    const lfoRef = useRef(null);
    const gainRef = useRef(null);

    // Initialize audio context and setup audio nodes
    useEffect(() => {
        if (audioEnabled && !audioContextRef.current) {
            try {
                // Create audio context
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                audioContextRef.current = audioCtx;

                // Create oscillator (bass sound source)
                const oscillator = audioCtx.createOscillator();
                oscillator.type = 'sawtooth';
                oscillator.frequency.value = bassFreq;
                oscillatorRef.current = oscillator;

                // Create filter for the wobble effect
                const filter = audioCtx.createBiquadFilter();
                filter.type = 'lowpass';
                filter.frequency.value = 1000;
                filter.Q.value = filterQ;
                filterRef.current = filter;

                // Create LFO for filter modulation (the "wobble")
                const lfo = audioCtx.createOscillator();
                lfo.type = 'sine';
                lfo.frequency.value = lfoRate;
                lfoRef.current = lfo;

                // Create gain node for LFO
                const lfoGain = audioCtx.createGain();
                lfoGain.gain.value = 500;

                // Create main gain node for volume control
                const gain = audioCtx.createGain();
                gain.gain.value = 0;
                gainRef.current = gain;

                // Connect everything
                oscillator.connect(filter);
                filter.connect(gain);
                gain.connect(audioCtx.destination);

                lfo.connect(lfoGain);
                lfoGain.connect(filter.frequency);

                // Start oscillators
                oscillator.start();
                lfo.start();
            } catch (err) {
                console.error("Error initializing audio:", err);
                setAudioEnabled(false);
            }
        }

        return () => {
            if (audioContextRef.current) {
                oscillatorRef.current?.stop();
                lfoRef.current?.stop();
                audioContextRef.current.close();
            }
        };
    }, [audioEnabled, bassFreq, filterQ, lfoRate]);

    // Update audio parameters when sliders change
    useEffect(() => {
        if (audioContextRef.current) {
            if (oscillatorRef.current) {
                oscillatorRef.current.frequency.value = bassFreq;
            }
            if (filterRef.current) {
                filterRef.current.Q.value = filterQ;
            }
            if (lfoRef.current) {
                lfoRef.current.frequency.value = lfoRate;
            }
        }
    }, [bassFreq, filterQ, lfoRate]);

    // Start webcam capture
    const startCapture = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                setIsCapturing(true);
                requestAnimationFrame(processFrame);
            }
        } catch (err) {
            console.error("Error accessing webcam:", err);
        }
    };

    // Stop webcam capture
    const stopCapture = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            const tracks = videoRef.current.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            videoRef.current.srcObject = null;
            setIsCapturing(false);
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        }
    };

    // Handle video file upload
    const handleUpload = (event) => {
        const file = event.target.files[0];
        if (file && videoRef.current) {
            const url = URL.createObjectURL(file);
            videoRef.current.src = url;
            videoRef.current.srcObject = null;
            setIsCapturing(true);
            requestAnimationFrame(processFrame);
        }
    };

    // Process video frame - detect motion and update audio
    const processFrame = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const outputCanvas = outputCanvasRef.current;

        if (video && canvas && outputCanvas &&
            ((video.readyState === video.HAVE_ENOUGH_DATA) ||
                (video.src && !video.ended))) {

            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            const outputCtx = outputCanvas.getContext('2d');

            // Set canvas sizes to match video
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            outputCanvas.width = video.videoWidth || 640;
            outputCanvas.height = video.videoHeight || 480;

            // Draw current frame to hidden canvas
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Calculate motion by comparing with previous frame
            if (prevFrameRef.current) {
                setIsProcessing(true);
                const currentFrame = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const prevFrame = prevFrameRef.current;

                // Simple motion detection by comparing pixel differences
                let motionSum = 0;
                const motionData = new Uint8ClampedArray(currentFrame.data.length);

                for (let i = 0; i < currentFrame.data.length; i += 4) {
                    // Calculate pixel difference
                    const rdiff = Math.abs(currentFrame.data[i] - prevFrame.data[i]);
                    const gdiff = Math.abs(currentFrame.data[i + 1] - prevFrame.data[i + 1]);
                    const bdiff = Math.abs(currentFrame.data[i + 2] - prevFrame.data[i + 2]);

                    // Average difference for this pixel
                    const diff = (rdiff + gdiff + bdiff) / 3;

                    // Apply threshold to reduce noise
                    const motionValue = diff > motionThreshold ? 255 : 0;

                    // Store motion data for visualization
                    motionData[i] = 0;  // R
                    motionData[i + 1] = motionValue;  // G (green for motion)
                    motionData[i + 2] = 0;  // B
                    motionData[i + 3] = motionValue > 0 ? 150 : 0;  // Alpha (semi-transparent)

                    motionSum += motionValue;
                }

                // Calculate motion level (normalized)
                const pixelCount = currentFrame.data.length / 4;
                const newMotionLevel = motionSum / (255 * pixelCount);
                setMotionLevel(newMotionLevel);

                // Update audio if enabled
                if (audioEnabled && gainRef.current) {
                    // Map motion to volume (with smoothing)
                    const targetGain = Math.min(newMotionLevel * 20, 0.7);
                    gainRef.current.gain.setTargetAtTime(targetGain, audioContextRef.current.currentTime, 0.1);

                    // Optionally map motion to LFO rate for more dynamic wobble
                    if (lfoRef.current) {
                        const targetRate = 2 + (newMotionLevel * 10); // Scale between 2-12 Hz
                        lfoRef.current.frequency.setTargetAtTime(targetRate, audioContextRef.current.currentTime, 0.2);
                    }
                }

                // Draw video frame to output canvas
                outputCtx.drawImage(video, 0, 0, outputCanvas.width, outputCanvas.height);

                // Overlay motion visualization
                const motionImageData = new ImageData(motionData, currentFrame.width, currentFrame.height);
                outputCtx.putImageData(motionImageData, 0, 0);
            }

            // Store current frame for next comparison
            prevFrameRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
        }

        // Continue processing frames if still capturing
        if (isCapturing) {
            animationRef.current = requestAnimationFrame(processFrame);
        }
    };

    // Toggle audio
    const toggleAudio = () => setAudioEnabled(!audioEnabled);

    // Switch between webcam and upload
    const switchToWebcam = () => {
        stopCapture();
        setInputSource('webcam');
    };

    const switchToUpload = () => {
        stopCapture();
        setInputSource('upload');
    };

    return (
        <div className="flex flex-col items-center p-4 max-w-5xl mx-auto bg-gray-50 rounded-lg shadow-md">
            <h1 className="text-3xl font-bold mb-6 text-indigo-700">Video to Audio Transformer v0.0.1</h1>

            {/* Source Selection Tabs */}
            <div className="flex mb-4 w-full max-w-md">
                <button
                    className={`flex-1 py-2 rounded-tl-lg ${inputSource === 'webcam' ? 'bg-indigo-600 text-white' : 'bg-gray-300'}`}
                    onClick={switchToWebcam}
                >
                    Webcam
                </button>
                <button
                    className={`flex-1 py-2 rounded-tr-lg ${inputSource === 'upload' ? 'bg-indigo-600 text-white' : 'bg-gray-300'}`}
                    onClick={switchToUpload}
                >
                    Upload Video
                </button>
            </div>

            {/* Input Controls */}
            <div className="mb-4 flex flex-wrap justify-center gap-2">
                {inputSource === 'webcam' ? (
                    <button
                        onClick={isCapturing ? stopCapture : startCapture}
                        className={`px-4 py-2 rounded font-semibold ${isCapturing ? 'bg-red-500 hover:bg-red-600' : 'bg-blue-500 hover:bg-blue-600'} text-white`}
                        aria-label={isCapturing ? 'Stop Camera' : 'Start Camera'}
                    >
                        {isCapturing ? 'Stop Camera' : 'Start Camera'}
                    </button>
                ) : (
                    <label className="px-4 py-2 rounded font-semibold bg-blue-500 hover:bg-blue-600 text-white cursor-pointer">
                        Select Video File
                        <input
                            type="file"
                            ref={uploadRef}
                            accept="video/*"
                            onChange={handleUpload}
                            className="hidden"
                        />
                    </label>
                )}

                <button
                    onClick={toggleAudio}
                    className={`px-4 py-2 rounded font-semibold ${audioEnabled ? 'bg-green-500 hover:bg-green-600' : 'bg-gray-500 hover:bg-gray-600'} text-white`}
                    disabled={!isCapturing}
                    aria-label={audioEnabled ? 'Disable Audio' : 'Enable Audio'}
                >
                    {audioEnabled ? 'Audio On' : 'Audio Off'}
                </button>
            </div>

            {/* Main Content Area */}
            <div className="flex flex-col md:flex-row w-full gap-4">
                {/* Video Display */}
                <div className="md:w-3/5 relative">
                    <div className="relative border-2 border-gray-300 rounded-lg overflow-hidden bg-black">
                        <video
                            ref={videoRef}
                            autoPlay
                            muted
                            playsInline
                            controls={inputSource === 'upload'}
                            className={inputSource === 'webcam' ? "hidden" : "w-full h-auto"}
                        />
                        <canvas
                            ref={canvasRef}
                            className="hidden"
                        />
                        <canvas
                            ref={outputCanvasRef}
                            className="w-full h-auto"
                        />
                        <div className="absolute top-2 left-2 bg-black bg-opacity-70 text-white p-2 rounded text-sm">
                            Motion: {(motionLevel * 100).toFixed(1)}%
                        </div>
                    </div>
                    {isProcessing && (
                        <div className="mt-4 p-3 bg-blue-100 border border-blue-300 rounded-lg shadow-sm">
                            <h3 className="text-lg font-semibold text-blue-800 mb-1">Processing Status</h3>
                            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                                <div className="font-medium">Motion Detection:</div>
                                <div>Active</div>
                                <div className="font-medium">Threshold:</div>
                                <div>{motionThreshold}</div>
                                <div className="font-medium">Audio Mapping:</div>
                                <div>{audioEnabled ? 'Active' : 'Disabled'}</div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Controls Panel */}
                <div className="md:w-2/5 space-y-4">
                    {/* Motion Detection Controls */}
                    <div className="p-4 border border-gray-300 rounded-lg bg-white shadow-sm">
                        <h2 className="text-xl font-semibold mb-3 text-gray-800">Motion Detection</h2>
                        <div className="mb-3">
                            <label className="block mb-1 text-sm font-medium text-gray-700">
                                Motion Threshold: {motionThreshold}
                            </label>
                            <input
                                type="range"
                                min="5"
                                max="50"
                                value={motionThreshold}
                                onChange={(e) => setMotionThreshold(Number(e.target.value))}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                aria-label="Motion Threshold"
                            />
                            <p className="text-xs text-gray-500 mt-1">
                                Lower values detect more subtle movements
                            </p>
                        </div>
                    </div>

                    {/* Audio Controls */}
                    <div className="p-4 border border-gray-300 rounded-lg bg-white shadow-sm">
                        <h2 className="text-xl font-semibold mb-3 text-gray-800">Wobble Bass Controls</h2>

                        <div className="mb-3">
                            <label className="block mb-1 text-sm font-medium text-gray-700">
                                Bass Frequency: {bassFreq} Hz
                            </label>
                            <input
                                type="range"
                                min="40"
                                max="120"
                                value={bassFreq}
                                onChange={(e) => setBassFreq(Number(e.target.value))}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                disabled={!audioEnabled}
                                aria-label="Bass Frequency"
                            />
                        </div>

                        <div className="mb-3">
                            <label className="block mb-1 text-sm font-medium text-gray-700">
                                LFO Rate (Wobble Speed): {lfoRate} Hz
                            </label>
                            <input
                                type="range"
                                min="0.5"
                                max="15"
                                step="0.5"
                                value={lfoRate}
                                onChange={(e) => setLfoRate(Number(e.target.value))}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                disabled={!audioEnabled}
                                aria-label="LFO Rate"
                            />
                        </div>

                        <div className="mb-3">
                            <label className="block mb-1 text-sm font-medium text-gray-700">
                                Filter Resonance (Q): {filterQ}
                            </label>
                            <input
                                type="range"
                                min="1"
                                max="20"
                                value={filterQ}
                                onChange={(e) => setFilterQ(Number(e.target.value))}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                disabled={!audioEnabled}
                                aria-label="Filter Resonance"
                            />
                        </div>
                    </div>

                    {/* Visual Mapping */}
                    <div className="p-4 border border-gray-300 rounded-lg bg-white shadow-sm">
                        <h2 className="text-xl font-semibold mb-3 text-gray-800">Visual → Audio Mapping</h2>
                        <div className="text-sm">
                            <div className="flex items-center mb-2">
                                <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                                <div>Motion Intensity → Audio Volume</div>
                            </div>
                            <div className="flex items-center mb-2">
                                <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
                                <div>Motion Intensity → Wobble Speed</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Informational Section */}
            <div className="mt-6 p-4 bg-gray-100 rounded-lg border border-gray-300 w-full">
                <h3 className="text-lg font-semibold mb-2 text-gray-800">How It Works</h3>
                <p className="text-sm text-gray-700">
                    This prototype captures video from your webcam or an uploaded file and analyzes frame-to-frame
                    differences to detect motion. The detected motion controls the volume of a wobble bass sound
                    and modulates the wobble speed. The green overlay shows where motion is detected in real-time.
                    Adjust the sliders to change the character of the bass sound and sensitivity of motion detection.
                </p>
            </div>
        </div>
    );
};

export default VideoToAudioTransformer;
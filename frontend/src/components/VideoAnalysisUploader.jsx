import React, { useState, useRef } from 'react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

const API_URL = 'http://localhost:5002';

const VideoAnalysisUploader = () => {
    const [file, setFile] = useState(null);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [analysisResults, setAnalysisResults] = useState(null);
    const [error, setError] = useState(null);
    const [audioUrl, setAudioUrl] = useState(null);
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile && selectedFile.type.startsWith('video/')) {
            setFile(selectedFile);
            setError(null);
        } else {
            setFile(null);
            setError('Please select a valid video file');
        }
    };

    const handleUpload = async () => {
        if (!file) {
            setError('Please select a video file first');
            return;
        }

        setIsUploading(true);
        setUploadProgress(0);
        setError(null);
        setAnalysisResults(null);
        setAudioUrl(null);

        const formData = new FormData();
        formData.append('video', file);

        try {
            // Simulate progress for better UX
            const progressInterval = setInterval(() => {
                setUploadProgress((prev) => {
                    const newProgress = prev + 5;
                    return newProgress > 90 ? 90 : newProgress;
                });
            }, 500);

            const response = await fetch(`${API_URL}/analyze`, {
                method: 'POST',
                body: formData,
            });

            clearInterval(progressInterval);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to analyze video');
            }

            setUploadProgress(100);
            const data = await response.json();
            setAnalysisResults(data.analysis_results);

            // Create audio URL from base64 if available
            if (data.audio_info && data.audio_info.file_path) {
                // In a real implementation, we would either:
                // 1. Have the server return the audio file directly
                // 2. Have the server store the audio file and return a URL
                // For this POC, we'll just note that we'd need to handle this
                setAudioUrl(`${API_URL}/audio/${data.video_id}`);
            }
        } catch (err) {
            setError(err.message || 'An error occurred during analysis');
        } finally {
            setIsUploading(false);
        }
    };

    const renderFrequencyBandResults = (bandName, bandData) => {
        if (!bandData) return null;

        const bandLabels = {
            ulf: 'Ultra-Low Frequency (0.01-0.1 Hz)',
            lf: 'Low Frequency (0.1-1 Hz)',
            mf: 'Mid Frequency (1-5 Hz)',
            hf: 'High Frequency (5-15 Hz)',
            uhf: 'Ultra-High Frequency (15+ Hz)'
        };

        const bandColors = {
            ulf: 'bg-blue-500',
            lf: 'bg-green-500',
            mf: 'bg-yellow-500',
            hf: 'bg-orange-500',
            uhf: 'bg-red-500'
        };

        return (
            <Card className="p-4 mb-4">
                <h3 className="text-lg font-medium mb-2">{bandLabels[bandName] || bandName}</h3>

                <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                        <p className="text-sm text-gray-500">Mean Magnitude</p>
                        <p className="font-medium">{bandData.mean_magnitude.toFixed(4)}</p>
                    </div>
                    <div>
                        <p className="text-sm text-gray-500">Peak Magnitude</p>
                        <p className="font-medium">{bandData.peak_magnitude.toFixed(4)}</p>
                    </div>
                </div>

                <div className="mb-4">
                    <p className="text-sm text-gray-500 mb-1">Motion Energy</p>
                    <div className="h-4 w-full bg-gray-200 rounded-full overflow-hidden">
                        <div
                            className={`h-full ${bandColors[bandName] || 'bg-blue-500'}`}
                            style={{ width: `${Math.min(100, bandData.mean_magnitude * 100 * 5)}%` }}
                        ></div>
                    </div>
                </div>

                {bandData.temporal_evolution && (
                    <div className="mb-4">
                        <p className="text-sm text-gray-500 mb-1">Temporal Evolution</p>
                        <div className="flex items-center gap-4">
                            <div>
                                <p className="text-xs text-gray-500">Trend</p>
                                <p className="font-medium">{bandData.temporal_evolution.trend}</p>
                            </div>
                            <div>
                                <p className="text-xs text-gray-500">Periodicity</p>
                                <p className="font-medium">{bandData.temporal_evolution.periodicity.toFixed(2)}</p>
                            </div>
                        </div>
                    </div>
                )}
            </Card>
        );
    };

    return (
        <div className="w-full max-w-4xl mx-auto p-4">
            <h2 className="text-2xl font-bold mb-6">Multi-Band Motion Analysis</h2>

            <Card className="p-6 mb-6">
                <div className="mb-4">
                    <label className="block text-sm font-medium mb-2">
                        Upload Video for Analysis
                    </label>
                    <input
                        type="file"
                        accept="video/*"
                        onChange={handleFileChange}
                        ref={fileInputRef}
                        className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-md file:border-0
              file:text-sm file:font-semibold
              file:bg-primary file:text-white
              hover:file:bg-primary/90"
                    />
                </div>

                {file && (
                    <div className="mb-4">
                        <p className="text-sm">Selected file: {file.name}</p>
                    </div>
                )}

                {error && (
                    <div className="mb-4 p-3 bg-red-100 text-red-800 rounded-md">
                        {error}
                    </div>
                )}

                <Button
                    onClick={handleUpload}
                    disabled={!file || isUploading}
                    className="w-full"
                >
                    {isUploading ? 'Analyzing...' : 'Analyze Video'}
                </Button>

                {isUploading && (
                    <div className="mt-4">
                        <Progress value={uploadProgress} className="h-2" />
                        <p className="text-sm text-center mt-1">
                            {uploadProgress < 100 ? 'Processing...' : 'Analysis complete!'}
                        </p>
                    </div>
                )}
            </Card>

            {analysisResults && (
                <div className="mt-8">
                    <h3 className="text-xl font-bold mb-4">Analysis Results</h3>

                    <Tabs defaultValue="mf">
                        <TabsList className="grid grid-cols-5 mb-4">
                            <TabsTrigger value="ulf">ULF</TabsTrigger>
                            <TabsTrigger value="lf">LF</TabsTrigger>
                            <TabsTrigger value="mf">MF</TabsTrigger>
                            <TabsTrigger value="hf">HF</TabsTrigger>
                            <TabsTrigger value="uhf">UHF</TabsTrigger>
                        </TabsList>

                        <TabsContent value="ulf">
                            {renderFrequencyBandResults('ulf', analysisResults.ulf)}
                        </TabsContent>

                        <TabsContent value="lf">
                            {renderFrequencyBandResults('lf', analysisResults.lf)}
                        </TabsContent>

                        <TabsContent value="mf">
                            {renderFrequencyBandResults('mf', analysisResults.mf)}
                        </TabsContent>

                        <TabsContent value="hf">
                            {renderFrequencyBandResults('hf', analysisResults.hf)}
                        </TabsContent>

                        <TabsContent value="uhf">
                            {renderFrequencyBandResults('uhf', analysisResults.uhf)}
                        </TabsContent>
                    </Tabs>

                    {audioUrl && (
                        <Card className="p-6 mt-6">
                            <h3 className="text-lg font-medium mb-4">Generated Audio</h3>
                            <audio controls className="w-full">
                                <source src={audioUrl} type="audio/wav" />
                                Your browser does not support the audio element.
                            </audio>
                            <p className="text-sm text-gray-500 mt-2">
                                Wobble bass generated from motion analysis
                            </p>
                        </Card>
                    )}
                </div>
            )}
        </div>
    );
};

export default VideoAnalysisUploader;
import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import AnimationExample from './AnimationExample';
import VideoAnalysisUploader from './VideoAnalysisUploader';

const Main = () => {
    const [activeTab, setActiveTab] = useState('video-analysis');

    return (
        <div className="container mx-auto py-6">
            <h1 className="text-3xl font-bold mb-6">Project Sonique</h1>

            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="video-analysis">Multi-Band Analysis</TabsTrigger>
                    <TabsTrigger value="animation">Animation Example</TabsTrigger>
                </TabsList>

                <TabsContent value="video-analysis" className="mt-6">
                    <VideoAnalysisUploader />
                </TabsContent>

                <TabsContent value="animation" className="mt-6">
                    <AnimationExample />
                </TabsContent>
            </Tabs>
        </div>
    );
};

export default Main;
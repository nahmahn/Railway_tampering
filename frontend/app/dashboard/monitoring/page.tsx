"use client";

import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { AlertTriangle, Wifi, Thermometer, Camera, Eye, Layers, Maximize, Play, Pause, Zap, Database } from 'lucide-react';
import DatasetModal from '@/components/modals/DatasetModal';

// Real Mock Data from Datasets
const LIVE_FEED_IMAGES = [
    { src: '/mock-images/real_track_1.jpg', label: 'CAM-01: Main Line (Normal)' },
    { src: '/mock-images/real_obstacle_1.jpg', label: 'CAM-02: Sector 4 (Obstacle Potential)' },
    { src: '/mock-images/real_fault_1.jpg', label: 'CAM-03: Junction B (Critical Fault)' },
    { src: '/mock-images/real_rusty_metal.jpg', label: 'CAM-04: Siding (Foreign Object)' }, // User attached image
];
// ... (keep generateData) ...

// ... (inside component) ...



const generateData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
        time: i,
        vibration: Math.sin(i * 0.5) * 40 + 50 + (Math.random() * 10 - 5),
        threshold: 80
    }));
};

export default function MonitoringPage() {
    const [data, setData] = useState(generateData());
    const [currentCamIndex, setCurrentCamIndex] = useState(0);
    const [isPlaying, setIsPlaying] = useState(true);
    const [activeTab, setActiveTab] = useState('vibration');
    const [activeFilters, setActiveFilters] = useState({
        box: true,
        mask: false,
        thermal: false,
    });
    const [isDatasetModalOpen, setIsDatasetModalOpen] = useState(false);

    useEffect(() => {
        let interval: any;
        if (isPlaying) {
            interval = setInterval(() => {
                setCurrentCamIndex((prev) => (prev + 1) % 4); // Hardcoded to 4 feeds now
            }, 3000);
        }
        return () => clearInterval(interval);
    }, [isPlaying]);

    const toggleFilter = (key: keyof typeof activeFilters) => {
        setActiveFilters(prev => ({ ...prev, [key]: !prev[key] }));
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <div>
                    <h2 className="text-2xl font-bold text-govt-navy">Live Surveillance Control</h2>
                    <div className="flex items-center gap-2 text-gray-500">
                        <p>UAV-RSOD / Fixed CCTV Network Stream</p>
                        <button onClick={() => setIsDatasetModalOpen(true)} className="text-xs bg-gray-100 hover:bg-gray-200 px-2 py-0.5 rounded flex items-center gap-1 text-govt-blue">
                            <Database size={12} /> View Dataset Info
                        </button>
                    </div>
                </div>
                <div className="flex gap-2">
                    <span className="px-3 py-1 bg-red-100 text-red-700 rounded text-xs font-bold animate-pulse">LIVE: 12ms Latency</span>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Main Camera Feed */}
                <div className="lg:col-span-2 space-y-4">
                    <div className="bg-black rounded-xl overflow-hidden shadow-lg relative group aspect-video border-4 border-gray-800">

                        {/* Image Layer */}
                        <img
                            src={LIVE_FEED_IMAGES[currentCamIndex].src}
                            alt="Live Feed"
                            className={`w-full h-full object-cover transition-all duration-500 ${activeFilters.thermal ? 'brightness-50 contrast-125 sepia hue-rotate-180' : ''}`}
                        />

                        {/* Overlay: AI Bounding Box (Dataset v2 - Obstacle) */}
                        {activeFilters.box && currentCamIndex === 1 && (
                            /* Coords from V2 UAV-RSOD 1163.xml: xmin 26%, ymin 41%, w 27%, h 58% */
                            <div className="absolute border-4 border-red-500 rounded-sm animate-pulse shadow-[0_0_15px_red] flex items-start justify-center"
                                style={{ left: '26%', top: '41%', width: '27%', height: '58%' }}>
                                <span className="absolute -top-6 left-0 bg-red-600 text-white text-xs px-2 py-0.5 font-bold shadow-sm">
                                    OBSTACLE 98%
                                </span>
                            </div>
                        )}

                        {/* Overlay: AI Bounding Box (Dataset Fault Detection) */}
                        {activeFilters.box && currentCamIndex === 2 && (
                            /* Estimated from User Screenshot "Crack Detected" */
                            <div className="absolute border-x-0 border-y-4 border-yellow-500 animate-pulse flex flex-col items-center"
                                style={{ left: '25%', top: '65%', width: '50%', height: '15%' }}>
                                <span className="absolute -top-7 left-10 bg-yellow-600 text-white text-xs px-2 py-0.5 font-bold shadow-sm">
                                    CRACK DETECTED
                                </span>
                            </div>
                        )}

                        {/* Overlay: New Rusty Metal Detection (User Attached Image) */}
                        {activeFilters.box && currentCamIndex === 3 && (
                            <div className="absolute border-4 border-orange-500 rounded-sm animate-pulse shadow-[0_0_10px_orange] flex items-center justify-center"
                                style={{ left: '35%', top: '50%', width: '30%', height: '35%' }}>
                                <span className="absolute -top-6 right-0 bg-orange-600 text-white text-xs px-2 py-0.5 font-bold shadow-sm">
                                    FOREIGN OBJECT (99%)
                                </span>
                            </div>
                        )}

                        {/* Segmentation Mask Overlay */}
                        {activeFilters.mask && (currentCamIndex === 0 || currentCamIndex === 3) && (
                            <div className="absolute inset-0 pointer-events-none">
                                {currentCamIndex === 0 ? (
                                    <>
                                        {/* Cam 1 Segmentation Overlay */}
                                        <div className="absolute inset-0 w-full h-full">
                                            <img
                                                src="/mock-images/cam1_segmentation.png"
                                                alt="Segmentation Mask"
                                                className="absolute inset-0 w-full h-full object-cover"
                                            />
                                        </div>
                                        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/80 text-white px-3 py-1 rounded-full text-xs font-mono border border-green-500/50 backdrop-blur shadow-lg z-10">
                                            V1 UAV-RSOD Mask: Cam 1
                                        </div>
                                    </>
                                ) : (
                                    <>
                                        {/* Cam 4 Segmentation Overlay */}
                                        <div className="absolute inset-0 w-full h-full">
                                            <img
                                                src="/mock-images/segmentation_cam4.png"
                                                alt="Segmentation Mask"
                                                className="absolute inset-0 w-full h-full object-cover"
                                            />
                                        </div>
                                        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/80 text-white px-3 py-1 rounded-full text-xs font-mono border border-green-500/50 backdrop-blur shadow-lg z-10">
                                            V1 UAV-RSOD Mask: Cam 4
                                        </div>
                                    </>
                                )}
                            </div>
                        )}

                        {/* HUD Overlay */}
                        <div className="absolute top-4 left-4 bg-black/60 backdrop-blur px-3 py-1 rounded text-green-400 font-mono text-xs flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
                            REC
                        </div>
                        <div className="absolute top-4 right-4 text-white/80 font-mono text-xs">
                            {LIVE_FEED_IMAGES[currentCamIndex].label}
                        </div>

                        {/* Controls Overlay */}
                        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent flex justify-between items-end opacity-0 group-hover:opacity-100 transition-opacity">
                            <div className="flex gap-2">
                                <button
                                    onClick={() => setIsPlaying(!isPlaying)}
                                    className="p-2 rounded-full bg-white/20 hover:bg-white/40 text-white backdrop-blur"
                                >
                                    {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                                </button>
                            </div>
                            <div className="flex gap-2 font-mono text-xs text-white/80">
                                <span>ISO 800</span>
                                <span>f/2.8</span>
                                <span>1/2000s</span>
                            </div>
                        </div>
                    </div>

                    {/* Control Panel (Working Buttons) */}
                    <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm flex items-center justify-between">
                        <div className="flex gap-4">
                            <button
                                onClick={() => toggleFilter('box')}
                                className={`flex items-center gap-2 px-4 py-2 rounded text-sm font-medium transition-colors ${activeFilters.box ? 'bg-govt-blue/10 text-govt-blue border border-govt-blue' : 'bg-gray-100 text-gray-600'}`}
                            >
                                <Eye size={16} /> AI Detection
                            </button>
                            <button
                                onClick={() => toggleFilter('mask')}
                                disabled={currentCamIndex === 1 || currentCamIndex === 2}
                                className={`flex items-center gap-2 px-4 py-2 rounded text-sm font-medium transition-colors ${currentCamIndex === 1 || currentCamIndex === 2
                                    ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                                    : activeFilters.mask
                                        ? 'bg-green-100 text-green-700 border border-green-500'
                                        : 'bg-gray-100 text-gray-600'
                                    }`}
                            >
                                <Layers size={16} /> Segmentation
                            </button>
                            <button
                                onClick={() => toggleFilter('thermal')}
                                className={`flex items-center gap-2 px-4 py-2 rounded text-sm font-medium transition-colors ${activeFilters.thermal ? 'bg-orange-100 text-orange-700 border border-orange-500' : 'bg-gray-100 text-gray-600'}`}
                            >
                                <Thermometer size={16} /> Thermal Mode
                            </button>
                        </div>

                        <button className="flex items-center gap-2 px-4 py-2 text-gray-500 hover:text-govt-navy">
                            <Maximize size={16} /> Fullscreen
                        </button>
                    </div>
                </div>

                {/* Side Panel: Telemetry & Alerts */}
                <div className="space-y-6">

                    {/* SENSOR PANEL */}
                    <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200 flex flex-col">
                        <h3 className="text-lg font-bold text-govt-navy mb-4 border-b border-gray-100 pb-2">Sensor Telemetry</h3>

                        {/* Tabs */}
                        <div className="flex gap-2 mb-4">
                            <button
                                onClick={() => setActiveTab('vibration')}
                                className={`flex-1 py-1 text-xs font-bold rounded ${activeTab === 'vibration' ? 'bg-govt-blue text-white' : 'bg-gray-100 text-gray-600'}`}
                            >
                                Vibration
                            </button>
                            <button
                                onClick={() => setActiveTab('thermal')}
                                className={`flex-1 py-1 text-xs font-bold rounded ${activeTab === 'thermal' ? 'bg-govt-blue text-white' : 'bg-gray-100 text-gray-600'}`}
                            >
                                Thermal
                            </button>
                        </div>

                        <div className="flex-1 flex flex-col h-64">
                            {activeTab === 'vibration' ? (
                                <>
                                    <div className="flex items-center gap-2 mb-2">
                                        <Wifi size={16} className="text-govt-orange" />
                                        <span className="text-sm font-semibold">DAS Vibration Graph</span>
                                    </div>
                                    <div className="flex-1 min-h-0">
                                        <ResponsiveContainer width="100%" height={200}>
                                            <LineChart data={data}>
                                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                                                <XAxis dataKey="time" hide />
                                                <YAxis domain={[0, 100]} hide />
                                                <Tooltip
                                                    contentStyle={{ backgroundColor: '#1f2937', color: '#fff', border: 'none', fontSize: '12px' }}
                                                    itemStyle={{ color: '#fff' }}
                                                />
                                                <Line
                                                    type="monotone"
                                                    dataKey="vibration"
                                                    stroke="#0a2647"
                                                    strokeWidth={2}
                                                    dot={false}
                                                    isAnimationActive={false}
                                                />
                                                <Line
                                                    type="monotone"
                                                    dataKey="threshold"
                                                    stroke="#dc2626"
                                                    strokeWidth={1}
                                                    strokeDasharray="5 5"
                                                    dot={false}
                                                />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>
                                    <div className="mt-4 grid grid-cols-2 gap-2 text-center">
                                        <div className="bg-gray-50 p-2 rounded">
                                            <p className="text-[10px] text-gray-500">Max Amplitude</p>
                                            <p className="text-lg font-bold text-gray-800">8.2 mm</p>
                                        </div>
                                        <div className="bg-gray-50 p-2 rounded">
                                            <p className="text-[10px] text-gray-500">Frequency</p>
                                            <p className="text-lg font-bold text-gray-800">42 Hz</p>
                                        </div>
                                    </div>
                                </>
                            ) : (
                                <>
                                    <div className="flex items-center gap-2 mb-2 justify-between">
                                        <div className="flex items-center gap-2">
                                            <Thermometer size={16} className="text-red-500" />
                                            <span className="text-sm font-semibold">Thermal Matrix</span>
                                        </div>
                                        <span className="text-[10px] font-mono text-red-500 animate-pulse">LIVE SENSOR FEED</span>
                                    </div>

                                    {/* Dynamic Thermal Grid */}
                                    <div className="flex-1 bg-slate-900 rounded-lg p-1 relative overflow-hidden border border-slate-700">
                                        <div className="absolute inset-0 grid grid-cols-8 grid-rows-6 gap-[1px] opacity-80">
                                            {Array.from({ length: 48 }).map((_, i) => (
                                                <div
                                                    key={i}
                                                    className="bg-red-500 transition-opacity duration-1000 ease-in-out"
                                                    style={{
                                                        opacity: Math.random() * 0.3 + 0.1,
                                                        backgroundColor: i % 7 === 0 ? '#ef4444' : (i % 5 === 0 ? '#f59e0b' : '#3b82f6')
                                                    }}
                                                />
                                            ))}
                                        </div>

                                        {/* Mock Hotspot Overlay */}
                                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-24 h-24 bg-red-500/30 blur-xl rounded-full animate-pulse"></div>

                                        {/* HUD Elements */}
                                        <div className="absolute top-2 left-2 text-[9px] font-mono text-cyan-400">
                                            MAX: 42.1°C<br />
                                            MIN: 18.5°C
                                        </div>
                                        <div className="absolute bottom-2 right-2 text-[9px] font-mono text-slate-400">
                                            IR-SENSOR-04
                                        </div>
                                    </div>

                                    <div className="mt-3 flex items-center justify-between p-2 bg-gradient-to-r from-blue-50 to-red-50 border border-slate-200 rounded">
                                        <span className="text-xs font-bold text-slate-700">Status:</span>
                                        <span className="text-xs font-mono font-bold text-green-600">NOMINAL (32°C AVG)</span>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>

                    {/* Alert List below video */}
                    <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-4">
                        <h3 className="text-sm font-bold text-state-800 mb-3 flex items-center justify-between">
                            <span className="flex items-center gap-2"><AlertTriangle size={16} className="text-orange-500" /> Recent Detections</span>
                            <span className="text-[10px] bg-slate-100 text-slate-500 px-2 py-0.5 rounded-full">Last Hour: 2</span>
                        </h3>
                        <div className="space-y-2">
                            {/* Alert Item 1 */}
                            <div className="group flex items-start gap-3 p-2 rounded-lg hover:bg-red-50 transition-colors border border-transparent hover:border-red-100 cursor-pointer">
                                <div className="bg-red-100 text-red-600 p-1.5 rounded-full mt-0.5 group-hover:bg-red-200 transition-colors">
                                    <AlertTriangle size={14} />
                                </div>
                                <div className="flex-1">
                                    <div className="flex justify-between items-start">
                                        <p className="text-sm font-bold text-red-700 leading-none">Critical Fault</p>
                                        <span className="text-[10px] text-slate-400 font-mono">10:42:05</span>
                                    </div>
                                    <p className="text-xs text-red-600/80 mt-1 line-clamp-1">Crack detected on Rail A4. Confidence: 92%</p>
                                </div>
                            </div>

                            {/* Alert Item 2 */}
                            <div className="group flex items-start gap-3 p-2 rounded-lg hover:bg-yellow-50 transition-colors border border-transparent hover:border-yellow-100 cursor-pointer">
                                <div className="bg-yellow-100 text-yellow-600 p-1.5 rounded-full mt-0.5 group-hover:bg-yellow-200 transition-colors">
                                    <AlertTriangle size={14} />
                                </div>
                                <div className="flex-1">
                                    <div className="flex justify-between items-start">
                                        <p className="text-sm font-bold text-yellow-700 leading-none">Obstacle Warning</p>
                                        <span className="text-[10px] text-slate-400 font-mono">10:48:12</span>
                                    </div>
                                    <p className="text-xs text-yellow-600/80 mt-1 line-clamp-1">Foreign object on track. Confidence: 85%</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <DatasetModal isOpen={isDatasetModalOpen} onClose={() => setIsDatasetModalOpen(false)} />
        </div>
    );
}

"use client";

import { useState, useEffect } from 'react';
import { AlertTriangle, Activity, CheckCircle, Database, Eye, TrendingUp, TrendingDown } from 'lucide-react';

function StatCard({ label, value, subtext, icon: Icon, color, trend }: any) {
    return (
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-sm font-medium text-gray-500 uppercase tracking-wide">{label}</p>
                    <div className="flex items-end gap-3 mt-1">
                        <h3 className="text-3xl font-bold text-gray-900 leading-none">{value}</h3>
                        {trend && (
                            <span className={`text-xs font-bold mb-1 flex items-center ${trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
                                {trend > 0 ? <TrendingUp size={12} className="mr-1" /> : <TrendingDown size={12} className="mr-1" />}
                                {Math.abs(trend)}%
                            </span>
                        )}
                    </div>
                    <p className={`text-xs mt-3 font-medium ${subtext.includes('+') ? 'text-green-600' : 'text-gray-500'}`}>
                        {subtext}
                    </p>
                </div>
                <div className={`p-3 rounded-xl bg-${color}-50 text-${color}-600 border border-${color}-100`}>
                    <Icon size={24} />
                </div>
            </div>

            {/* Mini Sparkline Mock */}
            <div className="mt-4 flex items-end gap-1 h-8 opacity-20">
                {[40, 60, 45, 70, 50, 80, 65, 90].map((h, i) => (
                    <div key={i} className={`flex-1 rounded-t bg-${color}-600`} style={{ height: `${h}%` }}></div>
                ))}
            </div>
        </div>
    );
}



export default function OverviewPage() {
    const [expandedNode, setExpandedNode] = useState<string | null>(null);

    const [time, setTime] = useState<string>("");

    useEffect(() => {
        setTime(new Date().toLocaleTimeString());
        const timer = setInterval(() => setTime(new Date().toLocaleTimeString()), 1000);
        return () => clearInterval(timer);
    }, []);

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-2xl font-bold text-govt-navy">System Overview</h2>
                <p className="text-gray-500">Operational Real-time Snapshot</p>
            </div>



            {/* Operations Summary */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <StatCard label="Active Alerts" value="3" subtext="+2 since last hour" icon={AlertTriangle} color="red" trend={12} />
                <StatCard label="System Health" value="98.2%" subtext="Optimal Performance" icon={CheckCircle} color="green" trend={0.5} />
                <StatCard label="Data Stream" value="4.2 GB/s" subtext="Live Latency: 12ms" icon={Database} color="blue" trend={-2} />
                <StatCard label="Visual Coverage" value="85%" subtext="12 Cams Active" icon={Eye} color="orange" trend={5} />
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                {/* Pipeline Visualization (Premium Light Theme) */}
                <div className="lg:col-span-2 bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden relative transition-all duration-500">
                    {/* Technical Background Pattern (Dot Grid) */}
                    <div className="absolute inset-0 bg-[radial-gradient(#e2e8f0_1px,transparent_1px)] [background-size:20px_20px] opacity-40"></div>

                    <div className="flex justify-between items-center p-6 border-b border-slate-100 relative z-10 bg-white/50 backdrop-blur-sm">
                        <div>
                            <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                                <Activity className="text-blue-600" size={20} /> Pipeline Architecture
                            </h3>
                            <p className="text-xs text-slate-500 font-medium">Real-time Inference Flow & Node Health</p>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                            </span>
                            <span className="text-[10px] font-mono font-bold text-slate-500 uppercase tracking-widest">
                                Live Stream
                            </span>
                        </div>
                    </div>

                    {/* Main Pipeline Flow */}
                    <div className="relative p-8 z-10">
                        {/* Connecting Line (Behind Nodes) */}
                        <div className="absolute top-[68px] left-12 right-12 h-0.5 bg-slate-100 -translate-y-1/2">
                            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-blue-400 to-transparent w-full h-full animate-[shimmer_2s_infinite] opacity-30"></div>
                        </div>

                        <div className="grid grid-cols-4 gap-4 relative">
                            {/* NODE 1: SENSORS */}
                            <div className="relative group flex flex-col items-center">
                                <button
                                    onClick={() => setExpandedNode(expandedNode === 'sensors' ? null : 'sensors')}
                                    className={`w-28 h-20 bg-white border rounded-xl flex flex-col items-center justify-center relative transition-all duration-200 z-10 
                                        ${expandedNode === 'sensors'
                                            ? 'border-blue-500 shadow-[0_4px_12px_-2px_rgba(59,130,246,0.2)] scale-105'
                                            : 'border-slate-200 hover:border-blue-300 hover:shadow-md'}`}
                                >
                                    <div className={`p-2 rounded-full mb-1 transition-colors ${expandedNode === 'sensors' ? 'bg-blue-50 text-blue-600' : 'bg-slate-50 text-slate-400 group-hover:text-blue-500'}`}>
                                        <Activity size={20} />
                                    </div>
                                    <span className={`font-bold text-[10px] uppercase tracking-wider ${expandedNode === 'sensors' ? 'text-blue-700' : 'text-slate-600'}`}>Input Layer</span>

                                    {/* Active Indicator */}
                                    {expandedNode === 'sensors' && (
                                        <div className="absolute -bottom-1.5 w-3 h-3 bg-white border-b border-r border-blue-500 rotate-45"></div>
                                    )}
                                </button>
                                <div className="mt-2 text-[9px] font-medium text-slate-400 bg-slate-50 px-2 py-0.5 rounded-full border border-slate-100">Click to Inspect</div>
                            </div>

                            {/* NODE 2: VISION */}
                            <div className="relative group flex flex-col items-center">
                                <div className="w-28 h-20 bg-white border border-slate-200 rounded-xl flex flex-col items-center justify-center relative transition-all duration-200 z-10 hover:border-purple-300 hover:shadow-md">
                                    <div className="p-2 rounded-full mb-1 bg-slate-50 text-slate-400 group-hover:text-purple-500 transition-colors">
                                        <Eye size={20} />
                                    </div>
                                    <span className="font-bold text-[10px] text-slate-600 uppercase tracking-wider">Vision</span>
                                    <div className="absolute bottom-2 w-8 h-0.5 bg-slate-100 overflow-hidden rounded-full">
                                        <div className="h-full bg-purple-500 w-full animate-pulse"></div>
                                    </div>
                                </div>
                                <div className="mt-2 text-[9px] font-medium text-slate-400">YOLOv9</div>
                            </div>

                            {/* NODE 3: FEATURES */}
                            <div className="relative group flex flex-col items-center">
                                <button
                                    onClick={() => setExpandedNode(expandedNode === 'features' ? null : 'features')}
                                    className={`w-28 h-20 bg-white border rounded-xl flex flex-col items-center justify-center relative transition-all duration-200 z-10 
                                        ${expandedNode === 'features'
                                            ? 'border-cyan-500 shadow-[0_4px_12px_-2px_rgba(6,182,212,0.2)] scale-105'
                                            : 'border-slate-200 hover:border-cyan-300 hover:shadow-md'}`}
                                >
                                    <div className={`p-2 rounded-full mb-1 transition-colors ${expandedNode === 'features' ? 'bg-cyan-50 text-cyan-600' : 'bg-slate-50 text-slate-400 group-hover:text-cyan-500'}`}>
                                        <Database size={20} />
                                    </div>
                                    <span className={`font-bold text-[10px] uppercase tracking-wider ${expandedNode === 'features' ? 'text-cyan-700' : 'text-slate-600'}`}>Features</span>

                                    {/* Active Indicator */}
                                    {expandedNode === 'features' && (
                                        <div className="absolute -bottom-1.5 w-3 h-3 bg-white border-b border-r border-cyan-500 rotate-45"></div>
                                    )}
                                </button>
                                <div className="mt-2 text-[9px] font-medium text-slate-400 bg-slate-50 px-2 py-0.5 rounded-full border border-slate-100">Click to Inspect</div>
                            </div>

                            {/* NODE 4: ALERT */}
                            <div className="relative group flex flex-col items-center">
                                <div className="w-28 h-20 bg-red-50/50 border border-red-100 rounded-xl flex flex-col items-center justify-center relative transition-all duration-200 z-10 animate-pulse hover:shadow-red-100 hover:shadow-md">
                                    <div className="p-2 rounded-full mb-1 bg-white text-red-500 shadow-sm border border-red-50">
                                        <AlertTriangle size={20} />
                                    </div>
                                    <span className="font-bold text-[10px] text-red-700 uppercase tracking-wider">Alert</span>
                                </div>
                                <div className="mt-2 text-[9px] font-bold text-red-500 px-2 py-0.5">GENERATED</div>
                            </div>
                        </div>

                        {/* EXPANDABLE DETAIL PANEL (Drawer) */}
                        <div className={`mt-6 overflow-hidden transition-all duration-300 ease-in-out ${expandedNode ? 'max-h-64 opacity-100' : 'max-h-0 opacity-0'}`}>
                            <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 shadow-inner relative">
                                <div className="absolute top-0 left-0 w-1 h-full bg-blue-500"></div>

                                {expandedNode === 'sensors' && (
                                    <div className="animate-in fade-in slide-in-from-top-2 duration-300">
                                        <h4 className="flex items-center gap-2 text-xs font-bold text-slate-700 uppercase mb-3">
                                            <Activity size={14} className="text-blue-500" /> Active Sensor Array
                                        </h4>
                                        <div className="grid grid-cols-4 gap-3">
                                            {['Geometric Sensors', 'Accelerometers', 'DAS (Acoustic)', 'LiDAR Input'].map((item) => (
                                                <div key={item} className="bg-white border border-slate-200 p-3 rounded-lg text-center shadow-sm hover:border-blue-300 transition-colors cursor-default">
                                                    <div className="w-2 h-2 rounded-full bg-green-500 mx-auto mb-2"></div>
                                                    <span className="text-[10px] font-medium text-slate-600 block leading-tight">{item}</span>
                                                </div>
                                            ))}
                                            <div className="col-span-4 bg-blue-600/5 border border-blue-200 p-2 rounded-lg flex items-center justify-center gap-3">
                                                <span className="relative flex h-2 w-2">
                                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                                                    <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
                                                </span>
                                                <span className="text-[10px] font-bold text-blue-700">CCTV & Drone Video Feed Active</span>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {expandedNode === 'features' && (
                                    <div className="animate-in fade-in slide-in-from-top-2 duration-300">
                                        <h4 className="flex items-center gap-2 text-xs font-bold text-slate-700 uppercase mb-3">
                                            <Database size={14} className="text-cyan-500" /> Feature Extraction Matrix
                                        </h4>
                                        <div className="grid grid-cols-4 gap-3">
                                            <div className="bg-white border-l-2 border-l-orange-500 p-2 rounded shadow-sm">
                                                <p className="text-[9px] font-bold text-slate-400 uppercase">Time Domain</p>
                                                <p className="text-[10px] font-medium text-slate-700 mt-1">RMS, Kurtosis</p>
                                            </div>
                                            <div className="bg-white border-l-2 border-l-violet-500 p-2 rounded shadow-sm">
                                                <p className="text-[9px] font-bold text-slate-400 uppercase">Freq Domain</p>
                                                <p className="text-[10px] font-medium text-slate-700 mt-1">FFT, Spectrogram</p>
                                            </div>
                                            <div className="bg-white border-l-2 border-l-emerald-500 p-2 rounded shadow-sm">
                                                <p className="text-[9px] font-bold text-slate-400 uppercase">Thermal</p>
                                                <p className="text-[10px] font-medium text-slate-700 mt-1">Hotspot Values</p>
                                            </div>
                                            <div className="bg-white border-l-2 border-l-pink-500 p-2 rounded shadow-sm">
                                                <p className="text-[9px] font-bold text-slate-400 uppercase">Spatial</p>
                                                <p className="text-[10px] font-medium text-slate-700 mt-1">Embeddings</p>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                    </div>

                    <div className="grid grid-cols-3 divide-x divide-slate-100 border-t border-slate-100 bg-slate-50/50">
                        <div className="p-4 text-center">
                            <p className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold mb-1">Throughput</p>
                            <p className="font-mono font-bold text-lg text-slate-700">4.2 <span className="text-xs text-slate-400">GB/s</span></p>
                        </div>
                        <div className="p-4 text-center">
                            <p className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold mb-1">Latency</p>
                            <p className="font-mono font-bold text-lg text-green-600">12<span className="text-xs text-green-500/70">ms</span></p>
                        </div>
                        <div className="p-4 text-center">
                            <p className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold mb-1">Model</p>
                            <p className="font-mono font-bold text-sm text-purple-600 py-1">YOLOv9-Rail</p>
                        </div>
                    </div>
                </div>
                {/* Right Panel: Health */}
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                    <h3 className="text-lg font-bold text-govt-navy mb-4">Module Health</h3>
                    <div className="space-y-4">
                        <div className="flex items-center justify-between p-3 bg-green-50 rounded border border-green-100">
                            <span className="text-sm font-medium text-green-800">Optical Sensors</span>
                            <span className="text-xs font-bold text-green-600 bg-white px-2 py-1 rounded border border-green-200">ONLINE</span>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-green-50 rounded border border-green-100">
                            <span className="text-sm font-medium text-green-800">DAS / Vibration</span>
                            <span className="text-xs font-bold text-green-600 bg-white px-2 py-1 rounded border border-green-200">ONLINE</span>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-yellow-50 rounded border border-yellow-100">
                            <span className="text-sm font-medium text-yellow-800">Thermal Imaging</span>
                            <span className="text-xs font-bold text-yellow-600 bg-white px-2 py-1 rounded border border-yellow-200">CALIBRATING</span>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-green-50 rounded border border-green-100">
                            <span className="text-sm font-medium text-green-800">NLP Assistant</span>
                            <span className="text-xs font-bold text-green-600 bg-white px-2 py-1 rounded border border-green-200">READY</span>
                        </div>
                    </div>

                    <div className="mt-8 pt-6 border-t border-gray-100">
                        <p className="text-xs text-gray-400 font-mono">Last Sync: {time}</p>
                        <p className="text-xs text-gray-400 font-mono mt-1">Server: RLY-NORTH-04</p>
                    </div>
                </div>
            </div>
        </div>
    );
}

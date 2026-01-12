"use client";

import { useState, useEffect } from 'react';
import { AlertTriangle, Info, MapPin, Signal, Activity } from 'lucide-react';
import { useTelemetry } from '@/hooks/useTelemetry';
import { TelemetryPackage } from '@/types/telemetry';

export default function MapViewPage() {
    const [selectedSegment, setSelectedSegment] = useState<string | null>(null);
    const { data, loading } = useTelemetry();
    const [currentIndex, setCurrentIndex] = useState(0);

    // Effect to simulate live data stream by cycling through data
    useEffect(() => {
        if (data.length > 0) {
            const interval = setInterval(() => {
                setCurrentIndex((prev) => (prev + 1) % data.length);
            }, 3000); // Update every 3 seconds
            return () => clearInterval(interval);
        }
    }, [data]);

    const latest = data[currentIndex] || {} as TelemetryPackage;

    // Mapping segments to specific data points (mock logic for demo)
    const getSegmentData = (segmentId: string): TelemetryPackage | null => {
        if (!data.length) return null;
        if (segmentId === "seg-alert") return data[0]; // Always show critical data for alert
        if (segmentId === "seg-01") return data[1];
        if (segmentId === "seg-maint") return data[2];
        return latest;
    };

    const selectedData = selectedSegment ? getSegmentData(selectedSegment) : null;

    if (loading) return <div className="h-full flex items-center justify-center text-gray-500">Loading Telemetry Stream...</div>;

    return (
        <div className="h-full flex flex-col space-y-6">
            <div className="flex justify-between items-end">
                <div>
                    <h2 className="text-2xl font-bold text-govt-navy">TMS: Track Management System</h2>
                    <p className="text-gray-500 font-mono text-sm">Zone: Northern Railway | Section: NDLS-UMB | Scale: 1:5000</p>
                </div>
                <div className="flex gap-4 text-xs font-mono text-gray-500">
                    <span>lat: {latest.geo_positioning_input?.latitude_decimal?.toFixed(4) || '28.6139'}° N</span>
                    <span>long: {latest.geo_positioning_input?.longitude_decimal?.toFixed(4) || '77.2090'}° E</span>
                    <span className="text-green-600 font-bold animate-pulse">LIVE DATA STREAM</span>
                </div>
            </div>

            <div className="flex-1 flex gap-6 min-h-0">

                {/* Main SCADA Map Panel */}
                <div className="flex-1 bg-[#f8fafc] rounded-xl border border-gray-300 relative overflow-hidden shadow-inner group">

                    {/* Technical Grid Background */}
                    <div className="absolute inset-0"
                        style={{
                            backgroundImage: 'linear-gradient(#e2e8f0 1px, transparent 1px), linear-gradient(90deg, #e2e8f0 1px, transparent 1px)',
                            backgroundSize: '40px 40px'
                        }}>
                    </div>

                    {/* Map Container */}
                    <div className="relative w-full h-full p-10 flex items-center justify-center">

                        {/* SVG Schematic Track Layout */}
                        <svg className="w-full h-full max-w-4xl" viewBox="0 0 800 400">
                            <defs>
                                <pattern id="diagonalHatch" width="10" height="10" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
                                    <line x1="0" y1="0" x2="0" y2="10" style={{ stroke: 'orange', strokeWidth: 1 }} />
                                </pattern>
                            </defs>

                            {/* Main Line (Track 1) */}
                            <path d="M50,200 L750,200" stroke="#94a3b8" strokeWidth="6" fill="none" />
                            <path d="M50,200 L750,200" stroke="white" strokeWidth="2" strokeDasharray="10,10" fill="none" />

                            {/* Loop Line (Track 2) */}
                            <path d="M150,200 C200,100 600,100 650,200" stroke="#94a3b8" strokeWidth="4" fill="none" />

                            {/* Siding (Track 3) */}
                            <path d="M100,200 L200,300 L500,300" stroke="#cbd5e1" strokeWidth="4" fill="none" />

                            {/* Stations */}
                            <rect x="120" y="180" width="40" height="40" fill="#1e293b" rx="2" stroke="white" strokeWidth="2" />
                            <text x="140" y="170" textAnchor="middle" className="text-[10px] font-mono fill-gray-600 font-bold">STN-A</text>

                            <rect x="640" y="180" width="40" height="40" fill="#1e293b" rx="2" stroke="white" strokeWidth="2" />
                            <text x="660" y="170" textAnchor="middle" className="text-[10px] font-mono fill-gray-600 font-bold">STN-B</text>

                            {/* SEGMENTS / ALERTS */}

                            {/* Normal Segment */}
                            <line
                                x1="180" y1="200" x2="350" y2="200"
                                stroke="#22c55e" strokeWidth="8"
                                className={`cursor-pointer hover:stroke-[12] transition-all ${selectedSegment === 'seg-01' ? 'stroke-[12] opacity-100' : 'opacity-60'}`}
                                onClick={() => setSelectedSegment("seg-01")}
                            />

                            {/* Danger Segment (The specific alert) */}
                            <g className="cursor-pointer" onClick={() => setSelectedSegment("seg-alert")}>
                                <line
                                    x1="350" y1="200" x2="450" y2="200"
                                    stroke="#dc2626" strokeWidth={selectedSegment === 'seg-alert' ? 12 : 8}
                                    className="animate-pulse transition-all"
                                />
                                <circle cx="400" cy="200" r="15" fill="#dc2626" fillOpacity="0.2" className="animate-ping" />
                                <rect x="390" y="160" width="20" height="20" fill="#dc2626" rx="4" />
                                <text x="400" y="175" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">!</text>
                            </g>

                            {/* Maintenance Segment */}
                            <line
                                x1="450" y1="200" x2="620" y2="200"
                                stroke="#eab308" strokeWidth="8"
                                className={`cursor-pointer hover:stroke-[12] transition-all ${selectedSegment === 'seg-maint' ? 'stroke-[12] opacity-100' : 'opacity-60'}`}
                                onClick={() => setSelectedSegment("seg-maint")}
                            />

                            {/* Signals */}
                            <circle cx="250" cy="190" r="6" fill="#22c55e" />
                            <line x1="250" y1="190" x2="250" y2="200" stroke="gray" strokeWidth="2" />

                            <circle cx="550" cy="190" r="6" fill="#ef4444" />
                            <line x1="550" y1="190" x2="550" y2="200" stroke="gray" strokeWidth="2" />

                        </svg>

                        {/* Overlay UI Elements */}
                        <div className="absolute top-4 right-4 bg-white/90 backdrop-blur border border-gray-200 p-2 rounded shadow-sm text-xs font-mono">
                            <div className="flex items-center gap-2 mb-1"><div className="w-2 h-2 rounded-full bg-green-500"></div> Signal: CLEAR</div>
                            <div className="flex items-center gap-2 mb-1"><div className="w-2 h-2 rounded-full bg-red-500"></div> Track Circuit: FAIL</div>
                            <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-blue-500"></div> Power: OK</div>
                        </div>
                    </div>
                </div>

                {/* Side Panel Details */}
                <div className="w-80 flex flex-col gap-4">
                    {/* Context Panel */}
                    <div className="bg-white p-5 rounded-lg border border-gray-200 shadow-sm flex-1 overflow-y-auto">
                        <h3 className="font-bold text-govt-navy border-b border-gray-100 pb-2 mb-4 flex items-center gap-2">
                            <Activity size={18} /> Telemetry Data
                        </h3>

                        {selectedData ? (
                            <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                                {selectedSegment === "seg-alert" && (
                                    <div className="bg-red-50 border-l-4 border-red-500 p-3 rounded">
                                        <p className="text-xs font-bold text-red-800 uppercase">Critical Alert</p>
                                        <p className="text-sm font-bold text-gray-900 mt-1">Rail Fracture Detected</p>
                                        <p className="text-xs text-red-600 mt-2">Loc: KM-394 / Main Line</p>
                                    </div>
                                )}

                                <div className="space-y-3">
                                    <div className="bg-slate-50 p-3 rounded border border-slate-100">
                                        <p className="text-xs font-bold text-gray-500 uppercase mb-2 flex items-center gap-1">
                                            <MapPin size={12} /> Geo Positioning
                                        </p>
                                        <div className="grid grid-cols-2 gap-2 text-xs">
                                            <div>
                                                <span className="text-gray-400 block">Latitude</span>
                                                <span className="font-mono font-medium text-gray-700">{selectedData.geo_positioning_input.latitude_decimal.toFixed(5)}</span>
                                            </div>
                                            <div>
                                                <span className="text-gray-400 block">Longitude</span>
                                                <span className="font-mono font-medium text-gray-700">{selectedData.geo_positioning_input.longitude_decimal.toFixed(5)}</span>
                                            </div>
                                            <div>
                                                <span className="text-gray-400 block">Altitude</span>
                                                <span className="font-mono font-medium text-gray-700">{selectedData.geo_positioning_input.altitude_msl_m} m</span>
                                            </div>
                                            <div>
                                                <span className="text-gray-400 block">Sats</span>
                                                <span className="font-mono font-medium text-gray-700">{selectedData.geo_positioning_input.satellite_count}</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="bg-slate-50 p-3 rounded border border-slate-100">
                                        <p className="text-xs font-bold text-gray-500 uppercase mb-2 flex items-center gap-1">
                                            <Signal size={12} /> Sensor Readings
                                        </p>
                                        <div className="grid grid-cols-2 gap-2 text-xs">
                                            <div>
                                                <span className="text-gray-400 block">Temp</span>
                                                <span className="font-mono font-medium text-gray-700">{selectedData.environmental_sensor_readings.external_temp_c}°C</span>
                                            </div>
                                            <div>
                                                <span className="text-gray-400 block">Humidity</span>
                                                <span className="font-mono font-medium text-gray-700">{selectedData.environmental_sensor_readings.humidity_percent}%</span>
                                            </div>
                                            <div className="col-span-2">
                                                <span className="text-gray-400 block">Camera</span>
                                                <span className="font-mono font-medium text-gray-700 truncate">{selectedData.sensor_hardware_config.sensor_id}</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="bg-slate-50 p-3 rounded border border-slate-100">
                                        <p className="text-xs font-bold text-gray-500 uppercase mb-2 flex items-center gap-1">
                                            <Activity size={12} /> Locomotive Interface
                                        </p>
                                        <div className="grid grid-cols-2 gap-2 text-xs">
                                            <div>
                                                <span className="text-gray-400 block">Status</span>
                                                <span className={`font-mono font-bold ${selectedData.locomotive_telemetry_bus.train_interface_unit_status === 'ONLINE' ? 'text-green-600' : 'text-orange-500'}`}>
                                                    {selectedData.locomotive_telemetry_bus.train_interface_unit_status}
                                                </span>
                                            </div>
                                            <div>
                                                <span className="text-gray-400 block">Brake Pr.</span>
                                                <span className="font-mono font-medium text-gray-700">{selectedData.locomotive_telemetry_bus.brake_cylinder_pressure_kg_cm2} kg/cm²</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div className="pt-2">
                                    <h4 className="text-xs font-bold text-gray-500 uppercase mb-2">Live Feed Snapshot</h4>
                                    <div className="aspect-video bg-black rounded overflow-hidden relative group">
                                        <img
                                            src={`/dataset/${selectedData.source_file_attributes.filename}`}
                                            className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
                                            alt="Feed"
                                        />
                                        <div className="absolute top-1 right-1 bg-red-600 text-[8px] text-white px-1 rounded animate-pulse">LIVE</div>
                                    </div>
                                </div>

                                <button className="w-full bg-govt-navy text-white text-sm py-2 rounded font-medium hover:bg-govt-blue transition-colors shadow-sm">
                                    Request Manual Inspection
                                </button>
                            </div>
                        ) : (
                            <div className="text-center py-10 text-gray-400 h-full flex flex-col items-center justify-center">
                                <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mx-auto mb-4 border border-gray-100">
                                    <MapPin size={32} className="text-gray-300" />
                                </div>
                                <h4 className="text-gray-600 font-medium mb-1">No Segment Selected</h4>
                                <p className="text-sm text-gray-400 px-4">Select a track segment on the map to view real-time engineering telemetry and sensor packages.</p>
                            </div>
                        )}
                    </div>

                    {/* Legend Panel */}
                    <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
                        <h4 className="text-xs font-bold text-gray-500 uppercase mb-3">Symbology</h4>
                        <div className="space-y-2 text-xs text-gray-600">
                            <div className="flex items-center gap-2"><div className="w-8 h-1 bg-gray-400"></div> Active Track</div>
                            <div className="flex items-center gap-2"><div className="w-8 h-1 bg-gray-300 border-t border-dashed border-white"></div> Inactive // Siding</div>
                            <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-green-500"></div> Signal Post</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}


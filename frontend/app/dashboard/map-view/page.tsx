"use client";

import { useState, useEffect, useMemo } from 'react';
import { MapPin, AlertTriangle, Shield, Activity, ChevronRight, ExternalLink } from 'lucide-react';
import { useAlerts } from '@/contexts/AlertContext';
import { api } from '@/services/api';
import Link from 'next/link';
import dynamic from 'next/dynamic';

// Dynamically import the Map component with no SSR
const MapWithNoSSR = dynamic(() => import('@/components/map/MapComponent'), {
    ssr: false,
    loading: () => <div className="h-full w-full flex items-center justify-center bg-gray-900 text-white">Loading Map...</div>
});

// Zone definitions — 12 zones along the track (New Delhi -> Tughlakabad)
// These must match the IDs in MapComponent.tsx for data consistency
const ZONES = [
    { id: 'Z-01', name: 'New Delhi (NDLS)', lat: 28.6427, lng: 77.2209 },
    { id: 'Z-02', name: 'Tilak Bridge', lat: 28.6295, lng: 77.2356 },
    { id: 'Z-03', name: 'Pragati Maidan', lat: 28.6186, lng: 77.2452 },
    { id: 'Z-04', name: 'Hazrat Nizamuddin', lat: 28.5901, lng: 77.2581 },
    { id: 'Z-05', name: 'Lajpat Nagar', lat: 28.5702, lng: 77.2403 },
    { id: 'Z-06', name: 'Okhla', lat: 28.5565, lng: 77.2801 },
    { id: 'Z-07', name: 'Tughlakabad', lat: 28.5028, lng: 77.2941 },
    { id: 'Z-08', name: 'Badarpur', lat: 28.4901, lng: 77.2995 },
    { id: 'Z-09', name: 'Sarita Vihar', lat: 28.5255, lng: 77.2905 },
    { id: 'Z-10', name: 'Govind Puri', lat: 28.5432, lng: 77.2655 },
    { id: 'Z-11', name: 'Kalkaji Mandir', lat: 28.5501, lng: 77.2605 },
    { id: 'Z-12', name: 'Nehru Place', lat: 28.5522, lng: 77.2515 },
];

function assignAlertToZone(alertId: string, totalZones: number): number {
    let hash = 0;
    for (let i = 0; i < alertId.length; i++) {
        hash = ((hash << 5) - hash) + alertId.charCodeAt(i);
        hash |= 0;
    }
    return Math.abs(hash) % totalZones;
}

export default function MapViewPage() {
    const { alerts } = useAlerts();
    const [selectedZone, setSelectedZone] = useState<string | null>(null);
    const [history, setHistory] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const data = await api.getHistory();
                setHistory(data);
            } catch (e) {
                console.error("Failed to fetch history:", e);
            } finally {
                setLoading(false);
            }
        };
        fetchHistory();
    }, []);

    // Map alerts to zones
    const zoneAlerts = useMemo(() => {
        const mapping: Record<string, any[]> = {};
        ZONES.forEach(z => { mapping[z.id] = []; });

        // Map real alerts to zones
        alerts.forEach(alert => {
            const zoneIdx = assignAlertToZone(alert.alert_id, ZONES.length);
            const zoneId = ZONES[zoneIdx].id;
            mapping[zoneId].push(alert);
        });

        // Also map history sessions
        history.forEach(session => {
            const sessionId = session.id || session.session_id || '';
            const zoneIdx = assignAlertToZone(sessionId, ZONES.length);
            const zoneId = ZONES[zoneIdx].id;
            const riskLevel = session.overall_assessment?.risk_level?.toLowerCase() || 'low';
            if (!mapping[zoneId].find((a: any) => a.alert_id === sessionId)) {
                mapping[zoneId].push({
                    alert_id: sessionId,
                    severity: riskLevel === 'critical' ? 'critical' : riskLevel === 'high' ? 'high' : 'warning',
                    title: `Session ${sessionId?.slice(0, 12)}`,
                    message: `Risk: ${riskLevel} | Confidence: ${session.overall_assessment?.confidence || 'N/A'}`,
                    timestamp: session.timestamp || '',
                    source: 'history',
                });
            }
        });

        return mapping;
    }, [alerts, history]);

    // Zone severity calculation
    const getZoneSeverity = (zoneId: string) => {
        const zAlerts = zoneAlerts[zoneId] || [];
        if (zAlerts.some(a => a.severity === 'critical')) return 'critical';
        if (zAlerts.some(a => a.severity === 'high')) return 'high';
        if (zAlerts.length > 0) return 'warning';
        return 'safe';
    };

    // Stats
    const stats = useMemo(() => {
        let safe = 0, warning = 0, high = 0, critical = 0;
        ZONES.forEach(z => {
            const sev = getZoneSeverity(z.id);
            if (sev === 'safe') safe++;
            else if (sev === 'warning') warning++;
            else if (sev === 'high') high++;
            else critical++;
        });
        return { total: ZONES.length, safe, warning, high, critical };
    }, [zoneAlerts]);

    const selectedZoneData = ZONES.find(z => z.id === selectedZone);
    const selectedAlerts = selectedZone ? (zoneAlerts[selectedZone] || []) : [];

    if (loading) {
        return (
            <div className="h-full flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-govt-blue"></div>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col space-y-4">
            <div className="flex justify-between items-center">
                <div>
                    <h2 className="text-2xl font-bold text-govt-navy">Hotspot Map</h2>
                    <p className="text-gray-500">Geospatial alert visualization — Northern Division Sector 4</p>
                </div>
            </div>

            {/* Stats Bar */}
            <div className="grid grid-cols-5 gap-3">
                <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
                    <p className="text-2xl font-bold text-govt-navy">{stats.total}</p>
                    <p className="text-xs text-gray-500">Active Zones</p>
                </div>
                <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-center">
                    <p className="text-2xl font-bold text-green-600">{stats.safe}</p>
                    <p className="text-xs text-green-500">Safe</p>
                </div>
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 text-center">
                    <p className="text-2xl font-bold text-yellow-600">{stats.warning}</p>
                    <p className="text-xs text-yellow-500">Low Risk</p>
                </div>
                <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 text-center">
                    <p className="text-2xl font-bold text-orange-600">{stats.high}</p>
                    <p className="text-xs text-orange-500">High Risk</p>
                </div>
                <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-center">
                    <p className="text-2xl font-bold text-red-600">{stats.critical}</p>
                    <p className="text-xs text-red-500">Critical</p>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 flex gap-4 min-h-0 relative">
                {/* Map Container */}
                <div className="flex-1 rounded-xl shadow-lg border border-gray-200 overflow-hidden relative bg-gray-900">
                    <div className="absolute top-4 left-4 z-[400] bg-white/90 backdrop-blur px-3 py-1.5 rounded-lg shadow-md border border-gray-200">
                        <div className="flex items-center gap-2">


                        </div>
                    </div>

                    <MapWithNoSSR
                        alerts={alerts.concat(history.map((h: any) => {
                            const rawRisk = (h.overall_assessment?.risk_level || 'low').toLowerCase();
                            const severity = rawRisk === 'critical' ? 'critical' :
                                rawRisk === 'high' ? 'high' :
                                    (rawRisk === 'medium' || rawRisk === 'warning') ? 'warning' : 'info';

                            return {
                                alert_id: h.id || h.session_id,
                                severity,
                                title: `Ref: ${h.id?.slice(0, 8)}`,
                                timestamp: h.timestamp || new Date().toISOString(),
                                source: 'history',
                                message: `Analysed Session ${h.id?.slice(0, 8)}`,
                                action_required: false,
                                acknowledged: true
                            };
                        }))}
                        onZoneSelect={setSelectedZone}
                        selectedZoneId={selectedZone}
                    />

                    {/* Legend Overlay */}
                    <div className="absolute bottom-4 left-4 z-[400] bg-white/90 backdrop-blur p-3 rounded-lg shadow-md border border-gray-200">
                        <p className="text-xs font-bold text-gray-500 uppercase mb-2">Severity Legend</p>
                        <div className="space-y-1.5">
                            {[
                                { label: 'Safe', color: 'bg-green-500' },
                                { label: 'Low Risk', color: 'bg-yellow-500' },
                                { label: 'High Risk', color: 'bg-orange-500' },
                                { label: 'Critical', color: 'bg-red-500' },
                            ].map(item => (
                                <div key={item.label} className="flex items-center gap-2">
                                    <div className={`w-3 h-3 rounded-full ${item.color}`} />
                                    <span className="text-xs text-gray-700 font-medium">{item.label}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Side Panel Overlay or Sidebar */}
                <div className={`w-80 bg-white rounded-xl border border-gray-200 shadow-xl overflow-hidden flex flex-col transition-all duration-300 ${selectedZone ? 'translate-x-0 opacity-100' : 'translate-x-4 opacity-50'}`}>
                    {selectedZoneData ? (
                        <div className="flex-1 flex flex-col h-full">
                            <div className="p-5 border-b border-gray-100 bg-gray-50">
                                <div className="flex justify-between items-start mb-2">
                                    <h3 className="font-bold text-lg text-govt-navy leading-tight">{selectedZoneData.name}</h3>
                                    <span className={`px-2 py-0.5 text-xs rounded-full font-bold uppercase tracking-wider ${getZoneSeverity(selectedZoneData.id) === 'critical' ? 'bg-red-100 text-red-700' :
                                        getZoneSeverity(selectedZoneData.id) === 'high' ? 'bg-orange-100 text-orange-700' :
                                            getZoneSeverity(selectedZoneData.id) === 'warning' ? 'bg-yellow-100 text-yellow-700' :
                                                'bg-green-100 text-green-700'
                                        }`}>
                                        {getZoneSeverity(selectedZoneData.id)}
                                    </span>
                                </div>
                                <p className="text-xs text-gray-500 font-mono">Zone ID: {selectedZoneData.id} • {selectedZoneData.lat.toFixed(4)}, {selectedZoneData.lng.toFixed(4)}</p>
                            </div>

                            <div className="flex-1 p-5 overflow-y-auto">
                                <h4 className="font-semibold text-gray-700 mb-3 text-sm flex items-center gap-2">
                                    <AlertTriangle size={14} /> Active Alerts
                                </h4>

                                {selectedAlerts.length === 0 ? (
                                    <div className="text-center py-8 bg-gray-50 rounded-lg border border-dashed border-gray-200">
                                        <Shield size={24} className="mx-auto text-green-400 mb-2" />
                                        <p className="text-sm text-gray-500">No active threats detected.</p>
                                        <p className="text-xs text-gray-400 mt-1">Area is secure.</p>
                                    </div>
                                ) : (
                                    <div className="space-y-2">
                                        {selectedAlerts.map((alert, i) => (
                                            <Link
                                                key={i}
                                                href={`/dashboard/chatbot?alert_id=${alert.alert_id}`}
                                                className="block p-3 bg-white rounded-lg border border-gray-200 shadow-sm hover:border-govt-blue hover:shadow-md transition-all cursor-pointer group"
                                            >
                                                <div className="flex justify-between items-start">
                                                    <span className="font-mono text-[10px] text-gray-400 uppercase group-hover:text-govt-blue transition-colors">
                                                        {alert.alert_id?.slice(0, 12)}
                                                    </span>
                                                    {(alert.severity === 'critical' || alert.severity === 'high') && (
                                                        <span className={`w-2 h-2 rounded-full animate-pulse ${alert.severity === 'critical' ? 'bg-red-500' : 'bg-orange-500'}`} />
                                                    )}
                                                </div>
                                                <p className="text-sm font-semibold text-gray-800 mt-1 line-clamp-2 group-hover:text-govt-blue transition-colors">{alert.title}</p>
                                                <p className="text-xs text-gray-500 mt-1 line-clamp-2">{alert.message || 'No additional details.'}</p>
                                            </Link>
                                        ))}
                                    </div>
                                )}
                            </div>

                            <div className="p-4 border-t border-gray-200 bg-gray-50 space-y-2">
                                <Link
                                    href={selectedAlerts.length > 0 ? `/dashboard/chatbot?alert_id=${selectedAlerts[0].alert_id}` : "/dashboard/chatbot"}
                                    className="w-full bg-govt-navy text-white py-2.5 rounded-lg text-sm font-semibold flex items-center justify-center gap-2 hover:bg-blue-800 transition-colors shadow-sm"
                                >
                                    Dispatch Team <ExternalLink size={14} />
                                </Link>
                                <Link href="/dashboard/response" className="w-full bg-white border border-govt-blue text-govt-blue py-2.5 rounded-lg text-sm font-semibold flex items-center justify-center gap-2 hover:bg-blue-50 transition-colors">
                                    Route Risk <ChevronRight size={14} />
                                </Link>
                            </div>
                        </div>
                    ) : (
                        <div className="h-full flex flex-col items-center justify-center p-8 text-center bg-gray-50">
                            <MapPin size={48} className="text-gray-300 mb-4" />
                            <p className="text-gray-600 font-medium">Select a Zone</p>
                            <p className="text-gray-400 text-sm mt-2">Click on any colored zone circle on the map to view detailed alerts and sensor data.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

"use client";

import { useEffect } from 'react';
import { MapContainer, TileLayer, Circle, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { AlertTriangle, Shield, Activity } from 'lucide-react';
import L from 'leaflet';

// Fix for default marker icon in Next.js
// delete (L.Icon.Default.prototype as any)._getIconUrl;
// L.Icon.Default.mergeOptions({
//   iconRetinaUrl: '/leaflet/marker-icon-2x.png',
//   iconUrl: '/leaflet/marker-icon.png',
//   shadowUrl: '/leaflet/marker-shadow.png',
// });

// Constants for map view
const CENTER: [number, number] = [28.556, 77.258]; // Center of the 12 zones
const ZOOM_LEVEL = 12;

interface Zone {
    id: string;
    name: string;
    lat: number;
    lng: number;
    radius: number;
}

// 12 Zones along a track (New Delhi -> Tughlakabad approx)
const ZONES: Zone[] = [
    { id: 'Z-01', name: 'New Delhi (NDLS)', lat: 28.6427, lng: 77.2209, radius: 800 },
    { id: 'Z-02', name: 'Tilak Bridge', lat: 28.6295, lng: 77.2356, radius: 700 },
    { id: 'Z-03', name: 'Pragati Maidan', lat: 28.6186, lng: 77.2452, radius: 700 },
    { id: 'Z-04', name: 'Hazrat Nizamuddin', lat: 28.5901, lng: 77.2581, radius: 900 },
    { id: 'Z-05', name: 'Lajpat Nagar', lat: 28.5702, lng: 77.2403, radius: 700 },
    { id: 'Z-06', name: 'Okhla', lat: 28.5565, lng: 77.2801, radius: 800 },
    { id: 'Z-07', name: 'Tughlakabad', lat: 28.5028, lng: 77.2941, radius: 1000 },
    { id: 'Z-08', name: 'Badarpur', lat: 28.4901, lng: 77.2995, radius: 800 },
    { id: 'Z-09', name: 'Sarita Vihar', lat: 28.5255, lng: 77.2905, radius: 700 },
    { id: 'Z-10', name: 'Govind Puri', lat: 28.5432, lng: 77.2655, radius: 600 },
    { id: 'Z-11', name: 'Kalkaji Mandir', lat: 28.5501, lng: 77.2605, radius: 600 },
    { id: 'Z-12', name: 'Nehru Place', lat: 28.5522, lng: 77.2515, radius: 700 },
];

interface MapComponentProps {
    alerts: any[];
    onZoneSelect: (zoneId: string) => void;
    selectedZoneId: string | null;
}

const severityColors: Record<string, string> = {
    critical: '#ef4444', // red-500
    high: '#f97316',     // orange-500
    warning: '#eab308',  // yellow-500
    safe: '#22c55e',     // green-500
};

// Helper to determine zone severity
function getZoneSeverity(zoneId: string, zoneAlerts: any[]) {
    if (zoneAlerts.some(a => a.severity === 'critical')) return 'critical';
    if (zoneAlerts.some(a => a.severity === 'high')) return 'high';
    if (zoneAlerts.length > 0) return 'warning';
    return 'safe';
}

function assignAlertToZone(alertId: string, zones: any[]): string {
    let hash = 0;
    for (let i = 0; i < alertId.length; i++) {
        hash = ((hash << 5) - hash) + alertId.charCodeAt(i);
        hash |= 0;
    }
    const idx = Math.abs(hash) % zones.length;
    return zones[idx].id;
}

export default function MapComponent({ alerts, onZoneSelect, selectedZoneId }: MapComponentProps) {
    // Group alerts by zone
    const alertsByZone: Record<string, any[]> = {};
    ZONES.forEach(z => { alertsByZone[z.id] = []; });

    alerts.forEach(alert => {
        const zoneId = assignAlertToZone(alert.alert_id, ZONES);
        if (alertsByZone[zoneId]) alertsByZone[zoneId].push(alert);
    });

    return (
        <MapContainer
            center={CENTER}
            zoom={ZOOM_LEVEL}
            scrollWheelZoom={true}
            className="w-full h-full rounded-xl z-0"
            style={{ minHeight: '500px', background: '#0f172a' }} // Dark background
        >
            {/* Dark Mode Tiles - CartoDB Dark Matter */}
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            />

            {ZONES.map(zone => {
                const zoneAlerts = alertsByZone[zone.id];
                const severity = getZoneSeverity(zone.id, zoneAlerts);
                const color = severityColors[severity];
                const isSelected = selectedZoneId === zone.id;

                return (
                    <Circle
                        key={zone.id}
                        center={[zone.lat, zone.lng]}
                        pathOptions={{
                            color: color,
                            fillColor: color,
                            fillOpacity: isSelected ? 0.6 : 0.3,
                            weight: isSelected ? 3 : 1
                        }}
                        radius={zone.radius}
                        eventHandlers={{
                            click: () => onZoneSelect(zone.id),
                        }}
                    >
                        <Popup>
                            <div className="p-2 min-w-[200px]">
                                <h3 className="font-bold text-sm border-b pb-1 mb-2 flex justify-between">
                                    {zone.name}
                                    <span className={`text-xs px-2 py-0.5 rounded text-white capitalize bg-${severity === 'critical' ? 'red-500' : severity === 'high' ? 'orange-500' : severity === 'warning' ? 'yellow-500' : 'green-500'}`}>
                                        {severity}
                                    </span>
                                </h3>
                                <p className="text-xs text-gray-600 mb-2">Zone ID: {zone.id}</p>

                                {zoneAlerts.length > 0 ? (
                                    <div className="space-y-1">
                                        <p className="text-xs font-semibold">{zoneAlerts.length} Active Alerts:</p>
                                        <ul className="max-h-24 overflow-y-auto">
                                            {zoneAlerts.slice(0, 3).map((a, i) => (
                                                <li key={i} className="text-[10px] text-gray-700 truncate border-l-2 pl-1 border-gray-300">
                                                    {a.title}
                                                </li>
                                            ))}
                                            {zoneAlerts.length > 3 && <li className="text-[10px] text-gray-500 italic">+{zoneAlerts.length - 3} more</li>}
                                        </ul>
                                    </div>
                                ) : (
                                    <p className="text-xs text-green-600 flex items-center gap-1">
                                        <Shield size={12} /> No active threats
                                    </p>
                                )}
                            </div>
                        </Popup>
                    </Circle>
                );
            })}
        </MapContainer>
    );
}

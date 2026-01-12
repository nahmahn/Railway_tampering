import {
    AlertTriangle,
    Shield,
    Video,
    MapPin,
    Activity,
    Hammer,
    Ruler,
    Thermometer,
    Radio,
    Signal,
    Satellite,
    Wifi
} from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip, BarChart, Bar } from 'recharts';
import { useEffect, useState } from 'react';

function StatCard({ label, value, subtext, icon: Icon, color, trend }: any) {
    return (
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-sm font-medium text-gray-500 uppercase tracking-wide">{label}</p>
                    <div className="flex items-end gap-3 mt-1">
                        <h3 className="text-3xl font-bold text-gray-900 leading-none">{value}</h3>
                    </div>
                    <p className={`text-xs mt-3 font-medium ${subtext.includes('+') ? 'text-green-600' : 'text-gray-500'}`}>
                        {subtext}
                    </p>
                </div>
                <div className={`p-3 rounded-xl bg-${color}-50 text-${color}-600 border border-${color}-100`}>
                    <Icon size={24} />
                </div>
            </div>
            <div className="mt-4 flex items-end gap-1 h-8 opacity-20">
                {[40, 60, 45, 70, 50, 80, 65, 90].map((h, i) => (
                    <div key={i} className={`flex-1 rounded-t bg-${color}-600`} style={{ height: `${h}%` }}></div>
                ))}
            </div>
        </div>
    );
}

// Data Interface based on dataset/data.json
interface RealData {
    source_file_attributes: {
        filename: string;
    };
    acquisition_timestamp: {
        utc_time: string;
    };
    sensor_hardware_config: {
        sensor_id: string;
    };
    geo_positioning_input: {
        latitude_decimal: number;
        longitude_decimal: number;
        satellite_count: number;
    };
    locomotive_telemetry_bus: {
        loco_id: string;
        current_speed_kmh: number;
        train_interface_unit_status: string;
    };
    environmental_sensor_readings: {
        external_temp_c: number;
        humidity_percent: number;
    };
}

// Hook to load data
function useRealData() {
    const [data, setData] = useState<RealData[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch('/dataset/data.json')
            .then(res => res.json())
            .then(data => {
                setData(data);
                setLoading(false);
            })
            .catch(err => {
                console.error("Failed to load dataset:", err);
                setLoading(false);
            });
    }, []);

    return { data, loading };
}

export function RPFDashboard() {
    const { data, loading } = useRealData();

    if (loading) return <div>Loading Live Data...</div>;

    const latest = data[0] || {} as RealData;
    const activeThreats = data.filter(d => d.locomotive_telemetry_bus?.train_interface_unit_status !== 'ONLINE').length;

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-2xl font-bold text-govt-navy">Security Command Center</h2>
                <p className="text-gray-500">Railway Protection Force (RPF) -  Real-time Surveillance</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <StatCard label="Active Threats" value={activeThreats.toString()} subtext="Anomaly Detected" icon={Shield} color="red" />
                <StatCard label="Intrusions" value="5" subtext="+2 in last hour" icon={AlertTriangle} color="orange" />
                <StatCard label="CCTV Active" value={`${data.length}/12`} subtext="Feeds Online" icon={Video} color="blue" />
                <StatCard label="Patrol Units" value="4" subtext={`Lat: ${latest.geo_positioning_input?.latitude_decimal?.toFixed(4) || 'N/A'}`} icon={MapPin} color="green" />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                    <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                        <Video className="text-blue-500" size={18} /> Live Feed Snapshots
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                        {data.slice(0, 4).map((item, idx) => (
                            <div key={idx} className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center relative overflow-hidden group">
                                <img
                                    src={`/dataset/${item.source_file_attributes.filename}`}
                                    alt={`Feed ${idx}`}
                                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                                />
                                <span className={`absolute top-2 left-2 text-white text-[10px] px-2 py-0.5 rounded font-bold ${idx === 0 ? 'bg-red-600 animate-pulse' : 'bg-green-600'}`}>
                                    {idx === 0 ? 'LIVE - ALERT' : `REC - ${item.sensor_hardware_config.sensor_id.split('_').pop()}`}
                                </span>
                                <div className="absolute bottom-0 left-0 right-0 bg-black/60 p-2 text-[10px] text-white opacity-0 group-hover:opacity-100 transition-opacity">
                                    <p>CAM: {item.sensor_hardware_config.sensor_id}</p>
                                    <p>TIME: {new Date(item.acquisition_timestamp.utc_time).toLocaleTimeString()}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                    <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                        <Activity className="text-red-500" size={18} /> Alert Timeline
                    </h3>
                    <div className="space-y-4 max-h-[400px] overflow-y-auto pr-2">
                        {data.map((item, i) => (
                            <div key={i} className="flex gap-4 p-3 border-b border-gray-100 last:border-0 hover:bg-slate-50 rounded transition-colors">
                                <div className="text-xs font-mono text-gray-400 w-16 pt-1">
                                    {new Date(item.acquisition_timestamp.utc_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </div>
                                <div>
                                    <p className="text-sm font-medium text-gray-700">Motion Detected - Sector 4</p>
                                    <p className="text-xs text-gray-500">Camera {item.sensor_hardware_config.sensor_id} reported movement.</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

export function MaintenanceDashboard() {
    const { data, loading } = useRealData();
    if (loading) return <div>Loading...</div>;
    const latest = data[0] || {} as RealData;

    const tempTrend = data.map((d, i) => ({
        time: i,
        temp: d.environmental_sensor_readings.external_temp_c
    }));

    // Mock track health derived from temp/humidity logic
    const trackHealthData = data.slice(0, 5).map((d, i) => ({
        km: `Km ${12 + i}`,
        health: 100 - (d.environmental_sensor_readings.humidity_percent / 5) // Mock logic
    }));

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-2xl font-bold text-govt-navy">Engineering & Maintenance</h2>
                <p className="text-gray-500">Structural Health & Defect Analysis</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <StatCard label="Critical Defects" value="1" subtext="Needs Immediate Action" icon={Hammer} color="red" />
                <StatCard label="Track Health" value="94%" subtext="-2% degradation" icon={Activity} color="yellow" />
                <StatCard label="Gauge Deviation" value="0.4mm" subtext="Within Tolerance" icon={Ruler} color="blue" />
                <StatCard label="Rail Temp" value={`${latest.environmental_sensor_readings?.external_temp_c || 25}°C`} subtext="Real-time Sensor" icon={Thermometer} color="orange" />
            </div>

            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <Thermometer className="text-orange-500" size={18} /> Rail Temperature Gradient
                </h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={tempTrend}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                            <XAxis dataKey="time" hide />
                            <YAxis domain={['auto', 'auto']} axisLine={false} tickLine={false} tick={{ fill: '#64748B', fontSize: 12 }} />
                            <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                            <Area type="monotone" dataKey="temp" stroke="#F97316" fill="#FFEDD5" strokeWidth={2} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <Activity className="text-blue-500" size={18} /> Track Health Index (Last 5 Km)
                </h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={trackHealthData}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                            <XAxis dataKey="km" axisLine={false} tickLine={false} tick={{ fill: '#64748B', fontSize: 12 }} />
                            <YAxis domain={[60, 100]} axisLine={false} tickLine={false} tick={{ fill: '#64748B', fontSize: 12 }} />
                            <Tooltip cursor={{ fill: '#F1F5F9' }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                            <Bar dataKey="health" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}



export function CCRDashboard() {
    const { data, loading } = useRealData();
    if (loading) return <div>Loading...</div>;

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-2xl font-bold text-govt-navy">Central Control Room</h2>
                <p className="text-gray-500">Cross-Departmental Consolidated View</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <StatCard label="Total Active Alerts" value="8" subtext="All Departments" icon={AlertTriangle} color="red" />
                <StatCard label="System Status" value="Healthy" subtext="All Nodes Online" icon={Activity} color="green" />
                <StatCard label="Safety Index" value="96/100" subtext="Daily Score" icon={Shield} color="blue" />
                <StatCard label="Fleet Active" value={data.length.toString()} subtext="Locos on Track" icon={Radio} color="purple" />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                    <h3 className="text-lg font-bold text-gray-800 mb-4">Live Incident Map</h3>
                    <div className="bg-slate-100 rounded-lg h-96 overflow-hidden relative group">
                        {/* Mock Map using one of the dataset images blurred or a placeholder if preferred, 
                            but let's use a nice abstract map-like representation or just the image for now if map library isn't available.
                            Using the first image as a 'satellite view' background.
                        */}
                        <img src="/dataset/img002.jpeg" className="w-full h-full object-cover opacity-20" alt="Map Base" />
                        <div className="absolute inset-0 flex items-center justify-center">
                            <div className="relative w-full h-full p-10">
                                {/* Pins based on approximate relative positions logic or just mock pins on the image */}
                                <div className="absolute top-1/4 left-1/4 group/pin">
                                    <MapPin className="text-red-500 animate-bounce" size={32} />
                                    <div className="absolute top-8 left-0 bg-white p-2 rounded shadow text-xs whitespace-nowrap hidden group-hover/pin:block">
                                        Tampering Detected
                                    </div>
                                </div>
                                <div className="absolute bottom-1/3 right-1/3 group/pin">
                                    <div className="w-4 h-4 bg-blue-500 rounded-full animate-ping absolute"></div>
                                    <div className="w-4 h-4 bg-blue-500 rounded-full relative"></div>
                                    <div className="absolute top-6 left-0 bg-white p-2 rounded shadow text-xs whitespace-nowrap hidden group-hover/pin:block">
                                        Patrol Unit Alpha
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                    <h3 className="text-lg font-bold text-gray-800 mb-4">Department Load</h3>
                    <div className="space-y-4">
                        {[{ dept: 'RPF', count: 35 }, { dept: 'Maintenance', count: 45 }].map(d => (
                            <div key={d.dept}>
                                <div className="flex justify-between text-sm mb-1">
                                    <span className="font-medium text-gray-700">{d.dept}</span>
                                    <span className="text-gray-500">{d.count}%</span>
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-2">
                                    <div className="bg-govt-navy h-2 rounded-full" style={{ width: `${d.count}%` }}></div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

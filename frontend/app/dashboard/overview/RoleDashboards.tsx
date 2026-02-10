import {
    AlertTriangle,
    Shield,
    Video,
    MapPin,
    Activity,
    Hammer,
    Ruler,
    Thermometer,
    Radio
} from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip, BarChart, Bar } from 'recharts';
import { useEffect, useState } from 'react';
import { useAlerts } from '@/contexts/AlertContext';
import { api } from '@/services/api';

function StatCard({ label, value, subtext, icon: Icon, color }: any) {
    return (
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-sm font-medium text-gray-500 uppercase tracking-wide">{label}</p>
                    <div className="flex items-end gap-3 mt-1">
                        <h3 className="text-3xl font-bold text-gray-900 leading-none">{value}</h3>
                    </div>
                    <p className={`text-xs mt-3 font-medium ${subtext.includes('+') || subtext.includes('No') ? 'text-green-600' : 'text-gray-500'}`}>
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

export function RPFDashboard() {
    const { alerts } = useAlerts();

    // Filter alerts for RPF (High/Critical security threats, visual anomalies)
    const securityAlerts = alerts.filter(a =>
        a.severity === 'critical' || a.severity === 'high' || a.source === 'VisualExpert'
    );

    const activeThreats = securityAlerts.filter(a => !a.acknowledged).length;

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-2xl font-bold text-govt-navy">Security Command Center</h2>
                <p className="text-gray-500">Railway Protection Force (RPF) - Real-time Surveillance</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <StatCard
                    label="Active Threats"
                    value={activeThreats.toString()}
                    subtext={activeThreats > 0 ? "Immediate Action Required" : "No Active Threats"}
                    icon={Shield}
                    color={activeThreats > 0 ? "red" : "green"}
                />
                <StatCard
                    label="Recent Intrusions"
                    value={securityAlerts.length.toString()}
                    subtext="Last 24 Hours"
                    icon={AlertTriangle}
                    color="orange"
                />
                <StatCard label="CCTV Active" value="12/12" subtext="All Feeds Online" icon={Video} color="blue" />
                <StatCard label="Patrol Units" value="4" subtext="Deployed in Sector 4" icon={MapPin} color="green" />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                    <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                        <Activity className="text-red-500" size={18} /> Live Alert Feed
                    </h3>
                    <div className="space-y-4 max-h-[400px] overflow-y-auto pr-2">
                        {securityAlerts.length === 0 ? (
                            <div className="text-center py-10 text-gray-400">No security alerts detected.</div>
                        ) : (
                            securityAlerts.map((alert, i) => (
                                <div key={i} className="flex gap-4 p-3 border-b border-gray-100 last:border-0 hover:bg-slate-50 rounded transition-colors">
                                    <div className="text-xs font-mono text-gray-400 w-16 pt-1">
                                        {new Date(alert.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    </div>
                                    <div>
                                        <p className="text-sm font-medium text-gray-700">{alert.title}</p>
                                        <p className="text-xs text-gray-500">{alert.message}</p>
                                        <span className={`inline-block mt-1 px-2 py-0.5 text-[10px] rounded font-bold uppercase ${alert.severity === 'critical' ? 'bg-red-100 text-red-700' :
                                                alert.severity === 'high' ? 'bg-orange-100 text-orange-700' : 'bg-blue-100 text-blue-700'
                                            }`}>
                                            {alert.severity}
                                        </span>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export function MaintenanceDashboard() {
    const { alerts } = useAlerts();
    const [history, setHistory] = useState<any[]>([]);

    useEffect(() => {
        api.getHistory().then(data => setHistory(data));
    }, []);

    const maintenanceAlerts = alerts.filter(a => a.source === 'StructuralExpert' || a.source === 'ThermalExpert');
    const criticalDefects = maintenanceAlerts.filter(a => a.severity === 'high' || a.severity === 'critical').length;

    // Calculate average track health from recent history (mock calculation based on confidence)
    const recentScans = history.slice(0, 10);
    const avgHealth = recentScans.length > 0
        ? Math.round(recentScans.reduce((acc, curr) => acc + (curr.overall_assessment?.confidence || 0), 0) / recentScans.length * 100)
        : 100;

    const trackHealthData = recentScans.map((d, i) => ({
        id: d.session_id.substring(0, 4),
        health: Math.round((d.overall_assessment?.confidence || 0) * 100)
    })).reverse();

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-2xl font-bold text-govt-navy">Engineering & Maintenance</h2>
                <p className="text-gray-500">Structural Health & Defect Analysis</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <StatCard
                    label="Critical Defects"
                    value={criticalDefects.toString()}
                    subtext={criticalDefects > 0 ? "Immediate Repair Needed" : "No Critical Defects"}
                    icon={Hammer}
                    color={criticalDefects > 0 ? "red" : "green"}
                />
                <StatCard
                    label="Track Health"
                    value={`${avgHealth}%`}
                    subtext="Average Condition"
                    icon={Activity}
                    color={avgHealth < 90 ? "yellow" : "green"}
                />
                <StatCard label="Gauge Deviation" value="0.4mm" subtext="Avg. Last 24h" icon={Ruler} color="blue" />
                <StatCard label="Rail Temp" value="25°C" subtext="Nominal" icon={Thermometer} color="orange" />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                    <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                        <Activity className="text-blue-500" size={18} /> Recent Track Health Scans
                    </h3>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={trackHealthData}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                                <XAxis dataKey="id" axisLine={false} tickLine={false} tick={{ fill: '#64748B', fontSize: 12 }} />
                                <YAxis domain={[0, 100]} axisLine={false} tickLine={false} tick={{ fill: '#64748B', fontSize: 12 }} />
                                <Tooltip cursor={{ fill: '#F1F5F9' }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                                <Bar dataKey="health" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                    <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                        <Hammer className="text-orange-500" size={18} /> Maintenance Alerts
                    </h3>
                    <div className="space-y-4 max-h-[400px] overflow-y-auto pr-2">
                        {maintenanceAlerts.length === 0 ? (
                            <div className="text-center py-10 text-gray-400">No maintenance alerts.</div>
                        ) : (
                            maintenanceAlerts.map((alert, i) => (
                                <div key={i} className="flex gap-4 p-3 border-b border-gray-100 last:border-0 hover:bg-slate-50 rounded transition-colors">
                                    <div className="text-xs font-mono text-gray-400 w-16 pt-1">
                                        {new Date(alert.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    </div>
                                    <div>
                                        <p className="text-sm font-medium text-gray-700">{alert.title}</p>
                                        <p className="text-xs text-gray-500">{alert.message}</p>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export function CCRDashboard() {
    const { alerts, activeAlertCount } = useAlerts();

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-2xl font-bold text-govt-navy">Central Control Room</h2>
                <p className="text-gray-500">Cross-Departmental Consolidated View</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <StatCard
                    label="Total Active Alerts"
                    value={activeAlertCount.toString()}
                    subtext="All Departments"
                    icon={AlertTriangle}
                    color={activeAlertCount > 0 ? "red" : "green"}
                />
                <StatCard label="System Status" value="Healthy" subtext="Nodes Online" icon={Activity} color="green" />
                <StatCard label="Safety Index" value="96/100" subtext="Daily Score" icon={Shield} color="blue" />
                <StatCard label="Fleet Active" value="14" subtext="Locos on Track" icon={Radio} color="purple" />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                    <h3 className="text-lg font-bold text-gray-800 mb-4">Live Alert Log</h3>
                    <div className="space-y-4 max-h-[400px] overflow-y-auto">
                        {alerts.map((alert, i) => (
                            <div key={i} className="flex gap-4 p-3 border-b border-gray-100 last:border-0 hover:bg-slate-50 rounded transition-colors">
                                <span className={`w-2 h-2 mt-2 rounded-full flex-shrink-0 ${alert.severity === 'critical' ? 'bg-red-500' : 'bg-blue-500'
                                    }`}></span>
                                <div>
                                    <p className="text-sm font-medium text-gray-700">{alert.title}</p>
                                    <p className="text-xs text-gray-500">{alert.message} • {alert.source}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

"use client";

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { Send, User, Bot, Briefcase, Users, CheckCircle2, Clock, AlertTriangle, Rocket, Wrench, Shield, Radio, Phone, MapPin, ChevronDown } from 'lucide-react';
import { api } from '@/services/api';
import { useAlerts } from '@/contexts/AlertContext';
import ReactMarkdown from 'react-markdown';

const SUGGESTED_QUERIES = [
    "Why was alert #102 flagged?",
    "Show similar incidents in Sector 4",
    "Generate inspection checklist",
];

// Equipment by alert type
const EQUIPMENT_MAP: Record<string, string[]> = {
    "Rail Fracture": ["Ultrasonic Flaw Detector (USFD)", "Hydraulic Jack", "Fish Plates & Bolts (Spare)", "Red Flag and Flares", "Torque Wrench"],
    "Obstacle": ["Heavy Crane", "Safety Flags", "Portable Lighting", "Radio Communication Set", "First Aid Kit"],
    "Track Displacement": ["Track Gauge", "Hydraulic Track Jack", "Rail Tongs", "Tamping Tools", "Spirit Level"],
    "default": ["USFD Equipment", "Safety Gear", "Portable Radio", "First Aid Kit", "Red/Green Flags", "Tool Kit"],
};

function ChatbotContent() {
    const searchParams = useSearchParams();
    const alertIdParam = searchParams.get('alert_id');

    const [messages, setMessages] = useState([
        { role: 'bot', content: "I am the AI Operational Assistant for Indian Railways. I have access to real-time track telemetry and historical data. How can I assist you with the latest alerts?" }
    ]);
    const [input, setInput] = useState('');
    const [activeTab, setActiveTab] = useState('chat');
    const [loading, setLoading] = useState(false);

    // Planning & Logistics state
    const { alerts } = useAlerts();
    const [crews, setCrews] = useState<any[]>([]);
    const [missions, setMissions] = useState<any[]>([]);
    const [loadingPlanning, setLoadingPlanning] = useState(false);
    const [selectedAlert, setSelectedAlert] = useState<string | null>(null);
    const [selectedCrew, setSelectedCrew] = useState<string>('');
    const [missionNotes, setMissionNotes] = useState('');
    const [checkedEquipment, setCheckedEquipment] = useState<Set<string>>(new Set());
    const [dispatchSuccess, setDispatchSuccess] = useState<string | null>(null);

    // Handle deeplinking from Map View or Analysis
    useEffect(() => {
        if (alertIdParam) {
            setActiveTab('planning');
            setSelectedAlert(alertIdParam);
        }
    }, [alertIdParam]);

    // Fetch crews & missions when Planning tab opens
    useEffect(() => {
        if (activeTab === 'planning') {
            fetchPlanningData();
        }
    }, [activeTab]);

    const fetchPlanningData = async () => {
        setLoadingPlanning(true);
        try {
            const [crewData, missionData] = await Promise.all([
                api.getCrews(),
                api.getMissions(),
            ]);
            setCrews(crewData);
            setMissions(missionData);
        } catch (e) {
            console.error("Failed to fetch planning data:", e);
        } finally {
            setLoadingPlanning(false);
        }
    };

    const handleDispatch = async () => {
        if (!selectedAlert || !selectedCrew) return;
        try {
            await api.createMission(selectedAlert, 'P2', selectedCrew, missionNotes);
            setDispatchSuccess(`Team dispatched successfully for alert ${selectedAlert}`);
            setSelectedAlert(null);
            setSelectedCrew('');
            setMissionNotes('');
            setCheckedEquipment(new Set());
            await fetchPlanningData();
            setTimeout(() => setDispatchSuccess(null), 4000);
        } catch (e) {
            console.error("Dispatch failed:", e);
        }
    };

    const getEquipmentList = () => {
        const alert = alerts.find(a => a.alert_id === selectedAlert);
        const alertTitle = alert?.title || '';
        for (const [key, items] of Object.entries(EQUIPMENT_MAP)) {
            if (alertTitle.toLowerCase().includes(key.toLowerCase())) return items;
        }
        return EQUIPMENT_MAP['default'];
    };

    const toggleEquipment = (item: string) => {
        setCheckedEquipment(prev => {
            const next = new Set(prev);
            next.has(item) ? next.delete(item) : next.add(item);
            return next;
        });
    };

    const handleStatusChange = async (crewId: string, newStatus: string) => {
        try {
            const success = await api.updateCrewStatus(crewId, newStatus);
            if (success) {
                setCrews(prev => prev.map(c => c.id === crewId ? { ...c, status: newStatus } : c));
            }
        } catch (e) {
            console.error("Failed to update crew status:", e);
        }
    };

    const STATUS_OPTIONS = [
        { value: 'available', label: 'Available', color: 'bg-green-500' },
        { value: 'standby', label: 'Standby', color: 'bg-yellow-500' },
        { value: 'off_duty', label: 'Off Duty', color: 'bg-gray-500' },
        { value: 'on_mission', label: 'On Mission', color: 'bg-blue-500' },
    ];

    const getStatusStyle = (status: string) => {
        switch (status) {
            case 'available': return { bg: 'bg-emerald-50 border-emerald-200', text: 'text-emerald-700', dot: 'bg-emerald-500' };
            case 'on_mission': return { bg: 'bg-blue-50 border-blue-200', text: 'text-blue-700', dot: 'bg-blue-500' };
            case 'standby': return { bg: 'bg-amber-50 border-amber-200', text: 'text-amber-700', dot: 'bg-amber-500' };
            case 'off_duty': return { bg: 'bg-gray-50 border-gray-300', text: 'text-gray-500', dot: 'bg-gray-400' };
            default: return { bg: 'bg-gray-50 border-gray-200', text: 'text-gray-600', dot: 'bg-gray-400' };
        }
    };

    // Show ALL alerts — not just action_required
    const missionSessionIds = new Set(missions.map(m => m.alert_session_id));
    const actionableAlerts = alerts; // Show every alert from the WebSocket feed
    const availableCrews = crews.filter(c => c.status === 'available' || c.status === 'standby');

    const sendMessage = async (text: string) => {
        if (!text.trim()) return;

        const userMsg = { role: 'user', content: text };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const aiResponseText = await api.query(text, { source: "dashboard_chat" });
            setMessages(prev => [...prev, { role: 'bot', content: aiResponseText }]);
        } catch (e) {
            console.error("Chat error:", e);
            setMessages(prev => [...prev, { role: 'bot', content: "Error connecting to AI service. Please check if the backend server is running." }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="h-full flex flex-col space-y-6">
            <div className="flex justify-between items-center">
                <div>
                    <h2 className="text-2xl font-bold text-govt-navy">AI Assistant</h2>
                    <p className="text-gray-500">Operational Decision Support</p>
                </div>

                {/* Toggle Mode */}
                <div className="bg-white border md:flex hidden border-gray-300 rounded-lg p-1">
                    <button
                        onClick={() => setActiveTab('chat')}
                        className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${activeTab === 'chat' ? 'bg-govt-navy text-white shadow' : 'text-gray-600 hover:bg-gray-100'}`}
                    >
                        Conversation Mode
                    </button>
                    <button
                        onClick={() => setActiveTab('planning')}
                        className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${activeTab === 'planning' ? 'bg-govt-navy text-white shadow' : 'text-gray-600 hover:bg-gray-100'}`}
                    >
                        Planning & Logistics
                    </button>
                </div>
            </div>

            <div className="flex-1 bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden flex">
                {activeTab === 'chat' && (
                    <>
                        {/* Chat Area */}
                        <div className="flex-1 flex flex-col">
                            <div className="flex-1 p-6 overflow-y-auto space-y-6">
                                {messages.map((msg, idx) => (
                                    <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                                        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${msg.role === 'bot' ? 'bg-govt-blue text-white' : 'bg-gray-200 text-gray-700'}`}>
                                            {msg.role === 'bot' ? <Bot size={18} /> : <User size={18} />}
                                        </div>
                                        <div className={`max-w-[70%] p-4 rounded-lg text-sm leading-relaxed ${msg.role === 'bot' ? 'bg-gray-100 text-gray-800' : 'bg-govt-blue text-white'}`}>
                                            {msg.role === 'bot' ? (
                                                <div className="prose prose-sm max-w-none prose-p:leading-relaxed prose-li:my-0 shadow-sm">
                                                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                                                </div>
                                            ) : (
                                                <div className="whitespace-pre-wrap">{msg.content}</div>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>

                            <div className="p-4 border-t border-gray-200 bg-gray-50">
                                <div className="flex gap-2 mb-4 overflow-x-auto pb-2">
                                    {SUGGESTED_QUERIES.map(q => (
                                        <button
                                            key={q}
                                            onClick={() => sendMessage(q)}
                                            className="px-3 py-1 bg-white border border-govt-blue text-govt-blue text-xs rounded-full hover:bg-blue-50 whitespace-nowrap"
                                        >
                                            {q}
                                        </button>
                                    ))}
                                </div>
                                <div className="flex gap-2">
                                    <input
                                        value={input}
                                        onChange={(e) => setInput(e.target.value)}
                                        onKeyDown={(e) => e.key === 'Enter' && sendMessage(input)}
                                        className="flex-1 border border-gray-300 rounded px-4 py-2 text-sm focus:outline-none focus:border-govt-blue"
                                        placeholder="Ask about anomalies, regulations, or historical data..."
                                        disabled={loading}
                                    />
                                    <button
                                        onClick={() => sendMessage(input)}
                                        disabled={loading}
                                        className={`bg-govt-navy text-white px-4 py-2 rounded hover:bg-govt-blue ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                                    >
                                        <Send size={18} />
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* Context Panel (Desktop) */}
                        <div className="w-80 border-l border-gray-200 bg-gray-50 p-6 hidden lg:block">
                            <h4 className="font-bold text-gray-700 mb-4">Current Context</h4>
                            <div className="bg-white p-4 rounded border border-gray-200 shadow-sm mb-4">
                                <p className="text-xs text-gray-500 uppercase">Active Alerts</p>
                                <p className="font-bold text-govt-navy text-2xl">{alerts.filter(a => !a.acknowledged).length}</p>
                            </div>
                            <div className="bg-white p-4 rounded border border-gray-200 shadow-sm mb-4">
                                <p className="text-xs text-gray-500 uppercase">Active Missions</p>
                                <p className="font-bold text-govt-navy text-2xl">{missions.filter(m => m.stage !== 'resolved').length}</p>
                            </div>

                            <h4 className="font-bold text-gray-700 mb-4 mt-8">Quick Actions</h4>
                            <button onClick={() => setActiveTab('planning')} className="w-full text-left px-3 py-2 bg-white border border-gray-200 rounded text-sm hover:bg-blue-50 mb-2 flex items-center gap-2">
                                <Briefcase size={14} className="text-govt-orange" /> Open Planning
                            </button>
                        </div>
                    </>
                )}

                {activeTab === 'planning' && (
                    <div className="w-full overflow-y-auto">
                        {loadingPlanning ? (
                            <div className="flex items-center justify-center h-64">
                                <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-govt-blue"></div>
                            </div>
                        ) : (
                            <div className="p-6 space-y-8">
                                {/* Success banner */}
                                {dispatchSuccess && (
                                    <div className="bg-green-50 border border-green-300 text-green-800 px-4 py-3 rounded-lg flex items-center gap-2 animate-pulse">
                                        <CheckCircle2 size={18} /> {dispatchSuccess}
                                    </div>
                                )}

                                {/* Stats Row */}
                                <div className="grid grid-cols-4 gap-4">
                                    <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
                                        <AlertTriangle className="mx-auto mb-1 text-red-500" size={22} />
                                        <p className="text-2xl font-bold text-red-700">{actionableAlerts.length}</p>
                                        <p className="text-xs text-red-500">Actionable Alerts</p>
                                    </div>
                                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
                                        <Rocket className="mx-auto mb-1 text-blue-500" size={22} />
                                        <p className="text-2xl font-bold text-blue-700">{missions.filter(m => m.stage !== 'resolved').length}</p>
                                        <p className="text-xs text-blue-500">Active Missions</p>
                                    </div>
                                    <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-center">
                                        <Users className="mx-auto mb-1 text-green-500" size={22} />
                                        <p className="text-2xl font-bold text-green-700">{availableCrews.length}</p>
                                        <p className="text-xs text-green-500">Available Crews</p>
                                    </div>
                                    <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 text-center">
                                        <CheckCircle2 className="mx-auto mb-1 text-purple-500" size={22} />
                                        <p className="text-2xl font-bold text-purple-700">{missions.filter(m => m.stage === 'resolved').length}</p>
                                        <p className="text-xs text-purple-500">Resolved</p>
                                    </div>
                                </div>

                                {/* ── Crew Roster ── */}
                                <div>
                                    <h3 className="text-lg font-bold text-govt-navy mb-4 flex items-center gap-2">
                                        <Users size={18} className="text-govt-blue" /> Crew Roster
                                        <span className="ml-auto text-xs font-normal text-gray-400">Click status to change · synced to DB</span>
                                    </h3>
                                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                                        {crews.map(crew => {
                                            const style = getStatusStyle(crew.status);
                                            return (
                                                <div key={crew.id} className={`relative rounded-xl border-2 p-4 transition-all hover:shadow-md ${style.bg}`}>
                                                    {/* Header */}
                                                    <div className="flex items-start justify-between mb-3">
                                                        <div className="flex items-center gap-3">
                                                            <div className="w-10 h-10 rounded-full bg-govt-navy text-white flex items-center justify-center font-bold text-sm">
                                                                {crew.team_lead?.split(' ').map((n: string) => n[0]).join('')}
                                                            </div>
                                                            <div>
                                                                <p className="font-semibold text-govt-navy text-sm">{crew.team_lead}</p>
                                                                <p className="text-xs text-gray-500 font-mono">{crew.id}</p>
                                                            </div>
                                                        </div>
                                                        {/* Status Dropdown */}
                                                        <div className="relative">
                                                            <select
                                                                value={crew.status}
                                                                onChange={(e) => handleStatusChange(crew.id, e.target.value)}
                                                                className={`appearance-none pl-5 pr-7 py-1 text-xs font-semibold rounded-full border cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-300 ${style.bg} ${style.text}`}
                                                            >
                                                                {STATUS_OPTIONS.map(opt => (
                                                                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                                                                ))}
                                                            </select>
                                                            <span className={`absolute left-2 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full ${style.dot}`} />
                                                            <ChevronDown size={10} className={`absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none ${style.text}`} />
                                                        </div>
                                                    </div>

                                                    {/* Details */}
                                                    <div className="space-y-1.5 text-xs text-gray-600">
                                                        <div className="flex items-center gap-2">
                                                            <Wrench size={12} className="text-gray-400 flex-shrink-0" />
                                                            <span>{crew.specialization}</span>
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <MapPin size={12} className="text-gray-400 flex-shrink-0" />
                                                            <span>{crew.zone} Zone</span>
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <Users size={12} className="text-gray-400 flex-shrink-0" />
                                                            <span>{crew.members || '—'} members</span>
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <Phone size={12} className="text-gray-400 flex-shrink-0" />
                                                            <span className="font-mono">{crew.contact || 'N/A'}</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>

                                {/* ── Alerts + Dispatch ── */}
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                    {/* Alerts Requiring Action */}
                                    <div>
                                        <h3 className="text-lg font-bold text-govt-navy mb-3 flex items-center gap-2">
                                            <AlertTriangle size={18} className="text-govt-orange" /> All Alerts
                                            <span className="ml-auto text-xs font-normal text-gray-400">{actionableAlerts.length} total · click to dispatch</span>
                                        </h3>
                                        {actionableAlerts.length === 0 ? (
                                            <div className="bg-green-50 p-4 rounded-lg text-green-700 text-sm border border-green-200">
                                                No alerts requiring immediate action.
                                            </div>
                                        ) : (
                                            <div className="space-y-2 max-h-72 overflow-y-auto">
                                                {alerts.map(alert => {
                                                    const alreadyDispatched = missionSessionIds.has(alert.alert_id);
                                                    return (
                                                        <div
                                                            key={alert.alert_id}
                                                            onClick={() => setSelectedAlert(alert.alert_id)}
                                                            className={`p-3 rounded-lg border cursor-pointer transition-all ${selectedAlert === alert.alert_id
                                                                ? 'border-govt-blue bg-blue-50 ring-2 ring-blue-300'
                                                                : 'border-gray-200 bg-white hover:border-govt-blue hover:shadow-sm'
                                                                }`}
                                                        >
                                                            <div className="flex justify-between items-center">
                                                                <span className="font-mono text-xs text-gray-500">{alert.alert_id}</span>
                                                                <div className="flex items-center gap-2">
                                                                    {alreadyDispatched && (
                                                                        <span className="px-2 py-0.5 text-xs rounded-full font-semibold bg-green-100 text-green-700">✓ Dispatched</span>
                                                                    )}
                                                                    <span className={`px-2 py-0.5 text-xs rounded-full font-semibold ${alert.severity === 'critical' ? 'bg-red-100 text-red-700' :
                                                                        alert.severity === 'high' ? 'bg-orange-100 text-orange-700' :
                                                                            alert.severity === 'warning' ? 'bg-yellow-100 text-yellow-700' :
                                                                                'bg-gray-100 text-gray-600'
                                                                        }`}>
                                                                        {(alert.severity || 'info').toUpperCase()}
                                                                    </span>
                                                                </div>
                                                            </div>
                                                            <p className="text-sm font-medium mt-1">{alert.title}</p>
                                                            <p className="text-xs text-gray-500 mt-0.5">{alert.message}</p>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        )}
                                    </div>

                                    {/* Dispatch Panel / Mission Info */}
                                    <div>
                                        {selectedAlert ? (
                                            missionSessionIds.has(selectedAlert) ? (
                                                /* Already dispatched — show mission info */
                                                <div className="bg-green-50 border-2 border-green-200 rounded-xl p-5 space-y-3 h-full">
                                                    <h4 className="font-bold flex items-center gap-2 text-lg text-green-800"><CheckCircle2 size={18} /> Mission Active</h4>
                                                    <p className="text-sm text-green-700">A crew has already been dispatched for this alert.</p>
                                                    {(() => {
                                                        const mission = missions.find(m => m.alert_session_id === selectedAlert);
                                                        if (!mission) return null;
                                                        return (
                                                            <div className="space-y-2 text-sm">
                                                                <div className="bg-white rounded-lg p-3 border border-green-200">
                                                                    <p className="text-xs text-gray-500 uppercase font-semibold">Mission ID</p>
                                                                    <p className="font-mono text-govt-navy">{mission.id}</p>
                                                                </div>
                                                                <div className="bg-white rounded-lg p-3 border border-green-200">
                                                                    <p className="text-xs text-gray-500 uppercase font-semibold">Stage</p>
                                                                    <p className="font-semibold text-govt-navy">{mission.stage?.replace('_', ' ').toUpperCase()}</p>
                                                                </div>
                                                                <div className="bg-white rounded-lg p-3 border border-green-200">
                                                                    <p className="text-xs text-gray-500 uppercase font-semibold">Assigned Crew</p>
                                                                    <p className="font-semibold text-govt-navy">{mission.crew_id || 'Unassigned'}</p>
                                                                </div>
                                                            </div>
                                                        );
                                                    })()}
                                                    <p className="text-xs text-green-600 mt-2">Go to <strong>Risk Response</strong> to advance this mission through stages.</p>
                                                </div>
                                            ) : (
                                                /* Not yet dispatched — show dispatch form */
                                                <div className="bg-gradient-to-br from-govt-navy to-blue-900 text-white rounded-xl p-5 space-y-4 h-full">
                                                    <h4 className="font-bold flex items-center gap-2 text-lg"><Rocket size={18} /> Dispatch Mission</h4>

                                                    {/* Equipment Checklist */}
                                                    <div>
                                                        <p className="text-xs uppercase text-blue-300 mb-2 font-semibold tracking-wider">Equipment Checklist</p>
                                                        <div className="space-y-1">
                                                            {getEquipmentList().map(item => (
                                                                <label key={item} className="flex items-center gap-2 text-sm cursor-pointer hover:bg-white/10 px-2 py-1 rounded">
                                                                    <input
                                                                        type="checkbox"
                                                                        checked={checkedEquipment.has(item)}
                                                                        onChange={() => toggleEquipment(item)}
                                                                        className="rounded"
                                                                    />
                                                                    {item}
                                                                </label>
                                                            ))}
                                                        </div>
                                                        {checkedEquipment.size === getEquipmentList().length && (
                                                            <p className="text-green-300 text-xs mt-1 flex items-center gap-1"><CheckCircle2 size={12} /> All equipment ready</p>
                                                        )}
                                                    </div>

                                                    {/* Crew Selection */}
                                                    <div>
                                                        <p className="text-xs uppercase text-blue-300 mb-1 font-semibold tracking-wider">Assign Crew</p>
                                                        <select
                                                            value={selectedCrew}
                                                            onChange={e => setSelectedCrew(e.target.value)}
                                                            className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-sm text-white"
                                                        >
                                                            <option value="" className="text-gray-800">Select crew...</option>
                                                            {availableCrews.map(c => (
                                                                <option key={c.id} value={c.id} className="text-gray-800">
                                                                    {c.id} — {c.team_lead} ({c.specialization})
                                                                </option>
                                                            ))}
                                                        </select>
                                                    </div>

                                                    {/* Notes */}
                                                    <div>
                                                        <p className="text-xs uppercase text-blue-300 mb-1 font-semibold tracking-wider">Mission Notes</p>
                                                        <textarea
                                                            value={missionNotes}
                                                            onChange={e => setMissionNotes(e.target.value)}
                                                            rows={2}
                                                            className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-sm text-white placeholder-blue-300"
                                                            placeholder="Additional instructions..."
                                                        />
                                                    </div>

                                                    <button
                                                        onClick={handleDispatch}
                                                        disabled={!selectedCrew}
                                                        className={`w-full py-3 rounded-lg font-bold text-sm flex items-center justify-center gap-2 transition-all ${selectedCrew ? 'bg-green-500 hover:bg-green-600 text-white' : 'bg-gray-500 text-gray-300 cursor-not-allowed'
                                                            }`}
                                                    >
                                                        <Rocket size={16} /> Dispatch Maintenance Team
                                                    </button>
                                                </div>
                                            )
                                        ) : (
                                            <div className="border-2 border-dashed border-gray-200 rounded-xl p-8 flex flex-col items-center justify-center text-center h-full min-h-[200px]">
                                                <Rocket size={32} className="text-gray-300 mb-3" />
                                                <p className="text-sm text-gray-400 font-medium">Select an alert to dispatch a crew</p>
                                                <p className="text-xs text-gray-300 mt-1">Click any alert on the left to begin mission setup</p>
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* ── Recent Missions ── */}
                                <div>
                                    <h3 className="text-lg font-bold text-govt-navy mb-3 flex items-center gap-2">
                                        <Clock size={18} className="text-govt-orange" /> Recent Missions
                                    </h3>
                                    {missions.length === 0 ? (
                                        <p className="text-sm text-gray-500 bg-gray-50 p-4 rounded-lg border border-gray-200">No missions created yet.</p>
                                    ) : (
                                        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                                            {missions.slice(0, 9).map(m => (
                                                <div key={m.id} className="p-3 bg-white border border-gray-200 rounded-lg flex items-center justify-between hover:shadow-sm transition-shadow">
                                                    <div className="min-w-0">
                                                        <p className="font-mono text-xs text-gray-500 truncate">{m.id}</p>
                                                        <p className="text-sm font-medium truncate">{m.alert_data?.risk_level ? `Risk: ${m.alert_data.risk_level}` : m.alert_session_id}</p>
                                                        <p className="text-xs text-gray-500">Crew: {m.crew_id || 'Unassigned'}</p>
                                                    </div>
                                                    <span className={`px-3 py-1 text-xs rounded-full font-semibold whitespace-nowrap ml-2 ${m.stage === 'resolved' ? 'bg-green-100 text-green-700' :
                                                        m.stage === 'in_progress' ? 'bg-blue-100 text-blue-700' :
                                                            m.stage === 'assigned' ? 'bg-purple-100 text-purple-700' :
                                                                m.stage === 'triaged' ? 'bg-yellow-100 text-yellow-700' :
                                                                    'bg-gray-100 text-gray-600'
                                                        }`}>
                                                        {m.stage.replace('_', ' ').toUpperCase()}
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div >
    );
}

export default function ChatbotPage() {
    return (
        <Suspense fallback={<div className="h-full flex items-center justify-center"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-govt-blue"></div></div>}>
            <ChatbotContent />
        </Suspense>
    );
}

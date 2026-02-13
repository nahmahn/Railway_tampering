"use client";

import { useState, useEffect, useCallback } from 'react';
import { AlertTriangle, ArrowRight, Users, Clock, CheckCircle2, Shield, ChevronDown, X, MessageSquare } from 'lucide-react';
import { api } from '@/services/api';
import { useAlerts } from '@/contexts/AlertContext';

const STAGES = [
    { key: 'new', label: 'NEW', color: 'border-red-400', bg: 'bg-red-50', headerBg: 'bg-red-500', icon: AlertTriangle },
    { key: 'triaged', label: 'TRIAGED', color: 'border-yellow-400', bg: 'bg-yellow-50', headerBg: 'bg-yellow-500', icon: Shield },
    { key: 'assigned', label: 'ASSIGNED', color: 'border-blue-400', bg: 'bg-blue-50', headerBg: 'bg-blue-500', icon: Users },
    { key: 'in_progress', label: 'IN PROGRESS', color: 'border-purple-400', bg: 'bg-purple-50', headerBg: 'bg-purple-500', icon: Clock },
    { key: 'resolved', label: 'RESOLVED', color: 'border-green-400', bg: 'bg-green-50', headerBg: 'bg-green-500', icon: CheckCircle2 },
];

const PRIORITIES = ['P1', 'P2', 'P3', 'P4'];

export default function ResponsePage() {
    const { alerts } = useAlerts();
    const [missions, setMissions] = useState<any[]>([]);
    const [crews, setCrews] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    // Modal state for advancing
    const [advanceModal, setAdvanceModal] = useState<{ mission: any; nextStage: string } | null>(null);
    const [modalPriority, setModalPriority] = useState('P2');
    const [modalCrewId, setModalCrewId] = useState('');
    const [resolveNotes, setResolveNotes] = useState('');

    const fetchData = useCallback(async () => {
        try {
            const [missionData, crewData] = await Promise.all([
                api.getMissions(),
                api.getCrews(),
            ]);
            setMissions(missionData);
            setCrews(crewData);
        } catch (e) {
            console.error("Failed to fetch:", e);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { fetchData(); }, [fetchData]);

    // Get alerts that don't have a mission yet (show ALL alerts for now to ensure visibility)
    const missionSessionIds = new Set(missions.map(m => m.alert_session_id));
    const unroutedAlerts = alerts.filter(a => !missionSessionIds.has(a.alert_id));

    // Create a mission from an unrouted alert
    const handleCreateMission = async (alertId: string) => {
        try {
            await api.createMission(alertId);
            await fetchData();
        } catch (e) {
            console.error("Create mission failed:", e);
        }
    };

    // Group missions by stage and merge virtual unrouted alerts
    const missionsByStage: Record<string, any[]> = {};
    STAGES.forEach(s => { missionsByStage[s.key] = []; });

    // Add virtual missions from unrouted alerts to NEW column
    unroutedAlerts.forEach(a => {
        missionsByStage['new'].push({
            id: `ALERT-${a.alert_id}`,
            stage: 'new',
            priority: 'P2',
            alert_session_id: a.alert_id,
            alert_data: {
                risk_level: a.severity,
                tampering_type: 'Unrouted Alert'
            },
            is_virtual: true
        });
    });

    missions.forEach(m => {
        if (missionsByStage[m.stage]) missionsByStage[m.stage].push(m);
    });

    // Advance mission (handles both virtual and real)
    const handleAdvance = async () => {
        if (!advanceModal) return;
        const { mission, nextStage } = advanceModal;

        try {
            let missionId = mission.id;

            // If virtual mission, create real mission first
            if (mission.is_virtual) {
                const alertId = mission.alert_session_id;
                const result = await api.createMission(alertId);
                if (result && result.mission) {
                    missionId = result.mission.id;
                } else {
                    throw new Error("Failed to create mission from alert");
                }
            }

            if (nextStage === 'resolved') {
                await api.resolveMission(missionId, resolveNotes || 'Resolved.');
            } else {
                await api.advanceMission(missionId, modalCrewId || undefined, modalPriority || undefined);
            }

            setAdvanceModal(null);
            setModalPriority('P2');
            setModalCrewId('');
            setResolveNotes('');
            await fetchData();
        } catch (e) {
            console.error("Advance failed:", e);
        }
    };

    const getNextStage = (current: string) => {
        const idx = STAGES.findIndex(s => s.key === current);
        return idx < STAGES.length - 1 ? STAGES[idx + 1].key : null;
    };

    const availableCrews = crews.filter(c => c.status === 'available' || c.status === 'standby');

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
                    <h2 className="text-2xl font-bold text-govt-navy">Risk Response Board</h2>
                    <p className="text-gray-500">Track and route detected risks through resolution</p>
                </div>
                <div className="flex gap-3 text-sm">
                    <div className="bg-red-50 px-3 py-1.5 rounded-lg border border-red-200">
                        <span className="text-red-600 font-bold">{unroutedAlerts.length}</span>
                        <span className="text-red-500 ml-1">Unrouted</span>
                    </div>
                    <div className="bg-blue-50 px-3 py-1.5 rounded-lg border border-blue-200">
                        <span className="text-blue-600 font-bold">{missions.filter(m => m.stage !== 'resolved').length}</span>
                        <span className="text-blue-500 ml-1">Active</span>
                    </div>
                    <div className="bg-green-50 px-3 py-1.5 rounded-lg border border-green-200">
                        <span className="text-green-600 font-bold">{missions.filter(m => m.stage === 'resolved').length}</span>
                        <span className="text-green-500 ml-1">Resolved</span>
                    </div>
                </div>
            </div>



            {/* Kanban Board */}
            <div className="flex-1 flex gap-3 overflow-x-auto pb-4">
                {STAGES.map(stage => {
                    const Icon = stage.icon;
                    const cards = missionsByStage[stage.key] || [];
                    return (
                        <div key={stage.key} className={`flex-1 min-w-[220px] flex flex-col rounded-lg border-t-4 ${stage.color} bg-white shadow-sm`}>
                            {/* Column Header */}
                            <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <Icon size={16} className="text-gray-600" />
                                    <span className="font-bold text-sm text-gray-700">{stage.label}</span>
                                </div>
                                <span className="bg-gray-100 px-2 py-0.5 rounded-full text-xs font-bold text-gray-600">{cards.length}</span>
                            </div>

                            {/* Cards */}
                            <div className="flex-1 p-3 space-y-2 overflow-y-auto max-h-[calc(100vh-320px)]">
                                {cards.length === 0 && (
                                    <div className="text-center py-8 text-gray-400 text-xs">No items</div>
                                )}
                                {cards.map(m => {
                                    const nextStage = getNextStage(m.stage);
                                    return (
                                        <div key={m.id} className={`p-3 rounded-lg border ${stage.bg} border-gray-200 hover:shadow-md transition-shadow`}>
                                            <div className="flex justify-between items-start mb-1">
                                                <span className="font-mono text-[10px] text-gray-400">{m.id}</span>
                                                <span className={`px-1.5 py-0.5 text-[10px] rounded font-bold ${m.priority === 'P1' ? 'bg-red-200 text-red-800' :
                                                    m.priority === 'P2' ? 'bg-orange-200 text-orange-800' :
                                                        m.priority === 'P3' ? 'bg-yellow-200 text-yellow-800' :
                                                            'bg-gray-200 text-gray-600'
                                                    }`}>
                                                    {m.priority || 'P2'}
                                                </span>
                                            </div>
                                            <p className="text-xs font-medium text-gray-800 mb-1">
                                                {m.alert_data?.risk_level ? `Risk: ${m.alert_data.risk_level}` : `Alert: ${m.alert_session_id?.slice(0, 15)}...`}
                                            </p>
                                            {m.alert_data?.tampering_type && m.alert_data.tampering_type !== 'Unknown' && (
                                                <p className="text-[11px] text-gray-500 mb-1">Type: {m.alert_data.tampering_type}</p>
                                            )}
                                            {m.crew_id && (
                                                <p className="text-[11px] text-blue-600 font-medium mb-1">
                                                    <Users size={10} className="inline mr-0.5" />{m.crew_id}
                                                </p>
                                            )}
                                            {m.resolution_notes && (
                                                <p className="text-[11px] text-green-600 italic mt-1">
                                                    <MessageSquare size={10} className="inline mr-0.5" />{m.resolution_notes}
                                                </p>
                                            )}

                                            {/* Advance Button */}
                                            {nextStage && (
                                                <button
                                                    onClick={() => {
                                                        setAdvanceModal({ mission: m, nextStage });
                                                        setModalPriority(m.priority || 'P2');
                                                    }}
                                                    className="mt-2 w-full bg-white border border-gray-300 text-gray-700 text-xs py-1.5 rounded hover:bg-gray-50 flex items-center justify-center gap-1 transition-colors"
                                                >
                                                    Advance <ArrowRight size={12} />
                                                </button>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Advance Modal */}
            {advanceModal && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setAdvanceModal(null)}>
                    <div className="bg-white rounded-xl shadow-2xl p-6 w-96 max-w-[90vw]" onClick={e => e.stopPropagation()}>
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="font-bold text-lg text-govt-navy">
                                Advance to {STAGES.find(s => s.key === advanceModal.nextStage)?.label}
                            </h3>
                            <button onClick={() => setAdvanceModal(null)} className="text-gray-400 hover:text-gray-600">
                                <X size={20} />
                            </button>
                        </div>

                        <p className="text-sm text-gray-500 mb-4">Mission: <span className="font-mono">{advanceModal.mission.id}</span></p>

                        {/* Triage → Priority */}
                        {advanceModal.nextStage === 'triaged' && (
                            <div className="mb-4">
                                <label className="text-sm font-semibold text-gray-700 block mb-1">Set Priority</label>
                                <div className="flex gap-2">
                                    {PRIORITIES.map(p => (
                                        <button key={p} onClick={() => setModalPriority(p)}
                                            className={`px-4 py-2 rounded font-bold text-sm border transition-all ${modalPriority === p
                                                ? p === 'P1' ? 'bg-red-500 text-white border-red-500' :
                                                    p === 'P2' ? 'bg-orange-500 text-white border-orange-500' :
                                                        p === 'P3' ? 'bg-yellow-500 text-white border-yellow-500' :
                                                            'bg-gray-500 text-white border-gray-500'
                                                : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50'
                                                }`}
                                        >
                                            {p}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Assigned → Crew */}
                        {advanceModal.nextStage === 'assigned' && (
                            <div className="mb-4">
                                <label className="text-sm font-semibold text-gray-700 block mb-1">Assign Crew</label>
                                <select value={modalCrewId} onChange={e => setModalCrewId(e.target.value)}
                                    className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:border-govt-blue">
                                    <option value="">Select crew...</option>
                                    {availableCrews.map(c => (
                                        <option key={c.id} value={c.id}>{c.id} — {c.team_lead} ({c.specialization})</option>
                                    ))}
                                </select>
                            </div>
                        )}

                        {/* Resolved → Notes */}
                        {advanceModal.nextStage === 'resolved' && (
                            <div className="mb-4">
                                <label className="text-sm font-semibold text-gray-700 block mb-1">Resolution Notes</label>
                                <textarea value={resolveNotes} onChange={e => setResolveNotes(e.target.value)}
                                    rows={3} className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:border-govt-blue"
                                    placeholder="Describe the resolution..."
                                />
                            </div>
                        )}

                        <div className="flex gap-2">
                            <button onClick={() => setAdvanceModal(null)} className="flex-1 py-2 border border-gray-300 rounded text-sm text-gray-600 hover:bg-gray-50">Cancel</button>
                            <button onClick={handleAdvance}
                                disabled={advanceModal.nextStage === 'assigned' && !modalCrewId}
                                className={`flex-1 py-2 rounded text-sm font-bold text-white flex items-center justify-center gap-1 ${advanceModal.nextStage === 'assigned' && !modalCrewId
                                    ? 'bg-gray-400 cursor-not-allowed'
                                    : 'bg-govt-blue hover:bg-blue-700'
                                    }`}
                            >
                                Confirm <ArrowRight size={14} />
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

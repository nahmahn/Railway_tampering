"use client";

import { useEffect, useState } from 'react';
import { api } from '@/services/api';
import { Clock, Download, ExternalLink, Filter, Search, Eye, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import Link from 'next/link';

export default function HistoryPage() {
    const [history, setHistory] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const data = await api.getHistory();
                setHistory(data);
            } catch (error) {
                console.error("Failed to fetch history:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchHistory();
    }, []);

    const filteredHistory = history.filter(item =>
        item.session_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (item.overall_assessment?.risk_level || '').toLowerCase().includes(searchTerm.toLowerCase())
    );

    const getRiskColor = (risk: string) => {
        switch (risk?.toLowerCase()) {
            case 'critical': return 'bg-red-100 text-red-700 border-red-200';
            case 'high': return 'bg-orange-100 text-orange-700 border-orange-200';
            case 'medium': return 'bg-yellow-100 text-yellow-700 border-yellow-200';
            case 'low': return 'bg-green-100 text-green-700 border-green-200';
            default: return 'bg-gray-100 text-gray-700 border-gray-200';
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-slate-800">Analysis History</h1>
                    <p className="text-slate-500 text-sm">Review past tampering detection sessions and reports.</p>
                </div>

                <div className="flex gap-2">
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
                        <input
                            type="text"
                            placeholder="Search Session ID..."
                            className="pl-9 pr-4 py-2 border border-slate-200 rounded-lg text-sm w-64 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                        />
                    </div>
                    <button className="flex items-center gap-2 px-3 py-2 border border-slate-200 rounded-lg text-slate-600 text-sm hover:bg-slate-50">
                        <Filter size={16} /> Filter
                    </button>
                    <button className="flex items-center gap-2 px-3 py-2 bg-govt-navy text-white rounded-lg text-sm hover:bg-blue-900 transition-colors shadow-sm">
                        <Download size={16} /> Export CSV
                    </button>
                </div>
            </div>

            {loading ? (
                <div className="flex flex-col items-center justify-center py-20 bg-white rounded-xl border border-slate-200 shadow-sm">
                    <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mb-4"></div>
                    <p className="text-slate-500">Loading history records...</p>
                </div>
            ) : filteredHistory.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-20 bg-white rounded-xl border border-slate-200 shadow-sm">
                    <div className="p-4 bg-slate-50 rounded-full mb-4">
                        <Clock size={32} className="text-slate-400" />
                    </div>
                    <h3 className="text-lg font-medium text-slate-700">No History Found</h3>
                    <p className="text-slate-500 text-sm mt-1 max-w-md text-center">
                        No analysis sessions have been recorded yet. Start a new analysis to see it appear here.
                    </p>
                    <Link href="/dashboard/analysis" className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors">
                        Start New Analysis
                    </Link>
                </div>
            ) : (
                <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-slate-50 text-slate-500 font-medium border-b border-slate-200">
                                <tr>
                                    <th className="px-6 py-4">Session ID</th>
                                    <th className="px-6 py-4">Date & Time</th>
                                    <th className="px-6 py-4">Risk Level</th>
                                    <th className="px-6 py-4">Findings</th>
                                    <th className="px-6 py-4">Confidence</th>
                                    <th className="px-6 py-4 text-right">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {filteredHistory.map((item) => (
                                    <tr key={item.session_id} className="hover:bg-slate-50 transition-colors">
                                        <td className="px-6 py-4">
                                            <div className="font-mono text-xs font-medium text-slate-600 bg-slate-100 px-2 py-1 rounded w-fit">
                                                {item.session_id}
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-slate-600">
                                            {new Date(item.timestamp).toLocaleString(undefined, {
                                                year: 'numeric', month: 'short', day: 'numeric',
                                                hour: '2-digit', minute: '2-digit'
                                            })}
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className={`px-2 py-1 rounded text-xs font-bold uppercase border ${getRiskColor(item.overall_assessment?.risk_level)}`}>
                                                {item.overall_assessment?.risk_level || 'UNKNOWN'}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="flex flex-col gap-1">
                                                {item.tampering_analysis?.tampering_assessment?.is_tampering_detected ? (
                                                    <span className="flex items-center gap-1.5 text-red-600 font-medium">
                                                        <AlertTriangle size={14} /> Tampering Detected
                                                    </span>
                                                ) : (
                                                    <span className="flex items-center gap-1.5 text-green-600 font-medium">
                                                        <CheckCircle size={14} /> No Anomalies
                                                    </span>
                                                )}
                                                <span className="text-xs text-slate-400">
                                                    {Object.keys(item.expert_results || {}).length} Experts Consulted
                                                </span>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-2">
                                                <div className="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                                                    <div
                                                        className={`h-full rounded-full ${item.overall_assessment?.confidence > 0.8 ? 'bg-green-500' : 'bg-blue-500'}`}
                                                        style={{ width: `${(item.overall_assessment?.confidence || 0) * 100}%` }}
                                                    ></div>
                                                </div>
                                                <span className="text-xs font-medium text-slate-600">
                                                    {Math.round((item.overall_assessment?.confidence || 0) * 100)}%
                                                </span>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <button className="p-2 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors group">
                                                <Eye size={18} />
                                                <span className="sr-only">View Details</span>
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    <div className="p-4 border-t border-slate-200 bg-slate-50 flex justify-between items-center text-xs text-slate-500">
                        <span>Showing {filteredHistory.length} of {history.length} records</span>
                        <div className="flex gap-2">
                            <button disabled className="px-3 py-1 border border-slate-200 rounded bg-white text-slate-300 cursor-not-allowed">Previous</button>
                            <button disabled className="px-3 py-1 border border-slate-200 rounded bg-white text-slate-300 cursor-not-allowed">Next</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

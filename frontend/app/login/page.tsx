"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Shield, ArrowRight } from 'lucide-react';

export default function LoginPage() {
    const router = useRouter();
    const [loading, setLoading] = useState(false);

    const handleLogin = (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        // Simulate delay
        setTimeout(() => {
            router.push('/dashboard/overview');
        }, 1000);
    };

    return (
        <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
            <div className="w-full max-w-md bg-white rounded-lg shadow-md overflow-hidden border-t-4 border-govt-navy">
                <div className="p-8 text-center border-b border-gray-100 flex flex-col items-center">
                    <div className="flex items-center gap-6 mb-6">
                        <img src="/assets/govt_logo.svg" alt="Govt of India" className="h-16 w-auto" />
                        <img src="/assets/railway_logo.png" alt="Indian Railways" className="h-16 w-auto" />
                    </div>
                    <h1 className="text-2xl font-bold text-govt-navy">Indian Railways</h1>
                    <p className="text-sm text-gray-500 uppercase tracking-wide mt-1">Government of India</p>
                    <h2 className="mt-6 text-lg font-semibold text-gray-800">Tampering Detection System</h2>
                    <p className="text-xs text-gray-400">Authorized Personnel Only</p>
                </div>

                <form onSubmit={handleLogin} className="p-8 space-y-6">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Employee ID</label>
                        <input
                            type="text"
                            defaultValue="EMP-8291"
                            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-govt-blue focus:border-govt-blue outline-none"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                        <input
                            type="password"
                            defaultValue="password"
                            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-govt-blue focus:border-govt-blue outline-none"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Department</label>
                        <select className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-govt-blue focus:border-govt-blue outline-none bg-white">
                            <option>Railway Protection Force (RPF)</option>
                            <option>Signal & Telecommunication</option>
                            <option>Track Maintenance</option>
                            <option>Central Control Room</option>
                        </select>
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-govt-navy hover:bg-govt-blue text-white font-medium py-2 rounded-md transition-colors flex items-center justify-center gap-2 group"
                    >
                        {loading ? 'Authenticating...' : 'Secure Authorization'}
                        {!loading && <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />}
                    </button>
                </form>

                <div className="bg-gray-50 px-8 py-4 text-center border-t border-gray-100">
                    <p className="text-xs text-gray-400">© 2026 Government of India. All rights reserved.</p>
                    <p className="text-[10px] text-gray-300 mt-1">System Version 2.4.1 | Secure Connection</p>
                </div>
            </div>
        </div>
    );
}

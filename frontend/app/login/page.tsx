"use client";

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Shield, ArrowRight } from 'lucide-react';
import { GoogleLogin } from '@react-oauth/google';
import { jwtDecode } from "jwt-decode";
import { useAuth } from '@/contexts/AuthContext';

export default function LoginPage() {
    const router = useRouter();
    const { login, user } = useAuth();
    const [loading, setLoading] = useState(false);

    const [role, setRole] = useState("Railway Protection Force (RPF)");

    // Auto-redirect if already logged in
    useEffect(() => {
        if (user) {
            router.push('/dashboard/overview');
        }
    }, [user, router]);

    const handleLogin = (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        // Simulator login for dev
        const mockUser = {
            name: "Demo User",
            email: "demo@railways.gov.in",
            picture: "",
            role: role
        };
        login("mock-token", mockUser);
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
                        <select
                            value={role}
                            onChange={(e) => setRole(e.target.value)}
                            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-govt-blue focus:border-govt-blue outline-none bg-white"
                        >
                            <option>Railway Protection Force (RPF)</option>

                            <option>Track Maintenance</option>
                            <option>Central Control Room</option>
                            <option>System Administrator</option>
                        </select>
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-govt-navy hover:bg-govt-blue text-white font-medium py-2 rounded-md transition-colors flex items-center justify-center gap-2 group mb-4"
                    >
                        {loading ? 'Authenticating...' : 'Secure Authorization'}
                        {!loading && <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />}
                    </button>

                    <div className="relative">
                        <div className="absolute inset-0 flex items-center">
                            <span className="w-full border-t border-gray-300" />
                        </div>
                        <div className="relative flex justify-center text-xs uppercase">
                            <span className="bg-white px-2 text-gray-500">Or continue with</span>
                        </div>
                    </div>

                    <div className="flex justify-center mt-4">
                        <GoogleLogin
                            onSuccess={credentialResponse => {
                                console.log(credentialResponse);
                                if (credentialResponse.credential) {
                                    setLoading(true);
                                    try {
                                        const decoded: any = jwtDecode(credentialResponse.credential);
                                        const userData = {
                                            name: decoded.name,
                                            email: decoded.email,
                                            picture: decoded.picture,
                                            role: role
                                        };
                                        login(credentialResponse.credential, userData);
                                    } catch (e) {
                                        console.error("Login failed", e);
                                        setLoading(false);
                                    }
                                }
                            }}
                            onError={() => {
                                console.log('Login Failed');
                            }}
                            theme="outline"
                            size="large"
                            width="250"
                        />
                    </div>
                </form>

                <div className="bg-gray-50 px-8 py-4 text-center border-t border-gray-100">
                    <p className="text-xs text-gray-400">© 2026 Government of India. All rights reserved.</p>
                    <p className="text-[10px] text-gray-300 mt-1">System Version 2.4.1 | Secure Connection</p>
                </div>
            </div>
        </div>
    );
}

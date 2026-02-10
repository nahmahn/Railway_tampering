"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LayoutDashboard, Activity, FileText, MessageSquare, LogOut, Map, History } from 'lucide-react';
import clsx from 'clsx';

const navItems = [
    { name: 'Overview', href: '/dashboard/overview', icon: LayoutDashboard },
    { name: 'Live Monitoring', href: '/dashboard/monitoring', icon: Activity },
    { name: 'Map View', href: '/dashboard/map-view', icon: Map },
    { name: 'Analysis & Reports', href: '/dashboard/analysis', icon: FileText },
    { name: 'History', href: '/dashboard/history', icon: History },
    { name: 'NLP Assistant', href: '/dashboard/chatbot', icon: MessageSquare },
];

export default function Sidebar() {
    const pathname = usePathname();

    return (
        <aside className="w-64 bg-govt-navy text-white flex flex-col h-[calc(100vh-80px)] sticky top-20">
            <div className="p-4 border-b border-govt-blue">
                <p className="text-xs text-gray-300 uppercase font-semibold tracking-wider">Main Navigation</p>
            </div>

            <nav className="flex-1 py-4">
                <ul className="space-y-1">
                    {navItems.map((item) => {
                        const isActive = pathname.startsWith(item.href);
                        return (
                            <li key={item.href}>
                                <Link
                                    href={item.href}
                                    className={clsx(
                                        "flex items-center gap-3 px-6 py-3 text-sm font-medium transition-colors",
                                        isActive
                                            ? "bg-govt-blue border-r-4 border-govt-orange text-white"
                                            : "text-gray-300 hover:bg-white/10 hover:text-white"
                                    )}
                                >
                                    <item.icon size={18} />
                                    {item.name}
                                </Link>
                            </li>
                        );
                    })}
                </ul>
            </nav>

            <div className="p-4 border-t border-govt-blue">
                <Link
                    href="/login"
                    className="flex items-center gap-3 px-4 py-2 text-sm text-red-300 hover:text-red-100 transition-colors"
                >
                    <LogOut size={18} />
                    Logout System
                </Link>
                <div className="mt-4 text-xs text-gray-400 text-center">
                    v1.0.4-rc
                </div>
            </div>
        </aside>
    );
}

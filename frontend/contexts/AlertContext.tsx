"use client";

import React, { createContext, useContext, useEffect, useState, ReactNode, useRef } from 'react';
import { api } from '@/services/api';

export interface Alert {
    alert_id: string;
    timestamp: string;
    severity: 'info' | 'warning' | 'high' | 'critical';
    source: string;
    title: string;
    message: string;
    action_required: boolean;
    acknowledged: boolean;
}

interface AlertContextType {
    alerts: Alert[];
    isConnected: boolean;
    activeAlertCount: number;
    acknowledgeAlert: (alertId: string) => void;
}

const AlertContext = createContext<AlertContextType | undefined>(undefined);


export function AlertProvider({ children }: { children: ReactNode }) {
    const [alerts, setAlerts] = useState<Alert[]>([]);
    const [isConnected, setIsConnected] = useState(false);
    const ws = useRef<WebSocket | null>(null);

    // Fetch initial active alerts from history
    useEffect(() => {
        const fetchInitialAlerts = async () => {
            try {
                const history = await api.getHistory();
                console.log("Fetched history for alerts:", history);

                // Each DB session = 1 active alert on the dashboard
                const activeAlerts: Alert[] = [];

                history.forEach((session: any) => {
                    const riskLevel = session.overall_assessment?.risk_level?.toLowerCase() || 'warning';
                    activeAlerts.push({
                        alert_id: session.session_id || session.id,
                        timestamp: session.timestamp,
                        severity: (riskLevel === 'high' || riskLevel === 'critical') ? riskLevel : 'warning',
                        source: 'CombinedExpert',
                        title: session.alerts?.[0]?.title || "Analysis Complete",
                        message: `Session ${session.session_id || session.id} - Risk: ${riskLevel}`,
                        action_required: riskLevel === 'high' || riskLevel === 'critical',
                        acknowledged: false
                    });
                });

                if (activeAlerts.length > 0) {
                    setAlerts(prev => {
                        // Filter out duplicates if any (based on ID)
                        const existingIds = new Set(prev.map(a => a.alert_id));
                        const newUnique = activeAlerts.filter(a => !existingIds.has(a.alert_id));
                        return [...newUnique, ...prev];
                    });
                }
            } catch (error) {
                console.error("Failed to fetch initial alerts:", error);
            }
        };

        fetchInitialAlerts();
    }, []);

    useEffect(() => {
        // Connect to WebSocket
        const connect = () => {
            const socket = new WebSocket('ws://localhost:8000/ws/alerts');

            socket.onopen = () => {
                console.log('✅ Connected to alert stream');
                setIsConnected(true);
            };

            socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'alert') {
                        setAlerts(prev => [data.data, ...prev]);
                    } else if (data.type === 'connected') {
                        console.log('Server confirmed connection:', data.message);
                    }
                } catch (e) {
                    console.error('Failed to parse alert:', e);
                }
            };

            socket.onclose = () => {
                console.log('❌ Disconnected from alert stream');
                setIsConnected(false);
                // Try to reconnect in 5 seconds
                setTimeout(connect, 5000);
            };

            socket.onerror = (error) => {
                console.error('WebSocket error:', error);
                socket.close();
            };

            ws.current = socket;
        };

        connect();

        return () => {
            if (ws.current) {
                ws.current.close();
            }
        };
    }, []);

    const acknowledgeAlert = (alertId: string) => {
        setAlerts(prev => prev.map(a =>
            a.alert_id === alertId ? { ...a, acknowledged: true } : a
        ));
        // Optionally send ack to backend here
    };

    const activeAlertCount = alerts.filter(a => !a.acknowledged).length;

    return (
        <AlertContext.Provider value={{ alerts, isConnected, activeAlertCount, acknowledgeAlert }}>
            {children}
        </AlertContext.Provider>
    );
}

export function useAlerts() {
    const context = useContext(AlertContext);
    if (context === undefined) {
        throw new Error('useAlerts must be used within an AlertProvider');
    }
    return context;
}

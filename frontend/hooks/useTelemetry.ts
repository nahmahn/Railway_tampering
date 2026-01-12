import { useState, useEffect } from 'react';
import { TelemetryPackage } from '@/types/telemetry';

export function useTelemetry() {
    const [data, setData] = useState<TelemetryPackage[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetch('/dataset/data.json')
            .then(res => {
                if (!res.ok) {
                    throw new Error('Failed to fetch data');
                }
                return res.json();
            })
            .then(data => {
                setData(data);
                setLoading(false);
            })
            .catch(err => {
                console.error("Failed to load dataset:", err);
                setError(err.message);
                setLoading(false);
            });
    }, []);

    return { data, loading, error };
}
